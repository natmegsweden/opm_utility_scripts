"""Synthetic HPI contracts. Run: python -m unittest discover -s tests -v."""

import argparse
from concurrent.futures import Future
from contextlib import ExitStack
from copy import deepcopy
import io as stdio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

from opm_utility_scripts.hpi import _core as core
from opm_utility_scripts.hpi import check, coregister
from opm_utility_scripts.hpi._options import add_fit_options, fit_options
from opm_utility_scripts import io
from opm_utility_scripts.tests import test_hpi_versions


def synthetic_raw():
    names = ['mag1', 'mag2', 'grad1', 'hpiout1', 'hpiout2', 'hpiout3', 'hpiout4']
    info = mne.create_info(names, 1000, ['mag', 'mag', 'grad'] + ['misc'] * 4)
    for i in range(3):
        info['chs'][i]['loc'] = np.r_[0.01 * (i + 1), 0.02, 0.1, np.eye(3).ravel()]
    t = np.arange(12001) / 1000
    data = np.zeros((len(names), len(t)))
    for i in range(4):
        data[3 + i] = 0.001 * np.sin(2 * np.pi * 33 * t) * ((t >= 1) & (t <= 5))
    return mne.io.RawArray(data, info, verbose=False)


def amplitude_mock(raw, **kwargs):
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    names = [raw.ch_names[p] for p in picks]
    proj = mne.Projection(
        data=dict(col_names=names, row_names=None, data=np.zeros((1, len(names))),
                  nrow=1, ncol=len(names)), desc='no-op amplitude projector',
        kind=1, active=False)
    return {'times': np.array([1.]), 'slopes': np.ones((1, 1, len(names))),
            'proj': proj}


def fieldline_info():
    """Real radial FieldLine geometry, with deterministic nonuniform coverage."""
    info = mne.create_info([f'OPM{i:02}' for i in range(20)], 1000, 'mag')
    directions = np.random.default_rng(7).normal(size=(20, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for ch, z in zip(info['chs'], directions):
        x = np.cross(z, [0., 0., 1.])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ch['loc'] = np.r_[.12 * z, x, y, z]
        ch['coil_type'] = mne.io.constants.FIFF.FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1
    return info


def assert_projectors_equal(test, actual, expected):
    test.assertEqual(len(actual), len(expected))
    for got, want in zip(actual, expected):
        test.assertEqual(got['desc'], want['desc'])
        test.assertEqual(got['active'], want['active'])
        test.assertEqual(got['kind'], want['kind'])
        test.assertEqual(got['data']['col_names'], want['data']['col_names'])
        test.assertEqual(got['data'].keys(), want['data'].keys())
        test.assertEqual(got['data'].get('row_names'), want['data'].get('row_names'))
        test.assertEqual(got['data']['nrow'], want['data']['nrow'])
        test.assertEqual(got['data']['ncol'], want['data']['ncol'])
        np.testing.assert_array_equal(got['data']['data'], want['data']['data'])


POINTS = np.array([[0, 0, 0.04], [0.03, 0, 0.04], [0, 0.03, 0.04], [0.03, 0.03, 0.04]])
POL = dict(source='fif', hpi_orig=POINTS, nasion=np.array([0, .1, 0]),
           lpa=np.array([-.07, 0, 0]), rpa=np.array([.07, 0, 0]),
           extra_pts=np.empty((0, 3)), eeg_pts=np.empty((0, 3)))


class PolicyTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_coregister_sidecars_default_on_check_remains_opt_in(self):
        with patch('sys.argv', ['coregister']):
            args = coregister._parse_args()
            self.assertEqual(args.settings_json, '')
            self.assertEqual(fit_options(args)['localization_grid'], 'medium')
            self.assertEqual(args.optimization, 'rigid_gof')
        with patch('sys.argv', ['coregister', '--localization-grid', 'mne_stock']):
            self.assertEqual(fit_options(coregister._parse_args())['localization_grid'], 'mne_stock')
        with patch('sys.argv', ['check']):
            args = check._parse_args()
            self.assertIsNone(args.settings_json)
            self.assertEqual(args.localization_grid, 'medium')
            self.assertEqual(args.optimization, 'rigid_gof')

    def test_invalid_options_fail_before_loading(self):
        for kwargs in ({'bad_channel_policy': 'other'},
                       {'activation_window_s': 0}, {'gof_limit': float('nan')},
                       {'gof_comparison': 'other'}, {'matching_strategy': 'other'},
                       {'localization_grid': 'other'},
                       {'unique_matches': None}, {'optim': 'legacy'},
                       {'bad_channel_policy': 'reference'},
                       {'center_matching': True, 'matching_strategy': 'coordinate_nearest'}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                core.fit_hpi('not-a-file', POL, 33, **kwargs)
        for freq in (0, -1, 500, float('nan')):
            with self.assertRaises(ValueError):
                core.fit_hpi_amplitudes('not-a-file', freq)
        with patch.object(core, 'fit_hpi_amplitudes') as amplitudes:
            with self.assertRaisesRegex(ValueError, 'localization_grid'):
                core.fit_hpi('not-a-file', POL, 33, localization_grid='unknown')
            amplitudes.assert_not_called()

    def test_gof_comparison(self):
        values = [.94, .95, .96, np.nan, np.inf]
        np.testing.assert_array_equal(core._gof_mask(values, .95, 'inclusive'),
                                      [False, True, True, False, False])
        np.testing.assert_array_equal(core._gof_mask(values, .95, 'strict'),
                                      [False, False, True, False, False])
        verdict = check._recommendation([.95, .95, .96], gof_comparison='strict')
        self.assertEqual(verdict[0], 'POOR')
        self.assertEqual(check._gof_color(.95, comparison='strict'), 'darkorange')

    def test_registration_recommendation_good_and_field_only_concerns(self):
        for fields, distances, expected in (
                ([.99] * 3, [1.] * 3, 'OK'),
                ([.972, 1., .560], [3.38, .43, 3.69], 'CHECK REGISTRATION'),
                ([.99, .99, .4], [1.] * 3, 'CHECK REGISTRATION')):
            with self.subTest(fields=fields):
                verdict = check._recommendation([.99] * 3, distances,
                                                [True] * 3, fields)
                self.assertEqual(verdict[0], 'OK')
                self.assertEqual(verdict[3], expected)
                if expected != 'OK':
                    self.assertIn('fail polhemus-position GOF', ' '.join(verdict[5]))
                    self.assertIn('field-model/registration consistency', ' '.join(verdict[5]))
                    self.assertIn('corroborating', ' '.join(verdict[5]))

    def test_registration_unavailable_fields_and_invalid_residuals(self):
        for fields in (None, [], [.99], .99, np.array([[.99] * 3]), [np.nan] * 3):
            with self.subTest(fields=fields):
                verdict = check._recommendation([.99] * 3, [1.] * 3,
                                                [True] * 3, fields)
                self.assertEqual(verdict[3], 'CHECK REGISTRATION')
                self.assertIn('Field GOF assessment unavailable', ' '.join(verdict[5]))
                self.assertNotIn('fail polhemus-position GOF', ' '.join(verdict[5]))
        for invalid in (np.nan, np.inf, -1.):
            with self.subTest(residual=invalid):
                verdict = check._recommendation([.99] * 3, [1., 1., invalid],
                                                [True] * 3, [.99] * 3)
                self.assertEqual(verdict[3], 'CHECK REGISTRATION')
                self.assertIn('Residual assessment unavailable', ' '.join(verdict[5]))

    def test_registration_requires_joint_passes_and_preserves_thresholds(self):
        verdict = check._recommendation([.99] * 4, [1., 1., 1., 6.],
                                        [True] * 4, [.4, .99, .99, .99])
        self.assertEqual(verdict[3], 'CHECK REGISTRATION')
        for comparison, expected in (('inclusive', 'OK'), ('strict', 'CHECK REGISTRATION')):
            with self.subTest(comparison=comparison):
                verdict = check._recommendation([.99] * 3, [4.999] * 3,
                                                [True] * 3, [.95] * 3,
                                                gof_comparison=comparison)
                self.assertEqual(verdict[3], expected)
        self.assertNotEqual(check._recommendation([.99] * 3, [5.] * 3,
                                                  [True] * 3, [.99] * 3)[3], 'OK')
        # Three jointly passing coils still suffice when a fourth is unavailable.
        self.assertEqual(check._recommendation([.99] * 4, [1., 1., 1., np.nan],
                                               [True] * 4, [.99, .99, .99, np.nan])[3], 'OK')

    def test_activation_bounds(self):
        self.assertEqual(core._activation_bounds(1, 5, 10, 1000, 2, 'coil'), (2, 4))
        self.assertEqual(core._activation_bounds(1, 5, 10, 1000, 3, 'coil'), (1.5, 4.5))
        for args in ((1, 2, 10, 1000, 2), (0, 1.998, 10, 1000, 2),
                     (8.002, 10, 10, 1000, 2)):
            with self.assertRaises(ValueError):
                core._activation_bounds(*args, 'coil')

    def test_noise_resolution(self):
        raw = synthetic_raw()
        with patch.object(core, 'find_bads', return_value=(['mag1'], None, [])) as detector:
            self.assertEqual(core._detect_noise(raw, raw, 33, return_policy=True)[2], 'reference')
            self.assertEqual(core._detect_noise(raw, None, 33, return_policy=True)[2], 'hpi_tail')
            self.assertEqual(core._detect_noise(raw, raw, 33, bad_channel_policy='none',
                                                return_policy=True), ([], None, 'none'))
            self.assertEqual(detector.call_count, 2)
            with self.assertWarnsRegex(RuntimeWarning, 'clean five-second'):
                self.assertEqual(core._detect_noise(raw, None, 33, peak_tlast_override=10,
                                                    return_policy=True)[2], 'none')
        raw._data[3:] = 0
        with patch.object(core, 'get_hpi_output_channels', return_value=([], [])), \
                self.assertWarns(RuntimeWarning):
            self.assertEqual(core._detect_noise(raw, None, 33, return_policy=True)[2], 'none')

    def test_opm_magnetometer_selection_and_invalid_geometry(self):
        raw = synthetic_raw()
        core._select_sensors(raw)
        self.assertEqual(raw.ch_names[:2], ['mag1', 'mag2'])
        self.assertNotIn('grad1', raw.ch_names)
        self.assertIn('hpiout1', raw.ch_names)
        raw = synthetic_raw()
        raw.info['chs'][2]['loc'][0] = np.nan
        core._select_sensors(raw)
        self.assertNotIn('grad1', raw.ch_names)

    def test_localization_grid_profiles(self):
        coils = [np.array([[0.0, 0.0, 0.08], [0.0, 0.0, -0.10]])]
        guesses = np.array([[0.0, 0.0, 0.075], [0.0, 0.0, -0.09],
                            [0.0, 0.0, 0.09], [0.0, 0.0, -0.11]])
        for mode, expected_radius, expected_grid, expected_mindist in (
                ('coarse', 0.10, 0.01, 0.005),
                ('medium', 0.10, 0.005, 0.002),
                ('fine', 0.10, 0.002, 0.001),
                ('mne_stock', 0.08, 0.01, 0.005)):
            with self.subTest(mode=mode), patch.object(
                    core, '_make_guesses', return_value=[{'rr': guesses}]) as make:
                result = core._make_opm_guesses(coils, mode)
            sphere = make.call_args.args[0]
            self.assertAlmostEqual(sphere['layers'][0]['rad'], expected_radius)
            self.assertEqual(make.call_args.args[1:], (expected_grid, 0.0, expected_mindist))
            if mode != 'mne_stock':
                np.testing.assert_array_equal(result, guesses[:2])
            else:
                np.testing.assert_array_equal(result, guesses)

    def test_localization_grid_uses_nearest_coil_radius(self):
        coils = [np.array([[.08, 0, 0], [0, .10, 0]])]
        guesses = np.array([[.075, 0, 0], [0, .09, 0], [.09, 0, 0],
                            [0, .11, 0], [.08, 0, 0], [0, 0, 0]])
        for mode in ('coarse', 'medium', 'fine'):
            with self.subTest(mode=mode), patch.object(
                    core, '_make_guesses', return_value=[{'rr': guesses}]):
                result = core._make_opm_guesses(coils, mode)
            # Retain the outer-sensor guess beyond the global minimum radius,
            # but reject a same-radius guess whose nearest sensor is inward.
            # Equality at the local radius and the origin remain included.
            np.testing.assert_array_equal(result, guesses[[0, 1, 4, 5]])

    def test_localization_grid_empty_results(self):
        coils = [np.array([[.08, 0, 0], [0, .10, 0]])]
        for mode in ('coarse', 'medium', 'fine'):
            for guesses in (np.empty((0, 3)), np.array([[.09, 0, 0]])):
                with self.subTest(mode=mode, guesses=guesses), patch.object(
                        core, '_make_guesses', return_value=[{'rr': guesses}]):
                    result = core._make_opm_guesses(coils, mode)
                self.assertEqual(result.shape, (0, 3))

    def test_grids_and_default_match_nearest_coil_construction(self):
        coils = [np.array([[0, 0, .08], [.061, 0, .08]])]
        radii = np.linalg.norm(coils[0], axis=1)
        sphere = mne.bem.ConductorModel(
            layers=[dict(rad=radii.max())], r0=np.zeros(3), is_sphere=True)
        for mode, spacing, clearance in (
                ('coarse', .01, .005), ('medium', .005, .002), ('fine', .002, .001)):
            with self.subTest(mode=mode):
                expected = mne.dipole._make_guesses(
                    sphere, spacing, 0., clearance, verbose=False)[0]['rr']
                # Independent brute-force oracle for the nearest integration point.
                distances = np.linalg.norm(expected[:, np.newaxis] - coils[0], axis=2)
                nearest = distances.argmin(axis=1)
                expected = expected[np.linalg.norm(expected, axis=1) <= radii[nearest]]
                np.testing.assert_array_equal(core._make_opm_guesses(coils, mode), expected)
                if mode == 'medium':
                    np.testing.assert_array_equal(core._make_opm_guesses(coils), expected)
        # Unequal radii distinguish stock's clearance from coarse's clipping.
        self.assertGreater(len(core._make_opm_guesses(coils, 'coarse')),
                           len(core._make_opm_guesses(coils, 'mne_stock')))
        with self.assertRaisesRegex(ValueError, 'localization_grid'):
            core._make_opm_guesses(coils, 'unknown')

    def test_mne_stock_exactly_matches_installed_stock_guess_construction(self):
        # Stop stock MNE immediately after guess construction: exercise its real
        # construction without running a different localizer or fitting data.
        class GuessesCaptured(Exception):
            pass

        info = mne.create_info(['mag1', 'mag2'], 1000, ['mag', 'mag'])
        info['dev_head_t'] = mne.transforms.Transform('meg', 'head')
        proj = mne.Projection(data=dict(col_names=info.ch_names, row_names=None,
                                       data=np.ones((1, 2)), nrow=1, ncol=2),
                              desc='test', kind=1, active=False)
        amplitudes = dict(proj=proj)
        geometries = (
            np.array([[0, 0, .08], [0, 0, .10]]),
            np.array([[.08, 0, 0], [0, -.08, 0], [0, 0, -.08]]),
            np.array([[.031, -.042, .057], [-.09, .015, .034], [.01, .11, -.02]]),
            # Multiple integration points per sensor; the inward point, not
            # a sensor centre or the outermost point, controls the sphere.
            np.array([[.079, 0, .002], [.081, 0, -.002],
                      [0, -.099, .001], [0, -.101, -.001]]),
        )
        for positions in geometries:
            with self.subTest(positions=positions), ExitStack() as stack:
                stack.enter_context(patch.object(mne.chpi, '_check_chpi_param'))
                stack.enter_context(patch.object(mne.chpi, '_create_meg_coils'))
                stack.enter_context(patch.object(mne.chpi, '_concatenate_coils',
                                                 return_value=[positions]))
                stack.enter_context(patch.object(mne.chpi, 'make_ad_hoc_cov'))
                stack.enter_context(patch.object(mne.chpi, 'compute_whitener',
                                                 return_value=(np.eye(2), None)))
                field = stack.enter_context(patch.object(
                    mne.chpi, '_magnetic_dipole_field_vec', side_effect=GuessesCaptured))
                with self.assertRaises(GuessesCaptured):
                    mne.chpi.compute_chpi_locs(info, amplitudes, verbose=False)
                expected = field.call_args.args[0]
                actual = core._make_opm_guesses([positions], 'mne_stock')
                self.assertGreater(len(actual), 0)
                np.testing.assert_array_equal(actual, expected)

    def test_reference_statistics_population_and_no_mutation(self):
        raw = synthetic_raw()
        original_names = raw.ch_names.copy()
        raw._data[:3] = np.random.default_rng(4).normal(scale=1e-13, size=raw._data[:3].shape)
        bads, fig, names = core.find_bads(raw, 33, match_channels=['mag2', 'mag1'])
        self.assertEqual(names, ['mag1', 'mag2'])
        self.assertEqual(bads, [])
        self.assertIsNotNone(fig)
        self.assertEqual(raw.ch_names, original_names)
        with self.assertRaisesRegex(ValueError, 'no usable sensors'):
            core.find_bads(raw, 33, match_channels=['missing'])

    def test_matching_strategies_and_safety(self):
        np.testing.assert_array_equal(core._match_points(POINTS + .1, POINTS,
                                                        'centroid_nearest', True), np.arange(4))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            core._match_points(POINTS + .1, POINTS, 'coordinate_nearest', True)
        # Repeated assignment is allowed only with sufficient remaining geometry.
        dev = np.vstack([POINTS[:3], POINTS[0] + [.001, .001, 0]])
        np.testing.assert_array_equal(core._match_points(dev, POINTS[:3], 'coordinate_nearest', False),
                                      [0, 1, 2, 0])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            core._match_points(POINTS + .1, POINTS, 'coordinate_nearest', False)
        line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'non-collinear'):
            core._match_points(line, line, 'centroid_nearest', True)
        with self.assertRaises(ValueError):
            core._match_points(POINTS[:2], POINTS, 'centroid_nearest', True)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(plt.close, 'all')
        self.stack.enter_context(patch.object(core, 'compute_chpi_amplitudes', side_effect=amplitude_mock))
        self.stack.enter_context(patch('sys.stdout', new=stdio.StringIO()))

    def mock_localization(self, points=POINTS, gofs=None):
        self.stack.enter_context(patch.object(core, 'compute_chpi_opm_locs',
                                             return_value={'rrs': [points], 'gofs': [gofs if gofs is not None else [.99] * len(points)]}))
        self.stack.enter_context(patch.object(core, 'compute_whitener', return_value=(np.eye(3), None)))
        self.stack.enter_context(patch.object(core, '_create_meg_coils', return_value=None))
        self.stack.enter_context(patch.object(core, '_concatenate_coils', return_value=None))
        self.stack.enter_context(patch.object(core, '_gof_at_fixed_pos', return_value=.98))

    def test_amplitude_window_population_and_missing_peak_indices(self):
        raw = synthetic_raw()
        raw._data[4] *= 1e-3  # non-flat, but below peak detection threshold
        expected = ['mag1', 'mag2']
        amp = core.fit_hpi_amplitudes(raw, 33, activation_window_s=3)
        self.assertEqual(amp['slope_ch_names'], expected)
        self.assertEqual(amp['slope'].shape, (3, len(expected)))
        np.testing.assert_array_equal(amp['original_coil_indices'], [0, 2, 3])
        self.assertEqual(amp['hpi_names'], ['hpiout1', 'hpiout3', 'hpiout4'])
        self.assertEqual(amp['coil_amplitudes']['proj']['data']['col_names'], expected)
        self.assertEqual(amp['settings']['activation_window']['duration_s'], 3)
        self.assertEqual(amp['settings']['sensor_selection'], 'opm_magnetometers')
        self.assertEqual(len(raw.ch_names), 7)  # no caller mutation

    def test_noise_exclusion_before_amplitudes(self):
        with patch.object(core, 'find_bads', return_value=(['mag2'], None, ['mag1', 'mag2'])):
            amp = core.fit_hpi_amplitudes(synthetic_raw(), 33, bad_channel_policy='auto')
        self.assertEqual(amp['slope_ch_names'], ['mag1'])
        self.assertEqual(amp['bads'], ['mag2'])
        self.assertEqual(amp['settings']['bad_channel_policy']['effective'], 'hpi_tail')

    def test_projector_alignment_error_and_no_peaks(self):
        bad_result = amplitude_mock(synthetic_raw())
        bad_result['proj']['data']['col_names'] = ['wrong']
        with patch.object(core, 'compute_chpi_amplitudes', return_value=bad_result):
            with self.assertRaisesRegex(ValueError, 'projector channel order'):
                core.fit_hpi_amplitudes(synthetic_raw(), 33)
        raw = synthetic_raw()
        raw._data[3:] *= 1e-3
        with self.assertRaisesRegex(ValueError, 'No HPI coils'):
            core.fit_hpi_amplitudes(raw, 33)

    def test_full_fit_schema_alias_and_sidecar(self):
        self.mock_localization()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none', optim='none', settings_json=path)
            np.testing.assert_allclose(fit['dev_to_head_trans']['trans'], np.eye(4), atol=1e-12)
            self.assertEqual(fit['slope_ch_names'], fit['raw_for_topomap'].ch_names)
            keys = {'dev_to_head_trans', 'hpi_dev', 'hpi_gofs', 'hpi_orig', 'hpi_names',
                    'nasion', 'lpa', 'rpa', 'pol_info', 'extra_pts', 'eeg_pts', 'slope',
                     'slope_ch_names', 'raw_for_topomap', 'dist', 'include_hpis', 'tree_indices',
                     'pol_gofs', 'bads', 'bads_fig', 'optim', 'opt_status', 'opt_success',
                     'opt_delta_deg', 'opt_delta_mm'}
            self.assertTrue(keys <= fit.keys())
            payload = json.loads(path.read_text())
            self.assertEqual(payload['schema_version'], 1)
            self.assertEqual(payload['settings']['matching']['strategy'], 'centroid_nearest')
            self.assertEqual(payload['settings']['localization_grid'], 'medium')
            self.assertEqual(payload['results_summary']['included_coils'], 4)
            self.assertIsNone(payload['results_summary']['optimizer_success'])
            fit_results = payload['fit_results']
            np.testing.assert_allclose(fit_results['transform_device_to_head']['matrix'], np.eye(4))
            self.assertEqual(fit_results['transform_device_to_head']['translation_unit'], 'm')
            self.assertEqual(len(fit_results['coils']), 4)
            self.assertTrue(fit_results['coils'][0]['included'])
            self.assertAlmostEqual(fit_results['coils'][0]['dipole_gof'], .99)
            self.assertAlmostEqual(fit_results['coils'][0]['polhemus_gof'], .98)
            self.assertIn('postfit_residual_m', fit_results['coils'][0])
            self.assertIsNone(fit_results['optimizer']['success'])
            self.assertIsNone(fit_results['optimizer']['delta_deg'])
            self.assertIsNone(fit_results['optimizer']['delta_mm'])
            core.write_settings_json(path, fit, hpifile='/private/subject/hpi.fif')
            self.assertEqual(json.loads(path.read_text())['inputs']['hpi'], 'hpi.fif')
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_center_matching_compatibility_alias(self):
        self.mock_localization()
        fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                           optim='none', center_matching=np.bool_(False))
        self.assertEqual(fit['settings']['matching']['strategy'], 'coordinate_nearest')

    def test_original_indices_seed_localization(self):
        raw = synthetic_raw()
        raw._data[4] *= 1e-3
        self.mock_localization(POINTS[[0, 2, 3]])
        fit = core.fit_hpi(raw, POL, 33, bad_channel_policy='none', optim='none')
        np.testing.assert_array_equal(fit['original_coil_indices'], [0, 2, 3])
        np.testing.assert_array_equal(fit['tree_indices'], [0, 2, 3])
        info = core.compute_chpi_opm_locs.call_args.args[0]
        self.assertEqual(core.compute_chpi_opm_locs.call_args.kwargs['localization_grid'], 'medium')
        dig_hpi = [d['r'] for d in info['dig'] if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_HPI]
        np.testing.assert_allclose(dig_hpi, POINTS[[0, 2, 3]])

    def test_fit_hpi_accepts_mne_stock_and_preserves_refinement_default(self):
        self.mock_localization(gofs=[.95, .99, .99, .99])
        success = SimpleNamespace(status=0, success=True, x=np.zeros(6), fun=-.98)
        with patch.object(core, 'minimize', return_value=success) as optimizer, \
                patch.object(mne.chpi, 'compute_chpi_locs',
                             side_effect=AssertionError('Must use the OPM localizer')):
            fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                               localization_grid='mne_stock')
        self.assertEqual(core.compute_chpi_opm_locs.call_args.kwargs['localization_grid'],
                         'mne_stock')
        self.assertEqual(fit['settings']['localization_grid'], 'mne_stock')
        self.assertEqual(fit['optim'], 'rigid_gof')
        self.assertEqual(fit['settings']['transform_refinement']['method'], 'rigid_gof')
        np.testing.assert_array_equal(fit['include_hpis'], [True] * 4)
        optimizer.assert_called_once()

    def test_prepare_polhemus_for_localization_frame_conversion_and_validation(self):
        original_indices = np.array([0, 2, 3])
        hpi_names = ['hpi1', 'hpi3', 'hpi4']
        raw_fif = synthetic_raw()
        result_fif = core._prepare_polhemus_for_localization(
            POL, raw_fif, hpi_names, original_indices)
        np.testing.assert_allclose(result_fif[1], POINTS)
        fif_dig_hpi = [d['r'] for d in raw_fif.info['dig']
                       if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_HPI]
        np.testing.assert_allclose(fif_dig_hpi, POINTS[original_indices])

        json_pol = dict(POL, source='json')
        raw_json = synthetic_raw()
        result_json = core._prepare_polhemus_for_localization(
            json_pol, raw_json, hpi_names, original_indices)
        isotrak_to_head = core.get_ras_to_neuromag_trans(POL['nasion'], POL['lpa'], POL['rpa'])
        expected_head = core.apply_trans(isotrak_to_head, POINTS)
        np.testing.assert_allclose(result_json[1], expected_head)
        json_dig_hpi = [d['r'] for d in raw_json.info['dig']
                        if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_HPI]
        np.testing.assert_allclose(json_dig_hpi, expected_head[original_indices])

        too_few = dict(POL, hpi_orig=POINTS[:2])
        with self.assertRaisesRegex(ValueError, 'Polhemus has 2 HPI dig point'):
            core._prepare_polhemus_for_localization(
                too_few, synthetic_raw(), hpi_names, original_indices)

    def test_strict_threshold_skips_optimizer_with_three_accepted(self):
        self.mock_localization(gofs=[.95, .99, .99, .99])
        with patch.object(core, 'minimize') as optimizer, \
                self.assertWarnsRegex(RuntimeWarning, 'at least 4 are required'):
            fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                               optim='rigid_gof', gof_comparison='strict')
        optimizer.assert_not_called()
        np.testing.assert_array_equal(fit['include_hpis'], [False, True, True, True])
        self.assertEqual(fit['optim'], 'none')
        self.assertIsNone(fit['opt_success'])
        np.testing.assert_allclose(fit['dev_to_head_trans']['trans'], np.eye(4), atol=1e-12)

    def test_failed_optimizer_with_four_accepted_retains_initial_transform(self):
        self.mock_localization(gofs=[.96, .99, .99, .99])
        failure = SimpleNamespace(status=2, success=False, x=np.zeros(6), fun=-.98)
        with patch.object(core, 'minimize', return_value=failure) as optimizer, self.assertWarnsRegex(RuntimeWarning, 'Retaining'):
            fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                               optim='rigid_gof', gof_comparison='strict')
        optimizer.assert_called_once()
        np.testing.assert_array_equal(fit['include_hpis'], [True] * 4)
        self.assertFalse(fit['opt_success'])
        np.testing.assert_allclose(fit['dev_to_head_trans']['trans'], np.eye(4), atol=1e-12)

    def test_three_accepted_gate_preserves_geometry_diagnostics_and_sidecar(self):
        # Test detected count separately from accepted count; alias metadata is normalized.
        for detected in (3, 4):
            for requested in ('rigid', 'rigid_gof'):
                with self.subTest(detected=detected, requested=requested), ExitStack() as stack:
                    raw = synthetic_raw()
                    if detected == 3:
                        raw._data[6] *= 1e-3
                    points = POINTS[:detected] + [.002, -.003, .001]
                    self.mock_localization(points, [.99] * 3 + ([.4] if detected == 4 else []))
                    optimizer = stack.enter_context(patch.object(core, 'minimize'))
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'settings.json'
                        with self.assertWarnsRegex(RuntimeWarning, 'only 3 HPI coils'):
                            fit = core.fit_hpi(raw, POL, 33, bad_channel_policy='none',
                                               optim=requested, settings_json=path)
                        payload = json.loads(path.read_text())
                    optimizer.assert_not_called()
                    expected = np.eye(4)
                    expected[:3, 3] = [-.002, .003, -.001]
                    np.testing.assert_allclose(fit['dev_to_head_trans']['trans'], expected, atol=1e-12)
                    np.testing.assert_allclose(fit['dist'], 0., atol=1e-12)
                    np.testing.assert_allclose(fit['pol_gofs'], .98)
                    self.assertTrue(np.isfinite(fit['hpi_gofs']).all())
                    self.assertEqual(fit['optim'], 'none')
                    refinement = dict(requested_method='rigid_gof', method='none',
                                      skip_reason='fewer_than_four_accepted_coils',
                                      rotation_bound_deg=None, translation_bound_mm=None)
                    self.assertEqual(fit['settings']['transform_refinement'], refinement)
                    self.assertEqual(payload['settings']['transform_refinement'], refinement)
                    self.assertEqual(payload['results_summary']['included_coils'], 3)
                    self.assertIsNone(payload['results_summary']['optimizer_success'])
                    for key in ('opt_status', 'opt_success', 'opt_delta_deg', 'opt_delta_mm'):
                        self.assertIsNone(fit[key])
                    self.assertEqual(payload['fit_results']['optimizer'], dict(
                        method='none', status=None, success=None, delta_deg=None, delta_mm=None))

    def test_explicit_none_with_three_accepted_does_not_warn_about_refinement(self):
        raw = synthetic_raw()
        raw._data[6] *= 1e-3
        self.mock_localization(POINTS[:3])
        with patch.object(core, 'minimize') as optimizer, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit = core.fit_hpi(raw, POL, 33, bad_channel_policy='none', optim='none')
        optimizer.assert_not_called()
        self.assertFalse(any('refinement disabled' in str(w.message) for w in caught))
        self.assertEqual(fit['settings']['transform_refinement'], dict(
            requested_method='none', method='none', skip_reason=None,
            rotation_bound_deg=None, translation_bound_mm=None))

    def test_fewer_than_three_accepted_still_rejected(self):
        self.mock_localization(gofs=[.99, .99, .4, np.nan])
        with patch.object(core, 'minimize') as optimizer, \
                self.assertRaisesRegex(ValueError, 'Only 2 HPI coil.*At least 3'):
            core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none')
        optimizer.assert_not_called()

    def test_malformed_field_gofs_have_consistent_figure_and_cli_verdicts(self):
        self.mock_localization()
        fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none', optim='none')
        for fields in (None, [], [.99], .99, np.array([[.99] * 4]), [np.nan] * 4):
            with self.subTest(fields=fields):
                malformed = dict(fit, pol_gofs=fields)
                output = stdio.StringIO()
                with patch('sys.stdout', new=output):
                    check._print_diagnostics_full(malformed, detailed=True)
                fig = check._build_figure_full(malformed, detailed=True)
                text = ' '.join(t.get_text() for ax in fig.axes for t in ax.texts)
                self.assertIn('Polhemus registration: CHECK REGISTRATION', output.getvalue())
                self.assertIn('CHECK REGISTRATION', text)
                self.assertIn('unavailable', output.getvalue())
                self.assertIn('unavailable', text)
                plt.close(fig)

    def test_nonfinite_optimizer_and_success(self):
        self.mock_localization()
        for success, x, expected in [(True, np.full(6, np.nan), False), (True, np.zeros(6), True)]:
            result = SimpleNamespace(status=0, success=success, x=x, fun=-.98)
            with patch.object(core, 'minimize', return_value=result):
                if expected:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'settings.json'
                        fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                                           optim='rigid', settings_json=path)
                        optimizer = json.loads(path.read_text())['fit_results']['optimizer']
                    np.testing.assert_allclose(optimizer['delta_deg'], [0, 0, 0])
                    np.testing.assert_allclose(optimizer['delta_mm'], [0, 0, 0])
                    np.testing.assert_allclose(fit['opt_delta_deg'], [0, 0, 0])
                    np.testing.assert_allclose(fit['opt_delta_mm'], [0, 0, 0])
                else:
                    with self.assertWarns(RuntimeWarning):
                        fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none', optim='rigid')
                    self.assertTrue(np.all(~np.isfinite(fit['opt_delta_deg'])))
                    self.assertTrue(np.all(~np.isfinite(fit['opt_delta_mm'])))
            self.assertEqual(bool(fit['opt_success']), expected)
            self.assertEqual(fit['settings']['transform_refinement']['method'], 'rigid_gof')
            if success:
                self.assertEqual(fit['optim'], 'rigid_gof')

    def test_version_comparison_refuses_existing_outputs_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = test_hpi_versions._parse_args([
                '--data', 'subject_raw.fif', '--hpi', 'hpi_raw.fif', '--pol', 'pol.json',
                '--output-dir', tmp,
            ])
            self.assertEqual(args.localization_grid, 'fine')
            legacy_grid_args = test_hpi_versions._parse_args([
                '--data', 'subject_raw.fif', '--hpi', 'hpi_raw.fif', '--pol', 'pol.json',
                '--output-dir', tmp, '--localization-grid', 'legacy',
            ])
            self.assertEqual(legacy_grid_args.localization_grid, 'legacy')
            target = Path(tmp) / 'v0.1.0' / 'v0.1.0_alignment.png'
            target.parent.mkdir()
            target.touch()
            with self.assertRaisesRegex(FileExistsError, '--overwrite'):
                test_hpi_versions._check_output_collisions(args)

            overwrite_args = test_hpi_versions._parse_args([
                '--data', 'subject_raw.fif', '--hpi', 'hpi_raw.fif', '--pol', 'pol.json',
                '--output-dir', tmp, '--overwrite',
            ])
            self.assertTrue(overwrite_args.overwrite)
            test_hpi_versions._check_output_collisions(overwrite_args)

    def test_sidecar_failures_clear_and_existing_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing' / 'settings.json'
            with self.assertRaisesRegex(OSError, 'settings sidecar'):
                core.write_settings_json(path, {'settings': {}, 'hpi_names': []})
            path = Path(tmp) / 'settings.json'
            path.write_text('original')
            with self.assertRaisesRegex(OSError, 'settings sidecar'):
                core.write_settings_json(path, {'settings': {'bad': np.nan}, 'hpi_names': []})
            self.assertEqual(path.read_text(), 'original')
            self.assertEqual(list(Path(tmp).iterdir()), [path])


class CallerTests(unittest.TestCase):
    def test_hpi_only_forwards_grid_and_preserves_medium_default(self):
        for kwargs, expected in (({}, 'medium'), ({'localization_grid': 'mne_stock'}, 'mne_stock')):
            raw = synthetic_raw()
            amp = dict(raw_orig=raw, coil_amplitudes=object(), hpi_indices=np.arange(4))
            with self.subTest(grid=expected), patch.object(
                    check, 'compute_chpi_opm_locs',
                    return_value=dict(rrs=[POINTS], gofs=[[.99] * 4])) as localize:
                result = check._resolve_hpi_only(amp, **kwargs)
            self.assertEqual(localize.call_args.kwargs['localization_grid'], expected)
            self.assertIs(localize.call_args.args[1], amp['coil_amplitudes'])
            self.assertEqual(len(raw.info['dig']), 4)
            np.testing.assert_array_equal(result['hpi_dev'], POINTS)

    def test_hpi_only_cli_forwards_grid_and_records_setting(self):
        for flags, expected in (([], 'medium'), (['--localization-grid', 'mne_stock'], 'mne_stock')):
            amp = dict(hpi_gofs=np.array([.99] * 4), settings={})
            with self.subTest(grid=expected), ExitStack() as stack:
                stack.enter_context(patch('sys.argv', ['check', '--hpi', 'hpi.fif'] + flags))
                stack.enter_context(patch('sys.stdout', new=stdio.StringIO()))
                stack.enter_context(patch.object(check, 'fit_hpi_amplitudes', return_value=amp))
                resolve = stack.enter_context(patch.object(check, '_resolve_hpi_only', return_value=amp))
                stack.enter_context(patch.object(check, '_print_diagnostics_hpi_only'))
                stack.enter_context(patch.object(check, '_build_figure_hpi_only'))
                stack.enter_context(patch.object(check.plt, 'show'))
                check.main()
            resolve.assert_called_once_with(amp, localization_grid=expected)
            self.assertEqual(amp['settings']['localization_grid'], expected)

    def test_output_sidecar_name(self):
        output = '/data/subject_proc-hpi+ds_raw.fif'
        self.assertEqual(coregister._settings_sidecar_path(output),
                         '/data/hpi_subject_proc-hpi+ds_raw.json')
        self.assertEqual(coregister._settings_sidecar_path(output, '/sidecars'),
                         '/sidecars/hpi_subject_proc-hpi+ds_raw.json')

    def test_cli_parsers_forward_options(self):
        flags = ['--bad-channel-policy', 'none', '--activation-window-s', '3',
                 '--gof-comparison', 'strict', '--matching-strategy', 'coordinate_nearest',
                 '--allow-repeated-matches', '--optimization', 'none', '--settings-json', 'fit.json',
                 '--localization-grid', 'mne_stock']
        for module in (check, coregister):
            with patch('sys.argv', ['program'] + flags):
                options = fit_options(module._parse_args())
            self.assertEqual(options, dict(bad_channel_policy='none',
                                          activation_window_s=3., gof_comparison='strict',
                                          matching_strategy='coordinate_nearest', unique_matches=False,
                                          localization_grid='mne_stock',
                                          optim='none', settings_json='fit.json'))
        parser = argparse.ArgumentParser()
        add_fit_options(parser)
        self.assertEqual(parser.parse_args(['--no-center-matching']).matching_strategy, 'coordinate_nearest')

    def test_cli_accepts_all_grids_and_rejects_unknown(self):
        for module in (check, coregister):
            for grid in ('coarse', 'medium', 'fine', 'mne_stock'):
                with self.subTest(module=module.__name__, grid=grid), \
                        patch('sys.argv', ['program', '--localization-grid', grid]):
                    self.assertEqual(fit_options(module._parse_args())['localization_grid'], grid)
            with self.subTest(module=module.__name__), \
                    patch('sys.argv', ['program', '--localization-grid', 'unknown']), \
                    patch('sys.stderr', new=stdio.StringIO()), self.assertRaises(SystemExit) as exc:
                module._parse_args()
            self.assertEqual(exc.exception.code, 2)

    def test_candidate_forwarding_and_naming(self):
        files = ['/one/hpi.fif', '/two/hpi.fif']
        kwargs = dict(bad_channel_policy='none', activation_window_s=3,
                      gof_comparison='strict', matching_strategy='coordinate_nearest',
                      unique_matches=False, optim='none', settings_json='batch.json')
        with patch.object(core, 'fit_hpi', side_effect=[{'hpi_gofs': [.95, .99]}, {'hpi_gofs': [.98, .98]}]) as fit:
            best, _ = io.select_best_hpi_file(files, POL, 33, n_jobs=1, **kwargs)
        self.assertEqual(best, files[0])
        sidecars = []
        for call in fit.call_args_list:
            for key in kwargs.keys() - {'settings_json'}:
                self.assertEqual(call.kwargs[key], kwargs[key])
            sidecars.append(call.kwargs['settings_json'])
        self.assertNotEqual(*sidecars)
        self.assertEqual(sidecars[0], io._candidate_settings_path('batch.json', files[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            io.select_best_hpi_file([files[0]] * 2, POL, 33, n_jobs=1, **kwargs)

    def test_worker_forwarding(self):
        options = dict(gof_comparison='strict', optim='none')
        with patch.object(core, 'fit_hpi', return_value={}) as fit:
            io._select_best_hpi_worker('hpi.fif', POL, 33, .95, None, None, options)
        self.assertEqual(fit.call_args.kwargs['gof_comparison'], 'strict')

    def test_process_pool_option_dispatch(self):
        class ImmediatePool:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, fn, *args):
                future = Future()
                future.set_result(fn(*args))
                return future
        with patch.object(io, 'ProcessPoolExecutor', ImmediatePool), \
                patch.object(core, 'fit_hpi', return_value={'hpi_gofs': [.99]}) as fit:
            io.select_best_hpi_file(['a.fif', 'b.fif'], POL, 33, n_jobs=2,
                                    bad_channel_policy='none',
                                    gof_comparison='strict', optim='none', settings_json='batch.json')
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            self.assertEqual(call.kwargs['gof_comparison'], 'strict')
            self.assertEqual(call.kwargs['optim'], 'none')
        self.assertNotEqual(fit.call_args_list[0].kwargs['settings_json'],
                            fit.call_args_list[1].kwargs['settings_json'])


class MNEIntegrationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_real_projected_dipole_metric_and_ordered_copy(self):
        info = fieldline_info()
        proj, proj_op, picks = mne.chpi._setup_ext_proj(info, 1)
        proj_before = deepcopy(proj)
        np.testing.assert_array_equal(picks, np.arange(20))
        original = info.copy()
        metric_info, whitener, coils = core._prepare_dipole_fit(info, proj)
        pos = np.array([.01, .02, .03])
        fwd = core._magnetic_dipole_field_vec(pos[None], coils, 'info')
        self.assertEqual(fwd.shape, (3, 20))
        field = np.array([1., 2., 3.]) @ fwd
        slope = proj_op @ field
        exact = core._gof_at_fixed_pos(slope, pos, whitener, coils)
        displaced = core._gof_at_fixed_pos(slope, pos + [.02, 0., 0.], whitener, coils)
        displaced_far = core._gof_at_fixed_pos(slope, pos + [.04, 0., 0.], whitener, coils)
        raw_cov = core.make_ad_hoc_cov(info, verbose=False)
        raw_whitener, _ = core.compute_whitener(raw_cov, info, verbose=False)
        unmatched = core._gof_at_fixed_pos(slope, pos, raw_whitener, coils)
        self.assertAlmostEqual(exact, 1., places=12)
        self.assertLess(displaced, exact - .01)
        # Sensitivity is geometry-dependent; a larger known displacement must
        # also cross the unchanged field-GOF acceptance threshold in this fixture.
        self.assertLess(displaced_far, check._GOF_ACCEPT)
        self.assertGreater(exact - unmatched, .1)
        self.assertAlmostEqual(core._gof_at_fixed_pos(field, pos, raw_whitener, coils), 1., places=12)
        self.assertEqual(np.linalg.matrix_rank(whitener), 17)
        self.assertEqual(np.linalg.matrix_rank(raw_whitener), 20)
        assert_projectors_equal(self, info['projs'], original['projs'])
        assert_projectors_equal(self, metric_info['projs'], [proj])
        assert_projectors_equal(self, [proj], [proj_before])

        # Existing SSP must be replaced, not accumulated. Ordered picks may be
        # neither all channels nor in the original Info order.
        info_with_proj = info.copy()
        with info_with_proj._unlock():
            info_with_proj['projs'] = [mne.Projection(
                data=dict(col_names=info.ch_names, row_names=None,
                          data=np.eye(1, 20), nrow=1, ncol=20),
                desc='original SSP', kind=1, active=False)]
        projected_before = info_with_proj.copy()
        order = np.array([9, 2, 18, 1, 12, 5, 7, 3, 16, 0])
        ordered_proj = deepcopy(proj)
        ordered_proj['data']['col_names'] = [info.ch_names[i] for i in order]
        ordered_proj['data']['data'] = proj['data']['data'][:, order].copy()
        # These synthetic channels are all good; normalize the intentionally
        # truncated SSP vectors so MNE need not warn about their magnitudes.
        ordered_proj['data']['data'] /= np.linalg.norm(
            ordered_proj['data']['data'], axis=1)[:, None]
        ordered_proj['data']['ncol'] = len(order)
        ordered_before = deepcopy(ordered_proj)
        prepared, _, prepared_coils = core._prepare_dipole_fit(info_with_proj, ordered_proj)
        self.assertEqual(prepared.ch_names, ordered_proj['data']['col_names'])
        np.testing.assert_allclose(np.array([c['loc'] for c in prepared['chs']]),
                                   np.array([info['chs'][i]['loc'] for i in order]))
        self.assertEqual(prepared_coils[0].shape[1], 3)
        assert_projectors_equal(self, prepared['projs'], [ordered_proj])
        assert_projectors_equal(self, info_with_proj['projs'], projected_before['projs'])
        assert_projectors_equal(self, [ordered_proj], [ordered_before])

    def test_fit_localization_and_refinement_reuse_real_projected_metric(self):
        sensor_info = fieldline_info()
        sensor_raw = mne.io.RawArray(np.zeros((20, 12001)), sensor_info, verbose=False)
        hpi_raw = synthetic_raw().pick(['hpiout1', 'hpiout2', 'hpiout3', 'hpiout4'])
        sensor_raw.add_channels([hpi_raw], force_update_info=True)
        original_proj = mne.Projection(
            data=dict(col_names=sensor_info.ch_names, row_names=None,
                      data=np.eye(1, 20), nrow=1, ncol=20),
            desc='original SSP', kind=1, active=False)
        sensor_raw.add_proj(original_proj, verbose=False)
        before = sensor_raw.info.copy()
        proj, proj_op, _ = mne.chpi._setup_ext_proj(sensor_info, 1)
        _, _, coils = core._prepare_dipole_fit(sensor_info, proj)
        fields = np.array([np.array([1., 2., 3.]) @
                           core._magnetic_dipole_field_vec(p[None], coils, 'info')
                           for p in POINTS])
        slopes = fields @ proj_op.T
        counter = iter(range(4))

        def amplitudes(raw, **kwargs):
            i = next(counter)
            return dict(times=np.array([1.]), slopes=slopes[i][None, None], proj=proj)

        prepared_metrics = []
        real_prepare = core._prepare_dipole_fit

        def prepare(info, projector):
            result = real_prepare(info, projector)
            prepared_metrics.append(result)
            return result

        def localize(field, x0, too_close, whitener, geometry, guesses):
            i = len(localizer.call_args_list) - 1
            gof = core._gof_at_fixed_pos(field, POINTS[i], whitener, geometry)
            return POINTS[i], gof, np.ones(3)

        objectives = []

        def optimize(objective, x0, **kwargs):
            objectives.extend([objective(x0), objective(np.array([0., 0., 0., .001, 0., 0.]))])
            return SimpleNamespace(status=0, success=True, x=np.zeros(6), fun=objectives[0])

        with patch.object(core, 'compute_chpi_amplitudes', side_effect=amplitudes), \
                patch.object(core, '_prepare_dipole_fit', side_effect=prepare), \
                patch.object(core, '_make_opm_guesses', return_value=POINTS), \
                patch.object(core, '_fit_magnetic_dipole', side_effect=localize) as localizer, \
                patch.object(core, '_gof_at_fixed_pos', wraps=core._gof_at_fixed_pos) as fixed, \
                patch.object(core, 'minimize', side_effect=optimize) as optimizer, \
                patch('sys.stdout', new=stdio.StringIO()):
            fit = core.fit_hpi(sensor_raw, POL, 33, bad_channel_policy='none')
        optimizer.assert_called_once()
        self.assertTrue(fit['opt_success'])
        self.assertEqual(fit['optim'], 'rigid_gof')
        # One real metric for localization, one for the fixed evaluations;
        # success must not build a third metric from unprojected plotting Info.
        self.assertEqual(len(prepared_metrics), 2)
        np.testing.assert_allclose(prepared_metrics[0][1], prepared_metrics[1][1])
        for a, b in zip(prepared_metrics[0][2], prepared_metrics[1][2]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(fixed.call_args_list), 20)  # 4 local + 4 initial + 8 objective + 4 final
        for call in fixed.call_args_list[:4]:
            self.assertIs(call.args[2], prepared_metrics[0][1])
            self.assertIs(call.args[3], prepared_metrics[0][2])
        for call in fixed.call_args_list[4:]:
            self.assertIs(call.args[2], prepared_metrics[1][1])
            self.assertIs(call.args[3], prepared_metrics[1][2])
        self.assertAlmostEqual(objectives[0], -1., places=12)
        self.assertGreater(objectives[1], objectives[0])
        np.testing.assert_allclose(fit['hpi_gofs'], 1., atol=1e-12)
        np.testing.assert_allclose(fit['pol_gofs'], 1., atol=1e-12)
        np.testing.assert_allclose(fit['dist'], 0., atol=1e-12)
        assert_projectors_equal(self, sensor_raw.info['projs'], before['projs'])
        assert_projectors_equal(self, fit['raw_for_topomap'].info['projs'], before['projs'])
        for prepared in prepared_metrics:
            assert_projectors_equal(self, prepared[0]['projs'], [proj])

    def test_actual_sequential_amplitude_estimation(self):
        # Exercise MNE's real projector and slope ordering, not its mock.
        n = 24
        names = [f'MEG{i:03}' for i in range(n)] + ['hpiout1']
        info = mne.create_info(names, 1000, ['mag'] * n + ['misc'])
        rng = np.random.default_rng(7)
        positions = rng.normal(size=(n, 3))
        positions *= .12 / np.linalg.norm(positions, axis=1)[:, None]
        for ch, pos in zip(info['chs'], positions):
            ch['loc'] = np.r_[pos, np.eye(3).ravel()]
        t = np.arange(6001) / 1000
        data = rng.normal(scale=1e-13, size=(n + 1, len(t)))
        data[-1] = .001 * np.sin(2 * np.pi * 33 * t) * ((t >= 1) & (t <= 5))
        raw = mne.io.RawArray(data, info, verbose=False)
        with self.assertWarnsRegex(RuntimeWarning, '1 HPIs active'):
            amp = core.fit_hpi_amplitudes(raw, 33)
        self.assertEqual(amp['slope'].shape, (1, n))
        self.assertEqual(amp['slope_ch_names'], names[:-1])
        self.assertTrue(np.isfinite(amp['slope']).all())


if __name__ == '__main__':
    unittest.main()
