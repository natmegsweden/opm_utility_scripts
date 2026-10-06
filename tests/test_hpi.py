"""Synthetic HPI contracts. Run: python -m unittest discover -s tests -v."""

import argparse
from concurrent.futures import Future
from contextlib import ExitStack
import io as stdio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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
    return {'times': np.array([1.]), 'slopes': np.ones((1, 1, len(names))),
            'proj': {'data': {'col_names': names}}}


POINTS = np.array([[0, 0, 0.04], [0.03, 0, 0.04], [0, 0.03, 0.04], [0.03, 0.03, 0.04]])
POL = dict(source='fif', hpi_orig=POINTS, nasion=np.array([0, .1, 0]),
           lpa=np.array([-.07, 0, 0]), rpa=np.array([.07, 0, 0]),
           extra_pts=np.empty((0, 3)), eeg_pts=np.empty((0, 3)))


class PolicyTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_coregister_sidecars_default_on_check_remains_opt_in(self):
        with patch('sys.argv', ['coregister']):
            self.assertEqual(coregister._parse_args().settings_json, '')
        with patch('sys.argv', ['check']):
            self.assertIsNone(check._parse_args().settings_json)

    def test_invalid_options_fail_before_loading(self):
        for kwargs in ({'bad_channel_policy': 'other'},
                       {'activation_window_s': 0}, {'gof_limit': float('nan')},
                       {'gof_comparison': 'other'}, {'matching_strategy': 'other'},
                       {'unique_matches': None}, {'optim': 'legacy'},
                       {'bad_channel_policy': 'reference'},
                       {'center_matching': True, 'matching_strategy': 'coordinate_nearest'}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                core.fit_hpi('not-a-file', POL, 33, **kwargs)
        for freq in (0, -1, 500, float('nan')):
            with self.assertRaises(ValueError):
                core.fit_hpi_amplitudes('not-a-file', freq)

    def test_gof_comparison(self):
        values = [.94, .95, .96, np.nan, np.inf]
        np.testing.assert_array_equal(core._gof_mask(values, .95, 'inclusive'),
                                      [False, True, True, False, False])
        np.testing.assert_array_equal(core._gof_mask(values, .95, 'strict'),
                                      [False, False, True, False, False])
        verdict = check._recommendation([.95, .95, .96], gof_comparison='strict')
        self.assertEqual(verdict[0], 'POOR')
        self.assertEqual(check._gof_color(.95, comparison='strict'), 'darkorange')

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
                    'pol_gofs', 'bads', 'bads_fig', 'optim', 'opt_status', 'opt_success'}
            self.assertTrue(keys <= fit.keys())
            payload = json.loads(path.read_text())
            self.assertEqual(payload['schema_version'], 1)
            self.assertEqual(payload['settings']['matching']['strategy'], 'centroid_nearest')
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
        dig_hpi = [d['r'] for d in info['dig'] if d['kind'] == mne.io.constants.FIFF.FIFFV_POINT_HPI]
        np.testing.assert_allclose(dig_hpi, POINTS[[0, 2, 3]])

    def test_strict_threshold_and_failed_optimizer(self):
        self.mock_localization(gofs=[.95, .99, .99, .99])
        failure = SimpleNamespace(status=2, success=False, x=np.zeros(6), fun=-.98)
        with patch.object(core, 'minimize', return_value=failure), self.assertWarnsRegex(RuntimeWarning, 'Retaining'):
            fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none',
                               optim='rigid_gof', gof_comparison='strict')
        np.testing.assert_array_equal(fit['include_hpis'], [False, True, True, True])
        self.assertFalse(fit['opt_success'])
        np.testing.assert_allclose(fit['dev_to_head_trans']['trans'], np.eye(4), atol=1e-12)

    def test_nonfinite_optimizer_and_success(self):
        self.mock_localization()
        for success, x, expected in [(True, np.full(6, np.nan), False), (True, np.zeros(6), True)]:
            result = SimpleNamespace(status=0, success=success, x=x, fun=-.98)
            with patch.object(core, 'minimize', return_value=result):
                if expected:
                    fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none', optim='rigid')
                else:
                    with self.assertWarns(RuntimeWarning):
                        fit = core.fit_hpi(synthetic_raw(), POL, 33, bad_channel_policy='none', optim='rigid')
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
    def test_output_sidecar_name(self):
        output = '/data/subject_proc-hpi+ds_raw.fif'
        self.assertEqual(coregister._settings_sidecar_path(output),
                         '/data/hpi_subject_proc-hpi+ds_raw.json')
        self.assertEqual(coregister._settings_sidecar_path(output, '/sidecars'),
                         '/sidecars/hpi_subject_proc-hpi+ds_raw.json')

    def test_cli_parsers_forward_options(self):
        flags = ['--bad-channel-policy', 'none', '--activation-window-s', '3',
                 '--gof-comparison', 'strict', '--matching-strategy', 'coordinate_nearest',
                 '--allow-repeated-matches', '--optimization', 'none', '--settings-json', 'fit.json']
        for module in (check, coregister):
            with patch('sys.argv', ['program'] + flags):
                options = fit_options(module._parse_args())
            self.assertEqual(options, dict(bad_channel_policy='none',
                                          activation_window_s=3., gof_comparison='strict',
                                          matching_strategy='coordinate_nearest', unique_matches=False,
                                          optim='none', settings_json='fit.json'))
        parser = argparse.ArgumentParser()
        add_fit_options(parser)
        self.assertEqual(parser.parse_args(['--no-center-matching']).matching_strategy, 'coordinate_nearest')

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
