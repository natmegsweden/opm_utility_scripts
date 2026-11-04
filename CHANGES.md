# Changes: `minor-bug-fixes` compared with `main`

Comparison: `opm_utility_scripts` branch `minor-bug-fixes` at `bc48d4f` against `main` at `5f7189f`. The branch changes five files, with 446 insertions and 203 deletions.

## Channel handling (`channels.py`)

- Correct zero/invalid MEG-channel location detection. The prior sum-based test could classify nonzero coordinates that cancel each other as an origin location. The updated check tests each coordinate component and rejects non-finite locations.
- Replace substring matching for HPI output channel indices with exact channel-name lookup. This handles overlapping names such as `hpiout1` and `hpiout11` correctly.
- Deprecate the `tolerance` argument, which did not affect selection.

## HPI fitting (`hpi/_core.py`)

- Improve input-channel cleanup and noisy-channel handling, including using consistent sensor ordering in amplitude fitting and downstream geometry.
- Change amplitude fitting from the previous six-second window to a centered two-second window around the activation midpoint.
- Correct per-coil drive-channel metadata, keep coil metadata aligned when coils are skipped, and raise a clear error if no HPI peaks are detected.
- Detect duplicate nearest-neighbour assignments between fitted coils and digitised Polhemus points and report them explicitly.
- Validate optimizer success. If refinement fails, retain the initial rigid transform instead of accepting a potentially invalid result.
- On successful refinement, recompute residual distances and fixed-position Polhemus GOFs for the refined transform.

## Diagnostic and orchestration changes

- `hpi/check.py`: use the configured GOF/residual thresholds in diagnostics and recommendations; use the OPM localizer for HPI-only checks.
- `hpi/coregister.py`: determine each coil's displayed acceptance status from the fit's actual inclusion mask rather than a hardcoded `GOF > 0.9` comparison.
- `io.py`: stop forwarding unsupported `n_jobs` to `fit_hpi`. Candidate recordings can still be processed in parallel; amplitude fitting within each candidate is sequential.

## Interpreting residuals after refinement

The initial rigid point fit assigns coils to Polhemus points and produces a geometric residual for each assignment. With rigid GOF optimization enabled, the transform is then refined to maximize dipole GOF at the digitised positions; that objective does not minimize geometric residuals. Consequently, a coil's residual can increase after optimization.

The assignment table is printed after the initial rigid fit. In `main`, the optimizer can update the returned transform without refreshing `fit['dist']`, so the residuals displayed by downstream diagnostics can remain the initial-fit values and may not describe the returned transform. In `minor-bug-fixes`, residuals are recomputed after successful refinement, so those diagnostics correspond to the returned transform. For example, a change from 4.20 mm to 8.56 mm for one coil can reflect the distinction between the initial and refined transforms, not a changed coil assignment.

The branch also changes amplitude fitting and channel/noise handling, so GOF values may differ between branches even when the GOF threshold and coil inclusion are unchanged. A particular numerical difference cannot be attributed to one change without comparing intermediate results on the same recording.
