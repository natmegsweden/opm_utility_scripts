# OPM Utility Scripts

Utilities for FieldLine OPM-MEG data processing, including HPI coregistration,
quality checks, visualization, and channel tools.

## Install

```bash
pip install .
```

Optional tool dependencies:

```bash
pip install ".[tools]"  # pandas-based utilities
pip install ".[vtk]"    # 3-D helmet visualization
```

This installs the `opmutil` command. For example:

```bash
opmutil check --hpi HPIbefore_raw.fif --pol digitisation.json
```

See the [documentation](docs/index.md) for installation options, CLI guides,
fit behavior, and the Python API.

To preview or build the documentation, install the docs extra and run
`zensical serve` or `zensical build` from the repository root:

```bash
pip install ".[docs]"
zensical serve
```
