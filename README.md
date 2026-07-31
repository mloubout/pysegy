# pysegy

[![CI](https://github.com/mloubout/pysegy/actions/workflows/main.yml/badge.svg)](https://github.com/mloubout/pysegy/actions/workflows/main.yml)
[![Docs](https://github.com/mloubout/pysegy/actions/workflows/docs.yml/badge.svg)](https://mloubout.github.io/pysegy)
[![codecov](https://codecov.io/gh/mloubout/pysegy/branch/main/graph/badge.svg)](https://codecov.io/gh/mloubout/pysegy)
[![PyPI version](https://badge.fury.io/py/pysegy.svg?icon=si%3Apython)](https://badge.fury.io/py/pysegy)

`pysegy` is a minimal Python library for working with SEGY Rev 1 data.  The
project provides helpers to read and write files as well as utilities to scan
large surveys without loading every trace in memory.

## Capabilities

- Read complete SEGY files with `segy_read` and access both binary and trace
  headers.
- Write new data sets using `segy_write` from NumPy arrays.
- Lazily inspect large archives via `segy_scan` and the `SegyScan` object.
- Retrieve individual header fields with automatic scaling through
  `get_header`.
- Compatible with any `fsspec` filesystem for local or remote storage.

## Installation

Install the project in editable mode from the repository root:

```bash
python -m pip install -e .
```

Or to install the latest pypi release

```
pip install pysegy
```

## Testing

Run the unit tests with `pytest`:

```bash
pytest -vs
```

The tests run automatically on GitHub Actions with coverage reports uploaded to Codecov.

## Scan benchmark

The scan benchmark compares sequential and threaded scanning of a file or
directory:

```bash
python benchmarks/scan_benchmark.py /path/to/large.segy
python benchmarks/scan_benchmark.py /path/to/survey --pattern '*.segy'
```

Without an argument the bundled `data` directory is scanned. Threading pays
off once several files or many shots are involved; a single small file is
dominated by the fixed cost of opening it.

## Inspiration

This project started as a lightweight port of the Julia package
[SegyIO.jl](https://github.com/slimgroup/SegyIO.jl).  The goal is to provide
a similar user experience for Python while keeping the code base small and
easy to understand.
