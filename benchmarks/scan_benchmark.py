"""Benchmark sequential and threaded SEGY scanning.

Run from the repository root, for example::

    python benchmarks/scan_benchmark.py survey.segy
    python benchmarks/scan_benchmark.py /path/to/directory --pattern '*.segy'

With no argument the bundled ``data`` directory is scanned.
"""

import argparse
import os
import time

import pysegy as seg

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PATTERN = 'overthrust_2D_shot_*.segy'


def benchmark(path, pattern, threads):
    """Return the elapsed scan time and the number of shots found."""
    started = time.perf_counter()
    scan = seg.segy_scan(path, pattern, threads=threads)
    return time.perf_counter() - started, len(scan)


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential and threaded SEGY scan throughput"
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="SEGY file or directory to scan (default: bundled data directory)",
    )
    parser.add_argument(
        "--pattern", default=None,
        help="glob selecting files when path is a directory (default: '*')",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="thread count for the threaded run (default: all cores)",
    )
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be at least 1")
    elif args.threads is None:
        args.threads = os.cpu_count() + 1

    path = args.path
    pattern = args.pattern
    if path is None:
        path = DATA_DIR
        if pattern is None:
            pattern = PATTERN
    if not os.path.exists(path):
        parser.error(f"path not found: {path}")
    if os.path.isdir(path) and pattern is None:
        pattern = "*"

    sequential, shots = benchmark(path, pattern, 1)
    threaded, _ = benchmark(path, pattern, args.threads)
    print(f"Scanned {path} ({shots} shots)")
    print(f"Sequential: {sequential:.3f}s")
    print(f"Threaded ({args.threads}):   {threaded:.3f}s")
    print(f"Speedup:    {sequential / threaded:.2f}x")


if __name__ == '__main__':
    main()
