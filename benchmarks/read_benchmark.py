"""Benchmark sequential and process-parallel SEGY decoding.

Run from the repository root, for example::

    python benchmarks/read_benchmark.py survey.segy --workers 1 2 4 8

Use a representative large file: process startup and IPC costs deliberately
make multiprocessing unattractive for small inputs.
"""

import argparse
import os
import statistics
import time

import pysegy as seg


def benchmark(path, workers, repeat):
    """Return elapsed read times while verifying every run reads all traces."""
    timings = []
    expected_traces = None
    for _ in range(repeat):
        started = time.perf_counter()
        block = seg.segy_read(path, workers=workers)
        timings.append(time.perf_counter() - started)
        traces = len(block.traceheaders)
        if expected_traces is None:
            expected_traces = traces
        elif traces != expected_traces:
            raise RuntimeError("benchmark runs returned different trace counts")
    return timings, expected_traces


def main():
    parser = argparse.ArgumentParser(
        description="Compare sequential and parallel SEGY read throughput"
    )
    parser.add_argument("path", help="SEGY file to read")
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 2, 4],
        help="worker counts to compare (default: 1 2 4)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="number of reads per worker count (default: 3)",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if any(worker < 1 for worker in args.workers):
        parser.error("--workers values must be at least 1")

    size_mib = os.path.getsize(args.path) / 1024 ** 2
    results = []
    for workers in args.workers:
        timings, traces = benchmark(args.path, workers, args.repeat)
        median = statistics.median(timings)
        results.append((workers, median, size_mib / median))
        samples = ", ".join(f"{timing:.3f}" for timing in timings)
        print(f"workers={workers}: [{samples}] seconds ({traces} traces)")

    baseline = next(
        (elapsed for workers, elapsed, _ in results if workers == 1),
        results[0][1],
    )
    print("\nworkers  median (s)  MiB/s    speedup  efficiency")
    for workers, elapsed, throughput in results:
        speedup = baseline / elapsed
        efficiency = speedup / workers
        print(
            f"{workers:7d}  {elapsed:10.3f}  {throughput:7.1f}  "
            f"{speedup:7.2f}x  {efficiency:9.1%}"
        )


if __name__ == "__main__":
    main()
