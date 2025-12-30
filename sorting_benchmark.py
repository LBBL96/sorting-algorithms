import random
import sys
import time
import copy
from typing import Callable, Any

sys.setrecursionlimit(50000)

import sorting_algorithms as sa


ALGORITHMS: dict[str, Callable] = {
    "bubble_sort": sa.bubble_sort,
    "selection_sort": sa.selection_sort,
    "insertion_sort": sa.insertion_sort,
    "merge_sort": sa.merge_sort,
    "quick_sort": sa.quick_sort,
    "heap_sort": sa.heap_sort,
    "counting_sort": sa.counting_sort,
    "radix_sort": sa.radix_sort,
    "bucket_sort": sa.bucket_sort,
    "shell_sort": sa.shell_sort,
    "tim_sort": sa.tim_sort,
    "intro_sort": sa.intro_sort,
    "tree_sort": sa.tree_sort,
}

FAST_ALGORITHMS: dict[str, Callable] = {
    "merge_sort": sa.merge_sort,
    "quick_sort": sa.quick_sort,
    "heap_sort": sa.heap_sort,
    "counting_sort": sa.counting_sort,
    "radix_sort": sa.radix_sort,
    "bucket_sort": sa.bucket_sort,
    "shell_sort": sa.shell_sort,
    "tim_sort": sa.tim_sort,
    "intro_sort": sa.intro_sort,
    "tree_sort": sa.tree_sort,
}

SLOW_ALGORITHMS: dict[str, Callable] = {
    "bubble_sort": sa.bubble_sort,
    "selection_sort": sa.selection_sort,
    "insertion_sort": sa.insertion_sort,
}


def generate_random(n: int, low: int = -10000, high: int = 10000) -> list[int]:
    return [random.randint(low, high) for _ in range(n)]


def generate_sorted(n: int) -> list[int]:
    return list(range(n))


def generate_reverse_sorted(n: int) -> list[int]:
    return list(range(n, 0, -1))


def generate_nearly_sorted(n: int, swaps: int = None) -> list[int]:
    if swaps is None:
        swaps = max(1, n // 20)
    data = list(range(n))
    for _ in range(swaps):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        data[i], data[j] = data[j], data[i]
    return data


def generate_many_duplicates(n: int, unique_values: int = 10) -> list[int]:
    return [random.randint(0, unique_values - 1) for _ in range(n)]


def generate_all_equal(n: int) -> list[int]:
    return [42] * n


DATA_GENERATORS: dict[str, Callable] = {
    "random": generate_random,
    "sorted": generate_sorted,
    "reverse_sorted": generate_reverse_sorted,
    "nearly_sorted": generate_nearly_sorted,
    "many_duplicates": generate_many_duplicates,
    "all_equal": generate_all_equal,
}


def time_algorithm(sort_fn: Callable, data: list[int], runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        data_copy = copy.copy(data)
        start = time.perf_counter()
        sort_fn(data_copy)
        end = time.perf_counter()
        times.append(end - start)
    return min(times)


def benchmark_single(
    algorithm_name: str,
    sort_fn: Callable,
    data: list[int],
    pattern_name: str,
    runs: int = 3,
) -> dict[str, Any]:
    elapsed = time_algorithm(sort_fn, data, runs)
    return {
        "algorithm": algorithm_name,
        "pattern": pattern_name,
        "size": len(data),
        "time_seconds": elapsed,
        "time_ms": elapsed * 1000,
    }


def benchmark_algorithms(
    algorithms: dict[str, Callable],
    sizes: list[int],
    patterns: list[str] = None,
    runs: int = 3,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if patterns is None:
        patterns = list(DATA_GENERATORS.keys())

    results = []

    for size in sizes:
        for pattern in patterns:
            generator = DATA_GENERATORS[pattern]
            data = generator(size)

            if verbose:
                print(f"\n{'='*60}")
                print(f"Size: {size:,} | Pattern: {pattern}")
                print(f"{'='*60}")

            for name, sort_fn in algorithms.items():
                result = benchmark_single(name, sort_fn, data, pattern, runs)
                results.append(result)
                if verbose:
                    print(f"  {name:20} : {result['time_ms']:10.3f} ms")

    return results


def run_all_sequential(
    sizes: list[int] = None,
    patterns: list[str] = None,
    include_slow: bool = True,
    runs: int = 3,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if sizes is None:
        if include_slow:
            sizes = [100, 500, 1000, 2000]
        else:
            sizes = [1000, 5000, 10000, 50000]

    if patterns is None:
        patterns = ["random", "sorted", "reverse_sorted", "nearly_sorted", "many_duplicates"]

    if include_slow:
        algorithms = ALGORITHMS
    else:
        algorithms = FAST_ALGORITHMS

    if verbose:
        print("\n" + "=" * 60)
        print("SORTING ALGORITHM BENCHMARK")
        print("=" * 60)
        print(f"Algorithms: {', '.join(algorithms.keys())}")
        print(f"Sizes: {sizes}")
        print(f"Patterns: {patterns}")
        print(f"Runs per test: {runs}")
        print("=" * 60)

    return benchmark_algorithms(algorithms, sizes, patterns, runs, verbose)


def run_fast_only(
    sizes: list[int] = None,
    patterns: list[str] = None,
    runs: int = 3,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if sizes is None:
        sizes = [1000, 5000, 10000, 50000, 100000]
    return run_all_sequential(sizes, patterns, include_slow=False, runs=runs, verbose=verbose)


def run_slow_only(
    sizes: list[int] = None,
    patterns: list[str] = None,
    runs: int = 3,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]

    if patterns is None:
        patterns = ["random", "sorted", "reverse_sorted", "nearly_sorted", "many_duplicates"]

    if verbose:
        print("\n" + "=" * 60)
        print("SLOW ALGORITHM BENCHMARK (O(n²))")
        print("=" * 60)
        print(f"Algorithms: {', '.join(SLOW_ALGORITHMS.keys())}")
        print(f"Sizes: {sizes}")
        print(f"Patterns: {patterns}")
        print("=" * 60)

    return benchmark_algorithms(SLOW_ALGORITHMS, sizes, patterns, runs, verbose)


def print_summary(results: list[dict[str, Any]]) -> None:
    from collections import defaultdict

    by_algorithm: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_algorithm[r["algorithm"]].append(r["time_ms"])

    print("\n" + "=" * 60)
    print("SUMMARY (average time in ms)")
    print("=" * 60)

    summaries = []
    for name, times in by_algorithm.items():
        avg = sum(times) / len(times)
        summaries.append((avg, name))

    summaries.sort()
    for avg, name in summaries:
        print(f"  {name:20} : {avg:10.3f} ms avg")


def compare_on_pattern(
    pattern: str,
    sizes: list[int] = None,
    include_slow: bool = False,
    runs: int = 3,
) -> list[dict[str, Any]]:
    if sizes is None:
        sizes = [1000, 5000, 10000]

    algorithms = ALGORITHMS if include_slow else FAST_ALGORITHMS

    print(f"\n{'='*60}")
    print(f"COMPARISON ON PATTERN: {pattern}")
    print(f"{'='*60}")

    return benchmark_algorithms(algorithms, sizes, [pattern], runs, verbose=True)


if __name__ == "__main__":
    print("Running benchmark with default settings...")
    print("Use run_all_sequential() for full benchmark")
    print("Use run_fast_only() for O(n log n) algorithms only")
    print("Use run_slow_only() for O(n²) algorithms only")
    print()

    results = run_all_sequential(
        sizes=[500, 1000, 2000],
        patterns=["random", "sorted", "nearly_sorted"],
        include_slow=True,
        runs=3,
    )
    print_summary(results)
