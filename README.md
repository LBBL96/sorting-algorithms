# Sorting Algorithms

![Sorting Algorithms Comparison](sorting_comparison.gif)

A collection of sorting algorithm implementations in Python with visualization and benchmarking tools. 

## Algorithms Included

- **Bubble Sort** - O(n²)
- **Selection Sort** - O(n²)
- **Insertion Sort** - O(n²)
- **Merge Sort** - O(n log n)
- **Quick Sort** - O(n log n) average
- **Heap Sort** - O(n log n)
- **Shell Sort** - O(n log n) to O(n²)
- **Counting Sort** - O(n + k)
- **Radix Sort** - O(nk)
- **Bucket Sort** - O(n + k)
- **Tim Sort** - O(n log n)
- **Intro Sort** - O(n log n)
- **Tree Sort** - O(n log n)

## Files

- `sorting_algorithms.py` - All sorting algorithm implementations
- `sorting_algorithms_test.py` - Test suite using pytest
- `sorting_benchmark.py` - Performance benchmarking tools
- `sorting_visualizer.py` - Animated visualization and GIF generation

## Running the Visualization

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install matplotlib pillow

# Generate the comparison GIF
python sorting_visualizer.py
```

This creates `sorting_comparison.gif` showing 9 algorithms sorting simultaneously.

## Running Benchmarks

```bash
python sorting_benchmark.py
```

Or use the functions directly:

```python
from sorting_benchmark import run_fast_only, run_all_sequential

# Benchmark O(n log n) algorithms only
run_fast_only()

# Benchmark all algorithms (slower)
run_all_sequential()
```

## Running Tests

```bash
pip install pytest
python -m pytest sorting_algorithms_test.py -v
```
