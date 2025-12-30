import copy
import random
from typing import Generator

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def bubble_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            yield arr[:], j, j + 1, "compare"
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                yield arr[:], j, j + 1, "swap"
    yield arr[:], -1, -1, "done"


def selection_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            yield arr[:], min_idx, j, "compare"
            if arr[j] < arr[min_idx]:
                min_idx = j
        if min_idx != i:
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            yield arr[:], i, min_idx, "swap"
    yield arr[:], -1, -1, "done"


def insertion_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            yield arr[:], j, j + 1, "compare"
            arr[j + 1] = arr[j]
            j -= 1
            yield arr[:], j + 1, j + 2, "swap"
        arr[j + 1] = key
    yield arr[:], -1, -1, "done"


def merge_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    def merge_sort_helper(arr, l, r):
        if l < r:
            m = (l + r) // 2
            yield from merge_sort_helper(arr, l, m)
            yield from merge_sort_helper(arr, m + 1, r)
            yield from merge(arr, l, m, r)

    def merge(arr, l, m, r):
        left = arr[l:m + 1]
        right = arr[m + 1:r + 1]
        i = j = 0
        k = l
        while i < len(left) and j < len(right):
            yield arr[:], l + i, m + 1 + j, "compare"
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            yield arr[:], k, k, "swap"
            k += 1
        while i < len(left):
            arr[k] = left[i]
            yield arr[:], k, k, "swap"
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            yield arr[:], k, k, "swap"
            j += 1
            k += 1

    yield from merge_sort_helper(arr, 0, len(arr) - 1)
    yield arr[:], -1, -1, "done"


def quick_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    def quick_sort_helper(arr, low, high):
        if low < high:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                yield arr[:], j, high, "compare"
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    yield arr[:], i, j, "swap"
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            yield arr[:], i + 1, high, "swap"
            pi = i + 1
            yield from quick_sort_helper(arr, low, pi - 1)
            yield from quick_sort_helper(arr, pi + 1, high)

    yield from quick_sort_helper(arr, 0, len(arr) - 1)
    yield arr[:], -1, -1, "done"


def heap_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    n = len(arr)

    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n:
            yield arr[:], largest, left, "compare"
            if arr[left] > arr[largest]:
                largest = left
        if right < n:
            yield arr[:], largest, right, "compare"
            if arr[right] > arr[largest]:
                largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr[:], i, largest, "swap"
            yield from heapify(arr, n, largest)

    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr[:], 0, i, "swap"
        yield from heapify(arr, i, 0)

    yield arr[:], -1, -1, "done"


def shell_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                yield arr[:], j - gap, j, "compare"
                arr[j] = arr[j - gap]
                yield arr[:], j - gap, j, "swap"
                j -= gap
            arr[j] = temp
        gap //= 2
    yield arr[:], -1, -1, "done"


def radix_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    if not arr:
        yield arr[:], -1, -1, "done"
        return

    max_val = max(arr)
    exp = 1

    while max_val // exp > 0:
        buckets: list[list[int]] = [[] for _ in range(10)]

        for i, num in enumerate(arr):
            digit = (num // exp) % 10
            buckets[digit].append(num)
            yield arr[:], i, i, "compare"

        idx = 0
        for bucket in buckets:
            for num in bucket:
                arr[idx] = num
                yield arr[:], idx, idx, "swap"
                idx += 1

        exp *= 10

    yield arr[:], -1, -1, "done"


def counting_sort_gen(arr: list[int]) -> Generator[tuple[list[int], int, int, str], None, None]:
    if not arr:
        yield arr[:], -1, -1, "done"
        return

    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val + 1

    count = [0] * range_val
    for num in arr:
        count[num - min_val] += 1

    idx = 0
    for i in range(range_val):
        while count[i] > 0:
            arr[idx] = i + min_val
            yield arr[:], idx, idx, "swap"
            idx += 1
            count[i] -= 1

    yield arr[:], -1, -1, "done"


VISUALIZABLE_ALGORITHMS: dict[str, Callable] = {
    "Bubble Sort": bubble_sort_gen,
    "Selection Sort": selection_sort_gen,
    "Insertion Sort": insertion_sort_gen,
    "Merge Sort": merge_sort_gen,
    "Quick Sort": quick_sort_gen,
    "Heap Sort": heap_sort_gen,
    "Shell Sort": shell_sort_gen,
    "Radix Sort": radix_sort_gen,
    "Counting Sort": counting_sort_gen,
}


class SortingVisualizer:
    def __init__(
        self,
        algorithms: dict[str, callable] = None,
        array_size: int = 30,
        seed: int = 42,
    ):
        if algorithms is None:
            algorithms = VISUALIZABLE_ALGORITHMS
        self.algorithms = algorithms
        self.array_size = array_size
        self.seed = seed
        self.base_array = self._generate_array()

    def _generate_array(self) -> list[int]:
        random.seed(self.seed)
        arr = list(range(1, self.array_size + 1))
        random.shuffle(arr)
        return arr

    def create_grid_animation(
        self,
        interval: int = 100,
        skip_frames: int = 1,
    ) -> animation.FuncAnimation:
        algo_names = list(self.algorithms.keys())
        n_algos = len(algo_names)

        cols = 3 if n_algos > 4 else 2
        rows = (n_algos + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten() if n_algos > 1 else [axes]

        for i in range(n_algos, len(axes)):
            axes[i].set_visible(False)

        generators: dict[str, Generator] = {}
        current_states: dict[str, tuple[list[int], int, int, str]] = {}
        bar_containers: dict[str, list] = {}
        step_counts: dict[str, int] = {}
        finished: dict[str, bool] = {}

        for i, name in enumerate(algo_names):
            arr = copy.copy(self.base_array)
            generators[name] = self.algorithms[name](arr)
            current_states[name] = (arr[:], -1, -1, "start")
            step_counts[name] = 0
            finished[name] = False

            ax = axes[i]
            ax.set_xlim(-0.5, self.array_size - 0.5)
            ax.set_ylim(0, self.array_size + 1)
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            bars = ax.bar(
                range(self.array_size),
                self.base_array,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )
            bar_containers[name] = bars

        fig.suptitle("Sorting Algorithm Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        def update(frame):
            all_done = True

            for i, name in enumerate(algo_names):
                if finished[name]:
                    continue

                all_done = False

                for _ in range(skip_frames):
                    try:
                        state: tuple[list[int], int, int, str] = next(generators[name])
                        current_states[name] = state
                        step_counts[name] += 1
                    except StopIteration:
                        finished[name] = True
                        break

                arr, idx1, idx2, action = current_states[name]
                bars = bar_containers[name]

                for j, bar in enumerate(bars):
                    bar.set_height(arr[j])
                    bar.set_color("steelblue")

                axes[i].set_title(f"{name} ({step_counts[name]} steps)", fontsize=11)

            if all_done:
                ani.event_source.stop()

            return [bar for bars in bar_containers.values() for bar in bars]

        def frame_generator():
            frame = 0
            max_frames = 500
            while frame < max_frames:
                if all(finished.values()):
                    for _ in range(24):
                        yield frame
                    return
                yield frame
                frame += 1

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frame_generator,
            interval=interval,
            blit=False,
            repeat=False,
            save_count=500,
        )

        return ani, fig

    def save_gif(
        self,
        filename: str = "sorting_comparison.gif",
        interval: int = 100,
        skip_frames: int = 2,
        fps: int = 12,
    ):
        ani, fig = self.create_grid_animation(interval=interval, skip_frames=skip_frames)
        print(f"Saving animation to {filename}...")
        writer = animation.PillowWriter(fps=fps)
        ani.save(filename, writer=writer)
        print(f"Saved {filename}")
        plt.close(fig)

    def show(self, interval: int = 100, skip_frames: int = 2):
        ani, fig = self.create_grid_animation(interval=interval, skip_frames=skip_frames)
        plt.show()


def create_comparison_gif(
    algorithms: list[str] = None,
    array_size: int = 30,
    filename: str = "sorting_comparison.gif",
    seed: int = 42,
    interval: int = 100,
    skip_frames: int = 2,
    fps: int = 12,
):
    if algorithms is None:
        algos = VISUALIZABLE_ALGORITHMS
    else:
        algos: dict[str, Callable] = {name: VISUALIZABLE_ALGORITHMS[name] for name in algorithms if name in VISUALIZABLE_ALGORITHMS}

    viz = SortingVisualizer(algorithms=algos, array_size=array_size, seed=seed)
    viz.save_gif(filename=filename, interval=interval, skip_frames=skip_frames, fps=fps)


def show_comparison(
    algorithms: list[str] = None,
    array_size: int = 30,
    seed: int = 42,
    interval: int = 100,
    skip_frames: int = 2,
):
    if algorithms is None:
        algos = VISUALIZABLE_ALGORITHMS
    else:
        algos: dict[str, Callable] = {name: VISUALIZABLE_ALGORITHMS[name] for name in algorithms if name in VISUALIZABLE_ALGORITHMS}

    viz = SortingVisualizer(algorithms=algos, array_size=array_size, seed=seed)
    viz.show(interval=interval, skip_frames=skip_frames)


if __name__ == "__main__":
    print("Sorting Algorithm Visualizer")
    print("=" * 40)
    print("Available algorithms:")
    for name in VISUALIZABLE_ALGORITHMS:
        print(f"  - {name}")
    print()
    print("Usage:")
    print("  show_comparison()  - Show live animation")
    print("  create_comparison_gif()  - Save as GIF")
    print()
    demo_algos = [
        "Bubble Sort", "Selection Sort", "Insertion Sort",
        "Merge Sort", "Quick Sort", "Heap Sort",
        "Shell Sort", "Radix Sort", "Counting Sort",
    ]

    print("Saving GIF...")
    create_comparison_gif(
        algorithms=demo_algos,
        array_size=20,
        filename="sorting_comparison.gif",
        skip_frames=1,
        fps=8,
    )
    print("GIF saved as sorting_comparison.gif")
 