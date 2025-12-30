import random

import pytest

import sorting_algorithms as sa


def run_sort(sort_fn, data):
    a = list(data)
    return sort_fn(a)


def test_bubble_sort_sorts_integers_with_negatives_and_duplicates():
    data = [3, -1, 2, -5, 2, 0]
    assert run_sort(sa.bubble_sort, data) == sorted(data)


def test_bubble_sort_sorts_empty_list():
    assert run_sort(sa.bubble_sort, []) == []


def test_selection_sort_sorts_integers_with_negatives_and_duplicates():
    data = [9, 1, 1, -3, 7, 0]
    assert run_sort(sa.selection_sort, data) == sorted(data)


def test_selection_sort_sorts_empty_list():
    assert run_sort(sa.selection_sort, []) == []


def test_insertion_sort_sorts_integers_with_negatives_and_duplicates():
    data = [5, -2, -2, 8, 3, 0]
    assert run_sort(sa.insertion_sort, data) == sorted(data)


def test_insertion_sort_sorts_empty_list():
    assert run_sort(sa.insertion_sort, []) == []


def test_merge_sort_sorts_integers_with_negatives_and_duplicates():
    data = [4, 0, -1, -1, 10, 2]
    assert run_sort(sa.merge_sort, data) == sorted(data)


def test_merge_sort_sorts_empty_list():
    assert run_sort(sa.merge_sort, []) == []


def test_quick_sort_sorts_integers_with_negatives_and_duplicates():
    data = [7, -4, 7, 1, 0, -4]
    assert run_sort(sa.quick_sort, data) == sorted(data)


def test_quick_sort_sorts_empty_list():
    assert run_sort(sa.quick_sort, []) == []


def test_heap_sort_sorts_integers_with_negatives_and_duplicates():
    data = [6, 2, -9, 2, 0, 5]
    assert run_sort(sa.heap_sort, data) == sorted(data)


def test_heap_sort_sorts_empty_list():
    assert run_sort(sa.heap_sort, []) == []


def test_counting_sort_sorts_integers_with_negatives_and_duplicates():
    data = [0, -1, -1, 3, 2, -5]
    assert run_sort(sa.counting_sort, data) == sorted(data)


def test_counting_sort_sorts_empty_list():
    assert run_sort(sa.counting_sort, []) == []


def test_radix_sort_sorts_integers_with_negatives_and_duplicates():
    data = [170, 45, 75, -90, -802, 24, 2, 66, -90]
    assert run_sort(sa.radix_sort, data) == sorted(data)


def test_radix_sort_sorts_empty_list():
    assert run_sort(sa.radix_sort, []) == []


def test_bucket_sort_sorts_integers_with_negatives_and_duplicates():
    data = [3, -1, 2, 0, -5, 2]
    assert run_sort(sa.bucket_sort, data) == sorted(data)


def test_bucket_sort_sorts_empty_list():
    assert run_sort(sa.bucket_sort, []) == []


def test_shell_sort_sorts_integers_with_negatives_and_duplicates():
    data = [12, 34, 54, 2, 3, -8, -8]
    assert run_sort(sa.shell_sort, data) == sorted(data)


def test_shell_sort_sorts_empty_list():
    assert run_sort(sa.shell_sort, []) == []


def test_tim_sort_sorts_integers_with_negatives_and_duplicates():
    data = [28, -30, 31, -24, -4, -11, -15, -32, 31, 34, -48, 1, -16, 18, 2, -13, -34, -14, 0, -23, -18, 33, 31, 50, -22, 44, 16, -1, -24, -12, -3, -49]
    assert run_sort(sa.tim_sort, data) == sorted(data)


def test_tim_sort_sorts_empty_list():
    assert run_sort(sa.tim_sort, []) == []


def test_intro_sort_sorts_integers_with_negatives_and_duplicates():
    data = [9, -10, 0, 3, -10, 5, 5, 2]
    assert run_sort(sa.intro_sort, data) == sorted(data)


def test_intro_sort_sorts_empty_list():
    assert run_sort(sa.intro_sort, []) == []


def test_tree_sort_sorts_integers_with_negatives_and_duplicates():
    data = [5, 3, 7, 3, 2, -1, 0]
    assert run_sort(sa.tree_sort, data) == sorted(data)


def test_tree_sort_sorts_empty_list():
    assert run_sort(sa.tree_sort, []) == []


def test_bucket_sort_sorts_all_equal_values():
    data = [7, 7, 7, 7]
    assert run_sort(sa.bucket_sort, data) == data


def test_all_sorts_match_python_sorted_for_random_small_input():
    rng = random.Random(0)
    data = [rng.randint(-25, 25) for _ in range(40)]
    expected = sorted(data)
    results = {
        "bubble_sort": run_sort(sa.bubble_sort, data),
        "selection_sort": run_sort(sa.selection_sort, data),
        "insertion_sort": run_sort(sa.insertion_sort, data),
        "merge_sort": run_sort(sa.merge_sort, data),
        "quick_sort": run_sort(sa.quick_sort, data),
        "heap_sort": run_sort(sa.heap_sort, data),
        "counting_sort": run_sort(sa.counting_sort, data),
        "radix_sort": run_sort(sa.radix_sort, data),
        "bucket_sort": run_sort(sa.bucket_sort, data),
        "shell_sort": run_sort(sa.shell_sort, data),
        "tim_sort": run_sort(sa.tim_sort, data),
        "intro_sort": run_sort(sa.intro_sort, data),
        "tree_sort": run_sort(sa.tree_sort, data),
    }
    assert all(result == expected for result in results.values())


def test_bubble_sort_raises_type_error_on_mixed_incomparable_types():
    with pytest.raises(TypeError):
        sa.bubble_sort([1, "a"])


def test_merge_sort_raises_type_error_on_mixed_incomparable_types():
    with pytest.raises(TypeError):
        sa.merge_sort([1, "a"])


def test_counting_sort_raises_type_error_on_float_values():
    with pytest.raises(TypeError):
        sa.counting_sort([1.5, 2.5])


def test_radix_sort_raises_type_error_on_float_values():
    with pytest.raises(TypeError):
        sa.radix_sort([1.5, 2.5])


def test_bucket_sort_raises_type_error_on_incomparable_types():
    with pytest.raises(TypeError):
        sa.bucket_sort([1, "a"])
