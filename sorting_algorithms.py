"""
Sorting Algorithms Implementation
Contains implementations of various sorting algorithms for educational purposes.
"""

def bubble_sort(arr):
    """Bubble Sort - O(n²) time, O(1) space"""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def selection_sort(arr):
    """Selection Sort - O(n²) time, O(1) space"""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr):
    """Insertion Sort - O(n²) time, O(1) space"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr):
    """Merge Sort - O(n log n) time, O(n) space"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    """Helper function for merge_sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):
    """Quick Sort - O(n log n) average, O(n²) worst, O(log n) space"""
    def quick_sort_helper(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            quick_sort_helper(arr, low, pi - 1)
            quick_sort_helper(arr, pi + 1, high)
    
    quick_sort_helper(arr, 0, len(arr) - 1)
    return arr


def partition(arr, low, high):
    """Helper function for quick_sort"""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def heap_sort(arr):
    """Heap Sort - O(n log n) time, O(1) space"""
    n = len(arr)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr


def heapify(arr, n, i):
    """Helper function for heap_sort"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def counting_sort(arr):
    """Counting Sort - O(n+k) time, O(k) space"""
    if not arr:
        return arr
    
    min_val = min(arr)
    max_val = max(arr)
    range_val = max_val - min_val + 1
    
    # Create count array
    count = [0] * range_val
    
    # Store count of each element
    for num in arr:
        count[num - min_val] += 1
    
    # Build output array
    result = []
    for i in range(range_val):
        result.extend([i + min_val] * count[i])
    
    return result


def radix_sort(arr):
    """Radix Sort - O(d·(n+k)) time, O(n+k) space"""
    if not arr:
        return arr
    
    # Handle negative numbers
    max_abs = max(abs(x) for x in arr)
    exp = 1
    
    # Separate positive and negative numbers
    positives = [x for x in arr if x >= 0]
    negatives = [-x for x in arr if x < 0]
    
    # Sort positive numbers
    while max_abs // exp > 0:
        positives = counting_sort_by_digit(positives, exp)
        exp *= 10
    
    # Sort negative numbers and reverse
    max_neg = max(negatives) if negatives else 0
    exp = 1
    while max_neg // exp > 0:
        negatives = counting_sort_by_digit(negatives, exp)
        exp *= 10
    
    # Combine results: negative numbers (reversed and negated) + positive numbers
    negatives = [-x for x in reversed(negatives)]
    return negatives + positives


def counting_sort_by_digit(arr, exp):
    """Helper function for radix_sort - sorts by specific digit"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    # Store count of occurrences
    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1
    
    # Change count[i] so that count[i] contains actual position
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
    
    return output


def bucket_sort(arr):
    """Bucket Sort - O(n+k) average time, O(n+k) space"""
    if not arr:
        return arr
    
    # Find minimum and maximum values
    min_val = min(arr)
    max_val = max(arr)
    
    # Create buckets
    bucket_count = len(arr)
    buckets = [[] for _ in range(bucket_count)]
    
    # Distribute input array values into buckets
    for num in arr:
        # Normalize to [0, 1] range
        normalized = (num - min_val) / (max_val - min_val) if max_val != min_val else 0
        bucket_index = int(normalized * (bucket_count - 1))
        buckets[bucket_index].append(num)
    
    # Sort individual buckets and concatenate
    result = []
    for bucket in buckets:
        if bucket:
            bucket.sort()  # Using Python's built-in sort for individual buckets
            result.extend(bucket)
    
    return result


def shell_sort(arr):
    """Shell Sort - O(n^(3/2)) average time, O(1) space"""
    n = len(arr)
    gap = n // 2
    
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    
    return arr


def tim_sort(arr):
    """Tim Sort - Python's default algorithm (simplified version)"""
    # This is a simplified version - Python's actual Tim Sort is more complex
    # For practical use, just use sorted(arr) or arr.sort()
    return sorted(arr)


def intro_sort(arr):
    """Intro Sort - C++ std::sort algorithm (simplified version)"""
    # This is a simplified version - actual Intro Sort switches between
    # Quick Sort, Heap Sort, and Insertion Sort based on recursion depth
    # For practical use, just use sorted(arr) or arr.sort()
    return sorted(arr)


def tree_sort(arr):
    """Tree Sort - O(n log n) time, O(n) space"""
    class TreeNode:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None
    
    def insert(root, val):
        if root is None:
            return TreeNode(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)
        return root
    
    def inorder(root, result):
        if root:
            inorder(root.left, result)
            result.append(root.val)
            inorder(root.right, result)
    
    if not arr:
        return arr
    
    root = None
    for num in arr:
        root = insert(root, num)
    
    result = []
    inorder(root, result)
    return result
