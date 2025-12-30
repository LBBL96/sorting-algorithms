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
    """Tim Sort - Python's actual algorithm implementation"""
    if len(arr) < 2:
        return arr[:]
    
    # Tim Sort parameters
    MIN_MERGE = 32
    MIN_GALLOP = 7
    
    def calc_min_run(n):
        """Calculate minimum run length for Tim Sort"""
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r
    
    def binary_insertion_sort(arr, left, right):
        """Binary insertion sort for small arrays"""
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = left
            while j < i and arr[j] <= key:
                j += 1
            while j < i:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                j += 1
    
    def merge(arr, l, m, r):
        """Merge two sorted runs with galloping optimization"""
        left = arr[l:m+1]
        right = arr[m+1:r+1]
        
        i = j = 0
        k = l
        left_galloping = right_galloping = False
        left_wins = right_wins = 0
        
        while i < len(left) and j < len(right):
            if left_galloping:
                # Galloping mode for left array
                pos = gallop_search(right, j, left[i])
                if pos > j:
                    # Copy elements from right
                    for idx in range(j, pos):
                        arr[k] = right[idx]
                        k += 1
                    j = pos
                left_galloping = False
                left_wins = 0
                right_wins = 0
            elif right_galloping:
                # Galloping mode for right array
                pos = gallop_search(left, i, right[j])
                if pos > i:
                    # Copy elements from left
                    for idx in range(i, pos):
                        arr[k] = left[idx]
                        k += 1
                    i = pos
                right_galloping = False
                left_wins = 0
                right_wins = 0
            else:
                # Normal comparison mode
                if left[i] <= right[j]:
                    arr[k] = left[i]
                    i += 1
                    left_wins += 1
                    right_wins = 0
                    # Enter galloping mode if we win enough times
                    if left_wins >= MIN_GALLOP:
                        left_galloping = True
                else:
                    arr[k] = right[j]
                    j += 1
                    right_wins += 1
                    left_wins = 0
                    # Enter galloping mode if we win enough times
                    if right_wins >= MIN_GALLOP:
                        right_galloping = True
                k += 1
        
        # Copy remaining elements
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
    
    def gallop_search(arr, start, key):
        """Binary search to find position where key should be inserted"""
        left = start
        right = len(arr)
        
        # Exponential search to find range
        last = start
        jump = 1
        while last < len(arr) and arr[last] <= key:
            left = last
            last += jump
            jump *= 2
        
        if last >= len(arr):
            right = len(arr)
        else:
            right = last
        
        # Binary search in the found range
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= key:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    n = len(arr)
    min_run = calc_min_run(n)
    
    # Sort individual runs using binary insertion sort
    for start in range(0, n, MIN_MERGE):
        end = min(start + MIN_MERGE - 1, n - 1)
        binary_insertion_sort(arr, start, end)
    
    # Merge runs
    size = MIN_MERGE
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size - 1
            right = min((left + 2 * size - 1), n - 1)
            
            if mid < right:
                merge(arr, left, mid, right)
        
        size *= 2
    
    return arr


def intro_sort(arr):
    """Intro Sort - C++ std::sort algorithm implementation"""
    if len(arr) < 2:
        return arr[:]
    
    # Intro Sort parameters
    MAX_DEPTH = 2 * (len(arr).bit_length())  # 2 * log2(n)
    
    def insertion_sort(arr, left, right):
        """Insertion sort for small arrays"""
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
    
    def heapify(arr, n, i):
        """Heapify helper for heap sort"""
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
    
    def heap_sort(arr, left, right):
        """Heap sort for fallback"""
        n = right - left + 1
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, left + n, left + i)
        
        # Extract elements
        for i in range(n - 1, 0, -1):
            arr[left], arr[left + i] = arr[left + i], arr[left]
            heapify(arr, left + i, left)
    
    def partition(arr, low, high):
        """Partition for quick sort"""
        pivot = arr[high]
        i = low - 1
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def intro_sort_helper(arr, left, right, depth):
        """Recursive intro sort helper"""
        if right - left <= 16:  # Use insertion sort for small arrays
            insertion_sort(arr, left, right)
        elif depth == 0:  # Use heap sort when depth limit reached
            heap_sort(arr, left, right)
        else:
            pivot = partition(arr, left, right)
            intro_sort_helper(arr, left, pivot - 1, depth - 1)
            intro_sort_helper(arr, pivot + 1, right, depth - 1)
    
    intro_sort_helper(arr, 0, len(arr) - 1, MAX_DEPTH)
    return arr


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
