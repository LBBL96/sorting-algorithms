"""
Sorting Algorithms Implementation
Contains implementations of various sorting algorithms for educational purposes.
"""

import bisect

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

    MIN_MERGE = 32

    def calc_min_run(n):
        """Calculate minimum run length for Tim Sort"""
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def binary_sort(a, lo, hi, start):
        if start == lo:
            start += 1
        for i in range(start, hi):
            pivot = a[i]
            pos = bisect.bisect_right(a, pivot, lo, i)
            j = i
            while j > pos:
                a[j] = a[j - 1]
                j -= 1
            a[pos] = pivot

    def count_run_and_make_ascending(a, lo, hi):
        run_hi = lo + 1
        if run_hi == hi:
            return 1

        if a[run_hi] < a[lo]:
            run_hi += 1
            while run_hi < hi and a[run_hi] < a[run_hi - 1]:
                run_hi += 1
            a[lo:run_hi] = reversed(a[lo:run_hi])
        else:
            run_hi += 1
            while run_hi < hi and a[run_hi] >= a[run_hi - 1]:
                run_hi += 1

        return run_hi - lo

    def gallop_left(key, a, base, length, hint):
        ofs = 1
        last_ofs = 0
        if key > a[base + hint]:
            max_ofs = length - hint
            while ofs < max_ofs and key > a[base + hint + ofs]:
                last_ofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = max_ofs
            if ofs > max_ofs:
                ofs = max_ofs
            last_ofs += hint
            ofs += hint
        else:
            max_ofs = hint + 1
            while ofs < max_ofs and key <= a[base + hint - ofs]:
                last_ofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = max_ofs
            if ofs > max_ofs:
                ofs = max_ofs
            tmp = last_ofs
            last_ofs = hint - ofs
            ofs = hint - tmp
        last_ofs += 1
        while last_ofs < ofs:
            m = last_ofs + ((ofs - last_ofs) >> 1)
            if key > a[base + m]:
                last_ofs = m + 1
            else:
                ofs = m
        return ofs

    def gallop_right(key, a, base, length, hint):
        ofs = 1
        last_ofs = 0
        if key < a[base + hint]:
            max_ofs = hint + 1
            while ofs < max_ofs and key < a[base + hint - ofs]:
                last_ofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = max_ofs
            if ofs > max_ofs:
                ofs = max_ofs
            tmp = last_ofs
            last_ofs = hint - ofs
            ofs = hint - tmp
        else:
            max_ofs = length - hint
            while ofs < max_ofs and key >= a[base + hint + ofs]:
                last_ofs = ofs
                ofs = (ofs << 1) + 1
                if ofs <= 0:
                    ofs = max_ofs
            if ofs > max_ofs:
                ofs = max_ofs
            last_ofs += hint
            ofs += hint
        last_ofs += 1
        while last_ofs < ofs:
            m = last_ofs + ((ofs - last_ofs) >> 1)
            if key < a[base + m]:
                ofs = m
            else:
                last_ofs = m + 1
        return ofs

    MIN_GALLOP = 7

    class _TimSortState:
        def __init__(self, a):
            self.a = a
            self.min_gallop = MIN_GALLOP
            self.run_base = []
            self.run_len = []

        def push_run(self, base, length):
            self.run_base.append(base)
            self.run_len.append(length)

        def merge_collapse(self):
            while len(self.run_len) > 1:
                n = len(self.run_len) - 2
                if (
                    (n > 0 and self.run_len[n - 1] <= self.run_len[n] + self.run_len[n + 1])
                    or (n > 1 and self.run_len[n - 2] <= self.run_len[n - 1] + self.run_len[n])
                ):
                    if self.run_len[n - 1] < self.run_len[n + 1]:
                        n -= 1
                    self.merge_at(n)
                elif self.run_len[n] <= self.run_len[n + 1]:
                    self.merge_at(n)
                else:
                    break

        def merge_force_collapse(self):
            while len(self.run_len) > 1:
                n = len(self.run_len) - 2
                if n > 0 and self.run_len[n - 1] < self.run_len[n + 1]:
                    n -= 1
                self.merge_at(n)

        def merge_at(self, i):
            a = self.a
            base1 = self.run_base[i]
            len1 = self.run_len[i]
            base2 = self.run_base[i + 1]
            len2 = self.run_len[i + 1]

            self.run_len[i] = len1 + len2
            if i == len(self.run_len) - 3:
                self.run_base[i + 1] = self.run_base[i + 2]
                self.run_len[i + 1] = self.run_len[i + 2]
            self.run_base.pop()
            self.run_len.pop()

            k = gallop_right(a[base2], a, base1, len1, 0)
            base1 += k
            len1 -= k
            if len1 == 0:
                return

            len2 = gallop_left(a[base1 + len1 - 1], a, base2, len2, len2 - 1)
            if len2 == 0:
                return

            if len1 <= len2:
                self.merge_lo(base1, len1, base2, len2)
            else:
                self.merge_hi(base1, len1, base2, len2)

        def merge_lo(self, base1, len1, base2, len2):
            a = self.a
            tmp = a[base1 : base1 + len1]
            cursor1 = 0
            cursor2 = base2
            dest = base1

            a[dest] = a[cursor2]
            dest += 1
            cursor2 += 1
            len2 -= 1
            if len2 == 0:
                a[dest : dest + len1] = tmp[cursor1 : cursor1 + len1]
                return
            if len1 == 1:
                a[dest : dest + len2] = a[cursor2 : cursor2 + len2]
                a[dest + len2] = tmp[cursor1]
                return

            min_gallop = self.min_gallop

            while True:
                count1 = 0
                count2 = 0

                while True:
                    if a[cursor2] < tmp[cursor1]:
                        a[dest] = a[cursor2]
                        dest += 1
                        cursor2 += 1
                        count2 += 1
                        count1 = 0
                        len2 -= 1
                        if len2 == 0:
                            break
                    else:
                        a[dest] = tmp[cursor1]
                        dest += 1
                        cursor1 += 1
                        count1 += 1
                        count2 = 0
                        len1 -= 1
                        if len1 == 1:
                            break
                    if (count1 | count2) >= min_gallop:
                        break

                if len2 == 0 or len1 == 1:
                    break

                while True:
                    count1 = gallop_right(a[cursor2], tmp, cursor1, len1, 0)
                    if count1 != 0:
                        a[dest : dest + count1] = tmp[cursor1 : cursor1 + count1]
                        dest += count1
                        cursor1 += count1
                        len1 -= count1
                        if len1 <= 1:
                            break

                    a[dest] = a[cursor2]
                    dest += 1
                    cursor2 += 1
                    len2 -= 1
                    if len2 == 0:
                        break

                    count2 = gallop_left(tmp[cursor1], a, cursor2, len2, 0)
                    if count2 != 0:
                        a[dest : dest + count2] = a[cursor2 : cursor2 + count2]
                        dest += count2
                        cursor2 += count2
                        len2 -= count2
                        if len2 == 0:
                            break

                    a[dest] = tmp[cursor1]
                    dest += 1
                    cursor1 += 1
                    len1 -= 1
                    if len1 == 1:
                        break

                    min_gallop -= 1
                    if not (count1 >= MIN_GALLOP or count2 >= MIN_GALLOP):
                        break

                if min_gallop < 0:
                    min_gallop = 0
                min_gallop += 2

                if len2 == 0 or len1 == 1:
                    break

            self.min_gallop = max(1, min_gallop)

            if len1 == 1:
                a[dest : dest + len2] = a[cursor2 : cursor2 + len2]
                a[dest + len2] = tmp[cursor1]
            elif len1 > 0:
                a[dest : dest + len1] = tmp[cursor1 : cursor1 + len1]

        def merge_hi(self, base1, len1, base2, len2):
            a = self.a
            tmp = a[base2 : base2 + len2]
            cursor1 = base1 + len1 - 1
            cursor2 = len2 - 1
            dest = base2 + len2 - 1

            a[dest] = a[cursor1]
            dest -= 1
            cursor1 -= 1
            len1 -= 1
            if len1 == 0:
                a[dest - len2 + 1 : dest + 1] = tmp[0:len2]
                return
            if len2 == 1:
                dest -= len1
                cursor1 -= len1
                a[dest + 1 : dest + 1 + len1] = a[cursor1 + 1 : cursor1 + 1 + len1]
                a[dest] = tmp[cursor2]
                return

            min_gallop = self.min_gallop

            while True:
                count1 = 0
                count2 = 0

                while True:
                    if tmp[cursor2] < a[cursor1]:
                        a[dest] = a[cursor1]
                        dest -= 1
                        cursor1 -= 1
                        count1 += 1
                        count2 = 0
                        len1 -= 1
                        if len1 == 0:
                            break
                    else:
                        a[dest] = tmp[cursor2]
                        dest -= 1
                        cursor2 -= 1
                        count2 += 1
                        count1 = 0
                        len2 -= 1
                        if len2 == 1:
                            break
                    if (count1 | count2) >= min_gallop:
                        break

                if len1 == 0 or len2 == 1:
                    break

                while True:
                    count1 = len1 - gallop_right(tmp[cursor2], a, base1, len1, len1 - 1)
                    if count1 != 0:
                        dest -= count1
                        cursor1 -= count1
                        len1 -= count1
                        a[dest + 1 : dest + 1 + count1] = a[cursor1 + 1 : cursor1 + 1 + count1]
                        if len1 == 0:
                            break

                    a[dest] = tmp[cursor2]
                    dest -= 1
                    cursor2 -= 1
                    len2 -= 1
                    if len2 == 1:
                        break

                    count2 = len2 - gallop_left(a[cursor1], tmp, 0, len2, len2 - 1)
                    if count2 != 0:
                        dest -= count2
                        cursor2 -= count2
                        len2 -= count2
                        a[dest + 1 : dest + 1 + count2] = tmp[cursor2 + 1 : cursor2 + 1 + count2]
                        if len2 <= 1:
                            break

                    a[dest] = a[cursor1]
                    dest -= 1
                    cursor1 -= 1
                    len1 -= 1
                    if len1 == 0:
                        break

                    min_gallop -= 1
                    if not (count1 >= MIN_GALLOP or count2 >= MIN_GALLOP):
                        break

                if min_gallop < 0:
                    min_gallop = 0
                min_gallop += 2

                if len1 == 0 or len2 == 1:
                    break

            self.min_gallop = max(1, min_gallop)

            if len2 == 1:
                dest -= len1
                cursor1 -= len1
                a[dest + 1 : dest + 1 + len1] = a[cursor1 + 1 : cursor1 + 1 + len1]
                a[dest] = tmp[cursor2]
            elif len2 > 0:
                a[dest - len2 + 1 : dest + 1] = tmp[0:len2]

    n = len(arr)
    min_run = calc_min_run(n)
    state = _TimSortState(arr)

    lo = 0
    remaining = n
    while remaining:
        run_len = count_run_and_make_ascending(arr, lo, n)
        if run_len < min_run:
            force = min(min_run, remaining)
            binary_sort(arr, lo, lo + force, lo + run_len)
            run_len = force

        state.push_run(lo, run_len)
        state.merge_collapse()

        lo += run_len
        remaining -= run_len

    state.merge_force_collapse()
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
