from collections import Counter, defaultdict
from collections import deque
from collections import Counter
from bisect import bisect_left, bisect_right, insort
from bisect import bisect_right, insort
from bisect import bisect_left, bisect_right, insort
import heapq
from functools import cmp_to_key
import bisect

# 1. Two Pointers - Find Pair with Given Sum (Sorted Array)
def find_pair_with_sum(nums, target):
    # Assumes nums is sorted
    left, right = 0, len(nums) - 1
    while left < right:
        curr_sum = nums[left] + nums[right]
        if curr_sum == target:
            return [left, right]
        elif curr_sum < target:
            left += 1
        else:
            right -= 1
    return []


# 2. Two Pointers - Remove Duplicates from Sorted Array
def remove_duplicates(nums):
    # Returns the length of array after removing duplicates in-place
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    return write


# 3. Two Pointers - Container With Most Water
def max_area(height):
    # Find max area between two lines
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        h = min(height[left], height[right])
        w = right - left
        max_area = max(max_area, h * w)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_area


# 4. Two Pointers - Merge Two Sorted Arrays
def merge_sorted_arrays(nums1, m, nums2, n):
    # Merge nums2 into nums1, assuming nums1 has enough space
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    while j >= 0:
        nums1[k] = nums2[j]
        j -= 1
        k -= 1


# 5. Two Pointers - Intersection of Two Arrays II
def intersect(nums1, nums2):
    # Returns intersection including duplicates
    nums1.sort()
    nums2.sort()
    i = j = 0
    res = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return res


# 6. Two Pointers - Move Zeroes to End
def move_zeroes(nums):
    # Moves all zeroes to end, maintains order of non-zero elements
    insert_pos = 0
    for num in nums:
        if num != 0:
            nums[insert_pos] = num
            insert_pos += 1
    for i in range(insert_pos, len(nums)):
        nums[i] = 0


# 7. Two Pointers - Sort Array of 0s, 1s, and 2s (Dutch National Flag)
def sort_colors(nums):
    # Sorts array in-place
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1


# 8. Two Pointers - Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    # Uses sliding window with two pointers
    char_index = {}
    left = max_len = 0
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len


# 9. Two Pointers - Trapping Rain Water
def trap(height):
    # Calculates total trapped water
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water


    # 10. Two Pointers - Valid Palindrome II
def valid_palindrome(s):
    # Returns True if can be palindrome by removing at most one char
    def is_palindrome(l, r):
        while l < r:
            if s[l] != s[r]:
                return False  # Not a palindrome
            l += 1
            r -= 1
        return True

    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # Try skipping left or right character
            return is_palindrome(left + 1, right) or is_palindrome(
                left, right - 1)
        left += 1
        right -= 1
    return True


# 11. Two Pointers - Partition Array Around a Pivot
def partition_array(nums, pivot):
    # Partitions nums so that elements < pivot come before >= pivot
    left, right = 0, len(nums) - 1
    while left <= right:
        if nums[left] < pivot:
            left += 1  # Already on correct side
        elif nums[right] >= pivot:
            right -= 1  # Already on correct side
        else:
            nums[left], nums[right] = nums[right], nums[left]  # Swap
            left += 1
            right -= 1


# 12. Two Pointers - Squares of a Sorted Array
def sorted_squares(nums):
    # Returns sorted squares of nums
    n = len(nums)
    res = [0] * n
    left, right = 0, n - 1
    pos = n - 1
    while left <= right:
        if abs(nums[left]) > abs(nums[right]):
            res[pos] = nums[left]**2  # Square left
            left += 1
        else:
            res[pos] = nums[right]**2  # Square right
            right -= 1
        pos -= 1
    return res


# 13. Two Pointers - Find All Triplets That Sum to Zero
def three_sum(nums):
    # Returns list of triplets that sum to zero
    nums.sort()
    res = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # Skip duplicates
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                res.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1  # Skip duplicates
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1  # Skip duplicates
            elif s < 0:
                left += 1
            else:
                right -= 1
    return res


# 14. Two Pointers - Minimum Size Subarray Sum
def min_subarray_len(target, nums):
    # Returns min length of subarray with sum >= target
    left = 0
    curr_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        curr_sum += nums[right]
        while curr_sum >= target:
            min_len = min(min_len, right - left + 1)
            curr_sum -= nums[left]
            left += 1
    return 0 if min_len == float('inf') else min_len


# 15. Two Pointers - Subarray Product Less Than K
def num_subarray_product_less_than_k(nums, k):
    # Returns count of subarrays with product < k
    if k <= 1:
        return 0
    prod = 1
    left = 0
    count = 0
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod //= nums[left]
            left += 1
        count += right - left + 1
    return count


# 16. Two Pointers - Remove Element In-Place
def remove_element(nums, val):
    # Removes all instances of val in-place, returns new length
    write = 0
    for read in range(len(nums)):
        if nums[read] != val:
            nums[write] = nums[read]  # Overwrite with non-val
            write += 1
    return write


# 17. Two Pointers - Find K Closest Elements
def find_closest_elements(arr, k, x):
    # Returns k closest elements to x in sorted order
    left, right = 0, len(arr) - k
    while left < right:
        mid = (left + right) // 2
        if x - arr[mid] > arr[mid + k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left + k]


# 18. Two Pointers - Sort Array By Parity
def sort_array_by_parity(nums):
    # Sorts nums so that even numbers come before odd numbers
    left, right = 0, len(nums) - 1
    while left < right:
        if nums[left] % 2 == 0:
            left += 1  # Even, correct side
        elif nums[right] % 2 == 1:
            right -= 1  # Odd, correct side
        else:
            nums[left], nums[right] = nums[right], nums[left]  # Swap
            left += 1
            right -= 1
    return nums


# 19. Two Pointers - Merge Intervals (Based on sorted ends)
def merge_intervals(intervals):
    # Merges overlapping intervals
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])  # Sort by start
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)  # Merge
        else:
            merged.append([start, end])
    return merged


# 20. Two Pointers - Valid Mountain Array
def valid_mountain_array(arr):
    # Returns True if arr is a valid mountain array
    n = len(arr)
    if n < 3:
        return False
    left, right = 0, n - 1
    while left + 1 < n and arr[left] < arr[left + 1]:
        left += 1  # Walk up
    while right - 1 > 0 and arr[right] < arr[right - 1]:
        right -= 1  # Walk down
    return 0 < left == right < n - 1


# 21. Two Pointers - Check If Array Is Monotonic
def is_monotonic(nums):
    # Returns True if nums is monotonic
    increasing = decreasing = True
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            decreasing = False
        if nums[i] < nums[i - 1]:
            increasing = False
    return increasing or decreasing


# 22. Two Pointers - Sum of Two Arrays
def sum_two_arrays(arr1, arr2):
    # Returns element-wise sum of two arrays
    n = max(len(arr1), len(arr2))
    res = []
    carry = 0
    i, j = len(arr1) - 1, len(arr2) - 1
    while i >= 0 or j >= 0 or carry:
        x = arr1[i] if i >= 0 else 0
        y = arr2[j] if j >= 0 else 0
        total = x + y + carry
        res.append(total % 10)
        carry = total // 10
        i -= 1
        j -= 1
    return res[::-1]


# 23. Two Pointers - Rotate Array by K Steps
def rotate(nums, k):
    # Rotates nums to the right by k steps
    n = len(nums)
    k %= n
    nums[:] = nums[-k:] + nums[:-k]  # Slice and concatenate


# 24. Two Pointers - Subarrays with Equal Number of 0s and 1s
def count_subarrays_equal_0_1(nums):
    # Returns count of subarrays with equal 0s and 1s
    count = 0
    prefix_sum = 0
    d = {0: 1}
    for num in nums:
        prefix_sum += 1 if num == 1 else -1
        count += d.get(prefix_sum, 0)
        d[prefix_sum] = d.get(prefix_sum, 0) + 1
    return count


# 25. Two Pointers - Max Consecutive Ones III
def longest_ones(nums, k):
    # Returns max length of subarray with at most k zeros
    left = 0
    zeros = 0
    max_len = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 26. Two Pointers - Longest Repeating Character Replacement
def character_replacement(s, k):
    # Returns length of longest substring with at most k replacements
    count = {}
    max_count = 0
    left = 0
    res = 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_count = max(max_count, count[s[right]])
        while (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1
        res = max(res, right - left + 1)
    return res


# 27. Two Pointers - Split Array into Consecutive Subsequences
def is_possible(nums):
    # Returns True if can split into consecutive subsequences
    count = Counter(nums)
    end = defaultdict(int)
    for x in nums:
        if count[x] == 0:
            continue
        if end[x - 1] > 0:
            end[x - 1] -= 1
            end[x] += 1
        elif count[x + 1] > 0 and count[x + 2] > 0:
            count[x + 1] -= 1
            count[x + 2] -= 1
            end[x + 2] += 1
        else:
            return False
        count[x] -= 1
    return True


# 28. Two Pointers - Find Duplicate Number
def find_duplicate(nums):
    # Uses Floyd's Tortoise and Hare (cycle detection)
    slow = fast = 0
    while True:
        slow = nums[slow]  # Move one step
        fast = nums[nums[fast]]  # Move two steps
        if slow == fast:
            break
    slow2 = 0
    while slow != slow2:
        slow = nums[slow]
        slow2 = nums[slow2]
    return slow


# 29. Two Pointers - Maximise Distance to Closest Person
def max_dist_to_closest(seats):
    # Returns max distance to closest person
    n = len(seats)
    prev = -1
    max_dist = 0
    for i, seat in enumerate(seats):
        if seat == 1:
            if prev == -1:
                max_dist = i  # Leading zeros
            else:
                max_dist = max(max_dist, (i - prev) // 2)  # Middle zeros
            prev = i
    max_dist = max(max_dist, n - 1 - prev)  # Trailing zeros
    return max_dist


# 30. Two Pointers - Find Peak Element
def find_peak_element(nums):
    # Returns index of a peak element
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1  # Peak is to the right
        else:
            right = mid  # Peak is at mid or to the left
    return left


    # 31. Sliding Window - Maximum Sum Subarray of Size K
def max_sum_subarray_k(nums, k):
    # Finds max sum of any subarray of size k
    max_sum = curr_sum = sum(nums[:k])
    for i in range(k, len(nums)):
        curr_sum += nums[i] - nums[i - k]  # Slide window
        max_sum = max(max_sum, curr_sum)
    return max_sum


# 32. Sliding Window - Minimum Window Substring
def min_window(s, t):
    # Finds the minimum window in s containing all chars of t
    need = Counter(t)
    missing = len(t)
    left = start = end = 0
    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if end == 0 or right - left < end - start:
                start, end = left, right
            need[s[left]] += 1
            missing += 1
            left += 1
    return s[start:end]


# 33. Sliding Window - Longest Substring with At Most K Distinct Characters
def length_of_longest_substring_k_distinct(s, k):
    # Returns length of longest substring with at most k distinct chars
    left = 0
    count = {}
    max_len = 0
    for right, char in enumerate(s):
        count[char] = count.get(char, 0) + 1
        while len(count) > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 34. Sliding Window - Count Number of Anagrams
def count_anagrams(s, p):
    # Counts number of anagrams of p in s
    res = 0
    need = Counter(p)
    window = Counter()
    for i, char in enumerate(s):
        window[char] += 1
        if i >= len(p):
            left_char = s[i - len(p)]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
        if window == need:
            res += 1
    return res


# 35. Sliding Window - Find All Anagrams in a String
def find_anagrams(s, p):
    # Returns starting indices of p's anagrams in s
    res = []
    need = Counter(p)
    window = Counter()
    for i, char in enumerate(s):
        window[char] += 1
        if i >= len(p):
            left_char = s[i - len(p)]
            window[left_char] -= 1
            if window[left_char] == 0:
                del window[left_char]
        if window == need:
            res.append(i - len(p) + 1)
    return res


# 36. Sliding Window - Longest Repeating Character Replacement
def character_replacement_sliding(s, k):
    # Returns length of longest substring with at most k replacements
    count = {}
    max_count = 0
    left = 0
    res = 0
    for right in range(len(s)):
        count[s[right]] = count.get(s[right], 0) + 1
        max_count = max(max_count, count[s[right]])
        while (right - left + 1) - max_count > k:
            count[s[left]] -= 1
            left += 1
        res = max(res, right - left + 1)
    return res


# 37. Sliding Window - Fruit Into Baskets
def total_fruit(tree):
    # Returns max number of fruits in two baskets (at most 2 types)
    count = {}
    left = 0
    max_fruits = 0
    for right, fruit in enumerate(tree):
        count[fruit] = count.get(fruit, 0) + 1
        while len(count) > 2:
            count[tree[left]] -= 1
            if count[tree[left]] == 0:
                del count[tree[left]]
            left += 1
        max_fruits = max(max_fruits, right - left + 1)
    return max_fruits


# 38. Sliding Window - Subarrays with K Different Integers
def subarrays_with_k_distinct(nums, k):
    # Returns number of subarrays with exactly k distinct integers
    def at_most_k(k):
        count = defaultdict(int)
        left = res = 0
        for right, x in enumerate(nums):
            count[x] += 1
            while len(count) > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1
            res += right - left + 1
        return res

    return at_most_k(k) - at_most_k(k - 1)


# 39. Sliding Window - Maximum Number of Vowels in Substring
def max_vowels(s, k):
    # Returns max number of vowels in any substring of length k
    vowels = set('aeiou')
    curr = sum(1 for c in s[:k] if c in vowels)
    res = curr
    for i in range(k, len(s)):
        curr += (s[i] in vowels) - (s[i - k] in vowels)
        res = max(res, curr)
    return res


# 40. Sliding Window - Minimum Size Subarray Sum
def min_subarray_len_sliding(target, nums):
    # Returns min length of subarray with sum >= target
    left = 0
    curr_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        curr_sum += nums[right]
        while curr_sum >= target:
            min_len = min(min_len, right - left + 1)
            curr_sum -= nums[left]
            left += 1
    return 0 if min_len == float('inf') else min_len


# 41. Sliding Window - Number of Subarrays with Sum K
def subarray_sum(nums, k):
    # Returns number of subarrays with sum == k
    count = defaultdict(int)
    count[0] = 1
    res = curr_sum = 0
    for num in nums:
        curr_sum += num
        res += count[curr_sum - k]
        count[curr_sum] += 1
    return res


# 42. Sliding Window - Longest Subarray with Ones After Replacement
def longest_ones_sliding(nums, k):
    # Returns max length of subarray with at most k zeros
    left = 0
    zeros = 0
    max_len = 0
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 43. Sliding Window - Longest Substring Without Repeating Characters
def length_of_longest_substring_sliding(s):
    # Returns length of longest substring without repeating chars
    char_index = {}
    left = max_len = 0
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len


# 44. Sliding Window - Maximum Average Subarray
def find_max_average(nums, k):
    # Returns max average of any subarray of length k
    curr_sum = sum(nums[:k])
    max_sum = curr_sum
    for i in range(k, len(nums)):
        curr_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, curr_sum)
    return max_sum / k


# 45. Sliding Window - Count Number of Nice Subarrays
def number_of_subarrays(nums, k):
    # Returns number of subarrays with exactly k odd numbers
    count = defaultdict(int)
    count[0] = 1
    res = curr = 0
    for num in nums:
        curr += num % 2
        res += count[curr - k]
        count[curr] += 1
    return res


# 46. Sliding Window - Longest Substring with At Most Two Distinct Characters
def length_of_longest_substring_two_distinct(s):
    # Returns length of longest substring with at most 2 distinct chars
    left = 0
    count = {}
    max_len = 0
    for right, char in enumerate(s):
        count[char] = count.get(char, 0) + 1
        while len(count) > 2:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 47. Sliding Window - Longest Substring with Exactly K Distinct Characters
def length_of_longest_substring_exactly_k_distinct(s, k):
    # Returns length of longest substring with exactly k distinct chars
    left = 0
    count = {}
    max_len = 0
    for right, char in enumerate(s):
        count[char] = count.get(char, 0) + 1
        while len(count) > k:
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1
        if len(count) == k:
            max_len = max(max_len, right - left + 1)
    return max_len


# 48. Sliding Window - Longest Substring Without Repeating Vowels
def longest_substring_no_repeating_vowels(s):
    # Returns length of longest substring without repeating vowels
    vowels = set('aeiou')
    last_seen = {}
    left = max_len = 0
    for right, char in enumerate(s):
        if char in vowels:
            if char in last_seen and last_seen[char] >= left:
                left = last_seen[char] + 1
            last_seen[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len


# 49. Sliding Window - Subarray Product Less Than K
def num_subarray_product_less_than_k_sliding(nums, k):
    # Returns count of subarrays with product < k
    if k <= 1:
        return 0
    prod = 1
    left = 0
    count = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        count += right - left + 1
    return count


# 50. Sliding Window - Number of Subarrays with Bounded Maximum
def num_subarray_bounded_max(nums, left, right):
    # Returns number of subarrays where max is in [left, right]
    def count(bound):
        ans = curr = 0
        for x in nums:
            curr = curr + 1 if x <= bound else 0
            ans += curr
        return ans

    return count(right) - count(left - 1)


# 51. Sliding Window - Longest Subarray with At Most K Odd Numbers
def longest_subarray_k_odds(nums, k):
    # Returns length of longest subarray with at most k odd numbers
    left = 0
    odds = 0
    max_len = 0
    for right, num in enumerate(nums):
        if num % 2 == 1:
            odds += 1
        while odds > k:
            if nums[left] % 2 == 1:
                odds -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 52. Sliding Window - Minimum Window Containing All Characters
def min_window_all_chars(s, t):
    # Finds the minimum window in s containing all chars of t
    need = Counter(t)
    missing = len(t)
    left = start = end = 0
    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if end == 0 or right - left < end - start:
                start, end = left, right
            need[s[left]] += 1
            missing += 1
            left += 1
    return s[start:end]


# 53. Sliding Window - Longest Substring With At Least K Repeating Characters
def longest_substring_at_least_k_repeats(s, k):
    # Returns length of longest substring with at least k repeats for each char
    max_len = 0
    for unique in range(1, 27):
        count = [0] * 26
        left = right = 0
        curr_unique = curr_k = 0
        while right < len(s):
            idx = ord(s[right]) - ord('a')
            if count[idx] == 0:
                curr_unique += 1
            count[idx] += 1
            if count[idx] == k:
                curr_k += 1
            right += 1
            while curr_unique > unique:
                idx2 = ord(s[left]) - ord('a')
                if count[idx2] == k:
                    curr_k -= 1
                count[idx2] -= 1
                if count[idx2] == 0:
                    curr_unique -= 1
                left += 1
            if curr_unique == curr_k:
                max_len = max(max_len, right - left)
    return max_len


# 54. Sliding Window - Maximum Number of Balls in a Box
def count_balls(lowLimit, highLimit):
    # Returns the max number of balls in any box (box = sum of digits)
    boxes = defaultdict(int)
    for i in range(lowLimit, highLimit + 1):
        box = sum(int(d) for d in str(i))
        boxes[box] += 1
    return max(boxes.values())


# 55. Sliding Window - Longest Continuous Subarray with Absolute Diff ≤ Limit
def longest_subarray_abs_diff(nums, limit):
    # Returns length of longest subarray with abs diff <= limit
    max_d = deque()
    min_d = deque()
    left = 0
    res = 0
    for right, num in enumerate(nums):
        while max_d and num > max_d[-1]:
            max_d.pop()
        while min_d and num < min_d[-1]:
            min_d.pop()
        max_d.append(num)
        min_d.append(num)
        while max_d[0] - min_d[0] > limit:
            if max_d[0] == nums[left]:
                max_d.popleft()
            if min_d[0] == nums[left]:
                min_d.popleft()
            left += 1
        res = max(res, right - left + 1)
    return res


# 56. Sliding Window - Maximum Number of Non-Overlapping Subarrays with Sum Equals Target
def max_non_overlapping(nums, target):
    # Returns max number of non-overlapping subarrays with sum == target
    seen = set([0])
    curr_sum = res = 0
    for num in nums:
        curr_sum += num
        if curr_sum - target in seen:
            res += 1
            curr_sum = 0
            seen = set([0])
        else:
            seen.add(curr_sum)
    return res


# 57. Sliding Window - Number of Subarrays With Odd Sum
def num_of_subarrays_with_odd_sum(arr):
    # Returns number of subarrays with odd sum
    odd = even = 0
    res = 0
    curr = 0
    for num in arr:
        curr = (curr + num) % 2
        if curr == 1:
            res += 1 + even
            odd += 1
        else:
            res += odd
            even += 1
    return res


# 58. Sliding Window - Count Subarrays with Median K
def count_subarrays_with_median_k(nums, k):
    # Returns number of subarrays where median is k
    idx = nums.index(k)
    count = Counter()
    count[0] = 1
    balance = 0
    res = 0
    for i in range(idx + 1, len(nums)):
        balance += 1 if nums[i] > k else -1
        count[balance] += 1
    balance = 0
    for i in range(idx, -1, -1):
        balance += 1 if nums[i] > k else -1
        res += count[-balance]
    return res


# 59. Sliding Window - Longest Substring Without Repeating Characters with Replacement
def length_of_longest_substring_no_repeat_with_replacement(s, k):
    # Returns length of longest substring with at most k replacements to avoid repeats
    count = {}
    left = max_len = 0
    for right, char in enumerate(s):
        count[char] = count.get(char, 0) + 1
        while (right - left + 1) - max(count.values()) > k:
            count[s[left]] -= 1
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# 60. Sliding Window - Count Good Substrings
def count_good_substrings(s):
    # Returns number of substrings of length 3 with all unique chars
    res = 0
    for i in range(len(s) - 2):
        window = s[i:i + 3]
        if len(set(window)) == 3:
            res += 1
    return res


    # 61. Prefix Sum - Subarray Sum Equals K
def subarray_sum_equals_k(nums, k):
    # Returns number of subarrays with sum == k
    count = defaultdict(int)
    count[0] = 1
    curr_sum = res = 0
    for num in nums:
        curr_sum += num
        res += count[curr_sum - k]
        count[curr_sum] += 1
    return res


# 62. Prefix Sum - Find Equilibrium Index
def equilibrium_index(nums):
    # Returns list of indices where left sum == right sum
    total = sum(nums)
    left_sum = 0
    res = []
    for i, num in enumerate(nums):
        if left_sum == total - left_sum - num:
            res.append(i)
        left_sum += num
    return res


# 63. Prefix Sum - Count Subarrays with Sum Divisible by K
def subarrays_div_by_k(nums, k):
    # Returns number of subarrays with sum % k == 0
    count = defaultdict(int)
    count[0] = 1
    curr_sum = res = 0
    for num in nums:
        curr_sum = (curr_sum + num) % k
        res += count[curr_sum]
        count[curr_sum] += 1
    return res


# 64. Prefix Sum - Maximum Size Subarray Sum Equals K
def max_subarray_len(nums, k):
    # Returns max length of subarray with sum == k
    prefix = {0: -1}
    curr_sum = res = 0
    for i, num in enumerate(nums):
        curr_sum += num
        if curr_sum - k in prefix:
            res = max(res, i - prefix[curr_sum - k])
        if curr_sum not in prefix:
            prefix[curr_sum] = i
    return res


# 65. Prefix Sum - Continuous Subarray Sum
def check_subarray_sum(nums, k):
    # Returns True if there is a subarray of at least length 2 with sum % k == 0
    prefix = {0: -1}
    curr_sum = 0
    for i, num in enumerate(nums):
        curr_sum += num
        mod = curr_sum % k if k != 0 else curr_sum
        if mod in prefix:
            if i - prefix[mod] > 1:
                return True
        else:
            prefix[mod] = i
    return False


# 66. Prefix Sum - Number of Subarrays with Bounded Maximum
def num_subarray_bounded_max_prefix(nums, left, right):
    # Returns number of subarrays where max is in [left, right]
    def count(bound):
        res = curr = 0
        for x in nums:
            curr = curr + 1 if x <= bound else 0
            res += curr
        return res

    return count(right) - count(left - 1)


# 67. Prefix Sum - Range Sum Query
class NumArray:
    # Supports sumRange(i, j) in O(1) after O(n) preprocessing
    def __init__(self, nums):
        self.prefix = [0]
        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, i, j):
        return self.prefix[j + 1] - self.prefix[i]


    # 68. Prefix Sum - Find Pivot Index
def pivot_index(nums):
    # Returns the index where left sum == right sum
    total = sum(nums)
    left_sum = 0
    for i, num in enumerate(nums):
        if left_sum == total - left_sum - num:
            return i
        left_sum += num
    return -1


# 69. Prefix Sum - Count Ways to Split Array into Equal Sum Parts
def ways_to_split_equal_sum(nums):
    # Returns number of ways to split into 2 parts with equal sum
    total = sum(nums)
    if total % 2 != 0:
        return 0
    half = total // 2
    curr = res = 0
    for i in range(len(nums) - 1):
        curr += nums[i]
        if curr == half:
            res += 1
    return res


# 70. Prefix Sum - Find Longest Balanced Subarray
def longest_balanced_subarray(nums):
    # Returns length of longest subarray with equal 0s and 1s
    d = {0: -1}
    max_len = curr = 0
    for i, num in enumerate(nums):
        curr += 1 if num == 1 else -1
        if curr in d:
            max_len = max(max_len, i - d[curr])
        else:
            d[curr] = i
    return max_len


# 71. Prefix Sum - Number of Subarrays with Sum in Range
def num_subarrays_with_sum_in_range(nums, lower, upper):
    # Returns number of subarrays with sum in [lower, upper]
    prefix = [0]
    curr = 0
    res = 0
    for num in nums:
        curr += num
        left = bisect_left(prefix, curr - upper)
        right = bisect_right(prefix, curr - lower)
        res += right - left
        insort(prefix, curr)
    return res


# 72. Prefix Sum - Maximum Length of Subarray With Positive Product
def get_max_len(nums):
    # Returns max length of subarray with positive product
    pos = neg = res = 0
    for num in nums:
        if num == 0:
            pos = neg = 0
        elif num > 0:
            pos += 1
            neg = neg + 1 if neg else 0
        else:
            pos, neg = neg, pos
            neg = neg + 1 if neg else 0
            pos = pos + 1 if pos else 0
        res = max(res, pos)
    return res


# 73. Prefix Sum - Count Number of Subarrays with Equal Number of 0s and 1s
def count_subarrays_equal_0_1_prefix(nums):
    # Returns count of subarrays with equal 0s and 1s
    d = defaultdict(int)
    d[0] = 1
    curr = res = 0
    for num in nums:
        curr += 1 if num == 1 else -1
        res += d[curr]
        d[curr] += 1
    return res


# 74. Prefix Sum - Number of Subarrays with Sum Equals Target
def num_subarrays_with_sum(nums, target):
    # Returns number of subarrays with sum == target
    count = defaultdict(int)
    count[0] = 1
    curr_sum = res = 0
    for num in nums:
        curr_sum += num
        res += count[curr_sum - target]
        count[curr_sum] += 1
    return res


# 75. Prefix Sum - Count Subarrays with Sum Less Than K
def count_subarrays_sum_less_than_k(nums, k):
    # Returns number of subarrays with sum < k
    res = 0
    curr_sum = left = 0
    for right, num in enumerate(nums):
        curr_sum += num
        while curr_sum >= k and left <= right:
            curr_sum -= nums[left]
            left += 1
        res += right - left + 1
    return res


# 76. Prefix Sum - Find Total Strength of Wizards
def total_strength(wizards):
    # Returns total strength as per Leetcode 2281
    MOD = 10**9 + 7
    n = len(wizards)
    stack = []
    left = [-1] * n
    right = [n] * n
    for i in range(n):
        while stack and wizards[stack[-1]] >= wizards[i]:
            right[stack.pop()] = i
        stack.append(i)
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and wizards[stack[-1]] > wizards[i]:
            left[stack.pop()] = i
        stack.append(i)
    prefix = [0] * (n + 2)
    for i in range(n):
        prefix[i + 1] = (prefix[i] + wizards[i]) % MOD
    prefix2 = [0] * (n + 2)
    for i in range(n + 1):
        prefix2[i + 1] = (prefix2[i] + prefix[i]) % MOD
    res = 0
    for i in range(n):
        l, r = left[i], right[i]
        total = (prefix2[r + 1] - prefix2[i + 1]) * (i - l) % MOD
        total -= (prefix2[i + 1] - prefix2[l + 1]) * (r - i) % MOD
        res = (res + wizards[i] * total) % MOD
    return res


# 77. Prefix Sum - Sum of Subarray Minimums
def sum_subarray_mins(arr):
    # Returns sum of minimums of all subarrays
    MOD = 10**9 + 7
    n = len(arr)
    stack = []
    left = [0] * n
    right = [0] * n
    for i in range(n):
        count = 1
        while stack and arr[stack[-1][0]] > arr[i]:
            count += stack.pop()[1]
        left[i] = count
        stack.append((i, count))
    stack = []
    for i in range(n - 1, -1, -1):
        count = 1
        while stack and arr[stack[-1][0]] >= arr[i]:
            count += stack.pop()[1]
        right[i] = count
        stack.append((i, count))
    return sum(a * l * r for a, l, r in zip(arr, left, right)) % MOD


# 78. Prefix Sum - Sum of Subarray Ranges
def subarray_ranges(nums):
    # Returns sum of (max - min) for all subarrays
    n = len(nums)
    res = 0
    stack = []
    # Sum of subarray maximums
    for sign in [1, -1]:
        stack = []
        for i in range(n + 1):
            curr = nums[i] if i < n else (
                float('-inf') if sign == 1 else float('inf'))
            while stack and (curr > nums[stack[-1]]
                             if sign == 1 else curr < nums[stack[-1]]):
                j = stack.pop()
                k = stack[-1] if stack else -1
                res += sign * nums[j] * (i - j) * (j - k)
            stack.append(i)
    return res


# 79. Prefix Sum - Find Number of Subarrays with Average Greater Than or Equal to K
def num_subarrays_avg_ge_k(nums, k):
    # Returns number of subarrays with average >= k
    n = len(nums)
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num - k)
    res = 0
    sorted_prefix = []
    for p in prefix:
        # Count number of prefix sums <= current
        res += bisect_right(sorted_prefix, p)
        insort(sorted_prefix, p)
    return res


# 80. Prefix Sum - Count of Range Sum
def count_range_sum(nums, lower, upper):
    # Returns number of range sums in [lower, upper]
    prefix = [0]
    curr = 0
    res = 0
    for num in nums:
        curr += num
        left = bisect_left(prefix, curr - upper)
        right = bisect_right(prefix, curr - lower)
        res += right - left
        insort(prefix, curr)
    return res


# 81. Prefix Sum - Minimum Size Subarray Sum
def min_subarray_len_prefix(target, nums):
    # Returns min length of subarray with sum >= target
    n = len(nums)
    left = curr_sum = 0
    min_len = float('inf')
    for right in range(n):
        curr_sum += nums[right]
        while curr_sum >= target:
            min_len = min(min_len, right - left + 1)
            curr_sum -= nums[left]
            left += 1
    return 0 if min_len == float('inf') else min_len


# 82. Prefix Sum - Maximum Average Subarray II
def find_max_average_ii(nums, k):
    # Returns max average of subarray with length >= k
    n = len(nums)
    left, right = min(nums), max(nums)
    eps = 1e-5

    def check(mid):
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num - mid)
        min_prefix = 0
        for i in range(k, n + 1):
            if prefix[i] - min_prefix >= 0:
                return True
            min_prefix = min(min_prefix, prefix[i - k + 1])
        return False

    while right - left > eps:
        mid = (left + right) / 2
        if check(mid):
            left = mid
        else:
            right = mid
    return left


# 83. Prefix Sum - Find Longest Arithmetic Subarray
def longest_arithmetic_subarray(nums):
    # Returns length of longest arithmetic subarray
    if len(nums) < 2:
        return len(nums)
    max_len = curr = 2
    diff = nums[1] - nums[0]
    for i in range(2, len(nums)):
        if nums[i] - nums[i - 1] == diff:
            curr += 1
        else:
            diff = nums[i] - nums[i - 1]
            curr = 2
        max_len = max(max_len, curr)
    return max_len


# 84. Prefix Sum - Count Subarrays with Product Less Than K
def num_subarray_product_less_than_k_prefix(nums, k):
    # Returns count of subarrays with product < k
    if k <= 1:
        return 0
    prod = 1
    left = res = 0
    for right, val in enumerate(nums):
        prod *= val
        while prod >= k:
            prod //= nums[left]
            left += 1
        res += right - left + 1
    return res


# 85. Prefix Sum - Sum of All Odd Length Subarrays
def sum_odd_length_subarrays(arr):
    # Returns sum of all odd length subarrays
    n = len(arr)
    res = 0
    for i in range(n):
        total = (i + 1) * (n - i)
        odd = (total + 1) // 2
        res += arr[i] * odd
    return res


# 86. Prefix Sum - Find Number of Subarrays with Sum Divisible by M
def subarrays_div_by_m(nums, m):
    # Returns number of subarrays with sum % m == 0
    count = defaultdict(int)
    count[0] = 1
    curr = res = 0
    for num in nums:
        curr = (curr + num) % m
        res += count[curr]
        count[curr] += 1
    return res


# 87. Prefix Sum - Number of Ways to Split Array Into Three Subarrays with Equal Sum
def ways_to_split_three_equal_sum(nums):
    # Returns number of ways to split into 3 parts with equal sum
    total = sum(nums)
    if total % 3 != 0:
        return 0
    part = total // 3
    n = len(nums)
    count = 0
    curr = 0
    ways = 0
    for i in range(n - 1):
        curr += nums[i]
        if curr == 2 * part:
            ways += count
        if curr == part:
            count += 1
    return ways


# 88. Prefix Sum - Find Subarray with Given Sum
def find_subarray_with_sum(nums, target):
    # Returns indices [start, end] of subarray with sum == target
    d = {0: -1}
    curr = 0
    for i, num in enumerate(nums):
        curr += num
        if curr - target in d:
            return [d[curr - target] + 1, i]
        d[curr] = i
    return []


# 89. Prefix Sum - Maximum Length of Subarray With Sum K
def max_len_subarray_sum_k(nums, k):
    # Returns max length of subarray with sum == k
    d = {0: -1}
    curr = res = 0
    for i, num in enumerate(nums):
        curr += num
        if curr - k in d:
            res = max(res, i - d[curr - k])
        if curr not in d:
            d[curr] = i
    return res


# 90. Prefix Sum - Number of Subarrays With Exactly K Odd Numbers
def number_of_subarrays_exactly_k_odds(nums, k):
    # Returns number of subarrays with exactly k odd numbers
    count = defaultdict(int)
    count[0] = 1
    curr = res = 0
    for num in nums:
        curr += num % 2
        res += count[curr - k]
        count[curr] += 1
    return res


    # 91. Binary Search - Find Peak Element
def find_peak_element_bs(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left


# 92. Binary Search - Find Minimum in Rotated Sorted Array
def find_min_rotated(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


# 93. Binary Search - Find Kth Smallest Element in Sorted Matrix
def kth_smallest_matrix(matrix, k):
    n = len(matrix)

    def count_less_equal(x):
        count = 0
        row, col = n - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] <= x:
                count += row + 1
                col += 1
            else:
                row -= 1
        return count

    left, right = matrix[0][0], matrix[-1][-1]
    while left < right:
        mid = (left + right) // 2
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    return left


# 94. Binary Search - Search in Rotated Sorted Array
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


# 95. Binary Search - Find First and Last Position of Element in Sorted Array
def search_range(nums, target):

    def find_bound(is_left):
        left, right = 0, len(nums) - 1
        bound = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                bound = mid
                if is_left:
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return bound

    return [find_bound(True), find_bound(False)]


# 96. Binary Search - Median of Two Sorted Arrays
def find_median_sorted_arrays(nums1, nums2):
    A, B = nums1, nums2
    if len(A) > len(B):
        A, B = B, A
    m, n = len(A), len(B)
    total = m + n
    half = total // 2
    left, right = 0, m
    while left <= right:
        i = (left + right) // 2
        j = half - i
        Aleft = A[i - 1] if i > 0 else float('-inf')
        Aright = A[i] if i < m else float('inf')
        Bleft = B[j - 1] if j > 0 else float('-inf')
        Bright = B[j] if j < n else float('inf')
        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min(Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        elif Aleft > Bright:
            right = i - 1
        else:
            left = i + 1


# 97. Binary Search - Find Smallest Letter Greater Than Target
def next_greatest_letter(letters, target):
    left, right = 0, len(letters)
    while left < right:
        mid = (left + right) // 2
        if letters[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return letters[left % len(letters)]


# 98. Binary Search - Split Array Largest Sum
def split_array_largest_sum(nums, m):

    def can_split(max_sum):
        count, curr = 1, 0
        for num in nums:
            if curr + num > max_sum:
                count += 1
                curr = 0
            curr += num
        return count <= m

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left


# 99. Binary Search - Capacity To Ship Packages Within D Days
def ship_within_days(weights, D):

    def can_ship(cap):
        days, curr = 1, 0
        for w in weights:
            if curr + w > cap:
                days += 1
                curr = 0
            curr += w
        return days <= D

    left, right = max(weights), sum(weights)
    while left < right:
        mid = (left + right) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    return left


# 100. Binary Search - Koko Eating Bananas
def min_eating_speed(piles, H):
    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        hours = sum((pile + mid - 1) // mid for pile in piles)
        if hours > H:
            left = mid + 1
        else:
            right = mid
    return left


# 101. Binary Search - Find Position to Insert Element
def search_insert(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left


# 102. Binary Search - Find Peak Index in Mountain Array
def peak_index_in_mountain_array(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left


# 103. Binary Search - Search in 2D Matrix
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    while left <= right:
        mid = (left + right) // 2
        val = matrix[mid // n][mid % n]
        if val == target:
            return True
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return False


# 104. Binary Search - Find Element in Infinite Sorted Array
def search_in_infinite_array(reader, target):
    left, right = 0, 1
    while reader[right] < target:
        left = right
        right *= 2
    while left <= right:
        mid = (left + right) // 2
        val = reader[mid]
        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# 105. Binary Search - Find Fixed Point (Index equals Value)
def fixed_point(nums):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == mid:
            return mid
        elif nums[mid] < mid:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# 106. Binary Search - Find First Bad Version
def first_bad_version(n, isBadVersion):
    left, right = 1, n
    while left < right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left


# 107. Binary Search - Find Square Root of Number
def my_sqrt(x):
    left, right = 0, x
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right


# 108. Binary Search - Find Minimum in Rotated Sorted Array II
def find_min_rotated_ii(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        elif nums[mid] < nums[right]:
            right = mid
        else:
            right -= 1
    return nums[left]


# 109. Binary Search - Find Median in Data Stream
class MedianFinder:

    def __init__(self):
        self.small = []  # max heap
        self.large = []  # min heap

    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2


    # 110. Binary Search - Find Rotation Count in Rotated Array
def rotation_count(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return left


# 111. Binary Search - Find Minimum Difference Element
def min_diff_element(nums, target):
    left, right = 0, len(nums) - 1
    res = float('inf')
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return nums[mid]
        if abs(nums[mid] - target) < abs(res - target):
            res = nums[mid]
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return res


# 112. Binary Search - Find Range Sum Query
def range_sum_query(nums, left, right):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix[right + 1] - prefix[left]


# 113. Binary Search - Find Maximum Average Subarray
def find_max_average_bs(nums, k):
    left, right = min(nums), max(nums)
    eps = 1e-5

    def check(mid):
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num - mid)
        min_prefix = 0
        for i in range(k, len(prefix)):
            if prefix[i] - min_prefix >= 0:
                return True
            min_prefix = min(min_prefix, prefix[i - k + 1])
        return False

    while right - left > eps:
        mid = (left + right) / 2
        if check(mid):
            left = mid
        else:
            right = mid
    return left


# 114. Binary Search - Find Element in Nearly Sorted Array
def search_nearly_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if mid - 1 >= left and nums[mid - 1] == target:
            return mid - 1
        if mid + 1 <= right and nums[mid + 1] == target:
            return mid + 1
        if nums[mid] < target:
            left = mid + 2
        else:
            right = mid - 2
    return -1


# 115. Binary Search - Find Closest Element to Target
def find_closest(nums, target):
    left, right = 0, len(nums) - 1
    res = nums[0]
    while left <= right:
        mid = (left + right) // 2
        if abs(nums[mid] - target) < abs(res - target):
            res = nums[mid]
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            return nums[mid]
    return res


# 116. Binary Search - Find Maximum in Bitonic Array
def max_in_bitonic_array(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return nums[left]


# 117. Binary Search - Find Missing Number
def missing_number(nums):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == mid:
            left = mid + 1
        else:
            right = mid - 1
    return left


# 118. Binary Search - Allocate Minimum Number of Pages
def min_pages(books, students):

    def is_possible(limit):
        count, curr = 1, 0
        for pages in books:
            if pages > limit:
                return False
            if curr + pages > limit:
                count += 1
                curr = 0
            curr += pages
        return count <= students

    left, right = max(books), sum(books)
    while left < right:
        mid = (left + right) // 2
        if is_possible(mid):
            right = mid
        else:
            left = mid + 1
    return left


# 119. Binary Search - Split Array to Minimize Largest Sum
def split_array_min_largest_sum(nums, m):

    def can_split(max_sum):
        count, curr = 1, 0
        for num in nums:
            if curr + num > max_sum:
                count += 1
                curr = 0
            curr += num
        return count <= m

    left, right = max(nums), sum(nums)
    while left < right:
        mid = (left + right) // 2
        if can_split(mid):
            right = mid
        else:
            left = mid + 1
    return left


# 120. Binary Search - Find Peak Element with Duplicates
def find_peak_with_duplicates(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        elif nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left += 1
    return left
    
    # 121. Sorting - Sort Colors (Dutch National Flag Problem)
def sort_colors_dnf(nums):
        low, mid, high = 0, 0, len(nums) - 1
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]  # Swap 0 to front
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1  # 1 is in correct place
            else:
                nums[mid], nums[high] = nums[high], nums[mid]  # Swap 2 to end
                high -= 1

    # 122. Sorting - Merge Intervals
def merge_intervals_sort(intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])  # Sort by start
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)  # Merge overlapping
            else:
                merged.append([start, end])  # Add new interval
        return merged

    # 123. Sorting - Find Kth Largest Element
def find_kth_largest(nums, k):
        nums.sort(reverse=True)  # Sort descending
        return nums[k - 1]  # Kth largest is at index k-1

    # 124. Sorting - Sort Array By Parity
def sort_array_by_parity_sort(nums):
        return [x for x in nums if x % 2 == 0
                ] + [x for x in nums if x % 2 == 1]  # Evens first, then odds

    # 125. Sorting - Wiggle Sort
def wiggle_sort(nums):
        for i in range(len(nums) - 1):
            if (i % 2 == 0
                    and nums[i] > nums[i + 1]) or (i % 2 == 1 and nums[i] < nums[i + 1]):
                nums[i], nums[i + 1] = nums[i + 1], nums[
                    i]  # Swap to maintain wiggle property

    # 126. Sorting - Sort Array of Squares
def sorted_squares_sort(nums):
        return sorted([x * x for x in nums])  # Square and sort

    # 127. Sorting - Minimum Number of Swaps to Sort Array
def min_swaps_to_sort(nums):
        arr = list(enumerate(nums))  # Pair index and value
        arr.sort(key=lambda x: x[1])  # Sort by value
        visited = [False] * len(nums)
        swaps = 0
        for i in range(len(nums)):
            if visited[i] or arr[i][0] == i:
                continue  # Already in place or visited
            cycle = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = arr[j][0]  # Next index in cycle
                cycle += 1
            if cycle > 0:
                swaps += cycle - 1  # Swaps needed for this cycle
        return swaps

    # 128. Sorting - Count Inversions in Array
def count_inversions(nums):
    
    def merge_sort(arr):
                if len(arr) <= 1:
                    return arr, 0
                mid = len(arr) // 2
                left, inv_left = merge_sort(arr[:mid])
                right, inv_right = merge_sort(arr[mid:])
                merged, inv_split = merge(left, right)
                return merged, inv_left + inv_right + inv_split
    
    def merge(left, right):
                merged = []
                i = j = inv = 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        merged.append(left[i])
                        i += 1
                    else:
                        merged.append(right[j])
                        inv += len(left) - i  # Count inversions
                        j += 1
                merged += left[i:]
                merged += right[j:]
                return merged, inv
        
    _, inv_count = merge_sort(nums)
    return inv_count
    
    # 129. Sorting - Relative Sort Array
def relative_sort_array(arr1, arr2):
        count = Counter(arr1)  # Count occurrences
        res = []
        for num in arr2:
            res += [num] * count[num]  # Add in order of arr2
            count[num] = 0
        for num in sorted(count.elements()):
            res.append(num)  # Add remaining in sorted order
        return res

    # 130. Sorting - Largest Number Formed by Array
from functools import cmp_to_key

def largest_number(nums):
    nums = list(map(str, nums))
    
    def compare(x, y):
        return int(y + x) - int(x + y)  # Custom sort: which order forms bigger number
    
    nums.sort(key=cmp_to_key(compare))
    result = ''.join(nums)
    return '0' if result[0] == '0' else result  # Handle all zeros

# 131. Partitioning - Dutch National Flag Problem
def dutch_national_flag(nums):
        low, mid, high = 0, 0, len(nums) - 1
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]  # Move 0 to front
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1  # 1 is in place
            else:
                nums[mid], nums[high] = nums[high], nums[mid]  # Move 2 to end
                high -= 1
    
    # 132. Partitioning - Partition Array into Disjoint Intervals
def partition_disjoint(nums):
        left_max = [nums[0]]
        for num in nums[1:]:
            left_max.append(max(left_max[-1], num))  # Max so far from left
        right_min = [0] * len(nums)
        right_min[-1] = nums[-1]
        for i in range(len(nums) - 2, -1, -1):
            right_min[i] = min(right_min[i + 1], nums[i])  # Min so far from right
        for i in range(len(nums) - 1):
            if left_max[i] <= right_min[i + 1]:
                return i + 1  # Partition point

    # 133. Partitioning - Sort Array According to Another Array
def sort_according_to_another(arr1, arr2):
        count = Counter(arr1)
        res = []
        for num in arr2:
            res += [num] * count[num]  # Add in arr2 order
            count[num] = 0
        for num in sorted(count.elements()):
            res.append(num)  # Add remaining sorted
        return res

    # 134. Partitioning - Partition Labels
def partition_labels(s):
        last = {c: i for i, c in enumerate(s)}  # Last occurrence of each char
        res = []
        start = end = 0
        for i, c in enumerate(s):
            end = max(end, last[c])  # Extend end to last occurrence
            if i == end:
                res.append(end - start + 1)  # Partition size
                start = i + 1
        return res

    # 135. Partitioning - Sort Array by Increasing Frequency
def frequency_sort(nums):
        count = Counter(nums)
        nums.sort(key=lambda x:(count[x], -x))  # Sort by freq, then value descending
        return nums

    # 136. Partitioning - Find the Kth Smallest Pair Distance
def smallest_distance_pair(nums, k):
        nums.sort()
        left, right = 0, nums[-1] - nums[0]

        def count_pairs(mid):
            count = left = 0
            for right_idx, val in enumerate(nums):
                while val - nums[left] > mid:
                    left += 1
                count += right_idx - left  # Pairs with distance <= mid
            return count

        while left < right:
            mid = (left + right) // 2
            if count_pairs(mid) < k:
                left = mid + 1
            else:
                right = mid
        return left

    # 137. Partitioning - Split Array Into Consecutive Subsequences
def is_possible_partition(nums):
        count = Counter(nums)
        end = defaultdict(int)
        for x in nums:
            if count[x] == 0:
                continue
            if end[x - 1] > 0:
                end[x - 1] -= 1
                end[x] += 1
            elif count[x + 1] > 0 and count[x + 2] > 0:
                count[x + 1] -= 1
                count[x + 2] -= 1
                end[x + 2] += 1
            else:
                return False  # Can't form required subsequence
            count[x] -= 1
        return True

    # 138. Partitioning - Minimum Number of Increments on Subarrays
def min_number_operations(target):
        res = prev = 0
        for num in target:
            res += max(0, num - prev)  # Only increment when current > prev
            prev = num
        return res

    # 139. Partitioning - Find Pivot Index
def find_pivot_index(nums):
        total = sum(nums)
        left_sum = 0
        for i, num in enumerate(nums):
            if left_sum == total - left_sum - num:
                return i  # Pivot index found
            left_sum += num
        return -1

    # 140. Partitioning - Longest Increasing Subsequence via Patience Sorting
def length_of_lis(nums):
        piles = []
        for num in nums:
            idx = bisect.bisect_left(piles, num)  # Find pile to put num
            if idx == len(piles):
                piles.append(num)  # New pile
            else:
                piles[idx] = num  # Replace top of pile
        return len(piles)

    # 141. Partitioning - Sort Characters By Frequency
def frequency_sort_string(s):
        count = Counter(s)
        return ''.join([char * freq for char, freq in count.most_common()])  # Most frequent first
    
    # 142. Partitioning - K Closest Points to Origin
def k_closest(points, k):
        points.sort(
            key=lambda x: x[0]**2 + x[1]**2)  # Sort by distance squared
        return points[:k]

    # 143. Partitioning - Sort Array by Parity II
def sort_array_by_parity_ii(nums):
        res = [0] * len(nums)
        even, odd = 0, 1
        for num in nums:
            if num % 2 == 0:
                res[even] = num  # Place even at even index
                even += 2
            else:
                res[odd] = num  # Place odd at odd index
                odd += 2
        return res

    # 144. Partitioning - Move All Negative Numbers to Beginning
def move_negatives(nums):
        j = 0
        for i in range(len(nums)):
            if nums[i] < 0:
                nums[i], nums[j] = nums[j], nums[i]  # Swap negative to front
                j += 1

    # 145. Partitioning - Sort Array By Increasing Frequency
def sort_by_freq(nums):
        count = Counter(nums)
        nums.sort(key=lambda x:(count[x], -x))  # Sort by freq, then value descending
        return nums

    # 146. Partitioning - Maximize Distance Between Same Elements
def max_distance_same(nums):
        last = {}
        max_dist = 0
        for i, num in enumerate(nums):
            if num in last:
                max_dist = max(max_dist, i - last[num])  # Update max distance
            last[num] = i  # Update last seen index
        return max_dist

    # 147. Partitioning - Minimum Number of Moves to Make Array Complementary
def min_moves_complementary(nums, limit):
        n = len(nums)
        diff = [0] * (2 * limit + 2)
        for i in range(n // 2):
            a, b = nums[i], nums[n - 1 - i]
            low = 1 + min(a, b)
            high = limit + max(a, b)
            diff[2] += 2
            diff[low] -= 1
            diff[a + b] -= 1
            diff[a + b + 1] += 1
            diff[high + 1] += 1
        res = n
        curr = 0
        for i in range(2, 2 * limit + 1):
            curr += diff[i]
            res = min(res, curr)  # Track minimum moves
        return res

    # 148. Partitioning - Partition Array for Maximum Sum
def max_sum_after_partitioning(arr, k):
        n = len(arr)
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            curr_max = 0
            for j in range(1, min(k, i) + 1):
                curr_max = max(curr_max, arr[i - j])  # Max in window
                dp[i] = max(dp[i], dp[i - j] + curr_max * j)  # DP transition
        return dp[n]

    # 149. Partitioning - Max Chunks To Make Sorted
def max_chunks_to_sorted(arr):
        max_so_far = res = 0
        for i, num in enumerate(arr):
            max_so_far = max(max_so_far, num)  # Track max till now
            if max_so_far == i:
                res += 1  # Can make a chunk here
        return res

    # 150. Partitioning - Sort Array with Odd Even Index Constraints
def sort_odd_even(nums):
        odds = sorted([nums[i] for i in range(1, len(nums), 2)],reverse=True)  # Odd indices descending
        evens = sorted([nums[i] for i in range(0, len(nums), 2)
                        ])  # Even indices ascending
        res = []
        even_idx, odd_idx = 0, 0
        for i in range(len(nums)):
            if i % 2 == 0:
                res.append(evens[even_idx])
                even_idx += 1
            else:
                res.append(odds[odd_idx])
                odd_idx += 1
        return res
