package arrays

import (
	"fmt"
	"sort"
	"strings"
)

// 1. Find Pair with Given Sum
func findPairWithSum(nums []int, target int) (int, int, bool) {
	left, right := 0, len(nums)-1
	for left < right {
		sum := nums[left] + nums[right]
		if sum == target {
			return nums[left], nums[right], true
		} else if sum < target {
			left++
		} else {
			right--
		}
	}
	return 0, 0, false
}

// 2. Remove Duplicates from Sorted Array
func removeDuplicates(nums []int) int {
	if len(nums) == 0 {
		return 0
	}
	j := 0
	for i := 1; i < len(nums); i++ {
		if nums[i] != nums[j] {
			j++
			nums[j] = nums[i]
		}
	}
	return j + 1
}

// 3. Container With Most Water
func maxArea(height []int) int {
	left, right := 0, len(height)-1
	maxA := 0
	for left < right {
		h := height[left]
		if height[right] < h {
			h = height[right]
		}
		area := (right - left) * h
		if area > maxA {
			maxA = area
		}
		if height[left] < height[right] {
			left++
		} else {
			right--
		}
	}
	return maxA
}

// 4. Merge Two Sorted Arrays
func mergeSortedArrays(nums1 []int, m int, nums2 []int, n int) {
	i, j, k := m-1, n-1, m+n-1
	for i >= 0 && j >= 0 {
		if nums1[i] > nums2[j] {
			nums1[k] = nums1[i]
			i--
		} else {
			nums1[k] = nums2[j]
			j--
		}
		k--
	}
	for j >= 0 {
		nums1[k] = nums2[j]
		j--
		k--
	}
}

// 5. Intersection of Two Arrays II
func intersect(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	i, j := 0, 0
	var res []int
	for i < len(nums1) && j < len(nums2) {
		if nums1[i] == nums2[j] {
			res = append(res, nums1[i])
			i++
			j++
		} else if nums1[i] < nums2[j] {
			i++
		} else {
			j++
		}
	}
	return res
}

// 6. Move Zeroes to End
func moveZeroes(nums []int) {
	j := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[j], nums[i] = nums[i], nums[j]
			j++
		}
	}
}

// 7. Sort Array of 0s, 1s, and 2s
func sortColors(nums []int) {
	low, mid, high := 0, 0, len(nums)-1
	for mid <= high {
		switch nums[mid] {
		case 0:
			nums[low], nums[mid] = nums[mid], nums[low]
			low++
			mid++
		case 1:
			mid++
		case 2:
			nums[mid], nums[high] = nums[high], nums[mid]
			high--
		}
	}
}

// 8. Longest Substring Without Repeating Characters
func lengthOfLongestSubstring(s string) int {
	m := make(map[byte]int)
	maxLen, left := 0, 0
	for right := 0; right < len(s); right++ {
		if idx, ok := m[s[right]]; ok && idx >= left {
			left = idx + 1
		}
		m[s[right]] = right
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 9. Trapping Rain Water
func trap(height []int) int {
	left, right := 0, len(height)-1
	leftMax, rightMax, res := 0, 0, 0
	for left < right {
		if height[left] < height[right] {
			if height[left] >= leftMax {
				leftMax = height[left]
			} else {
				res += leftMax - height[left]
			}
			left++
		} else {
			if height[right] >= rightMax {
				rightMax = height[right]
			} else {
				res += rightMax - height[right]
			}
			right--
		}
	}
	return res
}

// 10. Valid Palindrome II
func validPalindrome(s string) bool {
	isPalin := func(l, r int) bool {
		for l < r {
			if s[l] != s[r] {
				return false
			}
			l++
			r--
		}
		return true
	}
	l, r := 0, len(s)-1
	for l < r {
		if s[l] != s[r] {
			return isPalin(l+1, r) || isPalin(l, r-1)
		}
		l++
		r--
	}
	return true
}

// 11. Partition Array Around a Pivot
func partitionArray(nums []int, pivot int) []int {
	left := 0
	res := make([]int, len(nums))
	copy(res, nums)
	for i := 0; i < len(nums); i++ {
		if nums[i] < pivot {
			res[left] = nums[i]
			left++
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] == pivot {
			res[left] = nums[i]
			left++
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] > pivot {
			res[left] = nums[i]
			left++
		}
	}
	return res
}

// 12. Squares of a Sorted Array
func sortedSquares(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	left, right := 0, n-1
	for i := n - 1; i >= 0; i-- {
		if abs(nums[left]) > abs(nums[right]) {
			res[i] = nums[left] * nums[left]
			left++
		} else {
			res[i] = nums[right] * nums[right]
			right--
		}
	}
	return res
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// 13. Find All Triplets That Sum to Zero
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	var res [][]int
	for i := 0; i < len(nums)-2; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		l, r := i+1, len(nums)-1
		for l < r {
			sum := nums[i] + nums[l] + nums[r]
			if sum == 0 {
				res = append(res, []int{nums[i], nums[l], nums[r]})
				for l < r && nums[l] == nums[l+1] {
					l++
				}
				for l < r && nums[r] == nums[r-1] {
					r--
				}
				l++
				r--
			} else if sum < 0 {
				l++
			} else {
				r--
			}
		}
	}
	return res
}

// 14. Minimum Size Subarray Sum
func minSubArrayLen(target int, nums []int) int {
	left, sum, res := 0, 0, len(nums)+1
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for sum >= target {
			if right-left+1 < res {
				res = right - left + 1
			}
			sum -= nums[left]
			left++
		}
	}
	if res == len(nums)+1 {
		return 0
	}
	return res
}

// 15. Subarray Product Less Than K
func numSubarrayProductLessThanK(nums []int, k int) int {
	if k <= 1 {
		return 0
	}
	prod, left, res := 1, 0, 0
	for right := 0; right < len(nums); right++ {
		prod *= nums[right]
		for prod >= k {
			prod /= nums[left]
			left++
		}
		res += right - left + 1
	}
	return res
}

// 16. Remove Element In-Place
func removeElement(nums []int, val int) int {
	k := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] != val {
			nums[k] = nums[i]
			k++
		}
	}
	return k
}

// 17. Find K Closest Elements
func findClosestElements(arr []int, k int, x int) []int {
	left, right := 0, len(arr)-k
	for left < right {
		mid := (left + right) / 2
		if x-arr[mid] > arr[mid+k]-x {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return arr[left : left+k]
}

// 18. Sort Array By Parity
func sortArrayByParity(nums []int) []int {
	left, right := 0, len(nums)-1
	for left < right {
		if nums[left]%2 > nums[right]%2 {
			nums[left], nums[right] = nums[right], nums[left]
		}
		if nums[left]%2 == 0 {
			left++
		}
		if nums[right]%2 == 1 {
			right--
		}
	}
	return nums
}

// 19. Merge Intervals (Based on sorted ends)
type Interval struct {
	Start int
	End   int
}

func mergeIntervals(intervals []Interval) []Interval {
	if len(intervals) == 0 {
		return nil
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i].Start < intervals[j].Start
	})
	res := []Interval{intervals[0]}
	for i := 1; i < len(intervals); i++ {
		last := &res[len(res)-1]
		if intervals[i].Start <= last.End {
			if intervals[i].End > last.End {
				last.End = intervals[i].End
			}
		} else {
			res = append(res, intervals[i])
		}
	}
	return res
}

// 20. Valid Mountain Array
func validMountainArray(arr []int) bool {
	n := len(arr)
	if n < 3 {
		return false
	}
	i := 0
	for i+1 < n && arr[i] < arr[i+1] {
		i++
	}
	if i == 0 || i == n-1 {
		return false
	}
	for i+1 < n && arr[i] > arr[i+1] {
		i++
	}
	return i == n-1
}

// 21. Check If Array Is Monotonic
func isMonotonic(nums []int) bool {
	inc, dec := true, true
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			dec = false
		}
		if nums[i] < nums[i-1] {
			inc = false
		}
	}
	return inc || dec
}

// 22. Sum of Two Arrays
func sumOfTwoArrays(a, b []int) []int {
	n, m := len(a), len(b)
	i, j := n-1, m-1
	carry := 0
	var res []int
	for i >= 0 || j >= 0 || carry > 0 {
		sum := carry
		if i >= 0 {
			sum += a[i]
			i--
		}
		if j >= 0 {
			sum += b[j]
			j--
		}
		res = append([]int{sum % 10}, res...)
		carry = sum / 10
	}
	return res
}

// 23. Rotate Array by K Steps
func rotate(nums []int, k int) {
	n := len(nums)
	k %= n
	reverse := func(l, r int) {
		for l < r {
			nums[l], nums[r] = nums[r], nums[l]
			l++
			r--
		}
	}
	reverse(0, n-1)
	reverse(0, k-1)
	reverse(k, n-1)
}

// 24. Subarrays with Equal Number of 0s and 1s
func findMaxLength(nums []int) int {
	m := map[int]int{0: -1}
	maxLen, count := 0, 0
	for i, v := range nums {
		if v == 0 {
			count--
		} else {
			count++
		}
		if idx, ok := m[count]; ok {
			if i-idx > maxLen {
				maxLen = i - idx
			}
		} else {
			m[count] = i
		}
	}
	return maxLen
}

// 25. Max Consecutive Ones III
func longestOnes(nums []int, k int) int {
	left, maxLen, zeros := 0, 0, 0
	for right := 0; right < len(nums); right++ {
		if nums[right] == 0 {
			zeros++
		}
		for zeros > k {
			if nums[left] == 0 {
				zeros--
			}
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 26. Longest Repeating Character Replacement
func characterReplacement(s string, k int) int {
	count := [26]int{}
	maxCount, left, res := 0, 0, 0
	for right := 0; right < len(s); right++ {
		count[s[right]-'A']++
		if count[s[right]-'A'] > maxCount {
			maxCount = count[s[right]-'A']
		}
		for right-left+1-maxCount > k {
			count[s[left]-'A']--
			left++
		}
		if right-left+1 > res {
			res = right - left + 1
		}
	}
	return res
}

// 27. Split Array into Consecutive Subsequences
func isPossible(nums []int) bool {
	count := make(map[int]int)
	end := make(map[int]int)
	for _, v := range nums {
		count[v]++
	}
	for _, v := range nums {
		if count[v] == 0 {
			continue
		}
		if end[v-1] > 0 {
			end[v-1]--
			end[v]++
		} else if count[v+1] > 0 && count[v+2] > 0 {
			count[v+1]--
			count[v+2]--
			end[v+2]++
		} else {
			return false
		}
		count[v]--
	}
	return true
}

// 28. Find Duplicate Number (Floyd's Tortoise and Hare)
func findDuplicate(nums []int) int {
	slow, fast := nums[0], nums[nums[0]]
	for slow != fast {
		slow = nums[slow]
		fast = nums[nums[fast]]
	}
	slow = 0
	for slow != fast {
		slow = nums[slow]
		fast = nums[fast]
	}
	return slow
}

// 29. Maximise Distance to Closest Person
func maxDistToClosest(seats []int) int {
	prev, maxDist := -1, 0
	for i, seat := range seats {
		if seat == 1 {
			if prev == -1 {
				maxDist = i
			} else {
				d := (i - prev) / 2
				if d > maxDist {
					maxDist = d
				}
			}
			prev = i
		}
	}
	if len(seats)-1-prev > maxDist {
		maxDist = len(seats) - 1 - prev
	}
	return maxDist
}

// 30. Find Peak Element
func findPeakElement(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}

// 31. Sliding Window - Maximum Sum Subarray of Size K
func maxSumSubarrayOfSizeK(nums []int, k int) int {
	sum, maxSum := 0, 0
	for i := 0; i < len(nums); i++ {
		sum += nums[i]
		if i >= k-1 {
			if sum > maxSum || i == k-1 {
				maxSum = sum
			}
			sum -= nums[i-k+1]
		}
	}
	return maxSum
}

// 32. Sliding Window - Minimum Window Substring
func minWindow(s string, t string) string {
	need := make(map[byte]int)
	for i := 0; i < len(t); i++ {
		need[t[i]]++
	}
	have := make(map[byte]int)
	required := len(need)
	formed := 0
	l, r := 0, 0
	minLen, minL := len(s)+1, 0
	for r < len(s) {
		c := s[r]
		have[c]++
		if need[c] > 0 && have[c] == need[c] {
			formed++
		}
		for l <= r && formed == required {
			if r-l+1 < minLen {
				minLen = r - l + 1
				minL = l
			}
			have[s[l]]--
			if need[s[l]] > 0 && have[s[l]] < need[s[l]] {
				formed--
			}
			l++
		}
		r++
	}
	if minLen > len(s) {
		return ""
	}
	return s[minL : minL+minLen]
}

// 33. Sliding Window - Longest Substring with At Most K Distinct Characters
func lengthOfLongestSubstringKDistinct(s string, k int) int {
	if k == 0 {
		return 0
	}
	m := make(map[byte]int)
	left, maxLen := 0, 0
	for right := 0; right < len(s); right++ {
		m[s[right]]++
		for len(m) > k {
			m[s[left]]--
			if m[s[left]] == 0 {
				delete(m, s[left])
			}
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 34. Sliding Window - Count Number of Anagrams
func countAnagrams(s, p string) int {
	if len(p) > len(s) {
		return 0
	}
	countP, countS := [26]int{}, [26]int{}
	for i := 0; i < len(p); i++ {
		countP[p[i]-'a']++
		countS[s[i]-'a']++
	}
	res := 0
	if countP == countS {
		res++
	}
	for i := len(p); i < len(s); i++ {
		countS[s[i]-'a']++
		countS[s[i-len(p)]-'a']--
		if countP == countS {
			res++
		}
	}
	return res
}

// 35. Sliding Window - Find All Anagrams in a String
func findAnagrams(s string, p string) []int {
	if len(p) > len(s) {
		return nil
	}
	countP, countS := [26]int{}, [26]int{}
	for i := 0; i < len(p); i++ {
		countP[p[i]-'a']++
		countS[s[i]-'a']++
	}
	var res []int
	if countP == countS {
		res = append(res, 0)
	}
	for i := len(p); i < len(s); i++ {
		countS[s[i]-'a']++
		countS[s[i-len(p)]-'a']--
		if countP == countS {
			res = append(res, i-len(p)+1)
		}
	}
	return res
}

// 36. Sliding Window - Longest Repeating Character Replacement
func longestRepeatingCharReplacement(s string, k int) int {
	count := [26]int{}
	maxCount, left, res := 0, 0, 0
	for right := 0; right < len(s); right++ {
		count[s[right]-'A']++
		if count[s[right]-'A'] > maxCount {
			maxCount = count[s[right]-'A']
		}
		for right-left+1-maxCount > k {
			count[s[left]-'A']--
			left++
		}
		if right-left+1 > res {
			res = right - left + 1
		}
	}
	return res
}

// 37. Sliding Window - Fruit Into Baskets
func totalFruit(fruits []int) int {
	m := make(map[int]int)
	left, res := 0, 0
	for right := 0; right < len(fruits); right++ {
		m[fruits[right]]++
		for len(m) > 2 {
			m[fruits[left]]--
			if m[fruits[left]] == 0 {
				delete(m, fruits[left])
			}
			left++
		}
		if right-left+1 > res {
			res = right - left + 1
		}
	}
	return res
}

// 38. Sliding Window - Subarrays with K Different Integers
func subarraysWithKDistinct(nums []int, k int) int {
	return atMostKDistinct(nums, k) - atMostKDistinct(nums, k-1)
}
func atMostKDistinct(nums []int, k int) int {
	m := make(map[int]int)
	left, res := 0, 0
	for right := 0; right < len(nums); right++ {
		if m[nums[right]] == 0 {
			k--
		}
		m[nums[right]]++
		for k < 0 {
			m[nums[left]]--
			if m[nums[left]] == 0 {
				k++
			}
			left++
		}
		res += right - left + 1
	}
	return res
}

// 39. Sliding Window - Maximum Number of Vowels in Substring
func maxVowels(s string, k int) int {
	isVowel := func(c byte) bool {
		return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
	}
	count, maxCount := 0, 0
	for i := 0; i < len(s); i++ {
		if isVowel(s[i]) {
			count++
		}
		if i >= k && isVowel(s[i-k]) {
			count--
		}
		if count > maxCount {
			maxCount = count
		}
	}
	return maxCount
}

// 40. Sliding Window - Minimum Size Subarray Sum
func minSubArrayLenSlidingWindow(target int, nums []int) int {
	left, sum, res := 0, 0, len(nums)+1
	for right := 0; right < len(nums); right++ {
		sum += nums[right]
		for sum >= target {
			if right-left+1 < res {
				res = right - left + 1
			}
			sum -= nums[left]
			left++
		}
	}
	if res == len(nums)+1 {
		return 0
	}
	return res
}

// 41. Sliding Window - Number of Subarrays with Sum K
func subarraySum(nums []int, k int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		res += m[sum-k]
		m[sum]++
	}
	return res
}

// 42. Sliding Window - Longest Subarray with Ones After Replacement
func longestOnesAfterReplacement(nums []int, k int) int {
	left, maxLen, zeros := 0, 0, 0
	for right := 0; right < len(nums); right++ {
		if nums[right] == 0 {
			zeros++
		}
		for zeros > k {
			if nums[left] == 0 {
				zeros--
			}
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 43. Sliding Window - Longest Substring Without Repeating Characters
func lengthOfLongestSubstringSliding(s string) int {
	m := make(map[byte]int)
	maxLen, left := 0, 0
	for right := 0; right < len(s); right++ {
		if idx, ok := m[s[right]]; ok && idx >= left {
			left = idx + 1
		}
		m[s[right]] = right
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 44. Sliding Window - Maximum Average Subarray
func findMaxAverage(nums []int, k int) float64 {
	sum := 0
	for i := 0; i < k; i++ {
		sum += nums[i]
	}
	maxSum := sum
	for i := k; i < len(nums); i++ {
		sum += nums[i] - nums[i-k]
		if sum > maxSum {
			maxSum = sum
		}
	}
	return float64(maxSum) / float64(k)
}

// 45. Sliding Window - Count Number of Nice Subarrays
func numberOfSubarrays(nums []int, k int) int {
	return atMostKOdd(nums, k) - atMostKOdd(nums, k-1)
}
func atMostKOdd(nums []int, k int) int {
	res, left, count := 0, 0, 0
	for right := 0; right < len(nums); right++ {
		if nums[right]%2 == 1 {
			count++
		}
		for count > k {
			if nums[left]%2 == 1 {
				count--
			}
			left++
		}
		res += right - left + 1
	}
	return res
}

// 46. Sliding Window - Longest Substring with At Most Two Distinct Characters
func lengthOfLongestSubstringTwoDistinct(s string) int {
	m := make(map[byte]int)
	left, maxLen := 0, 0
	for right := 0; right < len(s); right++ {
		m[s[right]]++
		for len(m) > 2 {
			m[s[left]]--
			if m[s[left]] == 0 {
				delete(m, s[left])
			}
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 47. Sliding Window - Longest Substring with Exactly K Distinct Characters
func lengthOfLongestSubstringKDistinctExactly(s string, k int) int {
	m := make(map[byte]int)
	left, maxLen := 0, 0
	for right := 0; right < len(s); right++ {
		m[s[right]]++
		for len(m) > k {
			m[s[left]]--
			if m[s[left]] == 0 {
				delete(m, s[left])
			}
			left++
		}
		if len(m) == k && right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 48. Sliding Window - Longest Substring Without Repeating Vowels
func lengthOfLongestSubstringNoRepeatVowels(s string) int {
	isVowel := func(c byte) bool {
		return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u'
	}
	m := make(map[byte]int)
	left, maxLen := 0, 0
	for right := 0; right < len(s); right++ {
		if isVowel(s[right]) {
			if idx, ok := m[s[right]]; ok && idx >= left {
				left = idx + 1
			}
			m[s[right]] = right
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 49. Sliding Window - Subarray Product Less Than K
func numSubarrayProductLessThanKSliding(nums []int, k int) int {
	if k <= 1 {
		return 0
	}
	prod, left, res := 1, 0, 0
	for right := 0; right < len(nums); right++ {
		prod *= nums[right]
		for prod >= k {
			prod /= nums[left]
			left++
		}
		res += right - left + 1
	}
	return res
}

// 50. Sliding Window - Number of Subarrays with Bounded Maximum
func numSubarrayBoundedMax(nums []int, left int, right int) int {
	res, l, r := 0, -1, -1
	for i, v := range nums {
		if v > right {
			l = i
		}
		if v >= left {
			r = i
		}
		res += r - l
	}
	return res
}

// 51. Sliding Window - Longest Subarray with At Most K Odd Numbers
func longestSubarrayAtMostKOdd(nums []int, k int) int {
	left, count, maxLen := 0, 0, 0
	for right := 0; right < len(nums); right++ {
		if nums[right]%2 == 1 {
			count++
		}
		for count > k {
			if nums[left]%2 == 1 {
				count--
			}
			left++
		}
		if right-left+1 > maxLen {
			maxLen = right - left + 1
		}
	}
	return maxLen
}

// 52. Sliding Window - Minimum Window Containing All Characters
func minWindowAllChars(s string, t string) string {
	need := make(map[byte]int)
	for i := 0; i < len(t); i++ {
		need[t[i]]++
	}
	have := make(map[byte]int)
	required := len(need)
	formed := 0
	l, r := 0, 0
	minLen, minL := len(s)+1, 0
	for r < len(s) {
		c := s[r]
		have[c]++
		if need[c] > 0 && have[c] == need[c] {
			formed++
		}
		for l <= r && formed == required {
			if r-l+1 < minLen {
				minLen = r - l + 1
				minL = l
			}
			have[s[l]]--
			if need[s[l]] > 0 && have[s[l]] < need[s[l]] {
				formed--
			}
			l++
		}
		r++
	}
	if minLen > len(s) {
		return ""
	}
	return s[minL : minL+minLen]
}

// 53. Sliding Window - Longest Substring With At Least K Repeating Characters
func longestSubstringAtLeastKRepeats(s string, k int) int {
	maxLen := 0
	for unique := 1; unique <= 26; unique++ {
		count := [26]int{}
		left, right, currUnique, countAtLeastK := 0, 0, 0, 0
		for right < len(s) {
			if currUnique <= unique {
				idx := s[right] - 'a'
				if count[idx] == 0 {
					currUnique++
				}
				count[idx]++
				if count[idx] == k {
					countAtLeastK++
				}
				right++
			} else {
				idx := s[left] - 'a'
				if count[idx] == k {
					countAtLeastK--
				}
				count[idx]--
				if count[idx] == 0 {
					currUnique--
				}
				left++
			}
			if currUnique == unique && currUnique == countAtLeastK && right-left > maxLen {
				maxLen = right - left
			}
		}
	}
	return maxLen
}

// 54. Sliding Window - Maximum Number of Balls in a Box
func countBalls(lowLimit int, highLimit int) int {
	box := make(map[int]int)
	maxCount := 0
	for i := lowLimit; i <= highLimit; i++ {
		sum := 0
		n := i
		for n > 0 {
			sum += n % 10
			n /= 10
		}
		box[sum]++
		if box[sum] > maxCount {
			maxCount = box[sum]
		}
	}
	return maxCount
}

// 55. Sliding Window - Longest Continuous Subarray with Absolute Diff ≤ Limit
func longestSubarrayAbsDiffLimit(nums []int, limit int) int {
	maxDeque, minDeque := []int{}, []int{}
	left, res := 0, 0
	for right, v := range nums {
		for len(maxDeque) > 0 && nums[maxDeque[len(maxDeque)-1]] < v {
			maxDeque = maxDeque[:len(maxDeque)-1]
		}
		maxDeque = append(maxDeque, right)
		for len(minDeque) > 0 && nums[minDeque[len(minDeque)-1]] > v {
			minDeque = minDeque[:len(minDeque)-1]
		}
		minDeque = append(minDeque, right)
		for nums[maxDeque[0]]-nums[minDeque[0]] > limit {
			if maxDeque[0] == left {
				maxDeque = maxDeque[1:]
			}
			if minDeque[0] == left {
				minDeque = minDeque[1:]
			}
			left++
		}
		if right-left+1 > res {
			res = right - left + 1
		}
	}
	return res
}

// 56. Sliding Window - Maximum Number of Non-Overlapping Subarrays with Sum Equals Target
func maxNonOverlapping(nums []int, target int) int {
	m := map[int]int{0: -1}
	sum, res, lastEnd := 0, 0, -1
	for i, v := range nums {
		sum += v
		if idx, ok := m[sum-target]; ok && idx >= lastEnd {
			res++
			lastEnd = i
		}
		m[sum] = i
	}
	return res
}

// 57. Sliding Window - Number of Subarrays With Odd Sum
func numOfSubarraysWithOddSum(arr []int) int {
	const mod = 1e9 + 7
	odd, even, sum, res := 0, 1, 0, 0
	for _, v := range arr {
		sum += v
		if sum%2 == 0 {
			res = (res + odd) % mod
			even++
		} else {
			res = (res + even) % mod
			odd++
		}
	}
	return res
}

// 58. Sliding Window - Count Subarrays with Median K
func countSubarraysWithMedianK(nums []int, k int) int {
	pos := 0
	for i, v := range nums {
		if v == k {
			pos = i
			break
		}
	}
	m := map[int]int{0: 1}
	bal, res := 0, 0
	for i := pos - 1; i >= 0; i-- {
		if nums[i] < k {
			bal--
		} else {
			bal++
		}
		m[bal]++
	}
	bal = 0
	for i := pos; i < len(nums); i++ {
		if nums[i] < k {
			bal--
		} else if nums[i] > k {
			bal++
		}
		res += m[-bal] + m[1-bal]
	}
	return res
}

// 59. Sliding Window - Longest Substring Without Repeating Characters with Replacement
func lengthOfLongestSubstringWithReplacement(s string, k int) int {
	count := [256]int{}
	left, maxCount, res := 0, 0, 0
	for right := 0; right < len(s); right++ {
		count[s[right]]++
		if count[s[right]] > maxCount {
			maxCount = count[s[right]]
		}
		for right-left+1-maxCount > k {
			count[s[left]]--
			left++
		}
		if right-left+1 > res {
			res = right - left + 1
		}
	}
	return res
}

// 60. Sliding Window - Count Good Substrings
func countGoodSubstrings(s string) int {
	if len(s) < 3 {
		return 0
	}
	res := 0
	for i := 0; i <= len(s)-3; i++ {
		if s[i] != s[i+1] && s[i] != s[i+2] && s[i+1] != s[i+2] {
			res++
		}
	}
	return res
}

/*
61. Prefix Sum - Subarray Sum Equals K
*/
func subarraySumEqualsK(nums []int, k int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		res += m[sum-k]
		m[sum]++
	}
	return res
}

/*
62. Prefix Sum - Find Equilibrium Index
*/
func equilibriumIndex(nums []int) int {
	total := 0
	for _, v := range nums {
		total += v
	}
	leftSum := 0
	for i, v := range nums {
		if leftSum == total-leftSum-v {
			return i
		}
		leftSum += v
	}
	return -1
}

/*
63. Prefix Sum - Count Subarrays with Sum Divisible by K
*/
func subarraysDivByK(nums []int, k int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		mod := ((sum % k) + k) % k
		res += m[mod]
		m[mod]++
	}
	return res
}

/*
64. Prefix Sum - Maximum Size Subarray Sum Equals K
*/
func maxSubArrayLen(nums []int, k int) int {
	m := map[int]int{0: -1}
	sum, res := 0, 0
	for i, v := range nums {
		sum += v
		if idx, ok := m[sum-k]; ok {
			if i-idx > res {
				res = i - idx
			}
		}
		if _, ok := m[sum]; !ok {
			m[sum] = i
		}
	}
	return res
}

/*
65. Prefix Sum - Continuous Subarray Sum
*/
func checkSubarraySum(nums []int, k int) bool {
	m := map[int]int{0: -1}
	sum := 0
	for i, v := range nums {
		sum += v
		mod := sum
		if k != 0 {
			mod %= k
		}
		if idx, ok := m[mod]; ok {
			if i-idx > 1 {
				return true
			}
		} else {
			m[mod] = i
		}
	}
	return false
}

/*
66. Prefix Sum - Number of Subarrays with Bounded Maximum
*/
func numSubarrayBoundedMaxPrefix(nums []int, left int, right int) int {
	res, last1, last2 := 0, -1, -1
	for i, v := range nums {
		if v > right {
			last1 = i
		}
		if v >= left {
			last2 = i
		}
		res += last2 - last1
	}
	return res
}

/*
67. Prefix Sum - Range Sum Query
*/
type NumArray struct {
	prefix []int
}

func ConstructorNumArray(nums []int) NumArray {
	prefix := make([]int, len(nums)+1)
	for i := 0; i < len(nums); i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	return NumArray{prefix}
}

func (na *NumArray) SumRange(left int, right int) int {
	return na.prefix[right+1] - na.prefix[left]
}

/*
68. Prefix Sum - Find Pivot Index
*/
func pivotIndex(nums []int) int {
	total := 0
	for _, v := range nums {
		total += v
	}
	leftSum := 0
	for i, v := range nums {
		if leftSum == total-leftSum-v {
			return i
		}
		leftSum += v
	}
	return -1
}

/*
69. Prefix Sum - Count Ways to Split Array into Equal Sum Parts
*/
func waysToSplitArray(nums []int) int {
	total := 0
	for _, v := range nums {
		total += v
	}
	leftSum, res := 0, 0
	for i := 0; i < len(nums)-1; i++ {
		leftSum += nums[i]
		if leftSum*2 >= total {
			res++
		}
	}
	return res
}

/*
70. Prefix Sum - Find Longest Balanced Subarray
*/
func longestBalancedSubarray(nums []int) int {
	m := map[int]int{0: -1}
	bal, res := 0, 0
	for i, v := range nums {
		if v == 0 {
			bal--
		} else {
			bal++
		}
		if idx, ok := m[bal]; ok {
			if i-idx > res {
				res = i - idx
			}
		} else {
			m[bal] = i
		}
	}
	return res
}

/*
71. Prefix Sum - Number of Subarrays with Sum in Range
*/
func numSubarraysWithSumInRange(nums []int, lower int, upper int) int {
	return countRangeSum(nums, upper) - countRangeSum(nums, lower-1)
}

func countRangeSum(nums []int, k int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		res += m[sum-k]
		m[sum]++
	}
	return res
}

/*
72. Prefix Sum - Maximum Length of Subarray With Positive Product
*/
func getMaxLen(nums []int) int {
	pos, neg, res := 0, 0, 0
	for _, v := range nums {
		if v == 0 {
			pos, neg = 0, 0
		} else if v > 0 {
			pos++
			if neg > 0 {
				neg++
			}
		} else {
			pos, neg = neg, pos
			neg++
			if pos > 0 {
				pos++
			}
		}
		if pos > res {
			res = pos
		}
	}
	return res
}

/*
73. Prefix Sum - Count Number of Subarrays with Equal Number of 0s and 1s
*/
func countSubarraysEqual01(nums []int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		if v == 0 {
			sum--
		} else {
			sum++
		}
		res += m[sum]
		m[sum]++
	}
	return res
}

/*
74. Prefix Sum - Number of Subarrays with Sum Equals Target
*/
func numSubarraysWithSum(nums []int, goal int) int {
	m := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		res += m[sum-goal]
		m[sum]++
	}
	return res
}

/*
75. Prefix Sum - Count Subarrays with Sum Less Than K
*/
func countSubarraysSumLessThanK(nums []int, k int) int {
	res := 0
	for i := 0; i < len(nums); i++ {
		sum := 0
		for j := i; j < len(nums); j++ {
			sum += nums[j]
			if sum < k {
				res++
			} else {
				break
			}
		}
	}
	return res
}

/*
76. Prefix Sum - Find Total Strength of Wizards
*/
func totalStrength(wizards []int) int {
	const mod = 1_000_000_007
	n := len(wizards)
	prefix := make([]int, n+2)
	for i := 0; i < n; i++ {
		prefix[i+1] = (prefix[i] + wizards[i]) % mod
	}
	prefix2 := make([]int, n+2)
	for i := 0; i <= n; i++ {
		prefix2[i+1] = (prefix2[i] + prefix[i]) % mod
	}
	stack := []int{}
	res := 0
	for i := 0; i <= n; i++ {
		for len(stack) > 0 && (i == n || wizards[stack[len(stack)-1]] > func() int {
			if i == n {
				return 0
			}
			return wizards[i]
		}()) {
			j := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			l := -1
			if len(stack) > 0 {
				l = stack[len(stack)-1]
			}
			r := i
			left := (prefix2[j+1] - prefix2[l+1] + mod) % mod
			right := (prefix2[r+1] - prefix2[j+1] + mod) % mod
			cnt := ((right*(j-l)%mod-left*(r-j)%mod)%mod + mod) % mod
			res = (res + wizards[j]*cnt%mod) % mod
		}
		stack = append(stack, i)
	}
	return res
}

/*
77. Prefix Sum - Sum of Subarray Minimums
*/
func sumSubarrayMins(arr []int) int {
	const mod = 1_000_000_007
	n := len(arr)
	stack := []int{}
	prev := make([]int, n)
	for i := 0; i < n; i++ {
		for len(stack) > 0 && arr[stack[len(stack)-1]] > arr[i] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			prev[i] = -1
		} else {
			prev[i] = stack[len(stack)-1]
		}
		stack = append(stack, i)
	}
	stack = []int{}
	next := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		for len(stack) > 0 && arr[stack[len(stack)-1]] >= arr[i] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			next[i] = n
		} else {
			next[i] = stack[len(stack)-1]
		}
		stack = append(stack, i)
	}
	res := 0
	for i := 0; i < n; i++ {
		res = (res + arr[i]*(i-prev[i])*(next[i]-i)) % mod
	}
	return res
}

/*
78. Prefix Sum - Sum of Subarray Ranges
*/
func subArrayRanges(nums []int) int64 {
	n := len(nums)
	var res int64
	stack := []int{}
	for i := 0; i <= n; i++ {
		for len(stack) > 0 && (i == n || nums[stack[len(stack)-1]] > func() int {
			if i == n {
				return -1 << 31
			}
			return nums[i]
		}()) {
			j := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			k := -1
			if len(stack) > 0 {
				k = stack[len(stack)-1]
			}
			res -= int64(nums[j]) * int64((j-k)*(i-j))
		}
		stack = append(stack, i)
	}
	stack = []int{}
	for i := 0; i <= n; i++ {
		for len(stack) > 0 && (i == n || nums[stack[len(stack)-1]] < func() int {
			if i == n {
				return 1<<31 - 1
			}
			return nums[i]
		}()) {
			j := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			k := -1
			if len(stack) > 0 {
				k = stack[len(stack)-1]
			}
			res += int64(nums[j]) * int64((j-k)*(i-j))
		}
		stack = append(stack, i)
	}
	return res
}

/*
79. Prefix Sum - Find Number of Subarrays with Average Greater Than or Equal to K
*/
func numOfSubarraysAvgGEK(nums []int, k int) int {
	n := len(nums)
	prefix := make([]int, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + nums[i] - k
	}
	m := map[int]int{}
	res := 0
	for i := 0; i <= n; i++ {
		for key := range m {
			if prefix[i] >= key {
				res += m[key]
			}
		}
		m[prefix[i]]++
	}
	return res
}

/*
80. Prefix Sum - Count of Range Sum
*/
func countRangeSumPrefix(nums []int, lower int, upper int) int {
	n := len(nums)
	prefix := make([]int64, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + int64(nums[i])
	}
	return countRangeSumMerge(prefix, 0, n+1, int64(lower), int64(upper))
}

func countRangeSumMerge(sums []int64, left, right int, lower, upper int64) int {
	if right-left <= 1 {
		return 0
	}
	mid := (left + right) / 2
	count := countRangeSumMerge(sums, left, mid, lower, upper) + countRangeSumMerge(sums, mid, right, lower, upper)
	j, k := mid, mid
	cache := make([]int64, right-left)
	rIdx := 0
	for i := left; i < mid; i++ {
		for k < right && sums[k]-sums[i] < lower {
			k++
		}
		for j < right && sums[j]-sums[i] <= upper {
			j++
		}
		count += j - k
	}
	l, r := left, mid
	for l < mid && r < right {
		if sums[l] < sums[r] {
			cache[rIdx] = sums[l]
			l++
		} else {
			cache[rIdx] = sums[r]
			r++
		}
		rIdx++
	}
	for l < mid {
		cache[rIdx] = sums[l]
		l++
		rIdx++
	}
	for r < right {
		cache[rIdx] = sums[r]
		r++
		rIdx++
	}
	for i := 0; i < right-left; i++ {
		sums[left+i] = cache[i]
	}
	return count
}

/*
81. Prefix Sum - Minimum Size Subarray Sum
*/
func minSubArrayLenPrefix(target int, nums []int) int {
	n := len(nums)
	left, sum, res := 0, 0, n+1
	for right := 0; right < n; right++ {
		sum += nums[right]
		for sum >= target {
			if right-left+1 < res {
				res = right - left + 1
			}
			sum -= nums[left]
			left++
		}
	}
	if res == n+1 {
		return 0
	}
	return res
}

/*
82. Prefix Sum - Maximum Average Subarray II
*/
func findMaxAverageII(nums []int, k int) float64 {
	left, right := -1e4, 1e4
	for right-left > 1e-5 {
		mid := (left + right) / 2
		if checkMaxAverage(nums, k, mid) {
			left = mid
		} else {
			right = mid
		}
	}
	return left
}

func checkMaxAverage(nums []int, k int, avg float64) bool {
	n := len(nums)
	prefix := make([]float64, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + float64(nums[i]) - avg
	}
	minPrefix := 0.0
	for i := k; i <= n; i++ {
		if prefix[i]-minPrefix >= 0 {
			return true
		}
		if prefix[i-k+1] < minPrefix {
			minPrefix = prefix[i-k+1]
		}
	}
	return false
}

/*
83. Prefix Sum - Find Longest Arithmetic Subarray
*/
func longestArithmeticSubarray(nums []int) int {
	if len(nums) < 2 {
		return len(nums)
	}
	maxLen, currLen, diff := 2, 2, nums[1]-nums[0]
	for i := 2; i < len(nums); i++ {
		if nums[i]-nums[i-1] == diff {
			currLen++
		} else {
			diff = nums[i] - nums[i-1]
			currLen = 2
		}
		if currLen > maxLen {
			maxLen = currLen
		}
	}
	return maxLen
}

/*
84. Prefix Sum - Count Subarrays with Product Less Than K
*/
func numSubarrayProductLessThanKPrefix(nums []int, k int) int {
	if k <= 1 {
		return 0
	}
	prod, left, res := 1, 0, 0
	for right := 0; right < len(nums); right++ {
		prod *= nums[right]
		for prod >= k {
			prod /= nums[left]
			left++
		}
		res += right - left + 1
	}
	return res
}

/*
85. Prefix Sum - Sum of All Odd Length Subarrays
*/
func sumOddLengthSubarrays(arr []int) int {
	n := len(arr)
	res := 0
	for i := 0; i < n; i++ {
		left := i + 1
		right := n - i
		total := left * right
		odd := (total + 1) / 2
		res += arr[i] * odd
	}
	return res
}

/*
86. Prefix Sum - Find Number of Subarrays with Sum Divisible by M
*/
func subarraysDivByM(nums []int, m int) int {
	count := map[int]int{0: 1}
	sum, res := 0, 0
	for _, v := range nums {
		sum += v
		mod := ((sum % m) + m) % m
		res += count[mod]
		count[mod]++
	}
	return res
}

/*
87. Prefix Sum - Number of Ways to Split Array Into Three Subarrays with Equal Sum
*/
func waysToSplit(nums []int) int {
	const mod = 1_000_000_007
	n := len(nums)
	prefix := make([]int, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + nums[i]
	}
	res := 0
	for i := 1; i < n-1; i++ {
		left := sort.Search(n, func(j int) bool { return prefix[j+1]-prefix[i] >= prefix[i] })
		right := sort.Search(n, func(j int) bool { return prefix[n]-prefix[j+1] < prefix[j+1]-prefix[i] })
		if left <= right && right < n {
			res = (res + right - left + 1) % mod
		}
	}
	return res
}

/*
88. Prefix Sum - Find Subarray with Given Sum
*/
func findSubarrayWithSum(nums []int, target int) (int, int, bool) {
	m := map[int]int{0: -1}
	sum := 0
	for i, v := range nums {
		sum += v
		if idx, ok := m[sum-target]; ok {
			return idx + 1, i, true
		}
		m[sum] = i
	}
	return -1, -1, false
}

/*
89. Prefix Sum - Maximum Length of Subarray With Sum K
*/
func maxLenSubarraySumK(nums []int, k int) int {
	m := map[int]int{0: -1}
	sum, res := 0, 0
	for i, v := range nums {
		sum += v
		if idx, ok := m[sum-k]; ok {
			if i-idx > res {
				res = i - idx
			}
		}
		if _, ok := m[sum]; !ok {
			m[sum] = i
		}
	}
	return res
}

/*
90. Prefix Sum - Number of Subarrays With Exactly K Odd Numbers
*/
func numberOfSubarraysExactlyKOdd(nums []int, k int) int {
	return atMostKOdd(nums, k) - atMostKOdd(nums, k-1)
}

/*
91. Binary Search - Find Peak Element
*/
func findPeakElementBinary(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[mid+1] {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}

/*
92. Binary Search - Find Minimum in Rotated Sorted Array
*/
func findMin(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return nums[left]
}

/*
93. Binary Search - Find Kth Smallest Element in Sorted Matrix
*/
func kthSmallest(matrix [][]int, k int) int {
	n := len(matrix)
	left, right := matrix[0][0], matrix[n-1][n-1]
	for left < right {
		mid := (left + right) / 2
		count := 0
		for i, j := n-1, 0; i >= 0 && j < n; {
			if matrix[i][j] <= mid {
				count += i + 1
				j++
			} else {
				i--
			}
		}
		if count < k {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

/*
94. Binary Search - Search in Rotated Sorted Array
*/
func searchRotated(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[left] <= nums[mid] {
			if nums[left] <= target && target < nums[mid] {
				right = mid - 1
			} else {
				left = mid + 1
			}
		} else {
			if nums[mid] < target && target <= nums[right] {
				left = mid + 1
			} else {
				right = mid - 1
			}
		}
	}
	return -1
}

/*
95. Binary Search - Find First and Last Position of Element in Sorted Array
*/
func searchRange(nums []int, target int) []int {
	left := sort.Search(len(nums), func(i int) bool { return nums[i] >= target })
	if left == len(nums) || nums[left] != target {
		return []int{-1, -1}
	}
	right := sort.Search(len(nums), func(i int) bool { return nums[i] > target })
	return []int{left, right - 1}
}

/*
96. Binary Search - Median of Two Sorted Arrays
*/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	if len(nums1) > len(nums2) {
		return findMedianSortedArrays(nums2, nums1)
	}
	m, n := len(nums1), len(nums2)
	imin, imax, half := 0, m, (m+n+1)/2
	for imin <= imax {
		i := (imin + imax) / 2
		j := half - i
		if i < m && nums2[j-1] > nums1[i] {
			imin = i + 1
		} else if i > 0 && nums1[i-1] > nums2[j] {
			imax = i - 1
		} else {
			maxLeft := 0
			if i == 0 {
				maxLeft = nums2[j-1]
			} else if j == 0 {
				maxLeft = nums1[i-1]
			} else {
				if nums1[i-1] > nums2[j-1] {
					maxLeft = nums1[i-1]
				} else {
					maxLeft = nums2[j-1]
				}
			}
			if (m+n)%2 == 1 {
				return float64(maxLeft)
			}
			minRight := 0
			if i == m {
				minRight = nums2[j]
			} else if j == n {
				minRight = nums1[i]
			} else {
				if nums1[i] < nums2[j] {
					minRight = nums1[i]
				} else {
					minRight = nums2[j]
				}
			}
			return float64(maxLeft+minRight) / 2.0
		}
	}
	return 0
}

/*
97. Binary Search - Find Smallest Letter Greater Than Target
*/
func nextGreatestLetter(letters []byte, target byte) byte {
	left, right := 0, len(letters)
	for left < right {
		mid := (left + right) / 2
		if letters[mid] <= target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return letters[left%len(letters)]
}

/*
98. Binary Search - Split Array Largest Sum
*/
func splitArray(nums []int, m int) int {
	maxNum, sum := 0, 0
	for _, v := range nums {
		if v > maxNum {
			maxNum = v
		}
		sum += v
	}
	left, right := maxNum, sum
	for left < right {
		mid := (left + right) / 2
		if canSplit(nums, m, mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
func canSplit(nums []int, m, maxSum int) bool {
	count, curr := 1, 0
	for _, v := range nums {
		if curr+v > maxSum {
			count++
			curr = v
		} else {
			curr += v
		}
	}
	return count <= m
}

/*
99. Binary Search - Capacity To Ship Packages Within D Days
*/
func shipWithinDays(weights []int, days int) int {
	maxW, sum := 0, 0
	for _, w := range weights {
		if w > maxW {
			maxW = w
		}
		sum += w
	}
	left, right := maxW, sum
	for left < right {
		mid := (left + right) / 2
		if canShip(weights, days, mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
func canShip(weights []int, days, cap int) bool {
	d, curr := 1, 0
	for _, w := range weights {
		if curr+w > cap {
			d++
			curr = w
		} else {
			curr += w
		}
	}
	return d <= days
}

/*
100. Binary Search - Koko Eating Bananas
*/
func minEatingSpeed(piles []int, h int) int {
	maxP := 0
	for _, p := range piles {
		if p > maxP {
			maxP = p
		}
	}
	left, right := 1, maxP
	for left < right {
		mid := (left + right) / 2
		if canEat(piles, h, mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
func canEat(piles []int, h, k int) bool {
	time := 0
	for _, p := range piles {
		time += (p + k - 1) / k
	}
	return time <= h
}

/*
101. Binary Search - Find Position to Insert Element
*/
func searchInsert(nums []int, target int) int {
	left, right := 0, len(nums)
	for left < right {
		mid := (left + right) / 2
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

/*
102. Binary Search - Find Peak Index in Mountain Array
*/
func peakIndexInMountainArray(arr []int) int {
	left, right := 0, len(arr)-1
	for left < right {
		mid := (left + right) / 2
		if arr[mid] < arr[mid+1] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

/*
103. Binary Search - Search in 2D Matrix
*/
func searchMatrix(matrix [][]int, target int) bool {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return false
	}
	m, n := len(matrix), len(matrix[0])
	left, right := 0, m*n-1
	for left <= right {
		mid := (left + right) / 2
		val := matrix[mid/n][mid%n]
		if val == target {
			return true
		} else if val < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return false
}

/*
104. Binary Search - Find Element in Infinite Sorted Array
*/
func searchInfiniteArray(reader func(int) int, target int) int {
	left, right := 0, 1
	for reader(right) < target {
		left = right
		right *= 2
	}
	for left <= right {
		mid := (left + right) / 2
		val := reader(mid)
		if val == target {
			return mid
		} else if val < target {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

/*
105. Binary Search - Find Fixed Point (Index equals Value)
*/
func fixedPoint(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == mid {
			return mid
		} else if nums[mid] < mid {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return -1
}

/*
106. Binary Search - Find First Bad Version
*/
func firstBadVersion(n int, isBadVersion func(int) bool) int {
	left, right := 1, n
	for left < right {
		mid := left + (right-left)/2
		if isBadVersion(mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}

/*
107. Binary Search - Find Square Root of Number
*/
func mySqrt(x int) int {
	if x < 2 {
		return x
	}
	left, right := 1, x/2
	for left <= right {
		mid := (left + right) / 2
		if mid*mid == x {
			return mid
		} else if mid*mid < x {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return right
}

/*
108. Binary Search - Find Minimum in Rotated Sorted Array II
*/
func findMinII(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[right] {
			left = mid + 1
		} else if nums[mid] < nums[right] {
			right = mid
		} else {
			right--
		}
	}
	return nums[left]
}

/*
109. Binary Search - Find Median in Data Stream
*/
type MedianFinder struct {
	minHeap, maxHeap *IntHeap
}

func ConstructorMedianFinder() MedianFinder {
	minH := &IntHeap{isMin: true}
	maxH := &IntHeap{isMin: false}
	return MedianFinder{minH, maxH}
}

func (mf *MedianFinder) AddNum(num int) {
	if mf.maxHeap.Len() == 0 || num <= (*mf.maxHeap).Peek() {
		mf.maxHeap.Push(num)
	} else {
		mf.minHeap.Push(num)
	}
	if mf.maxHeap.Len() > mf.minHeap.Len()+1 {
		mf.minHeap.Push(mf.maxHeap.Pop())
	} else if mf.minHeap.Len() > mf.maxHeap.Len() {
		mf.maxHeap.Push(mf.minHeap.Pop())
	}
}

func (mf *MedianFinder) FindMedian() float64 {
	if mf.maxHeap.Len() > mf.minHeap.Len() {
		return float64((*mf.maxHeap).Peek())
	}
	return (float64((*mf.maxHeap).Peek()) + float64((*mf.minHeap).Peek())) / 2.0
}

type IntHeap struct {
	data  []int
	isMin bool
}

func (h *IntHeap) Len() int {
	return len(h.data)
}
func (h *IntHeap) Push(x int) {
	h.data = append(h.data, x)
	h.up(h.Len() - 1)
}
func (h *IntHeap) Pop() int {
	n := h.Len() - 1
	h.swap(0, n)
	x := h.data[n]
	h.data = h.data[:n]
	h.down(0)
	return x
}
func (h *IntHeap) Peek() int {
	return h.data[0]
}
func (h *IntHeap) less(i, j int) bool {
	if h.isMin {
		return h.data[i] < h.data[j]
	}
	return h.data[i] > h.data[j]
}
func (h *IntHeap) swap(i, j int) {
	h.data[i], h.data[j] = h.data[j], h.data[i]
}
func (h *IntHeap) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !h.less(i, p) {
			break
		}
		h.swap(i, p)
		i = p
	}
}
func (h *IntHeap) down(i int) {
	n := h.Len()
	for {
		l, r, smallest := 2*i+1, 2*i+2, i
		if l < n && h.less(l, smallest) {
			smallest = l
		}
		if r < n && h.less(r, smallest) {
			smallest = r
		}
		if smallest == i {
			break
		}
		h.swap(i, smallest)
		i = smallest
	}
}

/*
110. Binary Search - Find Rotation Count in Rotated Array
*/
func findRotationCount(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[right] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

/*
111. Binary Search - Find Minimum Difference Element
*/
func findMinDiffElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	res := nums[0]
	for left <= right {
		mid := (left + right) / 2
		if abs(nums[mid]-target) < abs(res-target) {
			res = nums[mid]
		}
		if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			return nums[mid]
		}
	}
	return res
}

/*
112. Binary Search - Find Range Sum Query
*/
func rangeSumQuery(nums []int, left int, right int) int {
	sum := 0
	for i := left; i <= right; i++ {
		sum += nums[i]
	}
	return sum
}

/*
113. Binary Search - Find Maximum Average Subarray
*/
func findMaxAverageBinary(nums []int, k int) float64 {
	left, right := -1e4, 1e4
	for right-left > 1e-5 {
		mid := (left + right) / 2
		if checkMaxAverage(nums, k, mid) {
			left = mid
		} else {
			right = mid
		}
	}
	return left
}

/*
114. Binary Search - Find Element in Nearly Sorted Array
*/
func searchNearlySorted(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			return mid
		}
		if mid-1 >= left && nums[mid-1] == target {
			return mid - 1
		}
		if mid+1 <= right && nums[mid+1] == target {
			return mid + 1
		}
		if nums[mid] < target {
			left = mid + 2
		} else {
			right = mid - 2
		}
	}
	return -1
}

/*
115. Binary Search - Find Closest Element to Target
*/
func findClosestElement(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] < target {
			left = mid + 1
		} else {
			right = mid
		}
	}
	if left == 0 {
		return nums[0]
	}
	if abs(nums[left]-target) < abs(nums[left-1]-target) {
		return nums[left]
	}
	return nums[left-1]
}

/*
116. Binary Search - Find Maximum in Bitonic Array
*/
func findMaxInBitonic(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] < nums[mid+1] {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return nums[left]
}

/*
117. Binary Search - Find Missing Number
*/
func missingNumber(nums []int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == mid {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}
	return left
}

/*
118. Binary Search - Allocate Minimum Number of Pages
*/
func minPages(books []int, students int) int {
	maxB, sum := 0, 0
	for _, b := range books {
		if b > maxB {
			maxB = b
		}
		sum += b
	}
	left, right := maxB, sum
	for left < right {
		mid := (left + right) / 2
		if canAllocate(books, students, mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}
	return left
}
func canAllocate(books []int, students, maxPages int) bool {
	count, curr := 1, 0
	for _, b := range books {
		if curr+b > maxPages {
			count++
			curr = b
		} else {
			curr += b
		}
	}
	return count <= students
}

/*
119. Binary Search - Split Array to Minimize Largest Sum
*/
func splitArrayMinLargestSum(nums []int, m int) int {
	return splitArray(nums, m)
}

/*
120. Binary Search - Find Peak Element with Duplicates
*/
func findPeakElementWithDuplicates(nums []int) int {
	left, right := 0, len(nums)-1
	for left < right {
		mid := (left + right) / 2
		if nums[mid] > nums[mid+1] {
			right = mid
		} else if nums[mid] < nums[mid+1] {
			left = mid + 1
		} else {
			left++
		}
	}
	return left
}

// 121. Sorting - Sort Colors (Dutch National Flag Problem)
func sortColorsDNF(nums []int) {
	low, mid, high := 0, 0, len(nums)-1
	for mid <= high {
		switch nums[mid] {
		case 0:
			nums[low], nums[mid] = nums[mid], nums[low]
			low++
			mid++
		case 1:
			mid++
		case 2:
			nums[mid], nums[high] = nums[high], nums[mid]
			high--
		}
	}
}

// Sorts 0s, 1s, and 2s in-place using three pointers (Dutch National Flag).

// 122. Sorting - Merge Intervals
func mergeIntervalsSorting(intervals [][]int) [][]int {
	if len(intervals) == 0 {
		return nil
	}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})
	res := [][]int{intervals[0]}
	for _, curr := range intervals[1:] {
		last := res[len(res)-1]
		if curr[0] <= last[1] {
			if curr[1] > last[1] {
				res[len(res)-1][1] = curr[1]
			}
		} else {
			res = append(res, curr)
		}
	}
	return res
}

// Merges overlapping intervals after sorting by start time.

// 123. Sorting - Find Kth Largest Element
func findKthLargest(nums []int, k int) int {
	sort.Sort(sort.Reverse(sort.IntSlice(nums)))
	return nums[k-1]
}

// Sorts array in descending order and returns the k-th largest element.

// 124. Sorting - Sort Array By Parity
func sortArrayByParitySorting(nums []int) []int {
	left, right := 0, len(nums)-1
	for left < right {
		if nums[left]%2 > nums[right]%2 {
			nums[left], nums[right] = nums[right], nums[left]
		}
		if nums[left]%2 == 0 {
			left++
		}
		if nums[right]%2 == 1 {
			right--
		}
	}
	return nums
}

// Moves all even numbers to the front, odd to the back, in-place.

// 125. Sorting - Wiggle Sort
func wiggleSort(nums []int) {
	for i := 1; i < len(nums); i++ {
		if (i%2 == 1 && nums[i] < nums[i-1]) || (i%2 == 0 && nums[i] > nums[i-1]) {
			nums[i], nums[i-1] = nums[i-1], nums[i]
		}
	}
}

// Rearranges array so nums[0] < nums[1] > nums[2] < nums[3] ...

// 126. Sorting - Sort Array of Squares
func sortedSquaresSorting(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	left, right := 0, n-1
	for i := n - 1; i >= 0; i-- {
		if abs(nums[left]) > abs(nums[right]) {
			res[i] = nums[left] * nums[left]
			left++
		} else {
			res[i] = nums[right] * nums[right]
			right--
		}
	}
	return res
}

// Returns sorted squares of a sorted array using two pointers.

// 127. Sorting - Minimum Number of Swaps to Sort Array
func minSwapsToSort(nums []int) int {
	n := len(nums)
	arr := make([][2]int, n)
	for i, v := range nums {
		arr[i] = [2]int{v, i}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i][0] < arr[j][0] })
	visited := make([]bool, n)
	res := 0
	for i := 0; i < n; i++ {
		if visited[i] || arr[i][1] == i {
			continue
		}
		cycle := 0
		j := i
		for !visited[j] {
			visited[j] = true
			j = arr[j][1]
			cycle++
		}
		if cycle > 0 {
			res += cycle - 1
		}
	}
	return res
}

// Counts minimum swaps needed to sort array using cycle decomposition.

// 128. Sorting - Count Inversions in Array
func countInversions(nums []int) int {
	var mergeSort func([]int) ([]int, int)
	mergeSort = func(arr []int) ([]int, int) {
		if len(arr) <= 1 {
			return arr, 0
		}
		mid := len(arr) / 2
		left, invL := mergeSort(arr[:mid])
		right, invR := mergeSort(arr[mid:])
		merged := make([]int, 0, len(arr))
		i, j, inv := 0, 0, invL+invR
		for i < len(left) && j < len(right) {
			if left[i] <= right[j] {
				merged = append(merged, left[i])
				i++
			} else {
				merged = append(merged, right[j])
				inv += len(left) - i
				j++
			}
		}
		merged = append(merged, left[i:]...)
		merged = append(merged, right[j:]...)
		return merged, inv
	}
	_, inv := mergeSort(nums)
	return inv
}

// Uses merge sort to count number of inversions in array.

// 129. Sorting - Relative Sort Array
func relativeSortArray(arr1 []int, arr2 []int) []int {
	pos := make(map[int]int)
	for i, v := range arr2 {
		pos[v] = i
	}
	sort.Slice(arr1, func(i, j int) bool {
		pi, iok := pos[arr1[i]]
		pj, jok := pos[arr1[j]]
		if iok && jok {
			return pi < pj
		}
		if iok {
			return true
		}
		if jok {
			return false
		}
		return arr1[i] < arr1[j]
	})
	return arr1
}

// Sorts arr1 so that elements in arr2 come first in order, rest sorted.

// 130. Sorting - Largest Number Formed by Array
func largestNumber(nums []int) string {
	strs := make([]string, len(nums))
	for i, v := range nums {
		strs[i] = fmt.Sprintf("%d", v)
	}
	sort.Slice(strs, func(i, j int) bool {
		return strs[i]+strs[j] > strs[j]+strs[i]
	})
	if strs[0] == "0" {
		return "0"
	}
	return strings.Join(strs, "")
}

// Arranges numbers to form the largest possible concatenated number.

// 131. Partitioning - Dutch National Flag Problem
func dutchNationalFlag(nums []int) {
	low, mid, high := 0, 0, len(nums)-1
	for mid <= high {
		switch nums[mid] {
		case 0:
			nums[low], nums[mid] = nums[mid], nums[low]
			low++
			mid++
		case 1:
			mid++
		case 2:
			nums[mid], nums[high] = nums[high], nums[mid]
			high--
		}
	}
}

// Partitions array into 0s, 1s, and 2s in-place.

// 132. Partitioning - Partition Array into Disjoint Intervals
func partitionDisjoint(nums []int) int {
	n := len(nums)
	maxLeft := make([]int, n)
	minRight := make([]int, n)
	maxLeft[0] = nums[0]
	for i := 1; i < n; i++ {
		if nums[i] > maxLeft[i-1] {
			maxLeft[i] = nums[i]
		} else {
			maxLeft[i] = maxLeft[i-1]
		}
	}
	minRight[n-1] = nums[n-1]
	for i := n - 2; i >= 0; i-- {
		if nums[i] < minRight[i+1] {
			minRight[i] = nums[i]
		} else {
			minRight[i] = minRight[i+1]
		}
	}
	for i := 0; i < n-1; i++ {
		if maxLeft[i] <= minRight[i+1] {
			return i + 1
		}
	}
	return n
}

// Finds the smallest partition such that max(left) <= min(right).

// 133. Partitioning - Sort Array According to Another Array
func sortByAnotherArray(arr1, arr2 []int) []int {
	pos := make(map[int]int)
	for i, v := range arr2 {
		pos[v] = i
	}
	sort.Slice(arr1, func(i, j int) bool {
		pi, iok := pos[arr1[i]]
		pj, jok := pos[arr1[j]]
		if iok && jok {
			return pi < pj
		}
		if iok {
			return true
		}
		if jok {
			return false
		}
		return arr1[i] < arr1[j]
	})
	return arr1
}

// Sorts arr1 so that elements in arr2 come first in order, rest sorted.

// 134. Partitioning - Partition Labels
func partitionLabels(s string) []int {
	last := make(map[byte]int)
	for i := 0; i < len(s); i++ {
		last[s[i]] = i
	}
	var res []int
	start, end := 0, 0
	for i := 0; i < len(s); i++ {
		if last[s[i]] > end {
			end = last[s[i]]
		}
		if i == end {
			res = append(res, end-start+1)
			start = i + 1
		}
	}
	return res
}

// Splits string into as many parts as possible so that each letter appears in at most one part.

// 135. Partitioning - Sort Array by Increasing Frequency
func frequencySort(nums []int) []int {
	count := make(map[int]int)
	for _, v := range nums {
		count[v]++
	}
	sort.Slice(nums, func(i, j int) bool {
		if count[nums[i]] == count[nums[j]] {
			return nums[i] > nums[j]
		}
		return count[nums[i]] < count[nums[j]]
	})
	return nums
}

// Sorts array by increasing frequency, ties by decreasing value.

// 136. Partitioning - Find the Kth Smallest Pair Distance
func smallestDistancePair(nums []int, k int) int {
	sort.Ints(nums)
	left, right := 0, nums[len(nums)-1]-nums[0]
	for left < right {
		mid := (left + right) / 2
		count := 0
		j := 0
		for i := 0; i < len(nums); i++ {
			for j < len(nums) && nums[j]-nums[i] <= mid {
				j++
			}
			count += j - i - 1
		}
		if count < k {
			left = mid + 1
		} else {
			right = mid
		}
	}
	return left
}

// Finds the k-th smallest distance among all pairs using binary search.

// 137. Partitioning - Split Array Into Consecutive Subsequences
func isPossiblePartition(nums []int) bool {
	count := make(map[int]int)
	end := make(map[int]int)
	for _, v := range nums {
		count[v]++
	}
	for _, v := range nums {
		if count[v] == 0 {
			continue
		}
		if end[v-1] > 0 {
			end[v-1]--
			end[v]++
		} else if count[v+1] > 0 && count[v+2] > 0 {
			count[v+1]--
			count[v+2]--
			end[v+2]++
		} else {
			return false
		}
		count[v]--
	}
	return true
}

// Checks if array can be split into consecutive subsequences of length ≥ 3.

// 138. Partitioning - Minimum Number of Increments on Subarrays
func minNumberOperations(target []int) int {
	res, prev := 0, 0
	for _, v := range target {
		if v > prev {
			res += v - prev
		}
		prev = v
	}
	return res
}

// Returns minimum increments needed to make array equal to target by incrementing subarrays.

// 139. Partitioning - Find Pivot Index
func findPivotIndex(nums []int) int {
	total := 0
	for _, v := range nums {
		total += v
	}
	leftSum := 0
	for i, v := range nums {
		if leftSum == total-leftSum-v {
			return i
		}
		leftSum += v
	}
	return -1
}

// Returns index where sum of left equals sum of right.

// 140. Partitioning - Longest Increasing Subsequence via Patience Sorting
func lengthOfLIS(nums []int) int {
	tails := []int{}
	for _, v := range nums {
		i := sort.SearchInts(tails, v)
		if i == len(tails) {
			tails = append(tails, v)
		} else {
			tails[i] = v
		}
	}
	return len(tails)
}

// Uses patience sorting (binary search) to find LIS length in O(n log n).

// 141. Partitioning - Sort Characters By Frequency
func frequencySortString(s string) string {
	count := make(map[rune]int)
	for _, c := range s {
		count[c]++
	}
	type pair struct {
		c rune
		f int
	}
	arr := make([]pair, 0, len(count))
	for c, f := range count {
		arr = append(arr, pair{c, f})
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].f > arr[j].f
	})
	var b strings.Builder
	for _, p := range arr {
		for i := 0; i < p.f; i++ {
			b.WriteRune(p.c)
		}
	}
	return b.String()
}

// Sorts characters in string by decreasing frequency.

// 142. Partitioning - K Closest Points to Origin
func kClosest(points [][]int, k int) [][]int {
	sort.Slice(points, func(i, j int) bool {
		return points[i][0]*points[i][0]+points[i][1]*points[i][1] < points[j][0]*points[j][0]+points[j][1]*points[j][1]
	})
	return points[:k]
}

// Returns k points closest to origin using squared distance.

// 143. Partitioning - Sort Array by Parity II
func sortArrayByParityII(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	even, odd := 0, 1
	for _, v := range nums {
		if v%2 == 0 {
			res[even] = v
			even += 2
		} else {
			res[odd] = v
			odd += 2
		}
	}
	return res
}

// Places even numbers at even indices, odd at odd indices.

// 144. Partitioning - Move All Negative Numbers to Beginning
func moveNegativesToFront(nums []int) {
	j := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] < 0 {
			nums[i], nums[j] = nums[j], nums[i]
			j++
		}
	}
}

// Moves all negative numbers to the beginning of the array in-place.

// 145. Partitioning - Sort Array By Increasing Frequency
func sortByIncreasingFrequency(nums []int) []int {
	count := make(map[int]int)
	for _, v := range nums {
		count[v]++
	}
	sort.Slice(nums, func(i, j int) bool {
		if count[nums[i]] == count[nums[j]] {
			return nums[i] > nums[j]
		}
		return count[nums[i]] < count[nums[j]]
	})
	return nums
}

// Sorts array by increasing frequency, ties by decreasing value.

// 146. Partitioning - Maximize Distance Between Same Elements
func maxDistanceBetweenSame(nums []int) int {
	pos := make(map[int]int)
	maxDist := 0
	for i, v := range nums {
		if first, ok := pos[v]; ok {
			if i-first > maxDist {
				maxDist = i - first
			}
		} else {
			pos[v] = i
		}
	}
	return maxDist
}

// Returns the maximum distance between two same elements in array.

// 147. Partitioning - Minimum Number of Moves to Make Array Complementary
func minMovesToMakeComplementary(nums []int, limit int) int {
	n := len(nums)
	diff := make([]int, 2*limit+2)
	for i := 0; i < n/2; i++ {
		a, b := nums[i], nums[n-1-i]
		low := 1 + min(a, b)
		high := limit + max(a, b)
		sum := a + b
		diff[2] += 2
		diff[low] -= 1
		diff[sum] -= 1
		diff[sum+1] += 1
		diff[high+1] += 1
	}
	res, curr := n, 0
	for i := 2; i <= 2*limit; i++ {
		curr += diff[i]
		if curr < res {
			res = curr
		}
	}
	return res
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Uses prefix sum to minimize moves to make pairs sum to same value.

// 148. Partitioning - Partition Array for Maximum Sum
func maxSumAfterPartitioning(arr []int, k int) int {
	n := len(arr)
	dp := make([]int, n+1)
	for i := 1; i <= n; i++ {
		maxVal := 0
		for j := 1; j <= k && i-j >= 0; j++ {
			if arr[i-j] > maxVal {
				maxVal = arr[i-j]
			}
			if dp[i] < dp[i-j]+maxVal*j {
				dp[i] = dp[i-j] + maxVal*j
			}
		}
	}
	return dp[n]
}

// Dynamic programming to maximize sum after partitioning into subarrays of at most k length.

// 149. Partitioning - Max Chunks To Make Sorted
func maxChunksToSorted(arr []int) int {
	maxVal, res := 0, 0
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
		}
		if maxVal == i {
			res++
		}
	}
	return res
}

// Counts max chunks so that sorting each chunk individually sorts the array.

// 150. Partitioning - Sort Array with Odd Even Index Constraints
func sortOddEvenIndex(nums []int) []int {
	odds, evens := []int{}, []int{}
	for i, v := range nums {
		if i%2 == 0 {
			evens = append(evens, v)
		} else {
			odds = append(odds, v)
		}
	}
	sort.Ints(evens)
	sort.Sort(sort.Reverse(sort.IntSlice(odds)))
	res := make([]int, len(nums))
	ei, oi := 0, 0
	for i := 0; i < len(nums); i++ {
		if i%2 == 0 {
			res[i] = evens[ei]
			ei++
		} else {
			res[i] = odds[oi]
			oi++
		}
	}
	return res
}

// Sorts even indices in ascending, odd indices in descending order.
