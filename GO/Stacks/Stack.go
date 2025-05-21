package stacks

import (
	"strconv"
	"strings"
)

// 1. Stack Using Arrays
type ArrayStack struct {
	data []int
}

func NewArrayStack() *ArrayStack {
	return &ArrayStack{data: []int{}}
}

func (s *ArrayStack) Push(x int) {
	s.data = append(s.data, x)
}

func (s *ArrayStack) Pop() int {
	if len(s.data) == 0 {
		panic("stack is empty")
	}
	val := s.data[len(s.data)-1]
	s.data = s.data[:len(s.data)-1]
	return val
}

func (s *ArrayStack) Top() int {
	if len(s.data) == 0 {
		panic("stack is empty")
	}
	return s.data[len(s.data)-1]
}

func (s *ArrayStack) IsEmpty() bool {
	return len(s.data) == 0
}

// 2. Stack Using Linked List
type StackListNode struct {
	val  int
	next *StackListNode
}

type LinkedListStack struct {
	head *StackListNode
}

func NewLinkedListStack() *LinkedListStack {
	return &LinkedListStack{nil}
}

func (s *LinkedListStack) Push(x int) {
	node := &StackListNode{val: x, next: s.head}
	s.head = node
}

func (s *LinkedListStack) Pop() int {
	if s.head == nil {
		panic("stack is empty")
	}
	val := s.head.val
	s.head = s.head.next
	return val
}

func (s *LinkedListStack) Top() int {
	if s.head == nil {
		panic("stack is empty")
	}
	return s.head.val
}

func (s *LinkedListStack) IsEmpty() bool {
	return s.head == nil
}

// 3. Stack with getMin() in O(1)
type MinStack struct {
	stack    []int
	minStack []int
}

func NewMinStack() *MinStack {
	return &MinStack{stack: []int{}, minStack: []int{}}
}

func (s *MinStack) Push(x int) {
	s.stack = append(s.stack, x)
	if len(s.minStack) == 0 || x <= s.minStack[len(s.minStack)-1] {
		s.minStack = append(s.minStack, x)
	}
}

func (s *MinStack) Pop() int {
	if len(s.stack) == 0 {
		panic("stack is empty")
	}
	val := s.stack[len(s.stack)-1]
	s.stack = s.stack[:len(s.stack)-1]
	if val == s.minStack[len(s.minStack)-1] {
		s.minStack = s.minStack[:len(s.minStack)-1]
	}
	return val
}

func (s *MinStack) Top() int {
	if len(s.stack) == 0 {
		panic("stack is empty")
	}
	return s.stack[len(s.stack)-1]
}

func (s *MinStack) GetMin() int {
	if len(s.minStack) == 0 {
		panic("stack is empty")
	}
	return s.minStack[len(s.minStack)-1]
}

// 4. Evaluate Reverse Polish Notation
func EvalRPN(tokens []string) int {
	stack := []int{}
	for _, token := range tokens {
		switch token {
		case "+", "-", "*", "/":
			b := stack[len(stack)-1]
			a := stack[len(stack)-2]
			stack = stack[:len(stack)-2]
			switch token {
			case "+":
				stack = append(stack, a+b)
			case "-":
				stack = append(stack, a-b)
			case "*":
				stack = append(stack, a*b)
			case "/":
				stack = append(stack, a/b)
			}
		default:
			num, _ := strconv.Atoi(token)
			stack = append(stack, num)
		}
	}
	return stack[0]
}

// 5. Queue Using Stacks
type QueueWithStacks struct {
	inStack  []int
	outStack []int
}

func NewQueueWithStacks() *QueueWithStacks {
	return &QueueWithStacks{}
}

func (q *QueueWithStacks) Push(x int) {
	q.inStack = append(q.inStack, x)
}

func (q *QueueWithStacks) Pop() int {
	if len(q.outStack) == 0 {
		for len(q.inStack) > 0 {
			n := len(q.inStack)
			q.outStack = append(q.outStack, q.inStack[n-1])
			q.inStack = q.inStack[:n-1]
		}
	}
	if len(q.outStack) == 0 {
		panic("queue is empty")
	}
	val := q.outStack[len(q.outStack)-1]
	q.outStack = q.outStack[:len(q.outStack)-1]
	return val
}

func (q *QueueWithStacks) Peek() int {
	if len(q.outStack) == 0 {
		for len(q.inStack) > 0 {
			n := len(q.inStack)
			q.outStack = append(q.outStack, q.inStack[n-1])
			q.inStack = q.inStack[:n-1]
		}
	}
	if len(q.outStack) == 0 {
		panic("queue is empty")
	}
	return q.outStack[len(q.outStack)-1]
}

func (q *QueueWithStacks) Empty() bool {
	return len(q.inStack) == 0 && len(q.outStack) == 0
}

// 6. Valid Parentheses
func IsValidParentheses(s string) bool {
	stack := []rune{}
	mapping := map[rune]rune{')': '(', '}': '{', ']': '['}
	for _, ch := range s {
		if ch == '(' || ch == '{' || ch == '[' {
			stack = append(stack, ch)
		} else {
			if len(stack) == 0 || stack[len(stack)-1] != mapping[ch] {
				return false
			}
			stack = stack[:len(stack)-1]
		}
	}
	return len(stack) == 0
}

// 7. Generate Parentheses
func GenerateParentheses(n int) []string {
	var res []string
	var backtrack func(cur string, open, close int)
	backtrack = func(cur string, open, close int) {
		if len(cur) == 2*n {
			res = append(res, cur)
			return
		}
		if open < n {
			backtrack(cur+"(", open+1, close)
		}
		if close < open {
			backtrack(cur+")", open, close+1)
		}
	}
	backtrack("", 0, 0)
	return res
}

// 8. Longest Valid Parentheses
func LongestValidParentheses(s string) int {
	stack := []int{-1}
	maxLen := 0
	for i, ch := range s {
		if ch == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				if i-stack[len(stack)-1] > maxLen {
					maxLen = i - stack[len(stack)-1]
				}
			}
		}
	}
	return maxLen
}

// 9. Remove Invalid Parentheses
func RemoveInvalidParentheses(s string) []string {
	var res []string
	var dfs func(string, int, int, []rune)
	dfs = func(s string, last_i, last_j int, par []rune) {
		count := 0
		for i := last_i; i < len(s); i++ {
			if rune(s[i]) == par[0] {
				count++
			}
			if rune(s[i]) == par[1] {
				count--
			}
			if count >= 0 {
				continue
			}
			for j := last_j; j <= i; j++ {
				if rune(s[j]) == par[1] && (j == last_j || s[j-1] != s[j]) {
					dfs(s[:j]+s[j+1:], i, j, par)
				}
			}
			return
		}
		reversed := reverseString(s)
		if par[0] == '(' {
			dfs(reversed, 0, 0, []rune{')', '('})
		} else {
			res = append(res, reversed)
		}
	}
	dfs(s, 0, 0, []rune{'(', ')'})
	return uniqueStrings(res)
}

func reverseString(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

func uniqueStrings(arr []string) []string {
	m := make(map[string]struct{})
	var res []string
	for _, v := range arr {
		if _, ok := m[v]; !ok {
			m[v] = struct{}{}
			res = append(res, v)
		}
	}
	return res
}

// 10. Score of Parentheses
func ScoreOfParentheses(s string) int {
	stack := []int{0}
	for _, ch := range s {
		if ch == '(' {
			stack = append(stack, 0)
		} else {
			v := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			stack[len(stack)-1] += max(2*v, 1)
		}
	}
	return stack[0]
}

// 11. Next Greater Element I
func NextGreaterElement(nums1 []int, nums2 []int) []int {
	res := make([]int, len(nums1))
	m := make(map[int]int)
	stack := []int{}
	for i := len(nums2) - 1; i >= 0; i-- {
		for len(stack) > 0 && stack[len(stack)-1] <= nums2[i] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			m[nums2[i]] = -1
		} else {
			m[nums2[i]] = stack[len(stack)-1]
		}
		stack = append(stack, nums2[i])
	}
	for i, v := range nums1 {
		res[i] = m[v]
	}
	return res
}

// 12. Next Greater Element II (Circular Array)
func NextGreaterElementsCircular(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	for i := range res {
		res[i] = -1
	}
	stack := []int{}
	for i := 0; i < 2*n; i++ {
		num := nums[i%n]
		for len(stack) > 0 && nums[stack[len(stack)-1]] < num {
			res[stack[len(stack)-1]] = num
			stack = stack[:len(stack)-1]
		}
		if i < n {
			stack = append(stack, i)
		}
	}
	return res
}

// 13. Daily Temperatures
func DailyTemperatures(temperatures []int) []int {
	n := len(temperatures)
	res := make([]int, n)
	stack := []int{}
	for i := 0; i < n; i++ {
		for len(stack) > 0 && temperatures[i] > temperatures[stack[len(stack)-1]] {
			idx := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			res[idx] = i - idx
		}
		stack = append(stack, i)
	}
	return res
}

// 14. Largest Rectangle in Histogram
func LargestRectangleInHistogram(heights []int) int {
	stack := []int{}
	maxArea := 0
	heights = append(heights, 0)
	for i, h := range heights {
		for len(stack) > 0 && h < heights[stack[len(stack)-1]] {
			height := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			width := i
			if len(stack) > 0 {
				width = i - stack[len(stack)-1] - 1
			}
			area := height * width
			if area > maxArea {
				maxArea = area
			}
		}
		stack = append(stack, i)
	}
	return maxArea
}

// 15. Maximal Rectangle
func MaximalRectangle(matrix [][]byte) int {
	if len(matrix) == 0 {
		return 0
	}
	n := len(matrix[0])
	heights := make([]int, n)
	maxArea := 0
	for _, row := range matrix {
		for i := 0; i < n; i++ {
			if row[i] == '1' {
				heights[i]++
			} else {
				heights[i] = 0
			}
		}
		area := LargestRectangleInHistogram(heights)
		if area > maxArea {
			maxArea = area
		}
	}
	return maxArea
}

// 16. Min Stack (already implemented as MinStack above)

// 17. Stock Span Problem
func StockSpan(prices []int) []int {
	n := len(prices)
	res := make([]int, n)
	stack := []int{}
	for i := 0; i < n; i++ {
		for len(stack) > 0 && prices[i] >= prices[stack[len(stack)-1]] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			res[i] = i + 1
		} else {
			res[i] = i - stack[len(stack)-1]
		}
		stack = append(stack, i)
	}
	return res
}

// 18. Sliding Window Maximum
func SlidingWindowMaximum(nums []int, k int) []int {
	var res []int
	deque := []int{}
	for i := 0; i < len(nums); i++ {
		if len(deque) > 0 && deque[0] <= i-k {
			deque = deque[1:]
		}
		for len(deque) > 0 && nums[i] >= nums[deque[len(deque)-1]] {
			deque = deque[:len(deque)-1]
		}
		deque = append(deque, i)
		if i >= k-1 {
			res = append(res, nums[deque[0]])
		}
	}
	return res
}

// 19. Decode String
func DecodeString(s string) string {
	countStack := []int{}
	strStack := []string{}
	curStr := ""
	k := 0
	for _, ch := range s {
		if ch >= '0' && ch <= '9' {
			k = k*10 + int(ch-'0')
		} else if ch == '[' {
			countStack = append(countStack, k)
			strStack = append(strStack, curStr)
			curStr = ""
			k = 0
		} else if ch == ']' {
			n := countStack[len(countStack)-1]
			countStack = countStack[:len(countStack)-1]
			prevStr := strStack[len(strStack)-1]
			strStack = strStack[:len(strStack)-1]
			curStr = prevStr + strings.Repeat(curStr, n)
		} else {
			curStr += string(ch)
		}
	}
	return curStr
}

// 20. Basic Calculator I
func BasicCalculator(s string) int {
	stack := []int{}
	num := 0
	sign := 1
	res := 0
	for i, ch := range s {
		if ch >= '0' && ch <= '9' {
			num = num*10 + int(ch-'0')
		}
		if (ch < '0' && ch != ' ') || i == len(s)-1 {
			res += sign * num
			num = 0
			if ch == '+' {
				sign = 1
			} else if ch == '-' {
				sign = -1
			} else if ch == '(' {
				stack = append(stack, res)
				stack = append(stack, sign)
				res = 0
				sign = 1
			} else if ch == ')' {
				sign = stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				res = stack[len(stack)-1] + sign*res
				stack = stack[:len(stack)-1]
			}
		}
	}
	return res
}

// 21. Largest Rectangle in Histogram (already implemented as LargestRectangleInHistogram)

// 22. Maximal Rectangle in Binary Matrix (already implemented as MaximalRectangle)

// 23. Stack with Increment Operation
type CustomStack struct {
	stack []int
	max   int
}

func NewCustomStack(maxSize int) *CustomStack {
	return &CustomStack{stack: []int{}, max: maxSize}
}

func (s *CustomStack) Push(x int) {
	if len(s.stack) < s.max {
		s.stack = append(s.stack, x)
	}
}

func (s *CustomStack) Pop() int {
	if len(s.stack) == 0 {
		return -1
	}
	val := s.stack[len(s.stack)-1]
	s.stack = s.stack[:len(s.stack)-1]
	return val
}

func (s *CustomStack) Increment(k int, val int) {
	n := min(k, len(s.stack))
	for i := 0; i < n; i++ {
		s.stack[i] += val
	}
}

// 24. Simplify Path (Unix Style)
func SimplifyPath(path string) string {
	parts := strings.Split(path, "/")
	stack := []string{}
	for _, part := range parts {
		if part == "" || part == "." {
			continue
		}
		if part == ".." {
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, part)
		}
	}
	return "/" + strings.Join(stack, "/")
}

// 25. Remove K Digits
func RemoveKDigits(num string, k int) string {
	stack := []byte{}
	for i := 0; i < len(num); i++ {
		for k > 0 && len(stack) > 0 && stack[len(stack)-1] > num[i] {
			stack = stack[:len(stack)-1]
			k--
		}
		stack = append(stack, num[i])
	}
	stack = stack[:len(stack)-k]
	res := strings.TrimLeft(string(stack), "0")
	if res == "" {
		return "0"
	}
	return res
}

/*
26. Trapping Rain Water Using Stack
*/
func TrapRainWater(height []int) int {
	stack := []int{}
	water := 0
	for i, h := range height {
		for len(stack) > 0 && h > height[stack[len(stack)-1]] {
			top := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				break
			}
			distance := i - stack[len(stack)-1] - 1
			boundedHeight := min(h, height[stack[len(stack)-1]]) - height[top]
			water += distance * boundedHeight
		}
		stack = append(stack, i)
	}
	return water
}

/*
27. Longest Valid Parentheses (Stack-based)
*/
func LongestValidParenthesesStack(s string) int {
	stack := []int{}
	maxLen := 0
	lastInvalid := -1
	for i, ch := range s {
		if ch == '(' {
			stack = append(stack, i)
		} else {
			if len(stack) == 0 {
				lastInvalid = i
			} else {
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					maxLen = max(maxLen, i-lastInvalid)
				} else {
					maxLen = max(maxLen, i-stack[len(stack)-1])
				}
			}
		}
	}
	return maxLen
}

/*
28. Design Hit Counter Using Stack
*/
type HitCounter struct {
	times []int
}

func NewHitCounter() *HitCounter {
	return &HitCounter{times: []int{}}
}

func (h *HitCounter) Hit(timestamp int) {
	h.times = append(h.times, timestamp)
}

func (h *HitCounter) GetHits(timestamp int) int {
	for len(h.times) > 0 && h.times[0] <= timestamp-300 {
		h.times = h.times[1:]
	}
	return len(h.times)
}

/*
29. Next Smaller Element
*/
func NextSmallerElement(nums []int) []int {
	n := len(nums)
	res := make([]int, n)
	stack := []int{}
	for i := n - 1; i >= 0; i-- {
		for len(stack) > 0 && stack[len(stack)-1] >= nums[i] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			res[i] = -1
		} else {
			res[i] = stack[len(stack)-1]
		}
		stack = append(stack, nums[i])
	}
	return res
}

/*
30. Expression Add Operators
*/
func AddOperators(num string, target int) []string {
	var res []string
	var dfs func(path string, pos int, eval int, multed int)
	dfs = func(path string, pos int, eval int, multed int) {
		if pos == len(num) {
			if eval == target {
				res = append(res, path)
			}
			return
		}
		for i := pos; i < len(num); i++ {
			if i != pos && num[pos] == '0' {
				break
			}
			curStr := num[pos : i+1]
			cur, _ := strconv.Atoi(curStr)
			if pos == 0 {
				dfs(curStr, i+1, cur, cur)
			} else {
				dfs(path+"+"+curStr, i+1, eval+cur, cur)
				dfs(path+"-"+curStr, i+1, eval-cur, -cur)
				dfs(path+"*"+curStr, i+1, eval-multed+multed*cur, multed*cur)
			}
		}
	}
	dfs("", 0, 0, 0)
	return res
}
