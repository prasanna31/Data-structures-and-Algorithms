from collections import deque
from collections import deque
import heapq
from collections import Counter
import bisect
import itertools


# 1. Implement Stack Using Arrays
class ArrayStack:

    def __init__(self):
        self.stack = []

    def push(self, x):
        # Add element to top of stack
        self.stack.append(x)

    def pop(self):
        # Remove and return top element
        if not self.is_empty():
            return self.stack.pop()
        return None

    def top(self):
        # Return top element without removing
        if not self.is_empty():
            return self.stack[-1]
        return None

    def is_empty(self):
        # Check if stack is empty
        return len(self.stack) == 0


# 2. Implement Stack Using Linked List
class ListNode:

    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedListStack:

    def __init__(self):
        self.head = None

    def push(self, x):
        # Insert at head
        node = ListNode(x)
        node.next = self.head
        self.head = node

    def pop(self):
        # Remove head node
        if self.head:
            val = self.head.val
            self.head = self.head.next
            return val
        return None

    def top(self):
        # Return head value
        if self.head:
            return self.head.val
        return None

    def is_empty(self):
        return self.head is None


# 3. Design a Stack That Supports getMin() in O(1)
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, x):
        self.stack.append(x)
        # Push to min_stack if it's the new min
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            if val == self.min_stack[-1]:
                self.min_stack.pop()
            return val
        return None

    def top(self):
        if self.stack:
            return self.stack[-1]
        return None

    def getMin(self):
        # Return current minimum
        if self.min_stack:
            return self.min_stack[-1]
        return None


# 4. Evaluate Reverse Polish Notation
def evalRPN(tokens):
    # Use stack to evaluate postfix expression
    stack = []
    for token in tokens:
        if token in "+-*/":
            b, a = stack.pop(), stack.pop()
            if token == '+': stack.append(a + b)
            elif token == '-': stack.append(a - b)
            elif token == '*': stack.append(a * b)
            elif token == '/': stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]


# 5. Implement Queue Using Stacks
class MyQueue:

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self.peek()
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack


# 6. Valid Parentheses
def isValid(s):
    # Use stack to match parentheses
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
    return not stack


# 7. Generate Parentheses
def generateParenthesis(n):
    # Backtracking to generate all valid combinations
    res = []

    def backtrack(s, left, right):
        if len(s) == 2 * n:
            res.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)

    backtrack('', 0, 0)
    return res


# 8. Longest Valid Parentheses
def longestValidParentheses(s):
    # Use stack to track indices of valid substrings
    stack = [-1]
    max_len = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_len = max(max_len, i - stack[-1])
    return max_len


# 9. Remove Invalid Parentheses
def removeInvalidParentheses(s):
    # BFS to remove minimum invalid parentheses
    def is_valid(string):
        count = 0
        for c in string:
            if c == '(': count += 1
            if c == ')': count -= 1
            if count < 0: return False
        return count == 0

    visited = set([s])
    queue = deque([s])
    res = []
    found = False
    while queue:
        curr = queue.popleft()
        if is_valid(curr):
            res.append(curr)
            found = True
        if found: continue
        for i in range(len(curr)):
            if curr[i] not in '()': continue
            nxt = curr[:i] + curr[i + 1:]
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return res


# 10. Score of Parentheses
def scoreOfParentheses(s):
    # Stack to compute score based on rules
    stack = [0]
    for c in s:
        if c == '(':
            stack.append(0)
        else:
            v = stack.pop()
            stack[-1] += max(2 * v, 1)
    return stack[0]


# 11. Next Greater Element I
def nextGreaterElement(nums1, nums2):
    # Monotonic stack to find next greater for each element
    stack, mapping = [], {}
    for num in nums2:
        while stack and num > stack[-1]:
            mapping[stack.pop()] = num
        stack.append(num)
    return [mapping.get(x, -1) for x in nums1]


# 12. Next Greater Element II (Circular Array)
def nextGreaterElements(nums):
    # Traverse array twice for circular effect
    n = len(nums)
    res = [-1] * n
    stack = []
    for i in range(2 * n):
        num = nums[i % n]
        while stack and nums[stack[-1]] < num:
            res[stack.pop()] = num
        if i < n:
            stack.append(i)
    return res


# 13. Daily Temperatures
def dailyTemperatures(T):
    # Monotonic stack to find next warmer day
    res = [0] * len(T)
    stack = []
    for i, temp in enumerate(T):
        while stack and T[stack[-1]] < temp:
            idx = stack.pop()
            res[idx] = i - idx
        stack.append(i)
    return res


# 14. Largest Rectangle in Histogram
def largestRectangleArea(heights):
    # Stack to maintain indices of increasing bars
    stack = []
    max_area = 0
    heights.append(0)
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()
    return max_area


# 15. Maximal Rectangle
def maximalRectangle(matrix):
    # Use histogram approach for each row
    if not matrix: return 0
    n = len(matrix[0])
    heights = [0] * n
    max_area = 0
    for row in matrix:
        for i in range(n):
            heights[i] = heights[i] + 1 if row[i] == '1' else 0
        max_area = max(max_area, largestRectangleArea(heights))
    return max_area


# 16. Min Stack (Same as #3)
# See MinStack class above


# 17. Stock Span Problem
def calculateSpan(prices):
    # Stack to store indices of previous higher price
    n = len(prices)
    span = [1] * n
    stack = []
    for i in range(n):
        while stack and prices[i] >= prices[stack[-1]]:
            stack.pop()
        span[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)
    return span


# 18. Sliding Window Maximum
def maxSlidingWindow(nums, k):
    # Monotonic deque to keep max at front
    dq = deque()
    res = []
    for i, num in enumerate(nums):
        while dq and nums[dq[-1]] < num:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


# 19. Decode String
def decodeString(s):
    # Stack to decode nested patterns
    stack = []
    curr_num = 0
    curr_str = ''
    for c in s:
        if c.isdigit():
            curr_num = curr_num * 10 + int(c)
        elif c == '[':
            stack.append((curr_str, curr_num))
            curr_str, curr_num = '', 0
        elif c == ']':
            last_str, num = stack.pop()
            curr_str = last_str + num * curr_str
        else:
            curr_str += c
    return curr_str


# 20. Basic Calculator (I & II)
def calculate(s):
    # Evaluate expression with +, -, *, /
    stack = []
    num = 0
    sign = '+'
    s += '+'
    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c in '+-*/':
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                stack.append(int(stack.pop() / num))
            num = 0
            sign = c
    return sum(stack)

    # 21. Largest Rectangle in Histogram
    # (Already implemented above as largestRectangleArea)
    # Explanation:
    # Use a stack to keep indices of increasing bar heights. When a lower bar is found,
    # pop from the stack and calculate area with the popped bar as the smallest bar.
    # This ensures all rectangles are considered.

    # 22. Maximal Rectangle in Binary Matrix
    # (Already implemented above as maximalRectangle)
    # Explanation:
    # Treat each row as the base of a histogram and use the largestRectangleArea function
    # for each row to find the maximal rectangle of 1's.


    # 23. Implement a Stack with Increment Operation
    # Design a stack that supports push, pop, and increment the bottom k elements by val.
class CustomStack:

    def __init__(self, maxSize):
        self.stack = []
        self.maxSize = maxSize

    def push(self, x):
        if len(self.stack) < self.maxSize:
            self.stack.append(x)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return -1

    def increment(self, k, val):
        for i in range(min(k, len(self.stack))):
            self.stack[i] += val


# 24. Simplify Path (Unix Style)
# Given a Unix-style path, simplify it (e.g., "/a/./b/../../c/" -> "/c")
def simplifyPath(path):
    stack = []
    for part in path.split('/'):
        if part == '' or part == '.':
            continue
        elif part == '..':
            if stack:
                stack.pop()
        else:
            stack.append(part)
    return '/' + '/'.join(stack)


# 25. Remove K Digits
# Remove k digits from the number to make it the smallest possible
def removeKdigits(num, k):
    stack = []
    for digit in num:
        while k and stack and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    # Remove remaining digits from the end if k > 0
    final = stack[:-k] if k else stack
    # Remove leading zeros
    return ''.join(final).lstrip('0') or '0'


# 26. Trapping Rain Water Using Stack
# Given heights, compute how much water is trapped after raining
def trap(height):
    stack = []
    water = 0
    for i, h in enumerate(height):
        while stack and h > height[stack[-1]]:
            top = stack.pop()
            if not stack:
                break
            distance = i - stack[-1] - 1
            bounded_height = min(h, height[stack[-1]]) - height[top]
            water += distance * bounded_height
        stack.append(i)
    return water


# 27. Longest Valid Parentheses
# (Already implemented above as longestValidParentheses)
# Explanation:
# Use a stack to track indices of '(' and unmatched ')'. The difference between indices
# gives the length of valid substrings.


# 28. Design Hit Counter Using Stack
# Count hits in the past 5 minutes (300 seconds)
class HitCounter:

    def __init__(self):
        self.hits = []

    def hit(self, timestamp):
        self.hits.append(timestamp)

    def getHits(self, timestamp):
        # Remove hits older than 300 seconds
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.pop(0)
        return len(self.hits)


# 29. Next Smaller Element
# For each element, find the next smaller element to its right
def nextSmallerElement(nums):
    stack = []
    res = [-1] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] >= nums[i]:
            stack.pop()
        if stack:
            res[i] = stack[-1]
        stack.append(nums[i])
    return res


# 30. Expression Add Operators
# Given num as a string and target, return all expressions by adding +, -, * that evaluate to target
def addOperators(num, target):
    res = []

    def backtrack(index, path, value, prev):
        if index == len(num):
            if value == target:
                res.append(path)
            return
        for i in range(index + 1, len(num) + 1):
            tmp = num[index:i]
            if len(tmp) > 1 and tmp[0] == '0':
                break
            n = int(tmp)
            if index == 0:
                backtrack(i, tmp, n, n)
            else:
                backtrack(i, path + '+' + tmp, value + n, n)
                backtrack(i, path + '-' + tmp, value - n, -n)
                backtrack(i, path + '*' + tmp, value - prev + prev * n,
                          prev * n)

    backtrack(0, '', 0, 0)
    return res


