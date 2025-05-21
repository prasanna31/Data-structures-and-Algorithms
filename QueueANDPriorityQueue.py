from collections import deque
from collections import deque
import heapq
from collections import Counter
import bisect
import itertools

#1. Implement Queue using Stacks
class QueueWithStacks:

    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def enqueue(self, x):
        self.in_stack.append(x)

    def dequeue(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        if self.out_stack:
            return self.out_stack.pop()
        return None

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        if self.out_stack:
            return self.out_stack[-1]
        return None

    def is_empty(self):
        return not self.in_stack and not self.out_stack


# 2. Implement Circular Queue
class MyCircularQueue:

    def __init__(self, k):
        self.q = [0] * k
        self.head = 0
        self.tail = 0
        self.size = 0
        self.capacity = k

    def enQueue(self, value):
        if self.isFull():
            return False
        self.q[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True

    def deQueue(self):
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return True

    def Front(self):
        if self.isEmpty():
            return -1
        return self.q[self.head]

    def Rear(self):
        if self.isEmpty():
            return -1
        return self.q[(self.tail - 1) % self.capacity]

    def isEmpty(self):
        return self.size == 0

    def isFull(self):
        return self.size == self.capacity


# 3. Design a Queue with getMax() in O(1)
class MaxQueue:

    def __init__(self):
        self.q = deque()
        self.max_q = deque()

    def enqueue(self, x):
        self.q.append(x)
        while self.max_q and self.max_q[-1] < x:
            self.max_q.pop()
        self.max_q.append(x)

    def dequeue(self):
        if self.q:
            val = self.q.popleft()
            if val == self.max_q[0]:
                self.max_q.popleft()
            return val
        return None

    def getMax(self):
        if self.max_q:
            return self.max_q[0]
        return None

    def is_empty(self):
        return not self.q


# 4. Implement Deque (Double Ended Queue)
class MyDeque:

    def __init__(self):
        self.dq = deque()

    def push_front(self, x):
        self.dq.appendleft(x)

    def push_back(self, x):
        self.dq.append(x)

    def pop_front(self):
        if self.dq:
            return self.dq.popleft()
        return None

    def pop_back(self):
        if self.dq:
            return self.dq.pop()
        return None

    def front(self):
        if self.dq:
            return self.dq[0]
        return None

    def back(self):
        if self.dq:
            return self.dq[-1]
        return None

    def is_empty(self):
        return not self.dq


# 5. Design Hit Counter using Queue
class HitCounterQueue:

    def __init__(self):
        self.q = deque()

    def hit(self, timestamp):
        self.q.append(timestamp)

    def getHits(self, timestamp):
        while self.q and self.q[0] <= timestamp - 300:
            self.q.popleft()
        return len(self.q)


# 6. Sliding Window Maximum (using deque)
def maxSlidingWindowDeque(nums, k):
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


# 7. Moving Average from Data Stream
class MovingAverage:

    def __init__(self, size):
        self.size = size
        self.q = deque()
        self.total = 0

    def next(self, val):
        self.q.append(val)
        self.total += val
        if len(self.q) > self.size:
            self.total -= self.q.popleft()
        return self.total / len(self.q)


# 8. Number of Recent Calls
class RecentCounter:

    def __init__(self):
        self.q = deque()

    def ping(self, t):
        self.q.append(t)
        while self.q and self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)


# 9. Shortest Path in Binary Matrix (BFS)
def shortestPathBinaryMatrix(grid):
    n = len(grid)
    if grid[0][0] or grid[n - 1][n - 1]:
        return -1
    q = deque([(0, 0, 1)])
    visited = set((0, 0))
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),(1, 1)]
    while q:
        x, y, d = q.popleft()
        if (x, y) == (n - 1, n - 1):
            return d
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and not grid[nx][ny] and (
                    nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny, d + 1))
    return -1


# 10. Walls and Gates
def wallsAndGates(rooms):
    if not rooms:
        return
    m, n = len(rooms), len(rooms[0])
    q = deque()
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0:
                q.append((i, j))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and rooms[nx][ny] == 2147483647:
                rooms[nx][ny] = rooms[x][y] + 1
                q.append((nx, ny))


# 11. Binary Tree Level Order Traversal
def levelOrder(root):
    res = []
    if not root:
        return res
    q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(level)
    return res


# 12. Serialize and Deserialize Binary Tree (using Queue)
class Codec:

    def serialize(self, root):
        if not root:
            return ''
        q = deque([root])
        res = []
        while q:
            node = q.popleft()
            if node:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
            else:
                res.append('null')
        return ','.join(res)

    def deserialize(self, data):
        if not data:
            return None
        vals = data.split(',')
        root = TreeNode(int(vals[0]))
        q = deque([root])
        i = 1
        while q:
            node = q.popleft()
            if vals[i] != 'null':
                node.left = TreeNode(int(vals[i]))
                q.append(node.left)
            i += 1
            if vals[i] != 'null':
                node.right = TreeNode(int(vals[i]))
                q.append(node.right)
            i += 1
        return root


# 13. Course Schedule (Topological Sort using Queue)
def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses
    for a, b in prerequisites:
        graph[b].append(a)
        indegree[a] += 1
    q = deque([i for i in range(numCourses) if indegree[i] == 0])
    count = 0
    while q:
        node = q.popleft()
        count += 1
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                q.append(nei)
    return count == numCourses


# 14. Clone Graph (using BFS)
def cloneGraph(node):
    if not node:
        return None
    mapping = {node: Node(node.val)}
    q = deque([node])
    while q:
        curr = q.popleft()
        for nei in curr.neighbors:
            if nei not in mapping:
                mapping[nei] = Node(nei.val)
                q.append(nei)
            mapping[curr].neighbors.append(mapping[nei])
    return mapping[node]


# 15. Design Twitter (Recent Tweets Queue)
class Twitter:

    def __init__(self):
        self.time = 0
        self.tweets = {}  # userId: list of (time, tweetId)
        self.following = {}  # userId: set of followeeIds

    def postTweet(self, userId, tweetId):
        self.tweets.setdefault(userId, []).append((self.time, tweetId))
        self.time -= 1

    def getNewsFeed(self, userId):
        heap = []
        self.following.setdefault(userId, set()).add(userId)
        for uid in self.following[userId]:
            for t in self.tweets.get(uid, [])[-10:]:
                heapq.heappush(heap, t)
        return [tweetId for _, tweetId in heapq.nsmallest(10, heap)]

    def follow(self, followerId, followeeId):
        self.following.setdefault(followerId, set()).add(followeeId)

    def unfollow(self, followerId, followeeId):
        if followerId != followeeId:
            self.following.setdefault(followerId, set()).discard(followeeId)


# 16. Generate Binary Numbers from 1 to N using Queue
def generateBinaryNumbers(n):
    res = []
    q = deque(['1'])
    for _ in range(n):
        curr = q.popleft()
        res.append(curr)
        q.append(curr + '0')
        q.append(curr + '1')
    return res


# 17. First Non-Repeating Character in a Stream
def firstNonRepeating(stream):
    q = deque()
    count = Counter()
    res = []
    for c in stream:
        count[c] += 1
        q.append(c)
        while q and count[q[0]] > 1:
            q.popleft()
        res.append(q[0] if q else '#')
    return ''.join(res)


# 18. Interleaving the First Half and Second Half of the Queue
def interleaveQueue(q):
    n = len(q)
    half = n // 2
    first = deque()
    for _ in range(half):
        first.append(q.popleft())
    while first:
        q.append(first.popleft())
        q.append(q.popleft())
    return q


# 19. Circular Tour (Gas Station Problem)
def canCompleteCircuit(gas, cost):
    total, curr, start = 0, 0, 0
    for i in range(len(gas)):
        total += gas[i] - cost[i]
        curr += gas[i] - cost[i]
        if curr < 0:
            start = i + 1
            curr = 0
    return start if total >= 0 else -1


# 20. Implement Stack using Two Queues
class StackWithQueues:

    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()

    def push(self, x):
        self.q2.append(x)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1

    def pop(self):
        if self.q1:
            return self.q1.popleft()
        return None

    def top(self):
        if self.q1:
            return self.q1[0]
        return None

    def is_empty(self):
        return not self.q1


        # 1. Implement Min Heap
class MinHeap:

    def __init__(self):
        self.heap = []

    def push(self, x):
        heapq.heappush(self.heap, x)

    def pop(self):
        return heapq.heappop(self.heap) if self.heap else None

    def top(self):
        return self.heap[0] if self.heap else None

    def __len__(self):
        return len(self.heap)


# Explanation: Uses Python's heapq which is a min-heap by default.


# 2. Implement Max Heap
class MaxHeap:

    def __init__(self):
        self.heap = []

    def push(self, x):
        heapq.heappush(self.heap, -x)

    def pop(self):
        return -heapq.heappop(self.heap) if self.heap else None

    def top(self):
        return -self.heap[0] if self.heap else None

    def __len__(self):
        return len(self.heap)


# Explanation: Negate values to simulate max-heap using heapq.


# 3. Kth Largest Element in an Array
def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]


# Explanation: nlargest returns k largest elements, last one is kth largest.


# 4. Merge K Sorted Lists
def mergeKLists(lists):
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))
    dummy = ListNode(0)
    curr = dummy
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next


# Explanation: Use heap to always pick the smallest node among k lists.


# 5. Find Median from Data Stream
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


# Explanation: Two heaps, max-heap for lower half, min-heap for upper half.


# 6. Top K Frequent Elements
def topKFrequent(nums, k):
    count = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, count.items(), key=lambda x: x[1])]


# Explanation: Count frequencies, use heap to get k most frequent.


# 7. K Closest Points to Origin
def kClosest(points, k):
    return heapq.nsmallest(k, points, key=lambda x: x[0]**2 + x[1]**2)


# Explanation: Use nsmallest with distance squared as key.


# 8. Sort Characters By Frequency
def frequencySort(s):
    count = Counter(s)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    res = []
    while heap:
        freq, char = heapq.heappop(heap)
        res.append(char * -freq)
    return ''.join(res)


# Explanation: Max-heap by frequency, pop and build result.


# 9. Find K Pairs with Smallest Sums
def kSmallestPairs(nums1, nums2, k):
    if not nums1 or not nums2:
        return []
    heap, res = [], []
    for i in range(min(k, len(nums1))):
        heapq.heappush(heap, (nums1[i] + nums2[0], i, 0))
    while heap and len(res) < k:
        total, i, j = heapq.heappop(heap)
        res.append([nums1[i], nums2[j]])
        if j + 1 < len(nums2):
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
    return res


# Explanation: Heap stores next possible pair, always expand smallest sum.


# 10. Reorganize String
def reorganizeString(s):
    count = Counter(s)
    heap = [(-freq, char) for char, freq in count.items()]
    heapq.heapify(heap)
    prev = (0, '')
    res = []
    while heap:
        freq, char = heapq.heappop(heap)
        res.append(char)
        if prev[0] < 0:
            heapq.heappush(heap, prev)
        freq += 1
        prev = (freq, char)
    return ''.join(res) if len(res) == len(s) else ''


# Explanation: Always pick most frequent char, avoid consecutive same chars.


# 11. Sliding Window Median
def medianSlidingWindow(nums, k):
    window = sorted(nums[:k])
    medians = []
    for i in range(k, len(nums) + 1):
        if k % 2:
            medians.append(float(window[k // 2]))
        else:
            medians.append((window[k // 2 - 1] + window[k // 2]) / 2)
        if i == len(nums):
            break
        window.pop(bisect.bisect_left(window, nums[i - k]))
        bisect.insort(window, nums[i])
    return medians


# Explanation: Maintain sorted window, use bisect for insertion/removal.


# 12. Maximum CPU Load (Interval Scheduling)
def maxCPULoad(jobs):
    jobs.sort(key=lambda x: x[0])
    heap = []
    max_load = curr_load = 0
    for start, end, load in jobs:
        while heap and heap[0][0] <= start:
            curr_load -= heapq.heappop(heap)[1]
        heapq.heappush(heap, (end, load))
        curr_load += load
        max_load = max(max_load, curr_load)
    return max_load


# Explanation: Min-heap by end time, track overlapping loads.


# 13. Task Scheduler
def leastInterval(tasks, n):
    count = Counter(tasks)
    max_freq = max(count.values())
    max_count = sum(1 for v in count.values() if v == max_freq)
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)


# Explanation: Greedy, fill most frequent tasks first, then idle slots.


# 14. Kth Smallest Number in Sorted Matrix
def kthSmallest(matrix, k):
    n = len(matrix)
    heap = [(matrix[i][0], i, 0) for i in range(n)]
    heapq.heapify(heap)
    for _ in range(k - 1):
        val, r, c = heapq.heappop(heap)
        if c + 1 < n:
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
    return heap[0][0]


# Explanation: Heap of next smallest elements from each row.


# 15. Smallest Range Covering Elements from K Lists
def smallestRange(nums):
    heap = []
    max_val = float('-inf')
    for i, row in enumerate(nums):
        heapq.heappush(heap, (row[0], i, 0))
        max_val = max(max_val, row[0])
    res = [float('-inf'), float('inf')]
    while True:
        min_val, i, j = heapq.heappop(heap)
        if max_val - min_val < res[1] - res[0]:
            res = [min_val, max_val]
        if j + 1 == len(nums[i]):
            break
        nxt = nums[i][j + 1]
        heapq.heappush(heap, (nxt, i, j + 1))
        max_val = max(max_val, nxt)
    return res


# Explanation: Heap tracks current min, update range, advance in list with min.
# 16. Dijkstra's Shortest Path
def dijkstra(graph, start):
    # graph: {u: [(v, weight), ...]}
    heap = [(0, start)]
    dist = {start: 0}
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.get(u, []):
            if v not in dist or d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(heap, (dist[v], v))
    return dist


# Explanation: Use min-heap to always expand the node with the smallest distance.


# 17. Connect Ropes with Minimum Cost
def connectRopes(ropes):
    heapq.heapify(ropes)
    cost = 0
    while len(ropes) > 1:
        a = heapq.heappop(ropes)
        b = heapq.heappop(ropes)
        cost += a + b
        heapq.heappush(ropes, a + b)
    return cost


# Explanation: Always connect two smallest ropes first using min-heap.


# 18. Schedule Tasks with Cooldown
def leastIntervalWithCooldown(tasks, n):
    count = Counter(tasks)
    heap = [(-v, k) for k, v in count.items()]
    heapq.heapify(heap)
    time = 0
    q = deque()
    while heap or q:
        time += 1
        if heap:
            freq, task = heapq.heappop(heap)
            if freq + 1 < 0:
                q.append((freq + 1, task, time + n))
        if q and q[0][2] == time:
            heapq.heappush(heap, (q[0][0], q[0][1]))
            q.popleft()
    return time


# Explanation: Use heap for most frequent task, queue for cooldown.


# 19. Median Sliding Window (Heap version)
def medianSlidingWindowHeap(nums, k):
    window = sorted(nums[:k])
    medians = []
    for i in range(k, len(nums) + 1):
        if k % 2:
            medians.append(float(window[k // 2]))
        else:
            medians.append((window[k // 2 - 1] + window[k // 2]) / 2)
        if i == len(nums):
            break
        window.pop(bisect.bisect_left(window, nums[i - k]))
        bisect.insort(window, nums[i])
    return medians


# Explanation: Maintain sorted window for median, use bisect for efficiency.


# 20. Find Top K Frequent Words
def topKFrequentWords(words, k):
    count = Counter(words)
    heap = [(-freq, word) for word, freq in count.items()]
    heapq.heapify(heap)
    res = []
    for _ in range(k):
        res.append(heapq.heappop(heap)[1])
    return res


# Explanation: Use heap with (-freq, word) to get k most frequent words.


# 21. Maximum Average Subtree
def maximumAverageSubtree(root):
    res = [0]

    def dfs(node):
        if not node:
            return (0, 0)
        left_sum, left_count = dfs(node.left)
        right_sum, right_count = dfs(node.right)
        total = left_sum + right_sum + node.val
        count = left_count + right_count + 1
        res[0] = max(res[0], total / count)
        return (total, count)

    dfs(root)
    return res[0]


# Explanation: DFS to compute sum and count for each subtree, track max average.


# 22. Kth Smallest Element in a BST
def kthSmallest(root, k):
    stack = []
    while True:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            return root.val
        root = root.right


# Explanation: Inorder traversal yields sorted order, kth element is answer.


# 23. Trapping Rain Water II
def trapRainWater(heightMap):
    if not heightMap or not heightMap[0]:
        return 0
    m, n = len(heightMap), len(heightMap[0])
    visited = [[False] * n for _ in range(m)]
    heap = []
    for i in range(m):
        for j in [0, n - 1]:
            heapq.heappush(heap, (heightMap[i][j], i, j))
            visited[i][j] = True
    for j in range(n):
        for i in [0, m - 1]:
            if not visited[i][j]:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = True
    res = 0
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while heap:
        h, x, y = heapq.heappop(heap)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny]:
                res += max(0, h - heightMap[nx][ny])
                heapq.heappush(heap, (max(h, heightMap[nx][ny]), nx, ny))
                visited[nx][ny] = True
    return res


# Explanation: Use min-heap for boundary, expand inward, trap water by elevation.


# 24. Minimum Number of Refueling Stops
def minRefuelStops(target, startFuel, stations):
    heap = []
    res = i = 0
    while startFuel < target:
        while i < len(stations) and stations[i][0] <= startFuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1
        if not heap:
            return -1
        startFuel += -heapq.heappop(heap)
        res += 1
    return res


# Explanation: Always refuel at the station with the most fuel within reach.


# 25. Sliding Window Maximum (Heap version)
def maxSlidingWindowHeap(nums, k):
    heap = [(-nums[i], i) for i in range(k)]
    heapq.heapify(heap)
    res = [-heap[0][0]]
    for i in range(k, len(nums)):
        heapq.heappush(heap, (-nums[i], i))
        while heap[0][1] <= i - k:
            heapq.heappop(heap)
        res.append(-heap[0][0])
    return res


# Explanation: Max-heap with indices, pop out-of-window elements.


# 26. Frequency Stack
class FreqStack:

    def __init__(self):
        self.freq = Counter()
        self.group = {}
        self.maxfreq = 0

    def push(self, x):
        f = self.freq[x] + 1
        self.freq[x] = f
        if f > self.maxfreq:
            self.maxfreq = f
        self.group.setdefault(f, []).append(x)

    def pop(self):
        x = self.group[self.maxfreq].pop()
        self.freq[x] -= 1
        if not self.group[self.maxfreq]:
            self.maxfreq -= 1
        return x


# Explanation: Track frequency and group by freq, pop most frequent/recent.


# 27. Ugly Numbers II
def nthUglyNumber(n):
    ugly = [1]
    i2 = i3 = i5 = 0
    for _ in range(1, n):
        next2, next3, next5 = ugly[i2] * 2, ugly[i3] * 3, ugly[i5] * 5
        nxt = min(next2, next3, next5)
        ugly.append(nxt)
        if nxt == next2: i2 += 1
        if nxt == next3: i3 += 1
        if nxt == next5: i5 += 1
    return ugly[-1]


# Explanation: Generate ugly numbers by multiplying previous by 2,3,5.


# 28. Kth Largest Element in a Stream
class KthLargest:

    def __init__(self, k, nums):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


# Explanation: Maintain min-heap of size k, top is kth largest.


# 29. Sort an Almost Sorted Array
def sortAlmostSortedArray(arr, k):
    heap = arr[:k + 1]
    heapq.heapify(heap)
    res = []
    for i in range(k + 1, len(arr)):
        res.append(heapq.heappop(heap))
        heapq.heappush(heap, arr[i])
    while heap:
        res.append(heapq.heappop(heap))
    return res


# Explanation: Min-heap of size k+1, always pop smallest.


# 30. Find Median of Two Sorted Arrays
def findMedianSortedArrays(nums1, nums2):
    A, B = nums1, nums2
    m, n = len(A), len(B)
    if m > n:
        A, B, m, n = B, A, n, m
    imin, imax, half = 0, m, (m + n + 1) // 2
    while imin <= imax:
        i = (imin + imax) // 2
        j = half - i
        if i < m and B[j - 1] > A[i]:
            imin = i + 1
        elif i > 0 and A[i - 1] > B[j]:
            imax = i - 1
        else:
            if i == 0: max_of_left = B[j - 1]
            elif j == 0: max_of_left = A[i - 1]
            else: max_of_left = max(A[i - 1], B[j - 1])
            if (m + n) % 2:
                return max_of_left
            if i == m: min_of_right = B[j]
            elif j == n: min_of_right = A[i]
            else: min_of_right = min(A[i], B[j])
            return (max_of_left + min_of_right) / 2


# Explanation: Binary search for partition, O(log(min(m, n))) time.


# 31. Maximum Number of Events That Can Be Attended
def maxEvents(events):
    events.sort()
    heap = []
    res = i = 0
    day = 0
    while heap or i < len(events):
        if not heap:
            day = events[i][0]
        while i < len(events) and events[i][0] == day:
            heapq.heappush(heap, events[i][1])
            i += 1
        heapq.heappop(heap)
        res += 1
        day += 1
        while heap and heap[0] < day:
            heapq.heappop(heap)
    return res


# Explanation: Attend earliest ending event each day using min-heap.


# 32. Maximum Product of Two Elements in an Array
def maxProduct(nums):
    a, b = heapq.nlargest(2, nums)
    return (a - 1) * (b - 1)


# Explanation: Product of two largest minus one.


# 33. Find the Kth Largest Sum of a Matrix With Sorted Rows
def kthLargestSum(matrix, k):
    rows = [row[:] for row in matrix]
    sums = [sum(row[i] for row in rows) for i in range(len(rows[0]))]
    heap = [(-sum(row), tuple(row)) for row in itertools.product(*matrix)]
    heapq.heapify(heap)
    seen = set()
    for _ in range(k - 1):
        total, comb = heapq.heappop(heap)
        # Not efficient for large input, but demonstrates the idea.
    return -heap[0][0]


# Explanation: Use max-heap to track largest sums, expand next possible.


# 34. Maximum Profit in Job Scheduling
def jobScheduling(startTime, endTime, profit):
    jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
    dp = [(0, 0)]
    for s, e, p in jobs:
        i = bisect.bisect_right(dp, (s, float('inf'))) - 1
        if dp[i][1] + p > dp[-1][1]:
            dp.append((e, dp[i][1] + p))
    return dp[-1][1]


# Explanation: DP with binary search for last non-overlapping job.

# 35. Sort Characters By Frequency (already implemented as frequencySort above)

# 36. K Closest Points to Origin (already implemented as kClosest above)

# 37. Reorganize String (already implemented as reorganizeString above)

# 38. Find Median from Data Stream (already implemented as MedianFinder above)

# 39. Sliding Window Median (already implemented as medianSlidingWindow above)

# 40. Merge K Sorted Lists (already implemented as mergeKLists above)

# 41. Kth Largest Element in a Stream (already implemented as KthLargest above)

# 42. Task Scheduler (already implemented as leastInterval above)

# 43. Maximum CPU Load (already implemented as maxCPULoad above)

# 44. Connect Ropes with Minimum Cost (already implemented as connectRopes above)

# 45. Maximum Number of Events That Can Be Attended (already implemented as maxEvents above)


# 46. Kth Smallest Number in Multiplication Table
def findKthNumber(m, n, k):

    def enough(x):
        return sum(min(x // i, n) for i in range(1, m + 1)) >= k

    lo, hi = 1, m * n
    while lo < hi:
        mid = (lo + hi) // 2
        if enough(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# Explanation: Binary search for value, count numbers <= mid.


# 47. Minimum Cost to Connect Sticks
def connectSticks(sticks):
    heapq.heapify(sticks)
    cost = 0
    while len(sticks) > 1:
        a = heapq.heappop(sticks)
        b = heapq.heappop(sticks)
        cost += a + b
        heapq.heappush(sticks, a + b)
    return cost


# Explanation: Same as connectRopes, always combine two smallest.


# 48. Maximize Capital
def findMaximizedCapital(k, W, Profits, Capital):
    projects = sorted(zip(Capital, Profits))
    heap = []
    i = 0
    for _ in range(k):
        while i < len(projects) and projects[i][0] <= W:
            heapq.heappush(heap, -projects[i][1])
            i += 1
        if not heap:
            break
        W -= heapq.heappop(heap)
    return W


# Explanation: Always pick most profitable project affordable at current capital.

# 49. Find Median from Data Stream (already implemented as MedianFinder above)

# 50. Maximum Average Subtree (already implemented as maximumAverageSubtree above)
