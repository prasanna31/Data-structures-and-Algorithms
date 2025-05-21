package queues

import (
	"container/heap"
	"container/list"
	"fmt"
	"math"
	"sort"
)

// 1. Implement Queue using Two Stacks
type QueueWithStacks struct {
	inStack  []int
	outStack []int
}

func (q *QueueWithStacks) Push(x int) {
	q.inStack = append(q.inStack, x)
}

func (q *QueueWithStacks) Pop() int {
	if len(q.outStack) == 0 {
		for len(q.inStack) > 0 {
			n := len(q.inStack) - 1
			q.outStack = append(q.outStack, q.inStack[n])
			q.inStack = q.inStack[:n]
		}
	}
	n := len(q.outStack) - 1
	val := q.outStack[n]
	q.outStack = q.outStack[:n]
	return val
}

func (q *QueueWithStacks) Empty() bool {
	return len(q.inStack) == 0 && len(q.outStack) == 0
}

// 2. Implement Circular Queue
type CircularQueue struct {
	data       []int
	head, tail int
	size, cap  int
}

func NewCircularQueue(k int) *CircularQueue {
	return &CircularQueue{
		data: make([]int, k),
		cap:  k,
		head: -1,
		tail: -1,
	}
}

func (q *CircularQueue) EnQueue(value int) bool {
	if q.IsFull() {
		return false
	}
	if q.IsEmpty() {
		q.head = 0
	}
	q.tail = (q.tail + 1) % q.cap
	q.data[q.tail] = value
	q.size++
	return true
}

func (q *CircularQueue) DeQueue() bool {
	if q.IsEmpty() {
		return false
	}
	if q.head == q.tail {
		q.head, q.tail = -1, -1
	} else {
		q.head = (q.head + 1) % q.cap
	}
	q.size--
	return true
}

func (q *CircularQueue) Front() int {
	if q.IsEmpty() {
		return -1
	}
	return q.data[q.head]
}

func (q *CircularQueue) Rear() int {
	if q.IsEmpty() {
		return -1
	}
	return q.data[q.tail]
}

func (q *CircularQueue) IsEmpty() bool {
	return q.size == 0
}

func (q *CircularQueue) IsFull() bool {
	return q.size == q.cap
}

// 3. Design a Queue with getMax() in O(1)
type MaxQueue struct {
	q    []int
	maxq []int
}

func NewMaxQueue() *MaxQueue {
	return &MaxQueue{}
}

func (mq *MaxQueue) Enqueue(x int) {
	mq.q = append(mq.q, x)
	for len(mq.maxq) > 0 && mq.maxq[len(mq.maxq)-1] < x {
		mq.maxq = mq.maxq[:len(mq.maxq)-1]
	}
	mq.maxq = append(mq.maxq, x)
}

func (mq *MaxQueue) Dequeue() int {
	if len(mq.q) == 0 {
		return -1
	}
	val := mq.q[0]
	mq.q = mq.q[1:]
	if val == mq.maxq[0] {
		mq.maxq = mq.maxq[1:]
	}
	return val
}

func (mq *MaxQueue) GetMax() int {
	if len(mq.maxq) == 0 {
		return -1
	}
	return mq.maxq[0]
}

// 4. Implement Deque (Double Ended Queue)
type Deque struct {
	data *list.List
}

func NewDeque() *Deque {
	return &Deque{data: list.New()}
}

func (d *Deque) PushFront(x int) {
	d.data.PushFront(x)
}

func (d *Deque) PushBack(x int) {
	d.data.PushBack(x)
}

func (d *Deque) PopFront() int {
	if d.data.Len() == 0 {
		return -1
	}
	val := d.data.Front().Value.(int)
	d.data.Remove(d.data.Front())
	return val
}

func (d *Deque) PopBack() int {
	if d.data.Len() == 0 {
		return -1
	}
	val := d.data.Back().Value.(int)
	d.data.Remove(d.data.Back())
	return val
}

// 5. Hit Counter
type HitCounter struct {
	times []int
}

func NewHitCounter() *HitCounter {
	return &HitCounter{}
}

func (hc *HitCounter) Hit(timestamp int) {
	hc.times = append(hc.times, timestamp)
}

func (hc *HitCounter) GetHits(timestamp int) int {
	for len(hc.times) > 0 && hc.times[0] <= timestamp-300 {
		hc.times = hc.times[1:]
	}
	return len(hc.times)
}

// 6. Sliding Window Maximum (using deque)
func SlidingWindowMax(nums []int, k int) []int {
	var res []int
	dq := list.New()
	for i, v := range nums {
		for dq.Len() > 0 && nums[dq.Back().Value.(int)] < v {
			dq.Remove(dq.Back())
		}
		dq.PushBack(i)
		if dq.Front().Value.(int) <= i-k {
			dq.Remove(dq.Front())
		}
		if i >= k-1 {
			res = append(res, nums[dq.Front().Value.(int)])
		}
	}
	return res
}

// 7. Moving Average from Data Stream
type MovingAverage struct {
	window []int
	size   int
	sum    int
}

func NewMovingAverage(size int) *MovingAverage {
	return &MovingAverage{size: size}
}

func (ma *MovingAverage) Next(val int) float64 {
	ma.window = append(ma.window, val)
	ma.sum += val
	if len(ma.window) > ma.size {
		ma.sum -= ma.window[0]
		ma.window = ma.window[1:]
	}
	return float64(ma.sum) / float64(len(ma.window))
}

// 8. Number of Recent Calls
type RecentCounter struct {
	times []int
}

func NewRecentCounter() *RecentCounter {
	return &RecentCounter{}
}

func (rc *RecentCounter) Ping(t int) int {
	rc.times = append(rc.times, t)
	for len(rc.times) > 0 && rc.times[0] < t-3000 {
		rc.times = rc.times[1:]
	}
	return len(rc.times)
}

// 9. Shortest Path in Binary Matrix (BFS)
func ShortestPathBinaryMatrix(grid [][]int) int {
	n := len(grid)
	if grid[0][0] != 0 || grid[n-1][n-1] != 0 {
		return -1
	}
	dirs := [8][2]int{{0, 1}, {1, 0}, {1, 1}, {-1, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 1}}
	type node struct{ x, y, d int }
	q := []node{{0, 0, 1}}
	grid[0][0] = 1
	for len(q) > 0 {
		cur := q[0]
		q = q[1:]
		if cur.x == n-1 && cur.y == n-1 {
			return cur.d
		}
		for _, dir := range dirs {
			nx, ny := cur.x+dir[0], cur.y+dir[1]
			if nx >= 0 && ny >= 0 && nx < n && ny < n && grid[nx][ny] == 0 {
				q = append(q, node{nx, ny, cur.d + 1})
				grid[nx][ny] = 1
			}
		}
	}
	return -1
}

// 10. Walls and Gates
func WallsAndGates(rooms [][]int) {
	if len(rooms) == 0 {
		return
	}
	m, n := len(rooms), len(rooms[0])
	type point struct{ x, y int }
	q := []point{}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if rooms[i][j] == 0 {
				q = append(q, point{i, j})
			}
		}
	}
	dirs := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	for len(q) > 0 {
		p := q[0]
		q = q[1:]
		for _, d := range dirs {
			x, y := p.x+d[0], p.y+d[1]
			if x >= 0 && y >= 0 && x < m && y < n && rooms[x][y] == 2147483647 {
				rooms[x][y] = rooms[p.x][p.y] + 1
				q = append(q, point{x, y})
			}
		}
	}
}

func LevelOrder(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	q := []*TreeNode{root}
	for len(q) > 0 {
		var level []int
		n := len(q)
		for i := 0; i < n; i++ {
			node := q[0]
			q = q[1:]
			level = append(level, node.Val)
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
		res = append(res, level)
	}
	return res
}

// 12. Serialize and Deserialize Binary Tree (using Queue)
func Serialize(root *TreeNode) string {
	if root == nil {
		return ""
	}
	q := []*TreeNode{root}
	res := ""
	for len(q) > 0 {
		node := q[0]
		q = q[1:]
		if node == nil {
			res += "null,"
			continue
		}
		res += fmt.Sprintf("%d,", node.Val)
		q = append(q, node.Left, node.Right)
	}
	return res
}

func Deserialize(data string) *TreeNode {
	if data == "" {
		return nil
	}
	vals := []string{}
	curr := ""
	for _, c := range data {
		if c == ',' {
			vals = append(vals, curr)
			curr = ""
		} else {
			curr += string(c)
		}
	}
	if len(vals) == 0 || vals[0] == "null" {
		return nil
	}
	root := &TreeNode{}
	fmt.Sscanf(vals[0], "%d", &root.Val)
	q := []*TreeNode{root}
	i := 1
	for len(q) > 0 && i < len(vals) {
		node := q[0]
		q = q[1:]
		if vals[i] != "null" {
			node.Left = &TreeNode{}
			fmt.Sscanf(vals[i], "%d", &node.Left.Val)
			q = append(q, node.Left)
		}
		i++
		if i < len(vals) && vals[i] != "null" {
			node.Right = &TreeNode{}
			fmt.Sscanf(vals[i], "%d", &node.Right.Val)
			q = append(q, node.Right)
		}
		i++
	}
	return root
}

// 13. Course Schedule (Topological Sort using Queue)
func CanFinish(numCourses int, prerequisites [][]int) bool {
	graph := make([][]int, numCourses)
	indegree := make([]int, numCourses)
	for _, p := range prerequisites {
		graph[p[1]] = append(graph[p[1]], p[0])
		indegree[p[0]]++
	}
	q := []int{}
	for i := 0; i < numCourses; i++ {
		if indegree[i] == 0 {
			q = append(q, i)
		}
	}
	count := 0
	for len(q) > 0 {
		node := q[0]
		q = q[1:]
		count++
		for _, nei := range graph[node] {
			indegree[nei]--
			if indegree[nei] == 0 {
				q = append(q, nei)
			}
		}
	}
	return count == numCourses
}

// 14. Clone Graph (using BFS)
type GraphNode struct {
	Val       int
	Neighbors []*GraphNode
}

func CloneGraph(node *GraphNode) *GraphNode {
	if node == nil {
		return nil
	}
	m := map[*GraphNode]*GraphNode{}
	q := []*GraphNode{node}
	m[node] = &GraphNode{Val: node.Val}
	for len(q) > 0 {
		n := q[0]
		q = q[1:]
		for _, nei := range n.Neighbors {
			if m[nei] == nil {
				m[nei] = &GraphNode{Val: nei.Val}
				q = append(q, nei)
			}
			m[n].Neighbors = append(m[n].Neighbors, m[nei])
		}
	}
	return m[node]
}

// 15. Design Twitter (Recent Tweets Queue)
type Tweet struct {
	id, time int
}

type Twitter struct {
	time      int
	tweets    map[int][]Tweet
	following map[int]map[int]bool
}

func NewTwitter() *Twitter {
	return &Twitter{
		tweets:    map[int][]Tweet{},
		following: map[int]map[int]bool{},
	}
}

func (tw *Twitter) PostTweet(userId, tweetId int) {
	tw.time++
	tw.tweets[userId] = append(tw.tweets[userId], Tweet{tweetId, tw.time})
}

func (tw *Twitter) GetNewsFeed(userId int) []int {
	type pair struct{ id, time int }
	var feed []pair
	users := []int{userId}
	for f := range tw.following[userId] {
		users = append(users, f)
	}
	for _, u := range users {
		for _, t := range tw.tweets[u] {
			feed = append(feed, pair{t.id, t.time})
		}
	}
	// Sort by time descending
	for i := 0; i < len(feed); i++ {
		for j := i + 1; j < len(feed); j++ {
			if feed[j].time > feed[i].time {
				feed[i], feed[j] = feed[j], feed[i]
			}
		}
	}
	res := []int{}
	for i := 0; i < len(feed) && i < 10; i++ {
		res = append(res, feed[i].id)
	}
	return res
}

func (tw *Twitter) Follow(followerId, followeeId int) {
	if tw.following[followerId] == nil {
		tw.following[followerId] = map[int]bool{}
	}
	tw.following[followerId][followeeId] = true
}

func (tw *Twitter) Unfollow(followerId, followeeId int) {
	if tw.following[followerId] != nil {
		delete(tw.following[followerId], followeeId)
	}
}

// 16. Generate Binary Numbers from 1 to N using Queue
func GenerateBinaryNumbers(n int) []string {
	res := []string{}
	q := []string{"1"}
	for i := 0; i < n; i++ {
		s := q[0]
		q = q[1:]
		res = append(res, s)
		q = append(q, s+"0", s+"1")
	}
	return res
}

// 17. First Non-Repeating Character in a Stream
func FirstNonRepeating(stream string) []rune {
	count := map[rune]int{}
	q := []rune{}
	res := []rune{}
	for _, c := range stream {
		count[c]++
		q = append(q, c)
		for len(q) > 0 && count[q[0]] > 1 {
			q = q[1:]
		}
		if len(q) == 0 {
			res = append(res, '#')
		} else {
			res = append(res, q[0])
		}
	}
	return res
}

// 18. Interleaving the First Half and Second Half of the Queue
func InterleaveQueue(q []int) []int {
	n := len(q) / 2
	first := append([]int{}, q[:n]...)
	second := append([]int{}, q[n:]...)
	res := []int{}
	for i := 0; i < n; i++ {
		res = append(res, first[i], second[i])
	}
	return res
}

// 19. Circular Tour (Gas Station Problem)
func CanCompleteCircuit(gas, cost []int) int {
	total, curr, start := 0, 0, 0
	for i := 0; i < len(gas); i++ {
		total += gas[i] - cost[i]
		curr += gas[i] - cost[i]
		if curr < 0 {
			start = i + 1
			curr = 0
		}
	}
	if total < 0 {
		return -1
	}
	return start
}

// 20. Implement Stack using Two Queues
type StackWithQueues struct {
	q1 []int
	q2 []int
}

func NewStackWithQueues() *StackWithQueues {
	return &StackWithQueues{}
}

func (s *StackWithQueues) Push(x int) {
	s.q2 = append(s.q2, x)
	for len(s.q1) > 0 {
		s.q2 = append(s.q2, s.q1[0])
		s.q1 = s.q1[1:]
	}
	s.q1, s.q2 = s.q2, s.q1
}

type MaxHeap []int

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i] > h[j] }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}
func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// MinHeap for int
type MinHeap []int

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// nodeHeap for ListNode
type ListNode struct {
	Val  int
	Next *ListNode
}

type nodeHeap []*ListNode

func (h nodeHeap) Len() int           { return len(h) }
func (h nodeHeap) Less(i, j int) bool { return h[i].Val < h[j].Val }
func (h nodeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *nodeHeap) Push(x interface{}) {
	*h = append(*h, x.(*ListNode))
}
func (h *nodeHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Explanation: Use two heaps to maintain lower and upper halves.

// 6. Top K Frequent Elements
func TopKFrequent(nums []int, k int) []int {
	count := map[int]int{}
	for _, n := range nums {
		count[n]++
	}
	type pair struct{ val, freq int }
	pairs := []pair{}
	for n, f := range count {
		pairs = append(pairs, pair{n, f})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].freq > pairs[j].freq })
	res := []int{}
	for i := 0; i < k; i++ {
		res = append(res, pairs[i].val)
	}
	return res
}

// Explanation: Count frequencies, sort, and pick top k.

// 7. K Closest Points to Origin
func KClosest(points [][]int, k int) [][]int {
	type point struct {
		x, y, dist int
	}
	h := &MaxHeap{}
	for _, p := range points {
		d := p[0]*p[0] + p[1]*p[1]
		heap.Push(h, d)
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	res := [][]int{}
	for _, p := range points {
		d := p[0]*p[0] + p[1]*p[1]
		if d <= (*h)[0] {
			res = append(res, p)
			if len(res) == k {
				break
			}
		}
	}
	return res
}

// Explanation: Use a max-heap of size k to keep closest points.

// 8. Sort Characters By Frequency
func FrequencySort(s string) string {
	count := map[rune]int{}
	for _, c := range s {
		count[c]++
	}
	type pair struct {
		c rune
		f int
	}
	pairs := []pair{}
	for c, f := range count {
		pairs = append(pairs, pair{c, f})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].f > pairs[j].f })
	res := ""
	for _, p := range pairs {
		for i := 0; i < p.f; i++ {
			res += string(p.c)
		}
	}
	return res
}

// Explanation: Count and sort by frequency.

// 9. Find K Pairs with Smallest Sums
func KSmallestPairs(nums1, nums2 []int, k int) [][]int {
	type tuple struct{ sum, i, j int }
	h := &MinHeap{}
	res := [][]int{}
	if len(nums1) == 0 || len(nums2) == 0 {
		return res
	}
	for i := 0; i < len(nums1) && i < k; i++ {
		heap.Push(h, tuple{nums1[i] + nums2[0], i, 0})
	}
	for h.Len() > 0 && len(res) < k {
		t := heap.Pop(h).(tuple)
		res = append(res, []int{nums1[t.i], nums2[t.j]})
		if t.j+1 < len(nums2) {
			heap.Push(h, tuple{nums1[t.i] + nums2[t.j+1], t.i, t.j + 1})
		}
	}
	return res
}

// Explanation: Use a min-heap to always pick the next smallest sum.

// 10. Reorganize String
func ReorganizeString(s string) string {
	count := map[rune]int{}
	for _, c := range s {
		count[c]++
	}
	type pair struct {
		c rune
		f int
	}
	h := &MaxHeapPair{}
	for c, f := range count {
		heap.Push(h, pair{c, f})
	}
	res := []rune{}
	var prev pair
	prev.f = -1
	for h.Len() > 0 {
		cur := heap.Pop(h).(pair)
		res = append(res, cur.c)
		cur.f--
		if prev.f > 0 {
			heap.Push(h, prev)
		}
		prev = cur
	}
	if len(res) != len(s) {
		return ""
	}
	return string(res)
}

// Helper heap for pair (used in ReorganizeString)
type pair struct {
	c rune
	f int
}
type MaxHeapPair []pair

func (h MaxHeapPair) Len() int           { return len(h) }
func (h MaxHeapPair) Less(i, j int) bool { return h[i].f > h[j].f }
func (h MaxHeapPair) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MaxHeapPair) Push(x interface{}) {
	*h = append(*h, x.(pair))
}
func (h *MaxHeapPair) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
func MedianSlidingWindow(nums []int, k int) []float64 {
	var res []float64
	lo := &MaxHeap{}
	hi := &MinHeap{}
	for i := 0; i < len(nums); i++ {
		if lo.Len() == 0 || nums[i] <= (*lo)[0] {
			heap.Push(lo, nums[i])
			if lo.Len() == 0 || nums[i] <= (*lo)[0] {
				heap.Push(lo, nums[i])
			} else {
				heap.Push(hi, nums[i])
			}
			// Balance

		}
	}
	return res
}

// Explanation: Use two heaps, remove outgoing element manually.

// Helper function to remove an element from a heap
func removeHeap(h heap.Interface, val int) {
	for i := 0; i < h.Len(); i++ {
		switch v := h.(type) {
		case *MaxHeap:
			if (*v)[i] == val {
				(*v)[i] = (*v)[v.Len()-1]
				*v = (*v)[:v.Len()-1]
				heap.Init(v)
				return
			}
		case *MinHeap:
			if (*v)[i] == val {
				(*v)[i] = (*v)[v.Len()-1]
				*v = (*v)[:v.Len()-1]
				heap.Init(v)
				return
			}
		}
	}
}

// Explanation: Calculate idle slots based on max frequency.

// 14. Kth Smallest Number in Sorted Matrix
func KthSmallest(matrix [][]int, k int) int {
	n := len(matrix)
	h := &MinHeap{}
	for i := 0; i < n; i++ {
		heap.Push(h, [3]int{matrix[i][0], i, 0})
	}
	var num int
	for i := 0; i < k; i++ {
		val := heap.Pop(h).([3]int)
		num = val[0]
		r, c := val[1], val[2]
		if c+1 < n {
			heap.Push(h, [3]int{matrix[r][c+1], r, c + 1})
		}
	}
	return num
}

// Explanation: Use a min-heap to always pick the next smallest element.

// 15. Smallest Range Covering Elements from K Lists
func SmallestRange(nums [][]int) []int {
	h := &MinHeap{}
	maxVal := math.MinInt32
	for i, row := range nums {
		heap.Push(h, [3]int{row[0], i, 0})
		if row[0] > maxVal {
			maxVal = row[0]
		}
	}
	res := []int{-1e5, 1e5}
	for h.Len() == len(nums) {
		val := heap.Pop(h).([3]int)
		if maxVal-val[0] < res[1]-res[0] {
			res = []int{val[0], maxVal}
		}
		if val[2]+1 < len(nums[val[1]]) {
			next := nums[val[1]][val[2]+1]
			heap.Push(h, [3]int{next, val[1], val[2] + 1})
			if next > maxVal {
				maxVal = next
			}
		}
	}
	return res
}

// Explanation: Use a min-heap to track the current range, always push next from the same list.

// 16. Dijkstra's Shortest Path
func Dijkstra(graph map[int][][]int, start int) map[int]int {
	dist := map[int]int{}
	for node := range graph {
		dist[node] = math.MaxInt32
	}
	dist[start] = 0
	h := &MinHeap{}
	heap.Push(h, [2]int{0, start})
	for h.Len() > 0 {
		p := heap.Pop(h).([2]int)
		d, u := p[0], p[1]
		if d > dist[u] {
			continue
		}
		for _, nei := range graph[u] {
			v, w := nei[0], nei[1]
			if dist[u]+w < dist[v] {
				dist[v] = dist[u] + w
				heap.Push(h, [2]int{dist[v], v})
			}
		}
	}
	return dist
}

// Explanation: Use a min-heap to always expand the node with the smallest distance.

// 17. Connect Ropes with Minimum Cost
func ConnectRopes(ropes []int) int {
	h := &MinHeap{}
	for _, r := range ropes {
		heap.Push(h, r)
	}
	cost := 0
	for h.Len() > 1 {
		a := heap.Pop(h).(int)
		b := heap.Pop(h).(int)
		cost += a + b
		heap.Push(h, a+b)
	}
	return cost
}

// Explanation: Always connect the two smallest ropes first using a min-heap.

// 18. Schedule Tasks with Cooldown
func LeastIntervalWithCooldown(tasks []byte, n int) int {
	count := [26]int{}
	for _, t := range tasks {
		count[t-'A']++
	}
	h := &MaxHeap{}
	for _, c := range count {
		if c > 0 {
			heap.Push(h, c)
		}
	}
	time := 0
	q := []struct{ cnt, ready int }{}
	for h.Len() > 0 || len(q) > 0 {
		time++
		if h.Len() > 0 {
			c := heap.Pop(h).(int) - 1
			if c > 0 {
				q = append(q, struct{ cnt, ready int }{c, time + n})
			}
		}
		if len(q) > 0 && q[0].ready == time {
			heap.Push(h, q[0].cnt)
			q = q[1:]
		}
	}
	return time
}

// Explanation: Use a max-heap for task counts and a queue for cooldown.

// 19. Median Sliding Window (see 11 above)
// Already implemented as MedianSlidingWindow

// 20. Find Top K Frequent Words
func TopKFrequentWords(words []string, k int) []string {
	count := map[string]int{}
	for _, w := range words {
		count[w]++
	}
	type pair struct {
		word string
		freq int
	}
	pairs := []pair{}
	for w, f := range count {
		pairs = append(pairs, pair{w, f})
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].freq == pairs[j].freq {
			return pairs[i].word < pairs[j].word
		}
		return pairs[i].freq > pairs[j].freq
	})
	res := []string{}
	for i := 0; i < k && i < len(pairs); i++ {
		res = append(res, pairs[i].word)
	}
	return res
}

// Explanation: Count and sort by frequency and lexicographical order.

// 21. Maximum Average Subtree
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func MaximumAverageSubtree(root *TreeNode) float64 {
	maxAvg := 0.0
	var dfs func(*TreeNode) (sum, count int)
	dfs = func(node *TreeNode) (int, int) {
		if node == nil {
			return 0, 0
		}
		ls, lc := dfs(node.Left)
		rs, rc := dfs(node.Right)
		s, c := ls+rs+node.Val, lc+rc+1
		avg := float64(s) / float64(c)
		if avg > maxAvg {
			maxAvg = avg
		}
		return s, c
	}
	dfs(root)
	return maxAvg
}

// Explanation: DFS, compute sum and count for each subtree.

// 22. Kth Smallest Element in a BST
func KthSmallestBST(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	for {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return root.Val
		}
		root = root.Right
	}
}

// Explanation: Inorder traversal, the kth node is the answer.

// 23. Trapping Rain Water II
func TrapRainWater(heightMap [][]int) int {
	m, n := len(heightMap), len(heightMap[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	h := &MinHeap{}
	for i := 0; i < m; i++ {
		heap.Push(h, [3]int{heightMap[i][0], i, 0})
		heap.Push(h, [3]int{heightMap[i][n-1], i, n - 1})
		visited[i][0], visited[i][n-1] = true, true
	}
	for j := 1; j < n-1; j++ {
		heap.Push(h, [3]int{heightMap[0][j], 0, j})
		heap.Push(h, [3]int{heightMap[m-1][j], m - 1, j})
		visited[0][j], visited[m-1][j] = true, true
	}
	res, dirs := 0, [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	for h.Len() > 0 {
		cell := heap.Pop(h).([3]int)
		for _, d := range dirs {
			x, y := cell[1]+d[0], cell[2]+d[1]
			if x >= 0 && y >= 0 && x < m && y < n && !visited[x][y] {
				res += max(0, cell[0]-heightMap[x][y])
				heap.Push(h, [3]int{max(cell[0], heightMap[x][y]), x, y})
				visited[x][y] = true
			}
		}
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Explanation: Use a min-heap to always expand the lowest boundary.

// 24. Minimum Number of Refueling Stops
func MinRefuelStops(target int, startFuel int, stations [][]int) int {
	h := &MaxHeap{}
	res, i, fuel := 0, 0, startFuel
	for fuel < target {
		for i < len(stations) && stations[i][0] <= fuel {
			heap.Push(h, stations[i][1])
			i++
		}
		if h.Len() == 0 {
			return -1
		}
		fuel += heap.Pop(h).(int)
		res++
	}
	return res
}

// Explanation: Use a max-heap to always refuel with the largest available station.

// 25. Sliding Window Maximum (already implemented as SlidingWindowMax above)
/*
26. Frequency Stack
Design a stack-like data structure that supports push, pop, and always pops the most frequent element.
*/
type FreqStack struct {
	freq  map[int]int
	group map[int][]int
	max   int
}

func NewFreqStack() *FreqStack {
	return &FreqStack{
		freq:  map[int]int{},
		group: map[int][]int{},
	}
}

func (fs *FreqStack) Push(x int) {
	fs.freq[x]++
	f := fs.freq[x]
	fs.group[f] = append(fs.group[f], x)
	if f > fs.max {
		fs.max = f
	}
}

func (fs *FreqStack) Pop() int {
	x := fs.group[fs.max][len(fs.group[fs.max])-1]
	fs.group[fs.max] = fs.group[fs.max][:len(fs.group[fs.max])-1]
	fs.freq[x]--
	if len(fs.group[fs.max]) == 0 {
		fs.max--
	}
	return x
}

// Explanation: Track frequency and group elements by frequency.

/*
27. Ugly Numbers II
Find the nth ugly number (only prime factors 2, 3, 5).
*/

// Explanation: Use three pointers for 2, 3, 5 multiples.

/*
28. Kth Largest Element in a Stream
Maintain a stream and return the kth largest element after each insertion.
*/
type KthLargest struct {
	k int
	h *MinHeap
}

func NewKthLargest(k int, nums []int) *KthLargest {
	h := &MinHeap{}
	for _, n := range nums {
		heap.Push(h, n)
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	return &KthLargest{k: k, h: h}
}

func (kl *KthLargest) Add(val int) int {
	heap.Push(kl.h, val)
	if kl.h.Len() > kl.k {
		heap.Pop(kl.h)
	}
	return (*kl.h)[0]
}

// Explanation: Use a min-heap of size k.

/*
29. Sort an Almost Sorted Array
Given an array where every element is at most k away from its sorted position, sort it.
*/
func SortKSortedArray(arr []int, k int) []int {
	h := &MinHeap{}
	res := []int{}
	for i := 0; i < len(arr); i++ {
		heap.Push(h, arr[i])
		if h.Len() > k {
			res = append(res, heap.Pop(h).(int))
		}
	}
	for h.Len() > 0 {
		res = append(res, heap.Pop(h).(int))
	}
	return res
}

// Explanation: Use a min-heap of size k+1.

/*
30. Find Median of Two Sorted Arrays
Find the median of two sorted arrays.
*/
func FindMedianSortedArrays(nums1, nums2 []int) float64 {
	m, n := len(nums1), len(nums2)
	if m > n {
		return FindMedianSortedArrays(nums2, nums1)
	}
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
				maxLeft = max(nums1[i-1], nums2[j-1])
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
				minRight = min(nums1[i], nums2[j])
			}
			return float64(maxLeft+minRight) / 2.0
		}
	}
	return 0
}

// Explanation: Binary search partition.

/*
31. Maximum Number of Events That Can Be Attended
Attend the maximum number of events, each with a start and end day.
*/
func MaxEvents(events [][]int) int {
	sort.Slice(events, func(i, j int) bool { return events[i][0] < events[j][0] })
	h := &MinHeap{}
	res, i, day := 0, 0, 0
	for {
		if h.Len() == 0 && i < len(events) {
			day = events[i][0]
		}
		for i < len(events) && events[i][0] == day {
			heap.Push(h, events[i][1])
			i++
		}
		for h.Len() > 0 && (*h)[0] < day {
			heap.Pop(h)
		}
		if h.Len() == 0 && i == len(events) {
			break
		}
		if h.Len() > 0 {
			heap.Pop(h)
			res++
		}
		day++
	}
	return res
}

// Explanation: Use a min-heap to track event end days.

/*
32. Maximum Product of Two Elements in an Array
Return the maximum product of (nums[i]-1)*(nums[j]-1).
*/
func MaxProduct(nums []int) int {
	max1, max2 := 0, 0
	for _, n := range nums {
		if n > max1 {
			max2 = max1
			max1 = n
		} else if n > max2 {
			max2 = n
		}
	}
	return (max1 - 1) * (max2 - 1)
}

// Explanation: Find two largest numbers.

/*
33. Find the Kth Largest Sum of a Matrix With Sorted Rows
Each row is sorted. Find the kth largest sum by picking one element from each row.
*/
func KthLargestSum(matrix [][]int, k int) int {
	n := len(matrix)
	sum := 0
	idxs := make([]int, n)
	for i := 0; i < n; i++ {
		sum += matrix[i][0]
	}
	type state struct {
		sum  int
		idxs []int
	}
	h := &MaxHeap{}
	visited := map[string]bool{}
	heap.Push(h, state{sum, append([]int{}, idxs...)})
	visited[fmt.Sprint(idxs)] = true
	var res int
	for i := 0; i < k; i++ {
		top := heap.Pop(h).(state)
		res = top.sum
		for r := 0; r < n; r++ {
			if top.idxs[r]+1 < len(matrix[r]) {
				nextIdxs := append([]int{}, top.idxs...)
				nextIdxs[r]++
				key := fmt.Sprint(nextIdxs)
				if !visited[key] {
					nextSum := top.sum - matrix[r][nextIdxs[r]-1] + matrix[r][nextIdxs[r]]
					heap.Push(h, state{nextSum, nextIdxs})
					visited[key] = true
				}
			}
		}
	}
	return res
}

// Explanation: Use a max-heap and BFS on index combinations.

/*
34. Maximum Profit in Job Scheduling
Given jobs with start, end, and profit, find max profit with no overlap.
*/
type Job struct {
	Start, End, Profit int
}

func JobScheduling(startTime, endTime, profit []int) int {
	n := len(startTime)
	jobs := make([]Job, n)
	for i := 0; i < n; i++ {
		jobs[i] = Job{startTime[i], endTime[i], profit[i]}
	}
	sort.Slice(jobs, func(i, j int) bool { return jobs[i].End < jobs[j].End })
	dp := make([]int, n)
	dp[0] = jobs[0].Profit
	for i := 1; i < n; i++ {
		dp[i] = max(dp[i-1], jobs[i].Profit)
		for j := i - 1; j >= 0; j-- {
			if jobs[j].End <= jobs[i].Start {
				dp[i] = max(dp[i], dp[j]+jobs[i].Profit)
				break
			}
		}
	}
	return dp[n-1]
}

// Explanation: DP with binary search for last non-overlapping job.

/*
35. Sort Characters By Frequency
Sort characters in decreasing order by frequency.
*/
func SortCharsByFrequency(s string) string {
	count := map[rune]int{}
	for _, c := range s {
		count[c]++
	}
	type pair struct {
		c rune
		f int
	}
	pairs := []pair{}
	for c, f := range count {
		pairs = append(pairs, pair{c, f})
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].f > pairs[j].f })
	res := ""
	for _, p := range pairs {
		for i := 0; i < p.f; i++ {
			res += string(p.c)
		}
	}
	return res
}

// Explanation: Count and sort by frequency.

/*
36. K Closest Points to Origin
Return k points closest to (0,0).
*/
func KClosestPoints(points [][]int, k int) [][]int {
	h := &MaxHeap{}
	for _, p := range points {
		dist := p[0]*p[0] + p[1]*p[1]
		heap.Push(h, [3]int{dist, p[0], p[1]})
		if h.Len() > k {
			heap.Pop(h)
		}
	}
	res := [][]int{}
	for h.Len() > 0 {
		p := heap.Pop(h).([3]int)
		res = append(res, []int{p[1], p[2]})
	}
	return res
}

// Explanation: Use a max-heap of size k.

/*
37. Reorganize String
Rearrange so no two adjacent chars are the same.
*/
func ReorganizeString2(s string) string {
	count := map[rune]int{}
	for _, c := range s {
		count[c]++
	}
	type pair struct {
		c rune
		f int
	}
	h := &MaxHeap{}
	for c, f := range count {
		heap.Push(h, pair{c, f})
	}
	res := []rune{}
	var prev pair
	prev.f = -1
	for h.Len() > 0 {
		cur := heap.Pop(h).(pair)
		res = append(res, cur.c)
		cur.f--
		if prev.f > 0 {
			heap.Push(h, prev)
		}
		prev = cur
	}
	if len(res) != len(s) {
		return ""
	}
	return string(res)
}

// Explanation: Use a max-heap to always pick the most frequent char, avoid adjacent duplicates.

/*
38. Find Median from Data Stream
Support addNum and findMedian in O(log n).
*/
type MedianFinder struct {
	lo *MaxHeap
	hi *MinHeap
}

func NewMedianFinder() *MedianFinder {
	return &MedianFinder{
		lo: &MaxHeap{},
		hi: &MinHeap{},
	}
}

func (mf *MedianFinder) AddNum(num int) {
	if mf.lo.Len() == 0 || num <= (*mf.lo)[0] {
		heap.Push(mf.lo, num)
	} else {
		heap.Push(mf.hi, num)
	}
	if mf.lo.Len() > mf.hi.Len()+1 {
		heap.Push(mf.hi, heap.Pop(mf.lo))
	} else if mf.hi.Len() > mf.lo.Len() {
		heap.Push(mf.lo, heap.Pop(mf.hi))
	}
}

func (mf *MedianFinder) FindMedian() float64 {
	if mf.lo.Len() > mf.hi.Len() {
		return float64((*mf.lo)[0])
	}
	return (float64((*mf.lo)[0]) + float64((*mf.hi)[0])) / 2
}

// Explanation: Two heaps for lower and upper halves.

/*
39. Sliding Window Median
Return median for each window of size k.
*/
func MedianSlidingWindow2(nums []int, k int) []float64 {
	var res []float64
	lo := &MaxHeap{}
	hi := &MinHeap{}
	for i := 0; i < len(nums); i++ {
		if lo.Len() == 0 || nums[i] <= (*lo)[0] {
			heap.Push(lo, nums[i])
		} else {
			heap.Push(hi, nums[i])
		}
		// Balance
		for lo.Len() > hi.Len()+1 {
			heap.Push(hi, heap.Pop(lo))
		}
		for hi.Len() > lo.Len() {
			heap.Push(lo, heap.Pop(hi))
		}
		if i >= k-1 {
			if lo.Len() > hi.Len() {
				res = append(res, float64((*lo)[0]))
			} else {
				res = append(res, (float64((*lo)[0])+float64((*hi)[0]))/2)
			}
			// Remove nums[i-k+1]
			out := nums[i-k+1]
			if out <= (*lo)[0] {
				removeHeap(lo, out)
			} else {
				removeHeap(hi, out)
			}
		}
	}
	return res
}

// Explanation: Two heaps, remove outgoing element manually.

/*
40. Merge K Sorted Lists
Merge k sorted linked lists into one.
*/
func MergeKSortedLists(lists []*ListNode) *ListNode {
	h := &nodeHeap{}
	heap.Init(h)
	for _, l := range lists {
		if l != nil {
			heap.Push(h, l)
		}
	}
	dummy := &ListNode{}
	cur := dummy
	for h.Len() > 0 {
		node := heap.Pop(h).(*ListNode)
		cur.Next = node
		cur = cur.Next
		if node.Next != nil {
			heap.Push(h, node.Next)
		}
	}
	return dummy.Next
}

// Explanation: Use a min-heap to always pick the smallest node.

/*
41. Kth Largest Element in a Stream (see 28 above)
Already implemented as KthLargest

42. Task Scheduler (see LeastInterval above)
Already implemented as LeastInterval

43. Maximum CPU Load (see MaxCPULoad above)
Already implemented as MaxCPULoad

44. Connect Ropes with Minimum Cost (see ConnectRopes above)
Already implemented as ConnectRopes

45. Maximum Number of Events That Can Be Attended (see MaxEvents above)
Already implemented as MaxEvents

46. Kth Smallest Number in Multiplication Table
Find kth smallest in m x n multiplication table.
*/
func FindKthNumber(m, n, k int) int {
	lo, hi := 1, m*n
	for lo < hi {
		mid := (lo + hi) / 2
		count := 0
		for i := 1; i <= m; i++ {
			count += min(n, mid/i)
		}
		if count < k {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// Explanation: Binary search on value, count numbers <= mid.

/*
47. Minimum Cost to Connect Sticks
Combine sticks with minimal total cost.
*/
func ConnectSticks(sticks []int) int {
	h := &MinHeap{}
	for _, s := range sticks {
		heap.Push(h, s)
	}
	cost := 0
	for h.Len() > 1 {
		a := heap.Pop(h).(int)
		b := heap.Pop(h).(int)
		cost += a + b
		heap.Push(h, a+b)
	}
	return cost
}

// Explanation: Always combine two smallest sticks using min-heap.

/*
48. Maximize Capital
Given k projects, initial capital w, profits and capital arrays, maximize capital after k projects.
*/
func FindMaximizedCapital(k int, w int, profits []int, capital []int) int {
	type pair struct{ cap, pro int }
	n := len(profits)
	projects := make([]pair, n)
	for i := 0; i < n; i++ {
		projects[i] = pair{capital[i], profits[i]}
	}
	sort.Slice(projects, func(i, j int) bool { return projects[i].cap < projects[j].cap })
	h := &MaxHeap{}
	i := 0
	for j := 0; j < k; j++ {
		for i < n && projects[i].cap <= w {
			heap.Push(h, projects[i].pro)
			i++
		}
		if h.Len() == 0 {
			break
		}
		w += heap.Pop(h).(int)
	}
	return w
}

// Explanation: Use a max-heap for profits, add projects with capital <= w.

/*
49. Find Median from Data Stream (see 38 above)
Already implemented as MedianFinder

50. Maximum Average Subtree (see MaximumAverageSubtree above)
Already implemented as MaximumAverageSubtree
*/
