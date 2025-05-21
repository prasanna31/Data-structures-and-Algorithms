package graphs

import (
	"container/heap"
	"fmt"
	"math"
	"sort"
	"strings"
)

// 1. Graph Representation

// Adjacency List
type GraphList struct {
	adj map[int][]int
}

func NewGraphList() *GraphList {
	return &GraphList{adj: make(map[int][]int)}
}

func (g *GraphList) AddEdge(u, v int) {
	g.adj[u] = append(g.adj[u], v)
}

// Adjacency Matrix
type GraphMatrix struct {
	mat [][]int
}

func NewGraphMatrix(n int) *GraphMatrix {
	mat := make([][]int, n)
	for i := range mat {
		mat[i] = make([]int, n)
	}
	return &GraphMatrix{mat: mat}
}

func (g *GraphMatrix) AddEdge(u, v int) {
	g.mat[u][v] = 1
}

// 2. DFS Traversal
func DFS(graph map[int][]int, start int, visited map[int]bool, res *[]int) {
	visited[start] = true
	*res = append(*res, start)
	for _, v := range graph[start] {
		if !visited[v] {
			DFS(graph, v, visited, res)
		}
	}
}

// 3. BFS Traversal
func BFS(graph map[int][]int, start int) []int {
	visited := make(map[int]bool)
	queue := []int{start}
	res := []int{}
	visited[start] = true
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			if !visited[v] {
				visited[v] = true
				queue = append(queue, v)
			}
		}
	}
	return res
}

// 4. Detect Cycle in Undirected Graph
func hasCycleUndirected(graph map[int][]int, n int) bool {
	visited := make([]bool, n)
	var dfs func(u, parent int) bool
	dfs = func(u, parent int) bool {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				if dfs(v, u) {
					return true
				}
			} else if v != parent {
				return true
			}
		}
		return false
	}
	for i := 0; i < n; i++ {
		if !visited[i] && dfs(i, -1) {
			return true
		}
	}
	return false
}

// 5. Detect Cycle in Directed Graph
func hasCycleDirected(graph map[int][]int, n int) bool {
	visited := make([]int, n) // 0=unvisited, 1=visiting, 2=visited
	var dfs func(u int) bool
	dfs = func(u int) bool {
		visited[u] = 1
		for _, v := range graph[u] {
			if visited[v] == 1 {
				return true
			}
			if visited[v] == 0 && dfs(v) {
				return true
			}
		}
		visited[u] = 2
		return false
	}
	for i := 0; i < n; i++ {
		if visited[i] == 0 && dfs(i) {
			return true
		}
	}
	return false
}

// 6. Topological Sort (Kahn’s Algorithm)
func TopoSortKahn(graph map[int][]int, n int) []int {
	indegree := make([]int, n)
	for _, vs := range graph {
		for _, v := range vs {
			indegree[v]++
		}
	}
	queue := []int{}
	for i := 0; i < n; i++ {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	res := []int{}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			indegree[v]--
			if indegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	if len(res) != n {
		return nil // cycle exists
	}
	return res
}

// 7. Topological Sort (DFS based)
func TopoSortDFS(graph map[int][]int, n int) []int {
	visited := make([]bool, n)
	res := []int{}
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v)
			}
		}
		res = append(res, u)
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i)
		}
	}
	for i, j := 0, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
	return res
}

// 8. Number of Islands (Grid as Graph)
func NumIslands(grid [][]byte) int {
	if len(grid) == 0 {
		return 0
	}
	m, n := len(grid), len(grid[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1' {
			return
		}
		grid[i][j] = '0'
		dfs(i+1, j)
		dfs(i-1, j)
		dfs(i, j+1)
		dfs(i, j-1)
	}
	count := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == '1' {
				count++
				dfs(i, j)
			}
		}
	}
	return count
}

// 9. Clone Graph
type Node struct {
	Val       int
	Neighbors []*Node
}

func CloneGraph(node *Node) *Node {
	if node == nil {
		return nil
	}
	visited := make(map[*Node]*Node)
	var clone func(*Node) *Node
	clone = func(n *Node) *Node {
		if n == nil {
			return nil
		}
		if v, ok := visited[n]; ok {
			return v
		}
		copy := &Node{Val: n.Val}
		visited[n] = copy
		for _, nei := range n.Neighbors {
			copy.Neighbors = append(copy.Neighbors, clone(nei))
		}
		return copy
	}
	return clone(node)
}

// 10. Bipartite Graph Check
func IsBipartite(graph [][]int) bool {
	n := len(graph)
	color := make([]int, n)
	for i := range color {
		color[i] = -1
	}
	var dfs func(u, c int) bool
	dfs = func(u, c int) bool {
		color[u] = c
		for _, v := range graph[u] {
			if color[v] == -1 {
				if !dfs(v, 1-c) {
					return false
				}
			} else if color[v] == c {
				return false
			}
		}
		return true
	}
	for i := 0; i < n; i++ {
		if color[i] == -1 && !dfs(i, 0) {
			return false
		}
	}
	return true
}

// 11. Count Connected Components in Undirected Graph
func CountComponents(n int, edges [][]int) int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	visited := make([]bool, n)
	count := 0
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v)
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			count++
			dfs(i)
		}
	}
	return count
}

// 12. Find Mother Vertex in Graph
func FindMotherVertex(graph map[int][]int, n int) int {
	visited := make([]bool, n)
	var lastV int
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v)
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i)
			lastV = i
		}
	}
	for i := range visited {
		visited[i] = false
	}
	dfs(lastV)
	for _, v := range visited {
		if !v {
			return -1
		}
	}
	return lastV
}

// 13. Detect Cycle in a Directed Graph using DFS
func DetectCycleDirectedDFS(graph map[int][]int, n int) bool {
	visited := make([]int, n)
	var dfs func(u int) bool
	dfs = func(u int) bool {
		visited[u] = 1
		for _, v := range graph[u] {
			if visited[v] == 1 {
				return true
			}
			if visited[v] == 0 && dfs(v) {
				return true
			}
		}
		visited[u] = 2
		return false
	}
	for i := 0; i < n; i++ {
		if visited[i] == 0 && dfs(i) {
			return true
		}
	}
	return false
}

// 14. Find Strongly Connected Components (Kosaraju’s Algorithm)
func KosarajuSCC(graph map[int][]int, n int) [][]int {
	stack := []int{}
	visited := make([]bool, n)
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v)
			}
		}
		stack = append(stack, u)
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i)
		}
	}
	rev := make(map[int][]int)
	for u, vs := range graph {
		for _, v := range vs {
			rev[v] = append(rev[v], u)
		}
	}
	for i := range visited {
		visited[i] = false
	}
	var scc [][]int
	var dfs2 func(u int, comp *[]int)
	dfs2 = func(u int, comp *[]int) {
		visited[u] = true
		*comp = append(*comp, u)
		for _, v := range rev[u] {
			if !visited[v] {
				dfs2(v, comp)
			}
		}
	}
	for i := len(stack) - 1; i >= 0; i-- {
		u := stack[i]
		if !visited[u] {
			comp := []int{}
			dfs2(u, &comp)
			scc = append(scc, comp)
		}
	}
	return scc
}

// 15. Find Strongly Connected Components (Tarjan’s Algorithm)
func TarjanSCC(graph map[int][]int, n int) [][]int {
	index := 0
	indices := make([]int, n)
	lowlink := make([]int, n)
	onStack := make([]bool, n)
	stack := []int{}
	for i := range indices {
		indices[i] = -1
	}
	var sccs [][]int
	var strongconnect func(v int)
	strongconnect = func(v int) {
		indices[v] = index
		lowlink[v] = index
		index++
		stack = append(stack, v)
		onStack[v] = true
		for _, w := range graph[v] {
			if indices[w] == -1 {
				strongconnect(w)
				if lowlink[w] < lowlink[v] {
					lowlink[v] = lowlink[w]
				}
			} else if onStack[w] {
				if indices[w] < lowlink[v] {
					lowlink[v] = indices[w]
				}
			}
		}
		if lowlink[v] == indices[v] {
			var scc []int
			for {
				w := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[w] = false
				scc = append(scc, w)
				if w == v {
					break
				}
			}
			sccs = append(sccs, scc)
		}
	}
	for v := 0; v < n; v++ {
		if indices[v] == -1 {
			strongconnect(v)
		}
	}
	return sccs
}

// 16. Graph Valid Tree
func ValidTree(n int, edges [][]int) bool {
	if len(edges) != n-1 {
		return false
	}
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	visited := make([]bool, n)
	var dfs func(u, parent int) bool
	dfs = func(u, parent int) bool {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				if !dfs(v, u) {
					return false
				}
			} else if v != parent {
				return false
			}
		}
		return true
	}
	if !dfs(0, -1) {
		return false
	}
	for _, v := range visited {
		if !v {
			return false
		}
	}
	return true
}

// 17. Minimum Number of Vertices to Reach All Nodes
func FindSmallestSetOfVertices(n int, edges [][]int) []int {
	indegree := make([]int, n)
	for _, e := range edges {
		indegree[e[1]]++
	}
	res := []int{}
	for i := 0; i < n; i++ {
		if indegree[i] == 0 {
			res = append(res, i)
		}
	}
	return res
}

// 18. Find Bridges in a Graph
func FindBridges(n int, edges [][]int) [][]int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	var bridges [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					bridges = append(bridges, []int{u, v})
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	return bridges
}

// 19. Find Articulation Points (Cut Vertices)
func FindArticulationPoints(n int, edges [][]int) []int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	ap := make([]bool, n)
	var dfs func(u, parent int, outEdge int)
	dfs = func(u, parent int, outEdge int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		children := 0
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				children++
				dfs(v, u, outEdge+1)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if parent != -1 && low[v] >= ids[u] {
					ap[u] = true
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
		if parent == -1 && children > 1 {
			ap[u] = true
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1, 0)
		}
	}
	var res []int
	for i, v := range ap {
		if v {
			res = append(res, i)
		}
	}
	return res
}

// 20. Find the Number of Provinces
func FindCircleNum(isConnected [][]int) int {
	n := len(isConnected)
	visited := make([]bool, n)
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for v := 0; v < n; v++ {
			if isConnected[u][v] == 1 && !visited[v] {
				dfs(v)
			}
		}
	}
	count := 0
	for i := 0; i < n; i++ {
		if !visited[i] {
			count++
			dfs(i)
		}
	}
	return count
}

// 21. Word Ladder (Shortest Path in Word Graph)
func LadderLength(beginWord string, endWord string, wordList []string) int {
	wordSet := make(map[string]bool)
	for _, w := range wordList {
		wordSet[w] = true
	}
	if !wordSet[endWord] {
		return 0
	}
	queue := []string{beginWord}
	visited := make(map[string]bool)
	visited[beginWord] = true
	level := 1
	for len(queue) > 0 {
		size := len(queue)
		for i := 0; i < size; i++ {
			word := queue[0]
			queue = queue[1:]
			if word == endWord {
				return level
			}
			for j := 0; j < len(word); j++ {
				for c := 'a'; c <= 'z'; c++ {
					next := word[:j] + string(c) + word[j+1:]
					if wordSet[next] && !visited[next] {
						visited[next] = true
						queue = append(queue, next)
					}
				}
			}
		}
		level++
	}
	return 0
}

// 22. Course Schedule I
func CanFinish(numCourses int, prerequisites [][]int) bool {
	graph := make(map[int][]int)
	for _, p := range prerequisites {
		graph[p[1]] = append(graph[p[1]], p[0])
	}
	visited := make([]int, numCourses)
	var dfs func(u int) bool
	dfs = func(u int) bool {
		visited[u] = 1
		for _, v := range graph[u] {
			if visited[v] == 1 {
				return false
			}
			if visited[v] == 0 && !dfs(v) {
				return false
			}
		}
		visited[u] = 2
		return true
	}
	for i := 0; i < numCourses; i++ {
		if visited[i] == 0 && !dfs(i) {
			return false
		}
	}
	return true
}

// 23. Course Schedule II
func FindOrder(numCourses int, prerequisites [][]int) []int {
	graph := make(map[int][]int)
	indegree := make([]int, numCourses)
	for _, p := range prerequisites {
		graph[p[1]] = append(graph[p[1]], p[0])
		indegree[p[0]]++
	}
	queue := []int{}
	for i := 0; i < numCourses; i++ {
		if indegree[i] == 0 {
			queue = append(queue, i)
		}
	}
	res := []int{}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			indegree[v]--
			if indegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	if len(res) != numCourses {
		return []int{}
	}
	return res
}

// 24. Number of Islands II (Dynamic Islands)
func NumIslands2(m int, n int, positions [][]int) []int {
	parent := make([]int, m*n)
	for i := range parent {
		parent[i] = -1
	}

	count := 0
	res := []int{}
	dirs := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	for _, pos := range positions {
		r, c := pos[0], pos[1]
		idx := r*n + c
		if parent[idx] != -1 {
			res = append(res, count)
			continue
		}
		parent[idx] = idx
		count++
		for _, d := range dirs {
			nr, nc := r+d[0], c+d[1]
			nidx := nr*n + nc
			if nr >= 0 && nr < m && nc >= 0 && nc < n && parent[nidx] != -1 {
				root1, root2 := find(idx), find(nidx)
				if root1 != root2 {
					parent[root1] = root2
					count--
				}
			}
		}
		res = append(res, count)
	}
	return res
}

// 25. Alien Dictionary (Topological Sort Application)
func AlienOrder(words []string) string {
	graph := make(map[byte][]byte)
	indegree := make(map[byte]int)
	for _, w := range words {
		for i := range w {
			indegree[w[i]] = 0
		}
	}
	for i := 0; i < len(words)-1; i++ {
		w1, w2 := words[i], words[i+1]
		minLen := len(w1)
		if len(w2) < minLen {
			minLen = len(w2)
		}
		for j := 0; j < minLen; j++ {
			if w1[j] != w2[j] {
				graph[w1[j]] = append(graph[w1[j]], w2[j])
				indegree[w2[j]]++
				break
			}
		}
	}
	queue := []byte{}
	for k, v := range indegree {
		if v == 0 {
			queue = append(queue, k)
		}
	}
	res := []byte{}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			indegree[v]--
			if indegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	if len(res) != len(indegree) {
		return ""
	}
	return string(res)
}

// 26. Reconstruct Itinerary
func FindItinerary(tickets [][]string) []string {
	graph := make(map[string][]string)
	for _, t := range tickets {
		graph[t[0]] = append(graph[t[0]], t[1])
	}
	for k := range graph {
		sort.Strings(graph[k])
	}
	res := []string{}
	var dfs func(u string)
	dfs = func(u string) {
		for len(graph[u]) > 0 {
			v := graph[u][0]
			graph[u] = graph[u][1:]
			dfs(v)
		}
		res = append(res, u)
	}
	dfs("JFK")
	for i, j := 0, len(res)-1; i < j; i, j = i+1, j-1 {
		res[i], res[j] = res[j], res[i]
	}
	return res
}

// 27. Network Delay Time
func NetworkDelayTime(times [][]int, n int, k int) int {
	graph := make(map[int][][2]int)
	for _, t := range times {
		graph[t[0]] = append(graph[t[0]], [2]int{t[1], t[2]})
	}
	dist := make([]int, n+1)
	for i := range dist {
		dist[i] = math.MaxInt32
	}
	dist[k] = 0
	h := &MinHeap{}
	heap.Init(h)
	heap.Push(h, [2]int{k, 0})
	for h.Len() > 0 {
		node := heap.Pop(h).([2]int)
		u, d := node[0], node[1]
		if d > dist[u] {
			continue
		}
		for _, v := range graph[u] {
			if dist[v[0]] > dist[u]+v[1] {
				dist[v[0]] = dist[u] + v[1]
				heap.Push(h, [2]int{v[0], dist[v[0]]})
			}
		}
	}
	max := 0
	for i := 1; i <= n; i++ {
		if dist[i] == math.MaxInt32 {
			return -1
		}
		if dist[i] > max {
			max = dist[i]
		}
	}
	return max
}

type MinHeap [][2]int

func (h MinHeap) Len() int           { return len(h) }
func (h MinHeap) Less(i, j int) bool { return h[i][1] < h[j][1] }
func (h MinHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) {
	*h = append(*h, x.([2]int))
}
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// 28. Cheapest Flights Within K Stops
func FindCheapestPrice(n int, flights [][]int, src int, dst int, K int) int {
	graph := make(map[int][][2]int)
	for _, f := range flights {
		graph[f[0]] = append(graph[f[0]], [2]int{f[1], f[2]})
	}
	cost := make([]int, n)
	for i := range cost {
		cost[i] = math.MaxInt32
	}
	cost[src] = 0
	type node struct{ u, stops, price int }
	q := []node{{src, 0, 0}}
	for len(q) > 0 {
		cur := q[0]
		q = q[1:]
		if cur.stops > K {
			continue
		}
		for _, v := range graph[cur.u] {
			if cur.price+v[1] < cost[v[0]] {
				cost[v[0]] = cur.price + v[1]
				q = append(q, node{v[0], cur.stops + 1, cost[v[0]]})
			}
		}
	}
	if cost[dst] == math.MaxInt32 {
		return -1
	}
	return cost[dst]
}

// 29. Redundant Connection
func FindRedundantConnection(edges [][]int) []int {
	n := len(edges)
	parent := make([]int, n+1)
	for i := range parent {
		parent[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	for _, e := range edges {
		u, v := e[0], e[1]
		pu, pv := find(u), find(v)
		if pu == pv {
			return e
		}
		parent[pu] = pv
	}
	return nil
}

// 30. Redundant Connection II
func FindRedundantDirectedConnection(edges [][]int) []int {
	n := len(edges)
	parent := make([]int, n+1)
	for i := range parent {
		parent[i] = i
	}
	// Define find function before its first use
	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	candA, candB := []int{}, []int{}
	child := make([]int, n+1)
	for i := range child {
		child[i] = -1
	}
	for i, e := range edges {
		if child[e[1]] == -1 {
			child[e[1]] = i
		} else {
			candA = edges[child[e[1]]]
			candB = e
			edges[i][1] = 0
		}
	}
	for i := range parent {
		parent[i] = i
	}
	for _, e := range edges {
		if e[1] == 0 {
			continue
		}
		u, v := e[0], e[1]
		pu, pv := find(u), find(v)
		if pu == pv {
			if len(candA) == 0 {
				return e
			}
			return candA
		}
		parent[pv] = pu
	}
	return candB
}

// 31. Evaluate Division (Graph with weights)
func CalcEquation(equations [][]string, values []float64, queries [][]string) []float64 {
	graph := make(map[string][][2]interface{})
	for i, eq := range equations {
		a, b := eq[0], eq[1]
		graph[a] = append(graph[a], [2]interface{}{b, values[i]})
		graph[b] = append(graph[b], [2]interface{}{a, 1 / values[i]})
	}
	var dfs func(a, b string, visited map[string]bool) float64
	dfs = func(a, b string, visited map[string]bool) float64 {
		if _, ok := graph[a]; !ok {
			return -1
		}
		if a == b {
			return 1
		}
		visited[a] = true
		for _, nei := range graph[a] {
			n, w := nei[0].(string), nei[1].(float64)
			if !visited[n] {
				res := dfs(n, b, visited)
				if res != -1 {
					return res * w
				}
			}
		}
		return -1
	}
	res := make([]float64, len(queries))
	for i, q := range queries {
		visited := make(map[string]bool)
		res[i] = dfs(q[0], q[1], visited)
	}
	return res
}

// 32. Graph Valid Tree (duplicate of 16)

// 33. Minimum Height Trees
func FindMinHeightTrees(n int, edges [][]int) []int {
	if n == 1 {
		return []int{0}
	}
	graph := make(map[int][]int)
	degree := make([]int, n)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
		degree[e[0]]++
		degree[e[1]]++
	}
	leaves := []int{}
	for i := 0; i < n; i++ {
		if degree[i] == 1 {
			leaves = append(leaves, i)
		}
	}
	remain := n
	for remain > 2 {
		remain -= len(leaves)
		newLeaves := []int{}
		for _, leaf := range leaves {
			for _, v := range graph[leaf] {
				degree[v]--
				if degree[v] == 1 {
					newLeaves = append(newLeaves, v)
				}
			}
		}
		leaves = newLeaves
	}
	return leaves
}

// 34. Find the Town Judge
func FindJudge(n int, trust [][]int) int {
	in := make([]int, n+1)
	out := make([]int, n+1)
	for _, t := range trust {
		out[t[0]]++
		in[t[1]]++
	}
	for i := 1; i <= n; i++ {
		if in[i] == n-1 && out[i] == 0 {
			return i
		}
	}
	return -1
}

// 35. Course Schedule III
// Not a graph traversal, but greedy scheduling, omitted for brevity

// 36. Sequence Reconstruction
func SequenceReconstruction(org []int, seqs [][]int) bool {
	graph := make(map[int][]int)
	indegree := make(map[int]int)
	for _, seq := range seqs {
		for i := 0; i < len(seq); i++ {
			if _, ok := indegree[seq[i]]; !ok {
				indegree[seq[i]] = 0
			}
			if i > 0 {
				graph[seq[i-1]] = append(graph[seq[i-1]], seq[i])
				indegree[seq[i]]++
			}
		}
	}
	queue := []int{}
	for k, v := range indegree {
		if v == 0 {
			queue = append(queue, k)
		}
	}
	res := []int{}
	for len(queue) > 0 {
		if len(queue) > 1 {
			return false
		}
		u := queue[0]
		queue = queue[1:]
		res = append(res, u)
		for _, v := range graph[u] {
			indegree[v]--
			if indegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}
	if len(res) != len(org) {
		return false
	}
	for i := range org {
		if org[i] != res[i] {
			return false
		}
	}
	return true
}

// 37. Graph Bipartition
func PossibleBipartition(n int, dislikes [][]int) bool {
	graph := make(map[int][]int)
	for _, d := range dislikes {
		graph[d[0]] = append(graph[d[0]], d[1])
		graph[d[1]] = append(graph[d[1]], d[0])
	}
	color := make([]int, n+1)
	for i := range color {
		color[i] = -1
	}
	var dfs func(u, c int) bool
	dfs = func(u, c int) bool {
		color[u] = c
		for _, v := range graph[u] {
			if color[v] == -1 {
				if !dfs(v, 1-c) {
					return false
				}
			} else if color[v] == c {
				return false
			}
		}
		return true
	}
	for i := 1; i <= n; i++ {
		if color[i] == -1 && !dfs(i, 0) {
			return false
		}
	}
	return true
}

// 38. Longest Increasing Path in a Matrix
func LongestIncreasingPath(matrix [][]int) int {
	if len(matrix) == 0 {
		return 0
	}
	m, n := len(matrix), len(matrix[0])
	dp := make([][]int, m)
	for i := range dp {
		dp[i] = make([]int, n)
	}
	dirs := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	var dfs func(i, j int) int
	dfs = func(i, j int) int {
		if dp[i][j] != 0 {
			return dp[i][j]
		}
		maxLen := 1
		for _, d := range dirs {
			ni, nj := i+d[0], j+d[1]
			if ni >= 0 && ni < m && nj >= 0 && nj < n && matrix[ni][nj] > matrix[i][j] {
				l := 1 + dfs(ni, nj)
				if l > maxLen {
					maxLen = l
				}
			}
		}
		dp[i][j] = maxLen
		return maxLen
	}
	res := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if l := dfs(i, j); l > res {
				res = l
			}
		}
	}
	return res
}

// 39. Critical Connections in a Network
func CriticalConnections(n int, connections [][]int) [][]int {
	graph := make(map[int][]int)
	for _, c := range connections {
		graph[c[0]] = append(graph[c[0]], c[1])
		graph[c[1]] = append(graph[c[1]], c[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	var res [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					res = append(res, []int{u, v})
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	return res
}

// 40. Reorganize String (Using Graph Theory)
// Not a graph problem, omitted for brevity

// 41. Find if Path Exists in Graph
func ValidPath(n int, edges [][]int, source int, destination int) bool {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	visited := make([]bool, n)
	var dfs func(u int) bool
	dfs = func(u int) bool {
		if u == destination {
			return true
		}
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] && dfs(v) {
				return true
			}
		}
		return false
	}
	return dfs(source)
}

// 42. Minimum Cost to Connect All Points (MST - Prim's)
func MinCostConnectPoints(points [][]int) int {
	n := len(points)
	visited := make([]bool, n)
	cost := make([]int, n)
	for i := range cost {
		cost[i] = math.MaxInt32
	}
	cost[0] = 0
	res := 0
	for i := 0; i < n; i++ {
		u := -1
		for j := 0; j < n; j++ {
			if !visited[j] && (u == -1 || cost[j] < cost[u]) {
				u = j
			}
		}
		visited[u] = true
		res += cost[u]
		for v := 0; v < n; v++ {
			if !visited[v] {
				d := abs(points[u][0]-points[v][0]) + abs(points[u][1]-points[v][1])
				if d < cost[v] {
					cost[v] = d
				}
			}
		}
	}
	return res
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

// 43. Minimum Spanning Tree (Kruskal’s Algorithm)
func KruskalMST(n int, edges [][]int) int {
	sort.Slice(edges, func(i, j int) bool {
		return edges[i][2] < edges[j][2]
	})
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	res := 0
	for _, e := range edges {
		u, v, w := e[0], e[1], e[2]
		pu, pv := find(u), find(v)
		if pu != pv {
			parent[pu] = pv
			res += w
		}
	}
	return res
}

// 44. Number of Connected Components
// Same as 11

// 45. Number of Distinct Islands
func NumDistinctIslands(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	shapes := make(map[string]bool)
	var dfs func(i, j, di, dj int, shape *[]string)
	dfs = func(i, j, di, dj int, shape *[]string) {
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0 || visited[i][j] {
			return
		}
		visited[i][j] = true
		*shape = append(*shape, fmt.Sprintf("%d,%d", di, dj))
		dfs(i+1, j, di+1, dj, shape)
		dfs(i-1, j, di-1, dj, shape)
		dfs(i, j+1, di, dj+1, shape)
		dfs(i, j-1, di, dj-1, shape)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				shape := []string{}
				dfs(i, j, 0, 0, &shape)
				shapes[strings.Join(shape, "|")] = true
			}
		}
	}
	return len(shapes)
}

// 46. Count Sub Islands
func CountSubIslands(grid1 [][]int, grid2 [][]int) int {
	m, n := len(grid2), len(grid2[0])
	var dfs func(i, j int) bool
	dfs = func(i, j int) bool {
		if i < 0 || i >= m || j < 0 || j >= n || grid2[i][j] == 0 {
			return true
		}
		grid2[i][j] = 0
		res := true
		if grid1[i][j] == 0 {
			res = false
		}
		res = dfs(i+1, j) && res
		res = dfs(i-1, j) && res
		res = dfs(i, j+1) && res
		res = dfs(i, j-1) && res
		return res
	}
	count := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid2[i][j] == 1 && dfs(i, j) {
				count++
			}
		}
	}
	return count
}

// 47. Find Eventual Safe States
func EventualSafeNodes(graph [][]int) []int {
	n := len(graph)
	state := make([]int, n) // 0=unknown, 1=safe, 2=unsafe
	var dfs func(u int) bool
	dfs = func(u int) bool {
		if state[u] > 0 {
			return state[u] == 1
		}
		state[u] = 2
		for _, v := range graph[u] {
			if !dfs(v) {
				return false
			}
		}
		state[u] = 1
		return true
	}
	res := []int{}
	for i := 0; i < n; i++ {
		if dfs(i) {
			res = append(res, i)
		}
	}
	return res
}

// 48. Surrounded Regions (Using Graph)
func Solve(board [][]byte) {
	if len(board) == 0 {
		return
	}
	m, n := len(board), len(board[0])
	var dfs func(i, j int)
	dfs = func(i, j int) {
		if i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O' {
			return
		}
		board[i][j] = 'E'
		dfs(i+1, j)
		dfs(i-1, j)
		dfs(i, j+1)
		dfs(i, j-1)
	}
	for i := 0; i < m; i++ {
		dfs(i, 0)
		dfs(i, n-1)
	}
	for j := 0; j < n; j++ {
		dfs(0, j)
		dfs(m-1, j)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			} else if board[i][j] == 'E' {
				board[i][j] = 'O'
			}
		}
	}
}

// 49. Walls and Gates
func WallsAndGates(rooms [][]int) {
	m, n := len(rooms), len(rooms[0])
	queue := [][2]int{}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if rooms[i][j] == 0 {
				queue = append(queue, [2]int{i, j})
			}
		}
	}
	dirs := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		for _, d := range dirs {
			ni, nj := p[0]+d[0], p[1]+d[1]
			if ni >= 0 && ni < m && nj >= 0 && nj < n && rooms[ni][nj] == 2147483647 {
				rooms[ni][nj] = rooms[p[0]][p[1]] + 1
				queue = append(queue, [2]int{ni, nj})
			}
		}
	}
}

// 50. Shortest Path in Binary Matrix
func ShortestPathBinaryMatrix(grid [][]int) int {
	n := len(grid)
	if grid[0][0] != 0 || grid[n-1][n-1] != 0 {
		return -1
	}
	queue := [][3]int{{0, 0, 1}}
	grid[0][0] = 1
	dirs := [][]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	for len(queue) > 0 {
		x, y, d := queue[0][0], queue[0][1], queue[0][2]
		queue = queue[1:]
		if x == n-1 && y == n-1 {
			return d
		}
		for _, dir := range dirs {
			nx, ny := x+dir[0], y+dir[1]
			if nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 0 {
				queue = append(queue, [3]int{nx, ny, d + 1})
				grid[nx][ny] = 1
			}
		}
	}
	return -1
}

// 51. Shortest Path in Weighted Graph (Dijkstra's Algorithm)
func Dijkstra(n int, edges [][]int, src int) []int {
	graph := make(map[int][][2]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], [2]int{e[1], e[2]})
	}
	dist := make([]int, n)
	for i := range dist {
		dist[i] = math.MaxInt32
	}
	dist[src] = 0
	h := &MinHeap{}
	heap.Init(h)
	heap.Push(h, [2]int{src, 0})
	for h.Len() > 0 {
		node := heap.Pop(h).([2]int)
		u, d := node[0], node[1]
		if d > dist[u] {
			continue
		}
		for _, v := range graph[u] {
			if dist[v[0]] > dist[u]+v[1] {
				dist[v[0]] = dist[u] + v[1]
				heap.Push(h, [2]int{v[0], dist[v[0]]})
			}
		}
	}
	return dist
}

// Dijkstra's algorithm finds the shortest path from a source to all nodes in a weighted graph with non-negative weights.

// 52. Bellman-Ford Algorithm
func BellmanFord(n int, edges [][]int, src int) ([]int, bool) {
	dist := make([]int, n)
	for i := range dist {
		dist[i] = math.MaxInt32
	}
	dist[src] = 0
	for i := 0; i < n-1; i++ {
		for _, e := range edges {
			u, v, w := e[0], e[1], e[2]
			if dist[u] != math.MaxInt32 && dist[u]+w < dist[v] {
				dist[v] = dist[u] + w
			}
		}
	}
	// Check for negative cycle
	for _, e := range edges {
		u, v, w := e[0], e[1], e[2]
		if dist[u] != math.MaxInt32 && dist[u]+w < dist[v] {
			return dist, false
		}
	}
	return dist, true
}

// Bellman-Ford computes shortest paths and detects negative cycles in graphs with negative weights.

// 53. Floyd Warshall Algorithm
func FloydWarshall(n int, edges [][]int) [][]int {
	dist := make([][]int, n)
	for i := range dist {
		dist[i] = make([]int, n)
		for j := range dist[i] {
			if i == j {
				dist[i][j] = 0
			} else {
				dist[i][j] = math.MaxInt32 / 2
			}
		}
	}
	for _, e := range edges {
		u, v, w := e[0], e[1], e[2]
		dist[u][v] = w
	}
	for k := 0; k < n; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if dist[i][j] > dist[i][k]+dist[k][j] {
					dist[i][j] = dist[i][k] + dist[k][j]
				}
			}
		}
	}
	return dist
}

// Floyd-Warshall finds shortest paths between all pairs of nodes in O(n^3) time.

// 54. Detect Negative Cycle in Graph
func HasNegativeCycle(n int, edges [][]int) bool {
	dist := make([]int, n)
	for i := range dist {
		dist[i] = 0
	}
	for i := 0; i < n; i++ {
		for _, e := range edges {
			u, v, w := e[0], e[1], e[2]
			if dist[u]+w < dist[v] {
				dist[v] = dist[u] + w
				if i == n-1 {
					return true
				}
			}
		}
	}
	return false
}

// Uses Bellman-Ford's last iteration to detect negative cycles.

// 55. Cheapest Flights Within K Stops (BFS/DP)
func CheapestFlightsKStops(n int, flights [][]int, src int, dst int, K int) int {
	graph := make(map[int][][2]int)
	for _, f := range flights {
		graph[f[0]] = append(graph[f[0]], [2]int{f[1], f[2]})
	}
	cost := make([]int, n)
	for i := range cost {
		cost[i] = math.MaxInt32
	}
	cost[src] = 0
	type node struct{ u, stops, price int }
	q := []node{{src, 0, 0}}
	for len(q) > 0 {
		cur := q[0]
		q = q[1:]
		if cur.stops > K {
			continue
		}
		for _, v := range graph[cur.u] {
			if cur.price+v[1] < cost[v[0]] {
				cost[v[0]] = cur.price + v[1]
				q = append(q, node{v[0], cur.stops + 1, cost[v[0]]})
			}
		}
	}
	if cost[dst] == math.MaxInt32 {
		return -1
	}
	return cost[dst]
}

// BFS with stops count, similar to Dijkstra but with stop constraint.

// 56. All Paths from Source to Target
func AllPathsSourceTarget(graph [][]int) [][]int {
	var res [][]int
	var dfs func(u int, path []int)
	n := len(graph)
	dfs = func(u int, path []int) {
		if u == n-1 {
			cp := make([]int, len(path))
			copy(cp, path)
			res = append(res, cp)
			return
		}
		for _, v := range graph[u] {
			dfs(v, append(path, v))
		}
	}
	dfs(0, []int{0})
	return res
}

// DFS to enumerate all paths from node 0 to n-1 in a DAG.

// 57. Find Critical Edges (Bridges)
func FindCriticalEdges(n int, edges [][]int) [][]int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	var res [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					res = append(res, []int{u, v})
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	return res
}

// Tarjan's algorithm for bridges (critical edges).

// 58. Count Paths in a DAG
func CountPathsDAG(graph map[int][]int, start, end int) int {
	memo := make(map[int]int)
	var dfs func(u int) int
	dfs = func(u int) int {
		if u == end {
			return 1
		}
		if v, ok := memo[u]; ok {
			return v
		}
		cnt := 0
		for _, v := range graph[u] {
			cnt += dfs(v)
		}
		memo[u] = cnt
		return cnt
	}
	return dfs(start)
}

// Uses memoized DFS to count all paths from start to end in a DAG.

// 59. Longest Path in a DAG
func LongestPathDAG(graph map[int][]int, n int) int {
	dp := make([]int, n)
	visited := make([]bool, n)
	var dfs func(u int) int
	dfs = func(u int) int {
		if visited[u] {
			return dp[u]
		}
		visited[u] = true
		maxLen := 0
		for _, v := range graph[u] {
			if l := dfs(v) + 1; l > maxLen {
				maxLen = l
			}
		}
		dp[u] = maxLen
		return maxLen
	}
	maxRes := 0
	for i := 0; i < n; i++ {
		if l := dfs(i); l > maxRes {
			maxRes = l
		}
	}
	return maxRes
}

// Memoized DFS for longest path in a DAG.

// 60. Cycle Detection in Directed Graph
func HasCycleDirectedGraph(graph map[int][]int, n int) bool {
	visited := make([]int, n) // 0=unvisited, 1=visiting, 2=visited
	var dfs func(u int) bool
	dfs = func(u int) bool {
		visited[u] = 1
		for _, v := range graph[u] {
			if visited[v] == 1 {
				return true
			}
			if visited[v] == 0 && dfs(v) {
				return true
			}
		}
		visited[u] = 2
		return false
	}
	for i := 0; i < n; i++ {
		if visited[i] == 0 && dfs(i) {
			return true
		}
	}
	return false
}

// Standard DFS-based cycle detection for directed graphs.

// 61. Hamiltonian Path Problem
// Returns true if there is a Hamiltonian Path in the graph (visits every node exactly once)
func HamiltonianPath(graph map[int][]int, n int) bool {
	visited := make([]bool, n)
	var dfs func(u, count int) bool
	dfs = func(u, count int) bool {
		if count == n {
			return true
		}
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] && dfs(v, count+1) {
				return true
			}
		}
		visited[u] = false
		return false
	}
	for i := 0; i < n; i++ {
		if dfs(i, 1) {
			return true
		}
	}
	return false
}

// Tries all possible starting points and uses backtracking to find a Hamiltonian path.

// 62. Eulerian Path and Circuit
// Returns (hasEulerianPath, hasEulerianCircuit)
func EulerianPathCircuit(graph map[int][]int, n int) (bool, bool) {
	degree := make([]int, n)

	for u, vs := range graph {
		for _, v := range vs {
			degree[u]++
			degree[v]++ // because undirected
		}
	}

	odd := 0
	for i := 0; i < n; i++ {
		if degree[i]%2 != 0 {
			odd++
		}
	}

	hasCircuit := odd == 0
	hasPath := odd == 2 || hasCircuit // circuit is also a path

	return hasPath, hasCircuit
}

// For undirected graphs: Eulerian circuit if all degrees even, path if exactly two odd degrees.

// 63. Find Bridges in Graph (Alternative)
func Bridges(n int, edges [][]int) [][]int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	var res [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					res = append(res, []int{u, v})
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	return res
}

// Tarjan's algorithm for finding all bridges (critical edges).

// 64. Articulation Points in Graph (Alternative)
func ArticulationPoints(n int, edges [][]int) []int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	ap := make([]bool, n)
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		children := 0
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				children++
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if parent != -1 && low[v] >= ids[u] {
					ap[u] = true
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
		if parent == -1 && children > 1 {
			ap[u] = true
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	var res []int
	for i, v := range ap {
		if v {
			res = append(res, i)
		}
	}
	return res
}

// Tarjan's algorithm for articulation points (cut vertices).

// 65. Graph Coloring Problem (m-coloring, backtracking)
func GraphColoring(graph map[int][]int, n, m int) bool {
	colors := make([]int, n)
	var isSafe func(u, c int) bool
	isSafe = func(u, c int) bool {
		for _, v := range graph[u] {
			if colors[v] == c {
				return false
			}
		}
		return true
	}
	var dfs func(u int) bool
	dfs = func(u int) bool {
		if u == n {
			return true
		}
		for c := 1; c <= m; c++ {
			if isSafe(u, c) {
				colors[u] = c
				if dfs(u + 1) {
					return true
				}
				colors[u] = 0
			}
		}
		return false
	}
	return dfs(0)
}

// Backtracking to assign m colors to n nodes so that no adjacent nodes share a color.

// 66. Maximal Bipartite Matching (Hopcroft-Karp)
func HopcroftKarp(graph map[int][]int, n1, n2 int) int {
	pairU := make([]int, n1)
	pairV := make([]int, n2)
	dist := make([]int, n1)
	for i := range pairU {
		pairU[i] = -1
	}
	for i := range pairV {
		pairV[i] = -1
	}
	var bfs func() bool
	bfs = func() bool {
		queue := []int{}
		for u := 0; u < n1; u++ {
			if pairU[u] == -1 {
				dist[u] = 0
				queue = append(queue, u)
			} else {
				dist[u] = math.MaxInt32
			}
		}
		found := false
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			for _, v := range graph[u] {
				if pairV[v] == -1 {
					found = true
				} else if dist[pairV[v]] == math.MaxInt32 {
					dist[pairV[v]] = dist[u] + 1
					queue = append(queue, pairV[v])
				}
			}
		}
		return found
	}
	var dfs func(u int) bool
	dfs = func(u int) bool {
		for _, v := range graph[u] {
			if pairV[v] == -1 || (dist[pairV[v]] == dist[u]+1 && dfs(pairV[v])) {
				pairU[u] = v
				pairV[v] = u
				return true
			}
		}
		dist[u] = math.MaxInt32
		return false
	}
	result := 0
	for bfs() {
		for u := 0; u < n1; u++ {
			if pairU[u] == -1 && dfs(u) {
				result++
			}
		}
	}
	return result
}

// Efficient O(sqrt(V)E) bipartite matching using BFS/DFS layers.

// 67. Minimum Vertex Cover in Bipartite Graph (König's theorem)
func MinVertexCoverBipartite(graph map[int][]int, n1, n2 int) ([]int, []int) {
	// Use Hopcroft-Karp to get maximum matching
	pairU := make([]int, n1)
	pairV := make([]int, n2)
	dist := make([]int, n1)
	for i := range pairU {
		pairU[i] = -1
	}
	for i := range pairV {
		pairV[i] = -1
	}
	var bfs func() bool
	bfs = func() bool {
		queue := []int{}
		for u := 0; u < n1; u++ {
			if pairU[u] == -1 {
				dist[u] = 0
				queue = append(queue, u)
			} else {
				dist[u] = math.MaxInt32
			}
		}
		found := false
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			for _, v := range graph[u] {
				if pairV[v] == -1 {
					found = true
				} else if dist[pairV[v]] == math.MaxInt32 {
					dist[pairV[v]] = dist[u] + 1
					queue = append(queue, pairV[v])
				}
			}
		}
		return found
	}
	var dfs func(u int) bool
	dfs = func(u int) bool {
		for _, v := range graph[u] {
			if pairV[v] == -1 || (dist[pairV[v]] == dist[u]+1 && dfs(pairV[v])) {
				pairU[u] = v
				pairV[v] = u
				return true
			}
		}
		dist[u] = math.MaxInt32
		return false
	}
	for bfs() {
		for u := 0; u < n1; u++ {
			if pairU[u] == -1 && dfs(u) {
				// nothing to do, just build matching
			}
		}
	}
	// Now, find minimum vertex cover using alternating paths
	visitedU := make([]bool, n1)
	visitedV := make([]bool, n2)
	var dfs2 func(u int)
	dfs2 = func(u int) {
		visitedU[u] = true
		for _, v := range graph[u] {
			if !visitedV[v] && pairU[u] != v {
				visitedV[v] = true
				if pairV[v] != -1 && !visitedU[pairV[v]] {
					dfs2(pairV[v])
				}
			}
		}
	}
	for u := 0; u < n1; u++ {
		if pairU[u] == -1 && !visitedU[u] {
			dfs2(u)
		}
	}
	leftCover := []int{}
	rightCover := []int{}
	for u := 0; u < n1; u++ {
		if !visitedU[u] {
			leftCover = append(leftCover, u)
		}
	}
	for v := 0; v < n2; v++ {
		if visitedV[v] {
			rightCover = append(rightCover, v)
		}
	}
	return leftCover, rightCover
}

// König's theorem: minimum vertex cover = maximum matching in bipartite graphs.

// 68. Maximum Flow (Ford-Fulkerson Algorithm, DFS-based)
func FordFulkerson(capacity [][]int, s, t int) int {
	n := len(capacity)
	flow := 0
	residual := make([][]int, n)
	for i := range residual {
		residual[i] = make([]int, n)
		copy(residual[i], capacity[i])
	}
	var dfs func(u int, minCap int, visited []bool) int
	dfs = func(u int, minCap int, visited []bool) int {
		if u == t {
			return minCap
		}
		visited[u] = true
		for v := 0; v < n; v++ {
			if !visited[v] && residual[u][v] > 0 {
				d := dfs(v, min(minCap, residual[u][v]), visited)
				if d > 0 {
					residual[u][v] -= d
					residual[v][u] += d
					return d
				}
			}
		}
		return 0
	}
	for {
		visited := make([]bool, n)
		f := dfs(s, math.MaxInt32, visited)
		if f == 0 {
			break
		}
		flow += f
	}
	return flow
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Finds max flow using DFS to find augmenting paths.

// 69. Edmonds-Karp Algorithm for Max Flow (BFS-based)
func EdmondsKarp(capacity [][]int, s, t int) int {
	n := len(capacity)
	flow := 0
	residual := make([][]int, n)
	for i := range residual {
		residual[i] = make([]int, n)
		copy(residual[i], capacity[i])
	}
	parent := make([]int, n)
	for {
		for i := range parent {
			parent[i] = -1
		}
		queue := []int{s}
		parent[s] = s
		for len(queue) > 0 && parent[t] == -1 {
			u := queue[0]
			queue = queue[1:]
			for v := 0; v < n; v++ {
				if parent[v] == -1 && residual[u][v] > 0 {
					parent[v] = u
					queue = append(queue, v)
				}
			}
		}
		if parent[t] == -1 {
			break
		}
		pathFlow := math.MaxInt32
		for v := t; v != s; v = parent[v] {
			u := parent[v]
			if residual[u][v] < pathFlow {
				pathFlow = residual[u][v]
			}
		}
		for v := t; v != s; v = parent[v] {
			u := parent[v]
			residual[u][v] -= pathFlow
			residual[v][u] += pathFlow
		}
		flow += pathFlow
	}
	return flow
}

// Edmonds-Karp is Ford-Fulkerson with BFS for shortest augmenting paths (O(VE^2)).

// 70. Dinic's Algorithm for Max Flow
type Dinic struct {
	n        int
	graph    [][]int
	capacity [][]int
	level    []int
	ptr      []int
}

func NewDinic(n int) *Dinic {
	g := make([][]int, n)
	c := make([][]int, n)
	for i := range c {
		c[i] = make([]int, n)
	}
	return &Dinic{n: n, graph: g, capacity: c}
}
func (d *Dinic) AddEdge(u, v, cap int) {
	d.graph[u] = append(d.graph[u], v)
	d.graph[v] = append(d.graph[v], u)
	d.capacity[u][v] += cap
}
func (d *Dinic) bfs(s, t int) bool {
	d.level = make([]int, d.n)
	for i := range d.level {
		d.level[i] = -1
	}
	queue := []int{s}
	d.level[s] = 0
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for _, v := range d.graph[u] {
			if d.level[v] == -1 && d.capacity[u][v] > 0 {
				d.level[v] = d.level[u] + 1
				queue = append(queue, v)
			}
		}
	}
	return d.level[t] != -1
}
func (d *Dinic) dfs(u, t, pushed int) int {
	if pushed == 0 || u == t {
		return pushed
	}
	for ; d.ptr[u] < len(d.graph[u]); d.ptr[u]++ {
		v := d.graph[u][d.ptr[u]]
		if d.level[v] == d.level[u]+1 && d.capacity[u][v] > 0 {
			tr := d.dfs(v, t, min(pushed, d.capacity[u][v]))
			if tr > 0 {
				d.capacity[u][v] -= tr
				d.capacity[v][u] += tr
				return tr
			}
		}
	}
	return 0
}
func (d *Dinic) MaxFlow(s, t int) int {
	flow := 0
	for d.bfs(s, t) {
		d.ptr = make([]int, d.n)
		for {
			pushed := d.dfs(s, t, math.MaxInt32)
			if pushed == 0 {
				break
			}
			flow += pushed
		}
	}
	return flow
}

// Dinic's algorithm uses BFS for level graph and DFS for blocking flow (O(V^2E)).

// 71. Network Flow with Lower Bounds
// Not implemented here due to complexity. It requires transforming the network to handle lower bounds and using max flow algorithms.
// Explanation: Lower bounds require adding a super source/sink and adjusting capacities to ensure feasibility.

// 72. Bipartite Graph Check (DFS and BFS)
func IsBipartiteDFS(graph map[int][]int, n int) bool {
	color := make([]int, n)
	for i := range color {
		color[i] = -1
	}
	var dfs func(u, c int) bool
	dfs = func(u, c int) bool {
		color[u] = c
		for _, v := range graph[u] {
			if color[v] == -1 {
				if !dfs(v, 1-c) {
					return false
				}
			} else if color[v] == c {
				return false
			}
		}
		return true
	}
	for i := 0; i < n; i++ {
		if color[i] == -1 && !dfs(i, 0) {
			return false
		}
	}
	return true
}
func IsBipartiteBFS(graph map[int][]int, n int) bool {
	color := make([]int, n)
	for i := range color {
		color[i] = -1
	}
	for i := 0; i < n; i++ {
		if color[i] == -1 {
			queue := []int{i}
			color[i] = 0
			for len(queue) > 0 {
				u := queue[0]
				queue = queue[1:]
				for _, v := range graph[u] {
					if color[v] == -1 {
						color[v] = 1 - color[u]
						queue = append(queue, v)
					} else if color[v] == color[u] {
						return false
					}
				}
			}
		}
	}
	return true
}

// DFS and BFS approaches to check if a graph is bipartite.

// 73. Word Ladder II (All shortest paths)
func FindLadders(beginWord, endWord string, wordList []string) [][]string {
	wordSet := make(map[string]bool)
	for _, w := range wordList {
		wordSet[w] = true
	}
	if !wordSet[endWord] {
		return nil
	}
	res := [][]string{}
	layer := map[string][][]string{beginWord: {{beginWord}}}
	for len(layer) > 0 {
		next := map[string][][]string{}
		for word, paths := range layer {
			if word == endWord {
				for _, p := range paths {
					res = append(res, p)
				}
			}
		}
		if len(res) > 0 {
			break
		}
		for word, paths := range layer {
			for i := 0; i < len(word); i++ {
				for c := 'a'; c <= 'z'; c++ {
					nextWord := word[:i] + string(c) + word[i+1:]
					if wordSet[nextWord] {
						for _, p := range paths {
							cp := make([]string, len(p))
							copy(cp, p)
							next[nextWord] = append(next[nextWord], append(cp, nextWord))
						}
					}
				}
			}
		}
		for w := range next {
			delete(wordSet, w)
		}
		layer = next
	}
	return res
}

// BFS layer by layer, tracking all paths, to find all shortest transformation sequences.

// 74. Number of Distinct Islands II (with rotations/reflections)
func NumDistinctIslands2(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	shapes := make(map[string]bool)
	var dfs func(i, j int, coords *[][2]int)
	dfs = func(i, j int, coords *[][2]int) {
		if i < 0 || i >= m || j < 0 || j >= n || grid[i][j] == 0 || visited[i][j] {
			return
		}
		visited[i][j] = true
		*coords = append(*coords, [2]int{i, j})
		dfs(i+1, j, coords)
		dfs(i-1, j, coords)
		dfs(i, j+1, coords)
		dfs(i, j-1, coords)
	}
	normalize := func(shape [][2]int) string {
		transforms := [][][2]int{}
		for _, t := range [][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
			for _, swap := range []bool{false, true} {
				tmp := make([][2]int, len(shape))
				for i, p := range shape {
					x, y := p[0]*t[0], p[1]*t[1]
					if swap {
						x, y = y, x
					}
					tmp[i] = [2]int{x, y}
				}
				sort.Slice(tmp, func(i, j int) bool {
					if tmp[i][0] != tmp[j][0] {
						return tmp[i][0] < tmp[j][0]
					}
					return tmp[i][1] < tmp[j][1]
				})
				ox, oy := tmp[0][0], tmp[0][1]
				for i := range tmp {
					tmp[i][0] -= ox
					tmp[i][1] -= oy
				}
				transforms = append(transforms, tmp)
			}
		}
		cands := []string{}
		for _, t := range transforms {
			s := ""
			for _, p := range t {
				s += fmt.Sprintf("%d,%d|", p[0], p[1])
			}
			cands = append(cands, s)
		}
		sort.Strings(cands)
		return cands[0]
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				coords := [][2]int{}
				dfs(i, j, &coords)
				shapes[normalize(coords)] = true
			}
		}
	}
	return len(shapes)
}

// Normalizes each island shape under all rotations/reflections and counts unique ones.

// 75. Cheapest Path with Discounts (Dijkstra with state)
func CheapestPathWithDiscounts(n int, edges [][]int, src, dst, k int) int {
	graph := make(map[int][][2]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], [2]int{e[1], e[2]})
	}
	type state struct{ u, used int }
	dist := make(map[state]int)
	h := &MinHeap{}
	heap.Init(h)
	heap.Push(h, [2]int{src, 0})
	dist[state{src, 0}] = 0
	for h.Len() > 0 {
		node := heap.Pop(h).([2]int)
		u, cost := node[0], node[1]
		used := 0
		for s := range dist {
			if s.u == u && dist[s] == cost {
				used = s.used
				break
			}
		}
		if u == dst {
			return cost
		}
		for _, v := range graph[u] {
			for d := 0; d <= 1 && used+d <= k; d++ {
				nextCost := cost + v[1]/(1+d)
				nextState := state{v[0], used + d}
				if old, ok := dist[nextState]; !ok || nextCost < old {
					dist[nextState] = nextCost
					heap.Push(h, [2]int{v[0], nextCost})
				}
			}
		}
	}
	return -1
}

// Dijkstra's algorithm with extra state for number of discounts used.

// 76. Count Ways to Reach Destination in DAG
func CountWaysDAG(graph map[int][]int, start, end int) int {
	memo := make(map[int]int)
	var dfs func(u int) int
	dfs = func(u int) int {
		if u == end {
			return 1
		}
		if v, ok := memo[u]; ok {
			return v
		}
		cnt := 0
		for _, v := range graph[u] {
			cnt += dfs(v)
		}
		memo[u] = cnt
		return cnt
	}
	return dfs(start)
}

// Memoized DFS to count all paths from start to end in a DAG.

// 77. Alien Dictionary II (All valid orders)
func AlienOrderAll(words []string) []string {
	graph := make(map[byte][]byte)
	indegree := make(map[byte]int)
	for _, w := range words {
		for i := range w {
			indegree[w[i]] = 0
		}
	}
	for i := 0; i < len(words)-1; i++ {
		w1, w2 := words[i], words[i+1]
		minLen := len(w1)
		if len(w2) < minLen {
			minLen = len(w2)
		}
		for j := 0; j < minLen; j++ {
			if w1[j] != w2[j] {
				graph[w1[j]] = append(graph[w1[j]], w2[j])
				indegree[w2[j]]++
				break
			}
		}
	}
	var res []string
	var dfs func(path []byte, indegree map[byte]int)
	dfs = func(path []byte, indegree map[byte]int) {
		if len(path) == len(indegree) {
			res = append(res, string(path))
			return
		}
		for k, v := range indegree {
			if v == 0 && (len(path) == 0 || !contains(path, k)) {
				indegree[k] = -1
				for _, nei := range graph[k] {
					indegree[nei]--
				}
				dfs(append(path, k), indegree)
				for _, nei := range graph[k] {
					indegree[nei]++
				}
				indegree[k] = 0
			}
		}
	}
	dfs([]byte{}, copyMap(indegree))
	return res
}
func contains(arr []byte, b byte) bool {
	for _, v := range arr {
		if v == b {
			return true
		}
	}
	return false
}
func copyMap(m map[byte]int) map[byte]int {
	cp := make(map[byte]int)
	for k, v := range m {
		cp[k] = v
	}
	return cp
}

// Backtracking all topological sorts for the alien dictionary.

// 78. Number of Paths Between Two Nodes (DFS)
func NumPaths(graph map[int][]int, start, end int) int {
	count := 0
	var dfs func(u int)
	dfs = func(u int) {
		if u == end {
			count++
			return
		}
		for _, v := range graph[u] {
			dfs(v)
		}
	}
	dfs(start)
	return count
}

// Simple DFS to count all paths from start to end (may be exponential).

// 79. Graph Is Bipartite? (returns bool, for adjacency list)
func GraphIsBipartite(graph map[int][]int, n int) bool {
	color := make([]int, n)
	for i := range color {
		color[i] = -1
	}
	for i := 0; i < n; i++ {
		if color[i] == -1 {
			queue := []int{i}
			color[i] = 0
			for len(queue) > 0 {
				u := queue[0]
				queue = queue[1:]
				for _, v := range graph[u] {
					if color[v] == -1 {
						color[v] = 1 - color[u]
						queue = append(queue, v)
					} else if color[v] == color[u] {
						return false
					}
				}
			}
		}
	}
	return true
}

// BFS-based bipartite check for adjacency list graphs.

// 80. Clone Graph with Weights
type WeightedNode struct {
	Val       int
	Neighbors []*WeightedNode
	Weights   []int
}

func CloneWeightedGraph(node *WeightedNode) *WeightedNode {
	if node == nil {
		return nil
	}
	visited := make(map[*WeightedNode]*WeightedNode)
	var clone func(*WeightedNode) *WeightedNode
	clone = func(n *WeightedNode) *WeightedNode {
		if n == nil {
			return nil
		}
		if v, ok := visited[n]; ok {
			return v
		}
		copy := &WeightedNode{Val: n.Val}
		visited[n] = copy
		for i, nei := range n.Neighbors {
			copy.Neighbors = append(copy.Neighbors, clone(nei))
			copy.Weights = append(copy.Weights, n.Weights[i])
		}
		return copy
	}
	return clone(node)
}

// DFS clone for graphs with weighted edges.

// 81. Reconstruct Original Digraph (from degree sequences)
// Given in/out degree sequences, reconstruct a digraph if possible (Havel-Hakimi for digraphs)
func ReconstructDigraph(inDeg, outDeg []int) [][]int {
	n := len(inDeg)
	type node struct{ idx, deg int }
	edges := [][]int{}
	for {
		// Find node with max out-degree
		maxIdx := -1
		for i := 0; i < n; i++ {
			if outDeg[i] > 0 && (maxIdx == -1 || outDeg[i] > outDeg[maxIdx]) {
				maxIdx = i
			}
		}
		if maxIdx == -1 {
			break
		}
		if outDeg[maxIdx] > n-1 {
			return nil // impossible
		}
		cnt := outDeg[maxIdx]
		outDeg[maxIdx] = 0
		// Connect to nodes with largest in-degree (excluding self)
		nodes := []node{}
		for i := 0; i < n; i++ {
			if i != maxIdx && inDeg[i] > 0 {
				nodes = append(nodes, node{i, inDeg[i]})
			}
		}
		sort.Slice(nodes, func(i, j int) bool { return nodes[i].deg > nodes[j].deg })
		if len(nodes) < cnt {
			return nil
		}
		for i := 0; i < cnt; i++ {
			edges = append(edges, []int{maxIdx, nodes[i].idx})
			inDeg[nodes[i].idx]--
		}
	}
	for _, d := range inDeg {
		if d != 0 {
			return nil
		}
	}
	return edges
}

// Explanation: Greedily connect nodes with highest out-degree to those with highest in-degree (excluding self), similar to Havel-Hakimi for digraphs.

// 82. Find Cycle in Undirected Graph (Union-Find)
func HasCycleUnionFind(n int, edges [][]int) bool {
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	for _, e := range edges {
		u, v := find(e[0]), find(e[1])
		if u == v {
			return true
		}
		parent[u] = v
	}
	return false
}

// Explanation: If two nodes are already connected, adding an edge forms a cycle.

// 83. Find Cycle in Directed Graph (DFS)
func FindCycleDirected(graph map[int][]int, n int) []int {
	visited := make([]int, n) // 0=unvisited, 1=visiting, 2=visited
	stack := []int{}
	var res []int
	var dfs func(u int) bool
	dfs = func(u int) bool {
		visited[u] = 1
		stack = append(stack, u)
		for _, v := range graph[u] {
			if visited[v] == 0 {
				if dfs(v) {
					return true
				}
			} else if visited[v] == 1 {
				// Found a cycle, extract it
				idx := len(stack) - 1
				for idx >= 0 && stack[idx] != v {
					idx--
				}
				res = append([]int{}, stack[idx:]...)
				return true
			}
		}
		stack = stack[:len(stack)-1]
		visited[u] = 2
		return false
	}
	for i := 0; i < n; i++ {
		if visited[i] == 0 && dfs(i) {
			return res
		}
	}
	return nil
}

// Explanation: Standard DFS with recursion stack to detect and extract a cycle in a directed graph.

// 84. Find Diameter of Tree
func TreeDiameter(n int, edges [][]int) int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	var farthest, maxDist int
	var dfs func(u, parent, dist int)
	dfs = func(u, parent, dist int) {
		if dist > maxDist {
			maxDist, farthest = dist, u
		}
		for _, v := range graph[u] {
			if v != parent {
				dfs(v, u, dist+1)
			}
		}
	}
	dfs(0, -1, 0)
	maxDist = 0
	dfs(farthest, -1, 0)
	return maxDist
}

// Explanation: Two DFS: first to find farthest node, second from that node to get diameter.

// 85. Count Trees in Forest
func CountTrees(n int, edges [][]int) int {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	visited := make([]bool, n)
	count := 0
	var dfs func(u, parent int) bool
	dfs = func(u, parent int) bool {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				if !dfs(v, u) {
					return false
				}
			} else if v != parent {
				return false
			}
		}
		return true
	}
	for i := 0; i < n; i++ {
		if !visited[i] && dfs(i, -1) {
			count++
		}
	}
	return count
}

// Explanation: Count connected components that are trees (no cycles).

// 86. Check if Graph is Tree
func IsTree(n int, edges [][]int) bool {
	if len(edges) != n-1 {
		return false
	}
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	visited := make([]bool, n)
	var dfs func(u, parent int) bool
	dfs = func(u, parent int) bool {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				if !dfs(v, u) {
					return false
				}
			} else if v != parent {
				return false
			}
		}
		return true
	}
	if !dfs(0, -1) {
		return false
	}
	for _, v := range visited {
		if !v {
			return false
		}
	}
	return true
}

// Explanation: A tree is connected and acyclic, so check both.

// 87. Find Eulerian Path in Undirected Graph
func EulerianPathUndirected(graph map[int][]int, n int) []int {
	degree := make([]int, n)
	for u := range graph {
		degree[u] = len(graph[u])
	}

	start := -1
	odd := 0
	for i := 0; i < n; i++ {
		if degree[i]%2 == 1 {
			odd++
			start = i
		}
	}

	// If no odd degree nodes, pick any node with edges
	if odd == 0 {
		for i := 0; i < n; i++ {
			if degree[i] > 0 {
				start = i
				break
			}
		}
	}

	if odd != 0 && odd != 2 {
		return nil // No Eulerian path
	}

	if start == -1 {
		return nil // No starting node with edges
	}

	used := make(map[[2]int]int)
	for u, vs := range graph {
		for _, v := range vs {
			used[[2]int{u, v}]++
		}
	}

	var path []int
	var dfs func(u int)
	dfs = func(u int) {
		for _, v := range graph[u] {
			if used[[2]int{u, v}] > 0 {
				used[[2]int{u, v}]--
				used[[2]int{v, u}]--
				dfs(v)
			}
		}
		path = append(path, u)
	}

	dfs(start)

	for _, cnt := range used {
		if cnt > 0 {
			return nil // Unused edge remains, graph is disconnected
		}
	}

	// Reverse path
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return path
}

// Explanation: Hierholzer's algorithm for Eulerian path in undirected graphs.

// 88. Find Eulerian Circuit in Directed Graph
func EulerianCircuitDirected(graph map[int][]int, n int) []int {
	inDeg := make([]int, n)
	outDeg := make([]int, n)
	for u, vs := range graph {
		outDeg[u] += len(vs)
		for _, v := range vs {
			inDeg[v]++
		}
	}
	for i := 0; i < n; i++ {
		if inDeg[i] != outDeg[i] {
			return nil
		}
	}
	path := []int{}
	idx := make([]int, n)
	var dfs func(u int)
	dfs = func(u int) {
		for idx[u] < len(graph[u]) {
			v := graph[u][idx[u]]
			idx[u]++
			dfs(v)
		}
		path = append(path, u)
	}
	dfs(0)
	if len(path) != len(graph[0])+1 {
		return nil
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

// Explanation: Hierholzer's algorithm for Eulerian circuit in directed graphs (all in-degree == out-degree).

// 89. Count Number of Islands III (hex grid)
func NumIslandsHex(grid [][]int) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	dirs := [][]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, 1}, {1, -1}}
	var dfs func(i, j int)
	dfs = func(i, j int) {
		visited[i][j] = true
		for _, d := range dirs {
			ni, nj := i+d[0], j+d[1]
			if ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1 && !visited[ni][nj] {
				dfs(ni, nj)
			}
		}
	}
	count := 0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				count++
				dfs(i, j)
			}
		}
	}
	return count
}

// Explanation: DFS for islands, but with 6 directions for hex grid.

// 90. Minimum Spanning Tree with Constraints (pre-included/forbidden edges)
func MSTWithConstraints(n int, edges [][]int, include, exclude map[[2]int]bool) int {
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	res := 0
	// Include mandatory edges first
	for e := range include {
		u, v := e[0], e[1]
		pu, pv := find(u), find(v)
		if pu == pv {
			return -1 // cycle
		}
		parent[pu] = pv
	}
	// Sort edges by weight
	sort.Slice(edges, func(i, j int) bool { return edges[i][2] < edges[j][2] })
	for _, e := range edges {
		u, v, w := e[0], e[1], e[2]
		if exclude[[2]int{u, v}] || exclude[[2]int{v, u}] {
			continue
		}
		if include[[2]int{u, v}] || include[[2]int{v, u}] {
			res += w
			continue
		}
		pu, pv := find(u), find(v)
		if pu != pv {
			parent[pu] = pv
			res += w
		}
	}
	roots := make(map[int]bool)
	for i := 0; i < n; i++ {
		roots[find(i)] = true
	}
	if len(roots) > 1 {
		return -1
	}
	return res
}

// Explanation: Kruskal's algorithm, but include/exclude edges as constraints.
// 91. Maximum Matching in General Graph (Edmonds' Blossom Algorithm)
// Note: Blossom algorithm is complex and not included here due to its length and complexity.
// Placeholder for reference:
// func BlossomMaximumMatching(graph map[int][]int, n int) int {
//     // Implement Edmonds' Blossom algorithm for general graphs
//     // Returns the size of the maximum matching
//     // Not implemented here
//     return 0
// }

// 92. Traveling Salesman Problem (TSP) - DP Bitmasking (O(n^2*2^n))
func TSP(graph [][]int) int {
	n := len(graph)
	dp := make([][]int, 1<<n)
	for i := range dp {
		dp[i] = make([]int, n)
		for j := range dp[i] {
			dp[i][j] = math.MaxInt32 / 2
		}
	}
	dp[1][0] = 0
	for mask := 1; mask < (1 << n); mask++ {
		for u := 0; u < n; u++ {
			if mask&(1<<u) == 0 {
				continue
			}
			for v := 0; v < n; v++ {
				if mask&(1<<v) != 0 && u != v {
					if dp[mask^(1<<u)][v]+graph[v][u] < dp[mask][u] {
						dp[mask][u] = dp[mask^(1<<u)][v] + graph[v][u]
					}
				}
			}
		}
	}
	res := math.MaxInt32
	for u := 1; u < n; u++ {
		if dp[(1<<n)-1][u]+graph[u][0] < res {
			res = dp[(1<<n)-1][u] + graph[u][0]
		}
	}
	return res
}

// 93. Count Components in Graph with Threshold
// Given a threshold, only connect nodes if edge weight > threshold
func CountComponentsWithThreshold(n int, edges [][]int, threshold int) int {
	graph := make(map[int][]int)
	for _, e := range edges {
		if e[2] > threshold {
			graph[e[0]] = append(graph[e[0]], e[1])
			graph[e[1]] = append(graph[e[1]], e[0])
		}
	}
	visited := make([]bool, n)
	count := 0
	var dfs func(u int)
	dfs = func(u int) {
		visited[u] = true
		for _, v := range graph[u] {
			if !visited[v] {
				dfs(v)
			}
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			count++
			dfs(i)
		}
	}
	return count
}

// 94. Minimum Cost to Connect All Cities (Prim's Algorithm)
func MinCostConnectCities(n int, connections [][]int) int {
	graph := make(map[int][][2]int)
	for _, c := range connections {
		u, v, w := c[0], c[1], c[2]
		graph[u] = append(graph[u], [2]int{v, w})
		graph[v] = append(graph[v], [2]int{u, w})
	}
	visited := make([]bool, n+1)
	h := &MinHeap{}
	heap.Init(h)
	heap.Push(h, [2]int{1, 0})
	res := 0
	count := 0
	for h.Len() > 0 && count < n {
		node := heap.Pop(h).([2]int)
		u, w := node[0], node[1]
		if visited[u] {
			continue
		}
		visited[u] = true
		res += w
		count++
		for _, v := range graph[u] {
			if !visited[v[0]] {
				heap.Push(h, [2]int{v[0], v[1]})
			}
		}
	}
	if count < n {
		return -1
	}
	return res
}

// 95. Maximum Bipartite Matching (Hungarian Algorithm, DFS-based)
func Hungarian(graph map[int][]int, n1, n2 int) int {
	match := make([]int, n2)
	for i := range match {
		match[i] = -1
	}
	var dfs func(u int, vis []bool) bool
	dfs = func(u int, vis []bool) bool {
		for _, v := range graph[u] {
			if !vis[v] {
				vis[v] = true
				if match[v] == -1 || dfs(match[v], vis) {
					match[v] = u
					return true
				}
			}
		}
		return false
	}
	res := 0
	for u := 0; u < n1; u++ {
		vis := make([]bool, n2)
		if dfs(u, vis) {
			res++
		}
	}
	return res
}

// 96. Find Critical Edges and Vertices (Bridges and Articulation Points)
func CriticalEdgesAndVertices(n int, edges [][]int) ([][]int, []int) {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	ap := make([]bool, n)
	var bridges [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		children := 0
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				children++
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					bridges = append(bridges, []int{u, v})
				}
				if parent != -1 && low[v] >= ids[u] {
					ap[u] = true
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
		if parent == -1 && children > 1 {
			ap[u] = true
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	var articulation []int
	for i, v := range ap {
		if v {
			articulation = append(articulation, i)
		}
	}
	return bridges, articulation
}

// 97. Find Bridges and Articulation Points (Tarjan’s Algorithm, alternative)
func TarjanBridgesArticulation(n int, edges [][]int) ([][]int, []int) {
	graph := make(map[int][]int)
	for _, e := range edges {
		graph[e[0]] = append(graph[e[0]], e[1])
		graph[e[1]] = append(graph[e[1]], e[0])
	}
	ids := make([]int, n)
	low := make([]int, n)
	visited := make([]bool, n)
	id := 0
	ap := make([]bool, n)
	var bridges [][]int
	var dfs func(u, parent int)
	dfs = func(u, parent int) {
		visited[u] = true
		id++
		ids[u] = id
		low[u] = id
		children := 0
		for _, v := range graph[u] {
			if v == parent {
				continue
			}
			if !visited[v] {
				children++
				dfs(v, u)
				if low[v] < low[u] {
					low[u] = low[v]
				}
				if low[v] > ids[u] {
					bridges = append(bridges, []int{u, v})
				}
				if parent != -1 && low[v] >= ids[u] {
					ap[u] = true
				}
			} else {
				if ids[v] < low[u] {
					low[u] = ids[v]
				}
			}
		}
		if parent == -1 && children > 1 {
			ap[u] = true
		}
	}
	for i := 0; i < n; i++ {
		if !visited[i] {
			dfs(i, -1)
		}
	}
	var articulation []int
	for i, v := range ap {
		if v {
			articulation = append(articulation, i)
		}
	}
	return bridges, articulation
}

// 98. Maximum Network Flow with Costs (Min-Cost Max-Flow, Successive Shortest Path)
type Edge struct {
	to, rev, cap, cost int
}
type MinCostMaxFlow struct {
	n     int
	graph [][]Edge
}

func NewMinCostMaxFlow(n int) *MinCostMaxFlow {
	g := make([][]Edge, n)
	return &MinCostMaxFlow{n: n, graph: g}
}
func (m *MinCostMaxFlow) AddEdge(u, v, cap, cost int) {
	m.graph[u] = append(m.graph[u], Edge{v, len(m.graph[v]), cap, cost})
	m.graph[v] = append(m.graph[v], Edge{u, len(m.graph[u]) - 1, 0, -cost})
}
func (m *MinCostMaxFlow) Flow(s, t int) (int, int) {
	n := m.n
	prevv := make([]int, n)
	preve := make([]int, n)
	const INF = 1 << 30
	flow, cost := 0, 0
	h := make([]int, n)
	for {
		dist := make([]int, n)
		for i := range dist {
			dist[i] = INF
		}
		dist[s] = 0
		inqueue := make([]bool, n)
		queue := []int{s}
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			inqueue[u] = false
			for i, e := range m.graph[u] {
				if e.cap > 0 && dist[e.to] > dist[u]+e.cost+h[u]-h[e.to] {
					dist[e.to] = dist[u] + e.cost + h[u] - h[e.to]
					prevv[e.to] = u
					preve[e.to] = i
					if !inqueue[e.to] {
						queue = append(queue, e.to)
						inqueue[e.to] = true
					}
				}
			}
		}
		if dist[t] == INF {
			break
		}
		for i := 0; i < n; i++ {
			h[i] += dist[i]
		}
		d := INF
		for v := t; v != s; v = prevv[v] {
			if m.graph[prevv[v]][preve[v]].cap < d {
				d = m.graph[prevv[v]][preve[v]].cap
			}
		}
		flow += d
		cost += d * h[t]
		for v := t; v != s; v = prevv[v] {
			e := &m.graph[prevv[v]][preve[v]]
			e.cap -= d
			m.graph[v][e.rev].cap += d
		}
	}
	return flow, cost
}

// 99. Graph Isomorphism Check (for small graphs, brute-force)
// Returns true if g1 and g2 are isomorphic (adjacency matrix form)
func GraphIsomorphic(g1, g2 [][]int) bool {
	n := len(g1)
	if n != len(g2) {
		return false
	}
	perm := make([]int, n)
	for i := 0; i < n; i++ {
		perm[i] = i
	}
	var check func() bool
	check = func() bool {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if g1[i][j] != g2[perm[i]][perm[j]] {
					return false
				}
			}
		}
		return true
	}
	for {
		if check() {
			return true
		}
		if !nextPerm(perm) {
			break
		}
	}
	return false
}
func nextPerm(a []int) bool {
	n := len(a)
	i := n - 2
	for i >= 0 && a[i] >= a[i+1] {
		i--
	}
	if i < 0 {
		return false
	}
	j := n - 1
	for a[j] <= a[i] {
		j--
	}
	a[i], a[j] = a[j], a[i]
	for l, r := i+1, n-1; l < r; l, r = l+1, r-1 {
		a[l], a[r] = a[r], a[l]
	}
	return true
}

// 100. Random Walk on Graph (returns path of given length)
func RandomWalk(graph map[int][]int, start, length int) []int {
	if length <= 0 {
		return []int{start}
	}
	path := []int{start}
	cur := start
	for i := 0; i < length; i++ {
		neighbors := graph[cur]
		if len(neighbors) == 0 {
			break
		}
		next := neighbors[randInt(len(neighbors))]
		path = append(path, next)
		cur = next
	}
	return path
}
func randInt(n int) int {
	return int(math.Floor(float64(n) * randFloat()))
}
func randFloat() float64 {
	return float64(randSeed()) / float64(1<<31)
}
func randSeed() int32 {
	// Simple LCG for demonstration (not cryptographically secure)
	const (
		a int64 = 1103515245
		c int64 = 12345
		m int64 = 1 << 31
	)
	var seed int64 = 42
	seed = (a*seed + c) % m
	return int32(seed)
}
