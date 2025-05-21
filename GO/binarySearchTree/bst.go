package binarysearchtree

import "sort"

// TreeNode defines a node in the binary search tree.
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 1. Validate Binary Search Tree
// Checks if a tree is a valid BST by ensuring all left < root < all right.
func IsValidBST(root *TreeNode) bool {
	return isValidBSTHelper(root, nil, nil)
}

func isValidBSTHelper(node *TreeNode, min, max *int) bool {
	if node == nil {
		return true
	}
	if min != nil && node.Val <= *min {
		return false
	}
	if max != nil && node.Val >= *max {
		return false
	}
	return isValidBSTHelper(node.Left, min, &node.Val) && isValidBSTHelper(node.Right, &node.Val, max)
}

// 2. Kth Smallest Element in a BST
// Inorder traversal yields sorted order; return kth element.
func KthSmallest(root *TreeNode, k int) int {
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

// 3. Lowest Common Ancestor of a BST
// Traverse from root, using BST property to find split point.
func LowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	for root != nil {
		if p.Val < root.Val && q.Val < root.Val {
			root = root.Left
		} else if p.Val > root.Val && q.Val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}

// 4. Convert Sorted Array to Binary Search Tree
// Recursively pick middle as root to ensure balance.
func SortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	mid := len(nums) / 2
	return &TreeNode{
		Val:   nums[mid],
		Left:  SortedArrayToBST(nums[:mid]),
		Right: SortedArrayToBST(nums[mid+1:]),
	}
}

// 5. Convert Sorted List to Binary Search Tree
// Use slow/fast pointers to find middle node for root.
type ListNode struct {
	Val  int
	Next *ListNode
}

func SortedListToBST(head *ListNode) *TreeNode {
	if head == nil {
		return nil
	}
	if head.Next == nil {
		return &TreeNode{Val: head.Val}
	}
	prev, slow, fast := (*ListNode)(nil), head, head
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	if prev != nil {
		prev.Next = nil
	}
	return &TreeNode{
		Val:   slow.Val,
		Left:  SortedListToBST(head),
		Right: SortedListToBST(slow.Next),
	}
}

// 6. Construct Binary Search Tree from Preorder Traversal
// Use bounds to construct recursively.
func BstFromPreorder(preorder []int) *TreeNode {
	idx := 0
	var helper func(bound int) *TreeNode
	helper = func(bound int) *TreeNode {
		if idx == len(preorder) || preorder[idx] > bound {
			return nil
		}
		root := &TreeNode{Val: preorder[idx]}
		idx++
		root.Left = helper(root.Val)
		root.Right = helper(bound)
		return root
	}
	return helper(1<<31 - 1)
}

// 7. Search in a Binary Search Tree
// Traverse left/right based on value.
func SearchBST(root *TreeNode, val int) *TreeNode {
	for root != nil {
		if val < root.Val {
			root = root.Left
		} else if val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}

// 8. Insert into a Binary Search Tree
// Recursively find position and insert.
func InsertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	if val < root.Val {
		root.Left = InsertIntoBST(root.Left, val)
	} else {
		root.Right = InsertIntoBST(root.Right, val)
	}
	return root
}

// 9. Delete Node in a BST
// Find node, replace with inorder successor if needed.
func DeleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return nil
	}
	if key < root.Val {
		root.Left = DeleteNode(root.Left, key)
	} else if key > root.Val {
		root.Right = DeleteNode(root.Right, key)
	} else {
		if root.Left == nil {
			return root.Right
		}
		if root.Right == nil {
			return root.Left
		}
		minNode := root.Right
		for minNode.Left != nil {
			minNode = minNode.Left
		}
		root.Val = minNode.Val
		root.Right = DeleteNode(root.Right, minNode.Val)
	}
	return root
}

// 10. Recover Binary Search Tree
// Find two swapped nodes in inorder traversal and swap them back.
func RecoverTree(root *TreeNode) {
	var first, second, prev *TreeNode
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		if prev != nil && node.Val < prev.Val {
			if first == nil {
				first = prev
			}
			second = node
		}
		prev = node
		inorder(node.Right)
	}
	inorder(root)
	if first != nil && second != nil {
		first.Val, second.Val = second.Val, first.Val
	}
}

// 11. Trim a Binary Search Tree
// Remove nodes outside [low, high].
func TrimBST(root *TreeNode, low, high int) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val < low {
		return TrimBST(root.Right, low, high)
	}
	if root.Val > high {
		return TrimBST(root.Left, low, high)
	}
	root.Left = TrimBST(root.Left, low, high)
	root.Right = TrimBST(root.Right, low, high)
	return root
}

// 12. Closest Binary Search Tree Value
// Traverse, keep track of closest value.
func ClosestValue(root *TreeNode, target float64) int {
	closest := root.Val
	for root != nil {
		if absFloat(float64(root.Val)-target) < absFloat(float64(closest)-target) {
			closest = root.Val
		}
		if target < float64(root.Val) {
			root = root.Left
		} else {
			root = root.Right
		}
	}
	return closest
}

func absFloat(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

// 13. Closest Binary Search Tree Value II
// Inorder traversal, collect k closest values.
func ClosestKValues(root *TreeNode, target float64, k int) []int {
	var inorder func(*TreeNode)
	vals := []int{}
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		vals = append(vals, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	sort.Slice(vals, func(i, j int) bool {
		return absFloat(float64(vals[i])-target) < absFloat(float64(vals[j])-target)
	})
	return vals[:k]
}

// 14. Find Mode in Binary Search Tree
// Inorder traversal, count frequencies.
func FindMode(root *TreeNode) []int {
	var res []int
	var prev *TreeNode
	maxCount, count := 0, 0
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		if prev != nil && node.Val == prev.Val {
			count++
		} else {
			count = 1
		}
		if count > maxCount {
			maxCount = count
			res = []int{node.Val}
		} else if count == maxCount {
			res = append(res, node.Val)
		}
		prev = node
		inorder(node.Right)
	}
	inorder(root)
	return res
}

// 15. Binary Search Tree Iterator
// Inorder traversal using stack.
type BSTIterator struct {
	stack []*TreeNode
}

func Constructor(root *TreeNode) BSTIterator {
	it := BSTIterator{}
	it.pushLeft(root)
	return it
}

func (it *BSTIterator) pushLeft(node *TreeNode) {
	for node != nil {
		it.stack = append(it.stack, node)
		node = node.Left
	}
}

func (it *BSTIterator) Next() int {
	node := it.stack[len(it.stack)-1]
	it.stack = it.stack[:len(it.stack)-1]
	it.pushLeft(node.Right)
	return node.Val
}

func (it *BSTIterator) HasNext() bool {
	return len(it.stack) > 0
}

// 16. Kth Largest Element in a BST
// Reverse inorder traversal.
func KthLargest(root *TreeNode, k int) int {
	stack := []*TreeNode{}
	for {
		for root != nil {
			stack = append(stack, root)
			root = root.Right
		}
		root = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return root.Val
		}
		root = root.Left
	}
}

// 17. Convert BST to Greater Tree
// Reverse inorder, accumulate sum.
func ConvertBST(root *TreeNode) *TreeNode {
	sum := 0
	var traverse func(*TreeNode)
	traverse = func(node *TreeNode) {
		if node == nil {
			return
		}
		traverse(node.Right)
		sum += node.Val
		node.Val = sum
		traverse(node.Left)
	}
	traverse(root)
	return root
}

// 18. Range Sum of BST
// Traverse, sum values in [low, high].
func RangeSumBST(root *TreeNode, low int, high int) int {
	if root == nil {
		return 0
	}
	if root.Val < low {
		return RangeSumBST(root.Right, low, high)
	}
	if root.Val > high {
		return RangeSumBST(root.Left, low, high)
	}
	return root.Val + RangeSumBST(root.Left, low, high) + RangeSumBST(root.Right, low, high)
}

// 19. Two Sum IV - Input is a BST
// Use set to check for complement.
func FindTarget(root *TreeNode, k int) bool {
	set := make(map[int]struct{})
	var dfs func(*TreeNode) bool
	dfs = func(node *TreeNode) bool {
		if node == nil {
			return false
		}
		if _, ok := set[k-node.Val]; ok {
			return true
		}
		set[node.Val] = struct{}{}
		return dfs(node.Left) || dfs(node.Right)
	}
	return dfs(root)
}

// 20. Binary Search Tree to Greater Sum Tree
// Similar to ConvertBST, but set node.Val to sum of greater values.
func BstToGst(root *TreeNode) *TreeNode {
	sum := 0
	var traverse func(*TreeNode)
	traverse = func(node *TreeNode) {
		if node == nil {
			return
		}
		traverse(node.Right)
		node.Val += sum
		sum = node.Val
		traverse(node.Left)
	}
	traverse(root)
	return root
}

// 21. Count Nodes Equal to Sum of Descendants
// Postorder, count nodes where node.Val == sum(left+right).
func CountNodesEqualToSumOfDescendants(root *TreeNode) int {
	count := 0
	var dfs func(*TreeNode) int
	dfs = func(node *TreeNode) int {
		if node == nil {
			return 0
		}
		left := dfs(node.Left)
		right := dfs(node.Right)
		if node.Val == left+right && (node.Left != nil || node.Right != nil) {
			count++
		}
		return node.Val + left + right
	}
	dfs(root)
	return count
}

// 22. Flatten BST to Sorted Doubly Linked List
// Inorder traversal, link nodes as doubly linked list.
func FlattenBSTToDLL(root *TreeNode) *TreeNode {
	var prev, head *TreeNode
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		if prev != nil {
			prev.Right = node
			node.Left = prev
		} else {
			head = node
		}
		prev = node
		inorder(node.Right)
	}
	inorder(root)
	return head
}

// 23. Find the Minimum Absolute Difference in BST
// Inorder traversal, track min difference.
func GetMinimumDifference(root *TreeNode) int {
	minDiff := 1<<31 - 1
	var prev *TreeNode
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		if prev != nil && node.Val-prev.Val < minDiff {
			minDiff = node.Val - prev.Val
		}
		prev = node
		inorder(node.Right)
	}
	inorder(root)
	return minDiff
}

// 24. Predecessor and Successor in BST
// Find predecessor (max in left subtree) and successor (min in right subtree).
func FindPreSuc(root *TreeNode, key int) (pre, suc *TreeNode) {
	for root != nil {
		if root.Val < key {
			pre = root
			root = root.Right
		} else if root.Val > key {
			suc = root
			root = root.Left
		} else {
			if root.Left != nil {
				tmp := root.Left
				for tmp.Right != nil {
					tmp = tmp.Right
				}
				pre = tmp
			}
			if root.Right != nil {
				tmp := root.Right
				for tmp.Left != nil {
					tmp = tmp.Left
				}
				suc = tmp
			}
			break
		}
	}
	return
}

// 25. Lowest Common Ancestor in BST (Iterative)
func LowestCommonAncestorIterative(root, p, q *TreeNode) *TreeNode {
	for root != nil {
		if p.Val < root.Val && q.Val < root.Val {
			root = root.Left
		} else if p.Val > root.Val && q.Val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return nil
}

// 26. Insert into a BST (Iterative)
func InsertIntoBSTIterative(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val: val}
	}
	curr := root
	for {
		if val < curr.Val {
			if curr.Left == nil {
				curr.Left = &TreeNode{Val: val}
				break
			}
			curr = curr.Left
		} else {
			if curr.Right == nil {
				curr.Right = &TreeNode{Val: val}
				break
			}
			curr = curr.Right
		}
	}
	return root
}

// 27. Delete Node in BST (Iterative)
func DeleteNodeIterative(root *TreeNode, key int) *TreeNode {
	dummy := &TreeNode{Right: root}
	parent, curr := dummy, root
	isLeft := false
	for curr != nil && curr.Val != key {
		parent = curr
		if key < curr.Val {
			curr = curr.Left
			isLeft = true
		} else {
			curr = curr.Right
			isLeft = false
		}
	}
	if curr == nil {
		return dummy.Right
	}
	var child *TreeNode
	if curr.Left == nil {
		child = curr.Right
	} else if curr.Right == nil {
		child = curr.Left
	} else {
		p, s := curr, curr.Right
		for s.Left != nil {
			p = s
			s = s.Left
		}
		if p != curr {
			p.Left = s.Right
			s.Right = curr.Right
		}
		s.Left = curr.Left
		child = s
	}
	if isLeft {
		parent.Left = child
	} else {
		parent.Right = child
	}
	return dummy.Right
}

// 28. Validate BST Preorder Sequence
func VerifyPreorder(preorder []int) bool {
	stack := []int{}
	lower := -1 << 31
	for _, v := range preorder {
		if v < lower {
			return false
		}
		for len(stack) > 0 && v > stack[len(stack)-1] {
			lower = stack[len(stack)-1]
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, v)
	}
	return true
}

// 29. Find K Closest Elements in BST
func KClosestValues(root *TreeNode, target float64, k int) []int {
	var inorder func(*TreeNode)
	vals := []int{}
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		vals = append(vals, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	sort.Slice(vals, func(i, j int) bool {
		return absFloat(float64(vals[i])-target) < absFloat(float64(vals[j])-target)
	})
	return vals[:k]
}

// 30. Merge Two BSTs
func MergeTwoBSTs(root1, root2 *TreeNode) []int {
	arr1, arr2 := []int{}, []int{}
	var inorder func(*TreeNode, *[]int)
	inorder = func(node *TreeNode, arr *[]int) {
		if node == nil {
			return
		}
		inorder(node.Left, arr)
		*arr = append(*arr, node.Val)
		inorder(node.Right, arr)
	}
	inorder(root1, &arr1)
	inorder(root2, &arr2)
	res := []int{}
	i, j := 0, 0
	for i < len(arr1) && j < len(arr2) {
		if arr1[i] < arr2[j] {
			res = append(res, arr1[i])
			i++
		} else {
			res = append(res, arr2[j])
			j++
		}
	}
	res = append(res, arr1[i:]...)
	res = append(res, arr2[j:]...)
	return res
}

// 31. BST Iterator (Inorder Traversal)
// See BSTIterator above.

// 32. Balanced BST from Sorted Array
// See SortedArrayToBST above.

// 33. BST to Sorted Doubly Linked List
// See FlattenBSTToDLL above.

// 34. Count Smaller Numbers After Self (using BST)
type CountNode struct {
	Val, Count, LeftCount int
	Left, Right           *CountNode
}

func CountSmaller(nums []int) []int {
	res := make([]int, len(nums))
	var root *CountNode
	for i := len(nums) - 1; i >= 0; i-- {
		root, res[i] = insertCount(root, nums[i])
	}
	return res
}

func insertCount(node *CountNode, val int) (*CountNode, int) {
	if node == nil {
		return &CountNode{Val: val, Count: 1}, 0
	}
	if val < node.Val {
		var cnt int
		node.Left, cnt = insertCount(node.Left, val)
		node.LeftCount++
		return node, cnt
	} else {
		var cnt int
		node.Right, cnt = insertCount(node.Right, val)
		cnt += node.LeftCount
		if val > node.Val {
			cnt += node.Count
		}
		if val == node.Val {
			node.Count++
		}
		return node, cnt
	}
}

// 35. Find if there exists two elements in BST such that their sum is equal to given target
// See FindTarget above.

// 36. Recover BST with Two Swapped Nodes (Iterative)
func RecoverTreeIterative(root *TreeNode) {
	stack := []*TreeNode{}
	var first, second, prev *TreeNode
	curr := root
	for len(stack) > 0 || curr != nil {
		for curr != nil {
			stack = append(stack, curr)
			curr = curr.Left
		}
		curr = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if prev != nil && curr.Val < prev.Val {
			if first == nil {
				first = prev
			}
			second = curr
		}
		prev = curr
		curr = curr.Right
	}
	if first != nil && second != nil {
		first.Val, second.Val = second.Val, first.Val
	}
}

// 37. Binary Search Tree Iterator II
// Like BSTIterator, but supports prev().
type BSTIteratorII struct {
	stack []*TreeNode
	vals  []int
	idx   int
}

func ConstructorII(root *TreeNode) BSTIteratorII {
	it := BSTIteratorII{}
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		it.vals = append(it.vals, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	it.idx = -1
	return it
}

func (it *BSTIteratorII) Next() int {
	it.idx++
	return it.vals[it.idx]
}

func (it *BSTIteratorII) HasNext() bool {
	return it.idx+1 < len(it.vals)
}

func (it *BSTIteratorII) Prev() int {
	it.idx--
	return it.vals[it.idx]
}

func (it *BSTIteratorII) HasPrev() bool {
	return it.idx > 0
}

// 38. Construct BST from Level Order Traversal
func BstFromLevelOrder(arr []int) *TreeNode {
	if len(arr) == 0 {
		return nil
	}
	root := &TreeNode{Val: arr[0]}
	for _, v := range arr[1:] {
		InsertIntoBST(root, v)
	}
	return root
}

// 39. Find Median in BST
func FindMedian(root *TreeNode) float64 {
	vals := []int{}
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		vals = append(vals, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	n := len(vals)
	if n%2 == 1 {
		return float64(vals[n/2])
	}
	return float64(vals[n/2-1]+vals[n/2]) / 2.0
}

// 40. Sum of Nodes in BST within Range
// See RangeSumBST above.

// 41. Count Nodes in Complete BST
func CountNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	lh, rh := 0, 0
	l, r := root, root
	for l != nil {
		lh++
		l = l.Left
	}
	for r != nil {
		rh++
		r = r.Right
	}
	if lh == rh {
		return (1 << lh) - 1
	}
	return 1 + CountNodes(root.Left) + CountNodes(root.Right)
}

// 42. Count Nodes with One Child in BST
func CountOneChildNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left := CountOneChildNodes(root.Left)
	right := CountOneChildNodes(root.Right)
	if (root.Left == nil) != (root.Right == nil) {
		return 1 + left + right
	}
	return left + right
}

// 43. Count Leaf Nodes in BST
func CountLeafNodes(root *TreeNode) int {
	if root == nil {
		return 0
	}
	if root.Left == nil && root.Right == nil {
		return 1
	}
	return CountLeafNodes(root.Left) + CountLeafNodes(root.Right)
}

// 44. Insert Duplicate Node in BST
func InsertDuplicateNode(root *TreeNode) {
	if root == nil {
		return
	}
	InsertDuplicateNode(root.Left)
	InsertDuplicateNode(root.Right)
	dup := &TreeNode{Val: root.Val, Left: root.Left}
	root.Left = dup
}

// 45. Largest BST in Binary Tree
type Info struct {
	size, min, max int
	isBST          bool
}

func LargestBSTSubtree(root *TreeNode) int {
	maxSize := 0
	var postorder func(*TreeNode) Info
	postorder = func(node *TreeNode) Info {
		if node == nil {
			return Info{0, 1<<31 - 1, -1 << 31, true}
		}
		l := postorder(node.Left)
		r := postorder(node.Right)
		if l.isBST && r.isBST && node.Val > l.max && node.Val < r.min {
			size := l.size + r.size + 1
			if size > maxSize {
				maxSize = size
			}
			return Info{size, minInt(node.Val, l.min), maxInt(node.Val, r.max), true}
		}
		return Info{0, 0, 0, false}
	}
	postorder(root)
	return maxSize
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// 46. Convert BST to Sorted Array
func BSTToSortedArray(root *TreeNode) []int {
	res := []int{}
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		res = append(res, node.Val)
		inorder(node.Right)
	}
	inorder(root)
	return res
}

// 47. Range Sum Query BST
// See RangeSumBST above.

// 48. Construct BST from Postorder Traversal
func BstFromPostorder(postorder []int) *TreeNode {
	idx := len(postorder) - 1
	var helper func(int) *TreeNode
	helper = func(bound int) *TreeNode {
		if idx < 0 || postorder[idx] < bound {
			return nil
		}
		root := &TreeNode{Val: postorder[idx]}
		idx--
		root.Right = helper(root.Val)
		root.Left = helper(bound)
		return root
	}
	return helper(-1 << 31)
}

// 49. Find Kth Smallest Element in a BST (Iterative)
func KthSmallestIterative(root *TreeNode, k int) int {
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

// 50. Find LCA of Two Nodes in BST (Iterative)
// See LowestCommonAncestorIterative above.
