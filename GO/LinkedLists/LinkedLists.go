package linkedLists

// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}

// 1. Reverse a Linked List
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		nextTemp := curr.Next
		curr.Next = prev
		prev = curr
		curr = nextTemp
	}
	return prev
}

// 2. Detect Cycle in a Linked List (Floyd's Tortoise and Hare)
func hasCycle(head *ListNode) bool {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			return true
		}
	}
	return false
}

// 3. Find the Middle of a Linked List
func findMiddle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}

// 4. Merge Two Sorted Linked Lists
func mergeTwoLists(l1, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	curr := dummy
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			curr.Next = l1
			l1 = l1.Next
		} else {
			curr.Next = l2
			l2 = l2.Next
		}
		curr = curr.Next
	}
	if l1 != nil {
		curr.Next = l1
	} else {
		curr.Next = l2
	}
	return dummy.Next
}

// 5. Remove Nth Node From End of List
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummy := &ListNode{Next: head}
	first, second := dummy, dummy
	for i := 0; i <= n; i++ {
		first = first.Next
	}
	for first != nil {
		first = first.Next
		second = second.Next
	}
	second.Next = second.Next.Next
	return dummy.Next
}

// 6. Delete a Node in a Linked List (Given only access to that node)
func deleteNode(node *ListNode) {
	if node == nil || node.Next == nil {
		return
	}
	node.Val = node.Next.Val
	node.Next = node.Next.Next
}

// 7. Intersection of Two Linked Lists
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	a, b := headA, headB
	for a != b {
		if a == nil {
			a = headB
		} else {
			a = a.Next
		}
		if b == nil {
			b = headA
		} else {
			b = b.Next
		}
	}
	return a
}

// 8. Check if Linked List is Palindrome
func isPalindrome(head *ListNode) bool {
	vals := []int{}
	for head != nil {
		vals = append(vals, head.Val)
		head = head.Next
	}
	i, j := 0, len(vals)-1
	for i < j {
		if vals[i] != vals[j] {
			return false
		}
		i++
		j--
	}
	return true
}

// 9. Copy List with Random Pointer
type RandomListNode struct {
	Val    int
	Next   *RandomListNode
	Random *RandomListNode
}

func copyRandomList(head *RandomListNode) *RandomListNode {
	if head == nil {
		return nil
	}
	oldToNew := make(map[*RandomListNode]*RandomListNode)
	curr := head
	for curr != nil {
		oldToNew[curr] = &RandomListNode{Val: curr.Val}
		curr = curr.Next
	}
	curr = head
	for curr != nil {
		oldToNew[curr].Next = oldToNew[curr.Next]
		oldToNew[curr].Random = oldToNew[curr.Random]
		curr = curr.Next
	}
	return oldToNew[head]
}

// 10. Remove Duplicates from Sorted Linked List
func deleteDuplicates(head *ListNode) *ListNode {
	curr := head
	for curr != nil && curr.Next != nil {
		if curr.Val == curr.Next.Val {
			curr.Next = curr.Next.Next
		} else {
			curr = curr.Next
		}
	}
	return head
}

/* 11. Reverse Nodes in k-Group */
func reverseKGroup(head *ListNode, k int) *ListNode {
	dummy := &ListNode{Next: head}
	groupPrev := dummy

	for {
		kth := groupPrev
		for i := 0; i < k && kth != nil; i++ {
			kth = kth.Next
		}
		if kth == nil {
			break
		}
		groupNext := kth.Next

		// Reverse group
		prev, curr := groupNext, groupPrev.Next
		for curr != groupNext {
			tmp := curr.Next
			curr.Next = prev
			prev = curr
			curr = tmp
		}
		tmp := groupPrev.Next
		groupPrev.Next = kth
		groupPrev = tmp
	}
	return dummy.Next
}

/* 12. Rotate List */
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	// Compute length and make it circular
	oldTail := head
	length := 1
	for oldTail.Next != nil {
		oldTail = oldTail.Next
		length++
	}
	oldTail.Next = head

	k = k % length
	if k == 0 {
		oldTail.Next = nil
		return head
	}
	newTail := head
	for i := 0; i < length-k-1; i++ {
		newTail = newTail.Next
	}
	newHead := newTail.Next
	newTail.Next = nil
	return newHead
}

/* 13. Flatten a Multilevel Doubly Linked List */
type DoublyNode struct {
	Val   int
	Prev  *DoublyNode
	Next  *DoublyNode
	Child *DoublyNode
}

func flatten(head *DoublyNode) *DoublyNode {
	curr := head
	for curr != nil {
		if curr.Child != nil {
			next := curr.Next
			child := flatten(curr.Child)
			curr.Next = child
			child.Prev = curr
			curr.Child = nil

			// Find tail of child
			tail := child
			for tail.Next != nil {
				tail = tail.Next
			}
			tail.Next = next
			if next != nil {
				next.Prev = tail
			}
		}
		curr = curr.Next
	}
	return head
}

/* 14. Swap Nodes in Pairs */
func swapPairs(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	curr := dummy
	for curr.Next != nil && curr.Next.Next != nil {
		first := curr.Next
		second := curr.Next.Next
		first.Next = second.Next
		second.Next = first
		curr.Next = second
		curr = first
	}
	return dummy.Next
}

/* 15. Add Two Numbers Represented by Linked Lists */
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	dummy := &ListNode{}
	curr := dummy
	carry := 0
	for l1 != nil || l2 != nil || carry != 0 {
		sum := carry
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}
		curr.Next = &ListNode{Val: sum % 10}
		carry = sum / 10
		curr = curr.Next
	}
	return dummy.Next
}

/* 16. Partition List Around a Value */
func partition(head *ListNode, x int) *ListNode {
	before := &ListNode{}
	after := &ListNode{}
	beforePtr, afterPtr := before, after
	for head != nil {
		if head.Val < x {
			beforePtr.Next = head
			beforePtr = beforePtr.Next
		} else {
			afterPtr.Next = head
			afterPtr = afterPtr.Next
		}
		head = head.Next
	}
	afterPtr.Next = nil
	beforePtr.Next = after.Next
	return before.Next
}

/* 17. Sort a Linked List using Merge Sort */
func sortList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// Split list
	mid := findMiddle(head)
	right := mid.Next
	mid.Next = nil
	left := sortList(head)
	right = sortList(right)
	return mergeTwoLists(left, right)
}

/* 18. Reorder List */
func reorderList(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	// Find middle
	mid := findMiddle(head)
	l1 := head
	l2 := mid.Next
	mid.Next = nil
	// Reverse second half
	l2 = reverseList(l2)
	// Merge
	for l1 != nil && l2 != nil {
		l1Next := l1.Next
		l2Next := l2.Next
		l1.Next = l2
		if l1Next == nil {
			break
		}
		l2.Next = l1Next
		l1 = l1Next
		l2 = l2Next
	}
}

/* 19. Remove Elements with Specific Value */
func removeElements(head *ListNode, val int) *ListNode {
	dummy := &ListNode{Next: head}
	curr := dummy
	for curr.Next != nil {
		if curr.Next.Val == val {
			curr.Next = curr.Next.Next
		} else {
			curr = curr.Next
		}
	}
	return dummy.Next
}

/* 20. Convert Sorted List to Binary Search Tree */
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func sortedListToBST(head *ListNode) *TreeNode {
	if head == nil {
		return nil
	}
	if head.Next == nil {
		return &TreeNode{Val: head.Val}
	}
	// Find middle
	prev := &ListNode{Next: head}
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	prev.Next = nil
	root := &TreeNode{Val: slow.Val}
	root.Left = sortedListToBST(head)
	root.Right = sortedListToBST(slow.Next)
	return root
}

/* 21. Find Start of Cycle in Linked List */
func detectCycle(head *ListNode) *ListNode {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			ptr := head
			for ptr != slow {
				ptr = ptr.Next
				slow = slow.Next
			}
			return ptr
		}
	}
	return nil
}

/* 22. Detect and Remove Loop in Linked List */
func detectAndRemoveLoop(head *ListNode) {
	if head == nil || head.Next == nil {
		return
	}
	slow, fast := head, head
	loopExists := false
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			loopExists = true
			break
		}
	}
	if !loopExists {
		return
	}
	slow = head
	var prev *ListNode
	for slow != fast {
		prev = fast
		slow = slow.Next
		fast = fast.Next
	}
	// Remove loop
	if prev != nil {
		prev.Next = nil
	}
}

/* 23. Check if Two Linked Lists Intersect (with Cycle) */
func listsIntersect(headA, headB *ListNode) bool {
	cycleA := detectCycle(headA)
	cycleB := detectCycle(headB)
	if cycleA == nil && cycleB == nil {
		// No cycles, check tail
		tailA, tailB := headA, headB
		for tailA != nil && tailA.Next != nil {
			tailA = tailA.Next
		}
		for tailB != nil && tailB.Next != nil {
			tailB = tailB.Next
		}
		return tailA == tailB && tailA != nil
	}
	if (cycleA == nil) != (cycleB == nil) {
		return false
	}
	// Both have cycles, check if cycles are the same
	ptr := cycleA
	for {
		if ptr == cycleB {
			return true
		}
		ptr = ptr.Next
		if ptr == cycleA {
			break
		}
	}
	return false
}

/* 24. Find Length of Loop in Linked List */
func loopLength(head *ListNode) int {
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		if slow == fast {
			// Count loop length
			count := 1
			ptr := slow.Next
			for ptr != slow {
				count++
				ptr = ptr.Next
			}
			return count
		}
	}
	return 0
}

/* 25. Find the Node where Cycle Begins */
func findCycleStart(head *ListNode) *ListNode {
	return detectCycle(head)
}

// 26. Implement Doubly Linked List Operations
type DoublyLinkedList struct {
	Head *DoublyNode
	Tail *DoublyNode
}

func (dll *DoublyLinkedList) Append(val int) {
	node := &DoublyNode{Val: val}
	if dll.Head == nil {
		dll.Head = node
		dll.Tail = node
		return
	}
	dll.Tail.Next = node
	node.Prev = dll.Tail
	dll.Tail = node
}

func (dll *DoublyLinkedList) Prepend(val int) {
	node := &DoublyNode{Val: val}
	if dll.Head == nil {
		dll.Head = node
		dll.Tail = node
		return
	}
	node.Next = dll.Head
	dll.Head.Prev = node
	dll.Head = node
}

func (dll *DoublyLinkedList) Delete(val int) {
	curr := dll.Head
	for curr != nil {
		if curr.Val == val {
			if curr.Prev != nil {
				curr.Prev.Next = curr.Next
			} else {
				dll.Head = curr.Next
			}
			if curr.Next != nil {
				curr.Next.Prev = curr.Prev
			} else {
				dll.Tail = curr.Prev
			}
			return
		}
		curr = curr.Next
	}
}

// 27. Insert Node into Sorted Doubly Linked List
func insertSortedDoubly(head *DoublyNode, val int) *DoublyNode {
	node := &DoublyNode{Val: val}
	if head == nil {
		return node
	}
	if val < head.Val {
		node.Next = head
		head.Prev = node
		return node
	}
	curr := head
	for curr.Next != nil && curr.Next.Val < val {
		curr = curr.Next
	}
	node.Next = curr.Next
	if curr.Next != nil {
		curr.Next.Prev = node
	}
	curr.Next = node
	node.Prev = curr
	return head
}

// 28. Convert Binary Tree to Doubly Linked List (Inorder)
func treeToDoublyList(root *TreeNode) *DoublyNode {
	var head, prev *DoublyNode
	var inorder func(*TreeNode)
	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}
		inorder(node.Left)
		curr := &DoublyNode{Val: node.Val}
		if prev == nil {
			head = curr
		} else {
			prev.Next = curr
			curr.Prev = prev
		}
		prev = curr
		inorder(node.Right)
	}
	inorder(root)
	return head
}

// 29. Implement Circular Linked List Insertion
type CircularNode struct {
	Val  int
	Next *CircularNode
}

func insertCircular(head *CircularNode, val int) *CircularNode {
	node := &CircularNode{Val: val}
	if head == nil {
		node.Next = node
		return node
	}
	curr := head
	for curr.Next != head {
		curr = curr.Next
	}
	curr.Next = node
	node.Next = head
	return head
}

// 30. Split Circular Linked List into Two Halves
func splitCircularList(head *CircularNode) (*CircularNode, *CircularNode) {
	if head == nil || head.Next == head {
		return head, nil
	}
	slow, fast := head, head
	for fast.Next != head && fast.Next.Next != head {
		slow = slow.Next
		fast = fast.Next.Next
	}
	head1 := head
	head2 := slow.Next
	slow.Next = head1
	if fast.Next.Next == head {
		fast = fast.Next
	}
	fast.Next = head2
	return head1, head2
}

// 31. Merge k Sorted Linked Lists
type Item struct {
	node *ListNode
	idx  int
}

type MinHeap []Item

func (h MinHeap) Len() int            { return len(h) }
func (h MinHeap) Less(i, j int) bool  { return h[i].node.Val < h[j].node.Val }
func (h MinHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *MinHeap) Push(x interface{}) { *h = append(*h, x.(Item)) }
func (h *MinHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func mergeKLists(lists []*ListNode) *ListNode {
	// Manual heap implementation to avoid importing "container/heap"
	h := MinHeap{}
	for i, node := range lists {
		if node != nil {
			h = append(h, Item{node, i})
		}
	}
	// heapify
	for i := len(h)/2 - 1; i >= 0; i-- {
		heapify(h, i)
	}
	dummy := &ListNode{}
	curr := dummy
	for len(h) > 0 {
		minIdx := 0
		for i := 1; i < len(h); i++ {
			if h[i].node.Val < h[minIdx].node.Val {
				minIdx = i
			}
		}
		item := h[minIdx]
		curr.Next = item.node
		curr = curr.Next
		if item.node.Next != nil {
			h[minIdx].node = item.node.Next
		} else {
			h = append(h[:minIdx], h[minIdx+1:]...)
		}
	}
	return dummy.Next
}

func heapify(h MinHeap, i int) {
	n := len(h)
	smallest := i
	l, r := 2*i+1, 2*i+2
	if l < n && h[l].node.Val < h[smallest].node.Val {
		smallest = l
	}
	if r < n && h[r].node.Val < h[smallest].node.Val {
		smallest = r
	}
	if smallest != i {
		h[i], h[smallest] = h[smallest], h[i]
		heapify(h, smallest)
	}
}

// 32. Clone a Linked List with Random Pointer
type RandomNode struct {
	Val    int
	Next   *RandomNode
	Random *RandomNode
}

func cloneRandomList(head *RandomNode) *RandomNode {
	if head == nil {
		return nil
	}
	// Step 1: Insert cloned nodes
	curr := head
	for curr != nil {
		next := curr.Next
		clone := &RandomNode{Val: curr.Val}
		curr.Next = clone
		clone.Next = next
		curr = next
	}
	// Step 2: Set random pointers
	curr = head
	for curr != nil {
		if curr.Random != nil {
			curr.Next.Random = curr.Random.Next
		}
		curr = curr.Next.Next
	}
	// Step 3: Separate lists
	curr = head
	cloneHead := head.Next
	for curr != nil {
		clone := curr.Next
		curr.Next = clone.Next
		if clone.Next != nil {
			clone.Next = clone.Next.Next
		}
		curr = curr.Next
	}
	return cloneHead
}

// 33. Remove Zero Sum Consecutive Nodes
func removeZeroSumSublists(head *ListNode) *ListNode {
	dummy := &ListNode{Next: head}
	prefix := 0
	m := map[int]*ListNode{}
	for node := dummy; node != nil; node = node.Next {
		prefix += node.Val
		m[prefix] = node
	}
	prefix = 0
	for node := dummy; node != nil; node = node.Next {
		prefix += node.Val
		node.Next = m[prefix].Next
	}
	return dummy.Next
}

// 34. Sort a Linked List of 0s, 1s, and 2s
func sort012List(head *ListNode) *ListNode {
	zero, one, two := &ListNode{}, &ListNode{}, &ListNode{}
	z, o, t := zero, one, two
	curr := head
	for curr != nil {
		switch curr.Val {
		case 0:
			z.Next = curr
			z = z.Next
		case 1:
			o.Next = curr
			o = o.Next
		case 2:
			t.Next = curr
			t = t.Next
		}
		curr = curr.Next
	}
	t.Next = nil
	o.Next = two.Next
	z.Next = one.Next
	return zero.Next
}

// 35. Delete Middle Node of Linked List
func deleteMiddle(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return nil
	}
	slow, fast := head, head
	var prev *ListNode
	for fast != nil && fast.Next != nil {
		prev = slow
		slow = slow.Next
		fast = fast.Next.Next
	}
	prev.Next = slow.Next
	return head
}

// 36. Add One to a Number Represented as Linked List
func addOne(head *ListNode) *ListNode {
	head = reverseList(head)
	carry := 1
	curr := head
	for curr != nil && carry > 0 {
		sum := curr.Val + carry
		curr.Val = sum % 10
		carry = sum / 10
		if curr.Next == nil && carry > 0 {
			curr.Next = &ListNode{Val: carry}
			break
		}
		curr = curr.Next
	}
	return reverseList(head)
}

// 37. Check if Linked List is a Palindrome (Recursive)
func isPalindromeRecursive(head *ListNode) bool {
	var front *ListNode = head
	var check func(*ListNode) bool
	check = func(curr *ListNode) bool {
		if curr == nil {
			return true
		}
		if !check(curr.Next) {
			return false
		}
		if curr.Val != front.Val {
			return false
		}
		front = front.Next
		return true
	}
	return check(head)
}

// 38. Intersection Point of Two Linked Lists
func intersectionPoint(headA, headB *ListNode) *ListNode {
	a, b := headA, headB
	for a != b {
		if a == nil {
			a = headB
		} else {
			a = a.Next
		}
		if b == nil {
			b = headA
		} else {
			b = b.Next
		}
	}
	return a
}

// 39. Find the Length of a Linked List
func lengthOfList(head *ListNode) int {
	count := 0
	for head != nil {
		count++
		head = head.Next
	}
	return count
}

// 40. Detect Intersection Node of Y Shaped Linked Lists
func getYIntersection(headA, headB *ListNode) *ListNode {
	lenA, lenB := lengthOfList(headA), lengthOfList(headB)
	for lenA > lenB {
		headA = headA.Next
		lenA--
	}
	for lenB > lenA {
		headB = headB.Next
		lenB--
	}
	for headA != nil && headB != nil {
		if headA == headB {
			return headA
		}
		headA = headA.Next
		headB = headB.Next
	}
	return nil
}

// 41. Flatten a Linked List with Next and Bottom Pointers
type BottomNode struct {
	Val    int
	Next   *BottomNode
	Bottom *BottomNode
}

func flattenBottomList(root *BottomNode) *BottomNode {
	if root == nil || root.Next == nil {
		return root
	}
	root.Next = flattenBottomList(root.Next)
	root = mergeBottom(root, root.Next)
	return root
}

func mergeBottom(a, b *BottomNode) *BottomNode {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	var result *BottomNode
	if a.Val < b.Val {
		result = a
		result.Bottom = mergeBottom(a.Bottom, b)
	} else {
		result = b
		result.Bottom = mergeBottom(a, b.Bottom)
	}
	result.Next = nil
	return result
}

// 42. Remove All Occurrences of a Given Value
func removeAllOccurrences(head *ListNode, val int) *ListNode {
	dummy := &ListNode{Next: head}
	curr := dummy
	for curr.Next != nil {
		if curr.Next.Val == val {
			curr.Next = curr.Next.Next
		} else {
			curr = curr.Next
		}
	}
	return dummy.Next
}

func nthNode(head *ListNode, n int) *ListNode {
	curr := head
	i := 1
	for curr != nil && i < n {
		curr = curr.Next
		i++
	}
	if i == n && curr != nil {
		return curr
	}
	return nil
}

// 44. Convert Sorted Array to Balanced Linked List
func sortedArrayToList(arr []int) *ListNode {
	var build func(int, int) *ListNode
	build = func(l, r int) *ListNode {
		if l > r {
			return nil
		}
		mid := (l + r) / 2
		node := &ListNode{Val: arr[mid]}
		node.Next = build(mid+1, r)
		return node
	}
	return build(0, len(arr)-1)
}

// 45. Segregate Even and Odd Nodes in Linked List
func segregateEvenOdd(head *ListNode) *ListNode {
	even, odd := &ListNode{}, &ListNode{}
	e, o := even, odd
	curr := head
	for curr != nil {
		if curr.Val%2 == 0 {
			e.Next = curr
			e = e.Next
		} else {
			o.Next = curr
			o = o.Next
		}
		curr = curr.Next
	}
	e.Next = odd.Next
	o.Next = nil
	return even.Next
}

// 46. Reverse Alternate K Nodes in Linked List
func reverseAlternateK(head *ListNode, k int) *ListNode {
	curr := head
	var prev *ListNode
	count := 0
	for curr != nil && count < k {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
		count++
	}
	if head != nil {
		head.Next = curr
	}
	count = 0
	for curr != nil && count < k-1 {
		curr = curr.Next
		count++
	}
	if curr != nil {
		curr.Next = reverseAlternateK(curr.Next, k)
	}
	return prev
}

// 47. Delete Nodes Which Have Greater Value on Right
func deleteGreaterRight(head *ListNode) *ListNode {
	head = reverseList(head)
	maxVal := head.Val
	curr := head
	for curr.Next != nil {
		if curr.Next.Val < maxVal {
			curr.Next = curr.Next.Next
		} else {
			maxVal = curr.Next.Val
			curr = curr.Next
		}
	}
	return reverseList(head)
}

// 48. Rotate Doubly Linked List by N Nodes
func rotateDoubly(head *DoublyNode, n int) *DoublyNode {
	if head == nil || n == 0 {
		return head
	}
	curr := head
	count := 1
	for count < n && curr != nil {
		curr = curr.Next
		count++
	}
	if curr == nil {
		return head
	}
	nthNode := curr
	tail := curr
	for tail.Next != nil {
		tail = tail.Next
	}
	tail.Next = head
	head.Prev = tail
	head = nthNode.Next
	if head != nil {
		head.Prev = nil
	}
	nthNode.Next = nil
	return head
}

// 49. Copy a Linked List with Next and Random Pointer (Same as 32)
func copyListWithRandom(head *RandomNode) *RandomNode {
	return cloneRandomList(head)
}

// 50. Merge Two Sorted Circular Linked Lists
func mergeCircularLists(head1, head2 *CircularNode) *CircularNode {
	if head1 == nil {
		return head2
	}
	if head2 == nil {
		return head1
	}
	// Break the circle
	last1, last2 := head1, head2
	for last1.Next != head1 {
		last1 = last1.Next
	}
	for last2.Next != head2 {
		last2 = last2.Next
	}
	last1.Next = nil
	last2.Next = nil
	// Merge
	var mergedHead, mergedTail *CircularNode
	p1, p2 := head1, head2
	for p1 != nil && p2 != nil {
		var node *CircularNode
		if p1.Val < p2.Val {
			node = p1
			p1 = p1.Next
		} else {
			node = p2
			p2 = p2.Next
		}
		if mergedHead == nil {
			mergedHead = node
			mergedTail = node
		} else {
			mergedTail.Next = node
			mergedTail = node
		}
	}
	if p1 != nil {
		mergedTail.Next = p1
		for mergedTail.Next != nil {
			mergedTail = mergedTail.Next
		}
	}
	if p2 != nil {
		mergedTail.Next = p2
		for mergedTail.Next != nil {
			mergedTail = mergedTail.Next
		}
	}
	mergedTail.Next = mergedHead
	return mergedHead
}
