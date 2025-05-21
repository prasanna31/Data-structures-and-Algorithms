from collections import Counter, defaultdict
from collections import deque
from collections import Counter
from bisect import bisect_left, bisect_right, insort
from bisect import bisect_right, insort
from bisect import bisect_left, bisect_right, insort
import heapq
from functools import cmp_to_key
import bisect


class ListNode:

    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random  # Only used if the problem requires it


# 1. Reverse a Linked List
def reverse_list(head):
    prev = None  # Previous node starts as None
    curr = head  # Current node starts as head
    while curr:  # Traverse the list
        nxt = curr.next  # Store next node
        curr.next = prev  # Reverse pointer
        prev = curr  # Move prev forward
        curr = nxt  # Move curr forward
    return prev  # New head is prev


# 2. Detect Cycle in a Linked List
def has_cycle(head):
    slow = fast = head  # Two pointers
    while fast and fast.next:  # Traverse while fast can move
        slow = slow.next  # Move slow by 1
        fast = fast.next.next  # Move fast by 2
        if slow == fast:  # Cycle detected
            return True
    return False  # No cycle


# 3. Find the Middle of a Linked List
def middle_node(head):
    slow = fast = head  # Two pointers
    while fast and fast.next:  # Move fast twice as fast as slow
        slow = slow.next  # Move slow by 1
        fast = fast.next.next  # Move fast by 2
    return slow  # Slow is at middle


# 4. Merge Two Sorted Linked Lists
def merge_two_lists(l1, l2):
    dummy = ListNode(0)  # Dummy node to start merged list
    curr = dummy  # Pointer to build merged list
    while l1 and l2:  # While both lists have nodes
        if l1.val < l2.val:  # l1 smaller
            curr.next = l1  # Link l1 node
            l1 = l1.next  # Move l1 forward
        else:
            curr.next = l2  # Link l2 node
            l2 = l2.next  # Move l2 forward
        curr = curr.next  # Move curr forward
    curr.next = l1 or l2  # Attach remaining nodes
    return dummy.next  # Return merged list head


# 5. Remove Nth Node From End of List
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)  # Dummy node before head
    fast = slow = dummy  # Two pointers
    for _ in range(n):  # Move fast n steps ahead
        fast = fast.next
    while fast.next:  # Move both until fast at end
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next  # Remove nth node
    return dummy.next  # Return new head


# 6. Delete a Node in a Linked List (Given only access to that node)
def delete_node(node):
    node.val = node.next.val  # Copy next node's value
    node.next = node.next.next  # Skip next node


# 7. Intersection of Two Linked Lists
def get_intersection_node(headA, headB):
    a, b = headA, headB  # Two pointers
    while a != b:  # Traverse until they meet
        a = a.next if a else headB  # Switch to other list at end
        b = b.next if b else headA
    return a  # Intersection node or None


# 8. Check if Linked List is Palindrome
def is_palindrome(head):
    vals = []  # Store values
    curr = head
    while curr:  # Traverse list
        vals.append(curr.val)  # Add value
        curr = curr.next
    return vals == vals[::-1]  # Check palindrome


# 9. Copy List with Random Pointer
def copy_random_list(head):
    if not head:
        return None  # Empty list
    old_to_new = {}  # Map old nodes to new nodes
    curr = head
    while curr:  # First pass: copy nodes
        old_to_new[curr] = ListNode(curr.val)
        curr = curr.next
    curr = head
    while curr:  # Second pass: set next and random
        old_to_new[curr].next = old_to_new.get(curr.next)
        old_to_new[curr].random = old_to_new.get(curr.random)
        curr = curr.next
    return old_to_new[head]  # Return new head


# 10. Remove Duplicates from Sorted Linked List
def delete_duplicates(head):
    curr = head  # Pointer to traverse
    while curr and curr.next:  # While next exists
        if curr.val == curr.next.val:  # Duplicate found
            curr.next = curr.next.next  # Skip duplicate
        else:
            curr = curr.next  # Move forward
    return head  # Return head


    # 11. Reverse Nodes in k-Group
def reverse_k_group(head, k):
    # Reverse every k nodes in the list
    dummy = ListNode(0, head)
    group_prev = dummy

    def get_kth(curr, k):
        # Find the kth node from curr
        while curr and k > 0:
            curr = curr.next
            k -= 1
        return curr

    while True:
        kth = get_kth(group_prev, k)
        if not kth:
            break
        group_next = kth.next
        # Reverse group
        prev, curr = kth.next, group_prev.next
        while curr != group_next:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        tmp = group_prev.next
        group_prev.next = kth
        group_prev = tmp
    return dummy.next


# 12. Rotate List
def rotate_right(head, k):
    # Rotate the list to the right by k places
    if not head or not head.next or k == 0:
        return head
    # Compute length and connect tail to head
    old_tail = head
    length = 1
    while old_tail.next:
        old_tail = old_tail.next
        length += 1
    old_tail.next = head
    # Find new tail and new head
    k = k % length
    new_tail = head
    for _ in range(length - k - 1):
        new_tail = new_tail.next
    new_head = new_tail.next
    new_tail.next = None
    return new_head


# 13. Flatten a Multilevel Doubly Linked List
class DoublyListNode:

    def __init__(self, val=0, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child


def flatten(head):
    # Flatten a multilevel doubly linked list
    if not head:
        return head
    pseudo_head = DoublyListNode(0, None, head, None)
    prev = pseudo_head
    stack = [head]
    while stack:
        curr = stack.pop()
        prev.next = curr
        curr.prev = prev
        if curr.next:
            stack.append(curr.next)
        if curr.child:
            stack.append(curr.child)
            curr.child = None
        prev = curr
    pseudo_head.next.prev = None
    return pseudo_head.next


# 14. Swap Nodes in Pairs
def swap_pairs(head):
    # Swap every two adjacent nodes
    dummy = ListNode(0, head)
    prev = dummy
    while head and head.next:
        first = head
        second = head.next
        # Swap
        prev.next = second
        first.next = second.next
        second.next = first
        # Move pointers
        prev = first
        head = first.next
    return dummy.next


# 15. Add Two Numbers Represented by Linked Lists
def add_two_numbers(l1, l2):
    # Each node contains a single digit, digits stored in reverse order
    dummy = ListNode(0)
    curr = dummy
    carry = 0
    while l1 or l2 or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        total = v1 + v2 + carry
        carry = total // 10
        curr.next = ListNode(total % 10)
        curr = curr.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return dummy.next


# 16. Partition List Around a Value
def partition(head, x):
    # Partition list so all nodes < x come before nodes >= x
    before = before_head = ListNode(0)
    after = after_head = ListNode(0)
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    after.next = None
    before.next = after_head.next
    return before_head.next


# 17. Sort a Linked List using Merge Sort
def sort_list(head):
    # Sort linked list in O(n log n) time using merge sort
    if not head or not head.next:
        return head
    # Split list into halves
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    mid = slow.next
    slow.next = None
    left = sort_list(head)
    right = sort_list(mid)
    # Merge sorted halves
    dummy = ListNode(0)
    curr = dummy
    while left and right:
        if left.val < right.val:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    curr.next = left or right
    return dummy.next


# 18. Reorder List
def reorder_list(head):
    # Reorder list to: L0→Ln→L1→Ln-1→L2→Ln-2→...
    if not head or not head.next:
        return
    # Find middle
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    # Reverse second half
    second = slow.next
    prev = slow.next = None
    while second:
        tmp = second.next
        second.next = prev
        prev = second
        second = tmp
    # Merge two halves
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2


# 19. Remove Elements with Specific Value
def remove_elements(head, val):
    # Remove all nodes with value == val
    dummy = ListNode(0, head)
    curr = dummy
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return dummy.next


# 20. Convert Sorted List to Binary Search Tree
class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def sorted_list_to_bst(head):
    # Convert sorted linked list to height-balanced BST
    def find_size(node):
        size = 0
        while node:
            size += 1
            node = node.next
        return size

    def convert(l, r):
        nonlocal head
        if l > r:
            return None
        mid = (l + r) // 2
        left = convert(l, mid - 1)
        root = TreeNode(head.val)
        root.left = left
        head = head.next
        root.right = convert(mid + 1, r)
        return root

    size = find_size(head)
    return convert(0, size - 1)


    # 21. Find Start of Cycle in Linked List
def detect_cycle(head):
    # Returns the node where the cycle begins, or None if no cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Cycle detected, find entry
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None


# 22. Detect and Remove Loop in Linked List
def detect_and_remove_loop(head):
    # Detects and removes a cycle if present
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Loop detected
            break
    else:
        return  # No loop
    slow = head
    prev = None
    while slow != fast:
        prev = fast
        slow = slow.next
        fast = fast.next
    # Remove loop
    while fast.next != slow:
        fast = fast.next
    fast.next = None


# 23. Check if Two Linked Lists Intersect (with Cycle)
def lists_intersect(headA, headB):
    # Returns True if two lists intersect (even with cycles)
    def get_cycle_node(head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return slow
        return None

    def get_tail(head):
        while head and head.next:
            head = head.next
        return head

    cycleA = get_cycle_node(headA)
    cycleB = get_cycle_node(headB)
    if not cycleA and not cycleB:
        # No cycles: check tails
        return get_tail(headA) is get_tail(headB)
    if (cycleA and not cycleB) or (cycleB and not cycleA):
        return False
    # Both have cycles: check if cycles are the same
    ptr = cycleA
    while True:
        ptr = ptr.next
        if ptr == cycleA or ptr == cycleB:
            break
    return ptr == cycleB


# 24. Find Length of Loop in Linked List
def loop_length(head):
    # Returns length of cycle, or 0 if no cycle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Cycle detected, count length
            count = 1
            fast = fast.next
            while fast != slow:
                fast = fast.next
                count += 1
            return count
    return 0


# 25. Find the Node where Cycle Begins (Alternative)
def find_cycle_start(head):
    # Same as detect_cycle, for clarity
    return detect_cycle(head)


# 26. Implement Doubly Linked List Operations
class DoublyLinkedList:

    def __init__(self):
        self.head = None

    def append(self, val):
        node = DoublyListNode(val)
        if not self.head:
            self.head = node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = node
        node.prev = curr

    def prepend(self, val):
        node = DoublyListNode(val)
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node

    def delete(self, val):
        curr = self.head
        while curr:
            if curr.val == val:
                if curr.prev:
                    curr.prev.next = curr.next
                else:
                    self.head = curr.next
                if curr.next:
                    curr.next.prev = curr.prev
                return
            curr = curr.next


# 27. Insert Node into Sorted Doubly Linked List
def insert_sorted_doubly(head, val):
    node = DoublyListNode(val)
    if not head or val < head.val:
        node.next = head
        if head:
            head.prev = node
        return node
    curr = head
    while curr.next and curr.next.val < val:
        curr = curr.next
    node.next = curr.next
    if curr.next:
        curr.next.prev = node
    curr.next = node
    node.prev = curr
    return head


# 28. Convert Binary Tree to Doubly Linked List
def tree_to_doubly_list(root):
    # Converts BST to sorted circular doubly linked list
    if not root:
        return None

    def helper(node):
        nonlocal last, first
        if not node:
            return
        helper(node.left)
        if last:
            last.next = node
            node.prev = last
        else:
            first = node
        last = node
        helper(node.right)

    first = last = None
    helper(root)
    # Make it circular
    if first and last:
        first.prev = last
        last.next = first
    return first


# 29. Implement Circular Linked List Insertion
class CircularListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def insert_circular(head, val):
    node = CircularListNode(val)
    if not head:
        node.next = node
        return node
    curr = head
    while curr.next != head:
        curr = curr.next
    curr.next = node
    node.next = head
    return head


# 30. Split Circular Linked List into Two Halves
def split_circular_list(head):
    if not head or head.next == head:
        return head, None
    slow = fast = head
    while fast.next != head and fast.next.next != head:
        slow = slow.next
        fast = fast.next.next
    head1 = head
    head2 = slow.next
    slow.next = head1
    curr = head2
    while curr.next != head:
        curr = curr.next
    curr.next = head2
    return head1, head2


# 31. Merge k Sorted Linked Lists
def merge_k_lists(lists):
    # Uses heap to merge k sorted lists
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


# 32. Clone a Linked List with Random Pointer
def clone_random_list(head):
    # O(1) space, O(n) time
    if not head:
        return None
    # Step 1: Clone nodes and insert next to originals
    curr = head
    while curr:
        nxt = curr.next
        curr.next = ListNode(curr.val, nxt)
        curr = nxt
    # Step 2: Set random pointers
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next
    # Step 3: Separate lists
    curr = head
    clone_head = head.next
    while curr:
        clone = curr.next
        curr.next = clone.next
        clone.next = clone.next.next if clone.next else None
        curr = curr.next
    return clone_head


# 33. Remove Zero Sum Consecutive Nodes
def remove_zero_sum_sublists(head):
    dummy = ListNode(0)
    dummy.next = head
    prefix = 0
    seen = {0: dummy}
    curr = head
    while curr:
        prefix += curr.val
        if prefix in seen:
            prev = seen[prefix]
            node = prev.next
            sum2 = prefix
            while node != curr:
                sum2 += node.val
                seen.pop(sum2, None)
                node = node.next
            prev.next = curr.next
        else:
            seen[prefix] = curr
        curr = curr.next
    return dummy.next


# 34. Sort a Linked List of 0s, 1s, and 2s
def sort_012_list(head):
    # Dutch National Flag for linked list
    zeroD = ListNode(0)
    oneD = ListNode(0)
    twoD = ListNode(0)
    zero, one, two = zeroD, oneD, twoD
    curr = head
    while curr:
        if curr.val == 0:
            zero.next = curr
            zero = zero.next
        elif curr.val == 1:
            one.next = curr
            one = one.next
        else:
            two.next = curr
            two = two.next
        curr = curr.next
    zero.next = oneD.next or twoD.next
    one.next = twoD.next
    two.next = None
    return zeroD.next


# 35. Delete Middle Node of Linked List
def delete_middle(head):
    if not head or not head.next:
        return None
    slow = fast = head
    prev = None
    while fast and fast.next:
        prev = slow
        slow = slow.next
        fast = fast.next.next
    prev.next = slow.next
    return head
