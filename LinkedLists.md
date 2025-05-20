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
