from collections import deque
from collections import deque
import heapq
from collections import Counter
import bisect
import itertools


# TreeNode class for binary tree problems
class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 1. Inorder Traversal
def inorderTraversal(root):
    # Left -> Root -> Right
    res, stack = [], []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        res.append(root.val)
        root = root.right
    return res


# Explanation: Use stack to traverse leftmost, then process node, then right.


# 2. Preorder Traversal
def preorderTraversal(root):
    # Root -> Left -> Right
    res, stack = [], []
    if root:
        stack.append(root)
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res


# Explanation: Stack, process node, push right then left.


# 3. Postorder Traversal
def postorderTraversal(root):
    # Left -> Right -> Root
    res, stack = [], []
    last = None
    while root or stack:
        if root:
            stack.append(root)
            root = root.left
        else:
            node = stack[-1]
            if node.right and last != node.right:
                root = node.right
            else:
                res.append(node.val)
                last = stack.pop()
    return res


# Explanation: Track last visited, process after children.


# 4. Level Order Traversal
def levelOrderTraversal(root):
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


# Explanation: BFS with queue, process level by level.


# 5. Zigzag Level Order Traversal
def zigzagLevelOrder(root):
    res = []
    if not root:
        return res
    q = deque([root])
    left_to_right = True
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        if not left_to_right:
            level.reverse()
        res.append(level)
        left_to_right = not left_to_right
    return res


# Explanation: Alternate reversing each level.


# 6. Maximum Depth of Binary Tree
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))


# Explanation: Recursively get max depth of left/right.


# 7. Minimum Depth of Binary Tree
def minDepth(root):
    if not root:
        return 0
    if not root.left:
        return 1 + minDepth(root.right)
    if not root.right:
        return 1 + minDepth(root.left)
    return 1 + min(minDepth(root.left), minDepth(root.right))


# Explanation: If one child is None, take the other.


# 8. Symmetric Tree
def isSymmetric(root):

    def isMirror(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)

    return isMirror(root, root)


# Explanation: Recursively check mirror symmetry.


# 9. Same Tree
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)


# Explanation: Recursively compare nodes.


# 10. Invert Binary Tree
def invertTree(root):
    if root:
        root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root


# Explanation: Swap left and right recursively.


# 11. Path Sum
def hasPathSum(root, targetSum):
    if not root:
        return False
    if not root.left and not root.right:
        return root.val == targetSum
    return hasPathSum(root.left, targetSum - root.val) or hasPathSum(
        root.right, targetSum - root.val)


# Explanation: Subtract node value, check leaf.


# 12. Path Sum II
def pathSum(root, targetSum):
    res = []

    def dfs(node, path, s):
        if not node:
            return
        if not node.left and not node.right and node.val == s:
            res.append(path + [node.val])
            return
        dfs(node.left, path + [node.val], s - node.val)
        dfs(node.right, path + [node.val], s - node.val)

    dfs(root, [], targetSum)
    return res


# Explanation: DFS, track path and sum.


# 13. Binary Tree Maximum Path Sum
def maxPathSum(root):
    res = [float('-inf')]

    def dfs(node):
        if not node:
            return 0
        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)
        res[0] = max(res[0], node.val + left + right)
        return node.val + max(left, right)

    dfs(root)
    return res[0]


# Explanation: At each node, consider max path through node.


# 14. Construct Binary Tree from Preorder and Inorder Traversal
def buildTreePreIn(preorder, inorder):
    if not preorder or not inorder:
        return None
    idx = inorder.index(preorder[0])
    root = TreeNode(preorder[0])
    root.left = buildTreePreIn(preorder[1:idx + 1], inorder[:idx])
    root.right = buildTreePreIn(preorder[idx + 1:], inorder[idx + 1:])
    return root


# Explanation: Root is first in preorder, split inorder.


# 15. Construct Binary Tree from Inorder and Postorder Traversal
def buildTreeInPost(inorder, postorder):
    if not inorder or not postorder:
        return None
    root_val = postorder[-1]
    idx = inorder.index(root_val)
    root = TreeNode(root_val)
    root.left = buildTreeInPost(inorder[:idx], postorder[:idx])
    root.right = buildTreeInPost(inorder[idx + 1:], postorder[idx:-1])
    return root


# Explanation: Root is last in postorder, split inorder.


# 16. Serialize and Deserialize Binary Tree
class CodecTree:

    def serialize(self, root):
        vals = []

        def dfs(node):
            if not node:
                vals.append('null')
                return
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ','.join(vals)

    def deserialize(self, data):
        vals = iter(data.split(','))

        def dfs():
            val = next(vals)
            if val == 'null':
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()


# Explanation: Preorder traversal, use 'null' for None.


# 17. Flatten Binary Tree to Linked List
def flatten(root):
    prev = None

    def dfs(node):
        nonlocal prev
        if not node:
            return
        dfs(node.right)
        dfs(node.left)
        node.right = prev
        node.left = None
        prev = node

    dfs(root)


# Explanation: Reverse preorder, link right pointers.


# 18. Populating Next Right Pointers in Each Node
class NodeNext:

    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect(root):
    if not root:
        return None
    leftmost = root
    while leftmost.left:
        head = leftmost
        while head:
            head.left.next = head.right
            if head.next:
                head.right.next = head.next.left
            head = head.next
        leftmost = leftmost.left
    return root


# Explanation: Use next pointers, perfect binary tree.


# 19. Populating Next Right Pointers in Each Node II
def connectII(root):
    curr = root
    dummy = NodeNext(0)
    while curr:
        tail = dummy
        while curr:
            if curr.left:
                tail.next = curr.left
                tail = tail.next
            if curr.right:
                tail.next = curr.right
                tail = tail.next
            curr = curr.next
        curr = dummy.next
        dummy.next = None
    return root


# Explanation: Use dummy node, works for any binary tree.


# 20. Balanced Binary Tree
def isBalanced(root):

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        if left == -1:
            return -1
        right = dfs(node.right)
        if right == -1:
            return -1
        if abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return dfs(root) != -1


# Explanation: Return -1 if unbalanced, else height.


# 21. Validate Binary Search Tree
def isValidBST(root):

    def dfs(node, low, high):
        if not node:
            return True
        if not (low < node.val < high):
            return False
        return dfs(node.left, low, node.val) and dfs(node.right, node.val,high)

    return dfs(root, float('-inf'), float('inf'))


# Explanation: Check value in (low, high) range.

# 22. Kth Smallest Element in a BST
# (Already implemented above as kthSmallest)


# 23. Lowest Common Ancestor of a Binary Tree
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left or right


# Explanation: If p and q in different subtrees, root is LCA.


# 24. Diameter of Binary Tree
def diameterOfBinaryTree(root):
    res = [0]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        res[0] = max(res[0], left + right)
        return 1 + max(left, right)

    dfs(root)
    return res[0]


# Explanation: Diameter is max left+right at any node.


# 25. Subtree of Another Tree
def isSubtree(s, t):

    def isSame(a, b):
        if not a and not b:
            return True
        if not a or not b or a.val != b.val:
            return False
        return isSame(a.left, b.left) and isSame(a.right, b.right)

    if not s:
        return False
    if isSame(s, t):
        return True
    return isSubtree(s.left, t) or isSubtree(s.right, t)


# Explanation: Check if t is same as s or any subtree of s.


# 26. Construct Binary Search Tree from Preorder Traversal
def bstFromPreorder(preorder):

    def helper(bound=float('inf')):
        nonlocal idx
        if idx == len(preorder) or preorder[idx] > bound:
            return None
        root = TreeNode(preorder[idx])
        idx += 1
        root.left = helper(root.val)
        root.right = helper(bound)
        return root

    idx = 0
    return helper()


# Explanation: Recursively build left subtree with upper bound as current value.


# 27. Find Leaves of Binary Tree
def findLeaves(root):
    res = []

    def dfs(node):
        if not node:
            return -1
        h = 1 + max(dfs(node.left), dfs(node.right))
        if h == len(res):
            res.append([])
        res[h].append(node.val)
        return h

    dfs(root)
    return res


# Explanation: Use height from bottom, group nodes by height.


# 28. Binary Tree Paths
def binaryTreePaths(root):
    res = []

    def dfs(node, path):
        if not node:
            return
        if not node.left and not node.right:
            res.append(path + str(node.val))
            return
        dfs(node.left, path + str(node.val) + "->")
        dfs(node.right, path + str(node.val) + "->")

    dfs(root, "")
    return res


# Explanation: DFS, build path string, add at leaf.


# 29. Sum of Left Leaves
def sumOfLeftLeaves(root):
    if not root:
        return 0
    res = 0
    if root.left and not root.left.left and not root.left.right:
        res += root.left.val
    res += sumOfLeftLeaves(root.left)
    res += sumOfLeftLeaves(root.right)
    return res


# Explanation: Check if left child is a leaf, sum up.


# 30. Find Bottom Left Tree Value
def findBottomLeftValue(root):
    q = deque([root])
    while q:
        node = q.popleft()
        if node.right:
            q.append(node.right)
        if node.left:
            q.append(node.left)
    return node.val


# Explanation: BFS, right first, last node is bottom-left.


# 31. Merge Two Binary Trees
def mergeTrees(t1, t2):
    if not t1 and not t2:
        return None
    if not t1:
        return t2
    if not t2:
        return t1
    root = TreeNode(t1.val + t2.val)
    root.left = mergeTrees(t1.left, t2.left)
    root.right = mergeTrees(t1.right, t2.right)
    return root


# Explanation: Recursively sum nodes, merge subtrees.


# 32. Binary Tree Tilt
def findTilt(root):
    res = [0]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        res[0] += abs(left - right)
        return node.val + left + right

    dfs(root)
    return res[0]


# Explanation: For each node, add abs(left sum - right sum).


# 33. Cousins in Binary Tree
def isCousins(root, x, y):

    def dfs(node, parent, depth):
        if not node:
            return None
        if node.val == x or node.val == y:
            return (parent, depth)
        left = dfs(node.left, node, depth + 1)
        right = dfs(node.right, node, depth + 1)
        return left or right

    x_info = dfs(root, None, 0)
    y_info = dfs(root, None, 0)
    return x_info and y_info and x_info[0] != y_info[0] and x_info[
        1] == y_info[1]


# Explanation: Nodes are cousins if same depth, different parent.


# 34. Binary Tree Vertical Order Traversal
def verticalOrder(root):
    if not root:
        return []
    col_table = {}
    q = deque([(root, 0)])
    min_col = max_col = 0
    while q:
        node, col = q.popleft()
        if col not in col_table:
            col_table[col] = []
        col_table[col].append(node.val)
        min_col = min(min_col, col)
        max_col = max(max_col, col)
        if node.left:
            q.append((node.left, col - 1))
        if node.right:
            q.append((node.right, col + 1))
    return [col_table[x] for x in range(min_col, max_col + 1)]


# Explanation: BFS, group nodes by column index.


# 35. Binary Tree Vertical Traversal
def verticalTraversal(root):
    nodes = []

    def dfs(node, row, col):
        if not node:
            return
        nodes.append((col, row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)

    dfs(root, 0, 0)
    nodes.sort()
    res, last_col = [], float('-inf')
    for col, row, val in nodes:
        if col != last_col:
            res.append([])
            last_col = col
        res[-1].append(val)
    return res


# Explanation: DFS, collect (col, row, val), sort, group by col.


# 36. Binary Tree Level Order Traversal II
def levelOrderBottom(root):
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
    return res[::-1]


# Explanation: Standard level order, reverse at end.


# 37. Binary Tree Cameras
def minCameraCover(root):
    res = [0]

    def dfs(node):
        if not node:
            return 1
        left = dfs(node.left)
        right = dfs(node.right)
        if left == 0 or right == 0:
            res[0] += 1
            return 2
        if left == 2 or right == 2:
            return 1
        return 0

    if dfs(root) == 0:
        res[0] += 1
    return res[0]


# Explanation: 0=needs camera, 1=covered, 2=has camera.


# 38. Largest BST Subtree
def largestBSTSubtree(root):

    def dfs(node):
        if not node:
            return (0, float('inf'), float('-inf'))
        l_size, l_min, l_max = dfs(node.left)
        r_size, r_min, r_max = dfs(node.right)
        if l_max < node.val < r_min:
            size = l_size + r_size + 1
            return (size, min(l_min, node.val), max(r_max, node.val))
        return (max(l_size, r_size), float('-inf'), float('inf'))

    return dfs(root)[0]


# Explanation: For each node, check BST property, return size.


# 39. Recover Binary Search Tree
def recoverTree(root):
    stack, x = [], None
    y = pred = None
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if pred and root.val < pred.val:
            y = root
            if not x:
                x = pred
            else:
                break
        pred = root
        root = root.right
    x.val, y.val = y.val, x.val


# Explanation: Inorder traversal, find two swapped nodes, swap back.


# 40. Trim a Binary Search Tree
def trimBST(root, low, high):
    if not root:
        return None
    if root.val < low:
        return trimBST(root.right, low, high)
    if root.val > high:
        return trimBST(root.left, low, high)
    root.left = trimBST(root.left, low, high)
    root.right = trimBST(root.right, low, high)
    return root


# Explanation: Recursively trim nodes outside [low, high].


# 41. Delete Node in a BST
def deleteNode(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        temp = root.right
        while temp.left:
            temp = temp.left
        root.val = temp.val
        root.right = deleteNode(root.right, temp.val)
    return root


# Explanation: Find node, replace with inorder successor if needed.


# 42. Count Complete Tree Nodes
def countNodes(root):
    if not root:
        return 0
    l, r = root, root
    lh = rh = 0
    while l:
        lh += 1
        l = l.left
    while r:
        rh += 1
        r = r.right
    if lh == rh:
        return (1 << lh) - 1
    return 1 + countNodes(root.left) + countNodes(root.right)


# Explanation: If left/right heights equal, it's full tree.


# 43. Boundary of Binary Tree
def boundaryOfBinaryTree(root):
    if not root:
        return []
    res = [root.val]

    def leftBoundary(node):
        while node:
            if node.left or node.right:
                res.append(node.val)
            node = node.left if node.left else node.right

    def leaves(node):
        if node and not node.left and not node.right and node != root:
            res.append(node.val)
        if node:
            leaves(node.left)
            leaves(node.right)

    def rightBoundary(node):
        tmp = []
        while node:
            if node.left or node.right:
                tmp.append(node.val)
            node = node.right if node.right else node.left
        res.extend(tmp[::-1])

    leftBoundary(root.left)
    leaves(root)
    rightBoundary(root.right)
    return res


# Explanation: Add left boundary, leaves, then right boundary in reverse.


# 44. Sum Root to Leaf Numbers
def sumNumbers(root):

    def dfs(node, curr):
        if not node:
            return 0
        curr = curr * 10 + node.val
        if not node.left and not node.right:
            return curr
        return dfs(node.left, curr) + dfs(node.right, curr)

    return dfs(root, 0)


# Explanation: DFS, build number along path, sum at leaves.


# 45. Convert BST to Greater Tree
def convertBST(root):
    total = 0

    def dfs(node):
        nonlocal total
        if node:
            dfs(node.right)
            total += node.val
            node.val = total
            dfs(node.left)

    dfs(root)
    return root


# Explanation: Reverse inorder, accumulate sum.


# 46. Count Univalue Subtrees
def countUnivalSubtrees(root):
    res = [0]

    def dfs(node):
        if not node:
            return True
        left = dfs(node.left)
        right = dfs(node.right)
        if left and right and (not node.left or node.left.val== node.val) and (not node.right or node.right.val == node.val):
            res[0] += 1
            return True
        return False

    dfs(root)
    return res[0]


# Explanation: Recursively check if subtree is univalue.


# 47. Closest Binary Search Tree Value
def closestValue(root, target):
    closest = root.val
    while root:
        if abs(root.val - target) < abs(closest - target):
            closest = root.val
        root = root.left if target < root.val else root.right
    return closest


# Explanation: Traverse BST, update closest value.


# 48. Closest Binary Search Tree Value II
def closestKValues(root, target, k):

    def inorder(node):
        if node:
            inorder(node.left)
            vals.append(node.val)
            inorder(node.right)

    vals = []
    inorder(root)
    vals.sort(key=lambda x: abs(x - target))
    return vals[:k]


# Explanation: Inorder traversal, sort by distance to target.


# 49. Find Mode in Binary Search Tree
def findMode(root):
    count = Counter()

    def inorder(node):
        if node:
            inorder(node.left)
            count[node.val] += 1
            inorder(node.right)

    inorder(root)
    max_freq = max(count.values())
    return [k for k, v in count.items() if v == max_freq]


# Explanation: Count frequencies, return most common.


# 50. Maximum Width of Binary Tree
def widthOfBinaryTree(root):
    if not root:
        return 0
    max_width = 0
    q = deque([(root, 0)])
    while q:
        size = len(q)
        _, first = q[0]
        for _ in range(size):
            node, idx = q.popleft()
            if node.left:
                q.append((node.left, 2 * idx))
            if node.right:
                q.append((node.right, 2 * idx + 1))
        max_width = max(max_width, idx - first + 1)
    return max_width


# Explanation: BFS, track indices as if full binary tree, width = last-first+1.
