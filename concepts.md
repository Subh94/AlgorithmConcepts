
### 1. Two Pointers
- **Concept**: Use two pointers to iterate through data structure.
- **Use Cases**: Finding pairs in sorted arrays, removing duplicates, etc.
- **Example**: Sorting and searching problems, such as finding two numbers that add up to a specific target in a sorted array.

### 2. Island (Matrix Traversal)
- **Concept**: Traverse a matrix/grid to identify distinct "islands" (connected components).
- **Use Cases**: Number of islands, flood fill, maze problems.
- **Techniques**: Depth-First Search (DFS), Breadth-First Search (BFS).

### 3. Fast and Slow Pointers
- **Concept**: Use two pointers moving at different speeds to detect cycles.
- **Use Cases**: Detecting cycles in linked lists, finding middle of linked list.
- **Example**: Floyd’s Tortoise and Hare algorithm.

### 4. Sliding Window
- **Concept**: Use a window that slides over data to solve problems involving subarrays or substrings.
- **Use Cases**: Maximum sum subarray of size k, longest substring without repeating characters.
- **Example**: Dynamic adjustment of window size to maintain a condition.

### 5. Merge Intervals
- **Concept**: Combine overlapping intervals.
- **Use Cases**: Meeting room scheduling, merging ranges.
- **Example**: Sort intervals by start time, then merge overlapping intervals.

### 6. Cyclic Sort
- **Concept**: Place elements in the correct index.
- **Use Cases**: Problems involving arrays with numbers in a given range.
- **Example**: Find missing number, find all duplicates.

### 7. In-place Reversal of Linked List
- **Concept**: Reverse linked list in-place without extra memory.
- **Use Cases**: Reverse entire linked list, reverse sublist.
- **Example**: Change pointers of nodes to reverse the list.

### 8. Tree Breadth-First Search (BFS)
- **Concept**: Traverse tree level by level.
- **Use Cases**: Shortest path in unweighted graph, level order traversal.
- **Example**: Use a queue to explore nodes level by level.

### 9. Tree Depth-First Search (DFS)
- **Concept**: Traverse tree depth-wise.
- **Use Cases**: Pathfinding, connectivity checking.
- **Example**: Use recursion or stack to explore nodes.

### 10. Two Heaps
- **Concept**: Use two heaps to solve problems involving running medians or k-largest elements.
- **Use Cases**: Find median in a data stream.
- **Example**: Maintain a max-heap for the lower half and a min-heap for the upper half.

### 11. Subsets
- **Concept**: Generate all possible subsets.
- **Use Cases**: Power set, combination problems.
- **Example**: Use backtracking or bit manipulation.

### 12. Modified Binary Search
- **Concept**: Adapt binary search for complex conditions.
- **Use Cases**: Search in rotated arrays, finding closest elements.
- **Example**: Adjust mid calculation based on conditions.

### 13. Top K Elements
- **Concept**: Find the top k elements in a collection.
- **Use Cases**: Top k frequent elements, k largest elements.
- **Example**: Use min-heap of size k.

### 14. Bitwise XOR
- **Concept**: Use XOR properties to solve problems.
- **Use Cases**: Find missing number, single non-repeated number.
- **Example**: XOR of a number with itself is 0, and XOR with 0 is the number.

### 15. Backtracking
- **Concept**: Explore all possible solutions and backtrack if a solution fails.
- **Use Cases**: Permutations, combinations, N-queens problem.
- **Example**: Recursively explore all paths and backtrack on failure.

### 16. 0/1 Knapsack (Dynamic Programming)
- **Concept**: Solve optimization problems using dynamic programming.
- **Use Cases**: Subset sum problem, partition problem.
- **Example**: Use a 2D DP array to track optimal solutions.

### 17. Topological Sort (Graph)
- **Concept**: Order vertices in a Directed Acyclic Graph (DAG) linearly.
- **Use Cases**: Task scheduling, course prerequisite ordering.
- **Example**: Use Kahn’s algorithm or DFS-based approach.

### 18. K-way Merge
- **Concept**: Merge k sorted arrays.
- **Use Cases**: External sorting, merging k sorted lists.
- **Example**: Use a min-heap to merge arrays efficiently.

### 19. Monotonic Stack
- **Concept**: Use a stack to maintain a sequence that is either entirely non-increasing or non-decreasing.
- **Use Cases**: Next greater element, largest rectangle in histogram.
- **Example**: Push and pop elements to maintain stack properties.

### 20. Multi-threaded Algorithms
- **Concept**: Utilize multiple threads to parallelize tasks.
- **Use Cases**: Concurrent data processing, web scraping.
- **Example**: Use thread pools or async programming to manage multiple threads.

Sure, here is the pseudocode for each of the algorithmic concepts:

### 1. Two Pointers

```pseudo
function findPairWithSum(arr, target):
    left = 0
    right = len(arr) - 1
    while left < right:
        sum = arr[left] + arr[right]
        if sum == target:
            return (left, right)
        elif sum < target:
            left += 1
        else:
            right -= 1
    return (-1, -1)
```

### 2. Island (Matrix Traversal)

```pseudo
function numIslands(grid):
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'  # Mark as visited
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count
```

### 3. Fast and Slow Pointers

```pseudo
function hasCycle(head):
    slow = head
    fast = head
    while fast != null and fast.next != null:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return true
    return false
```

### 4. Sliding Window

```pseudo
function maxSumSubarray(arr, k):
    max_sum = 0
    window_sum = 0
    for i in range(k):
        window_sum += arr[i]
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### 5. Merge Intervals

```pseudo
function mergeIntervals(intervals):
    if len(intervals) <= 1:
        return intervals
    intervals.sort(key=lambda x: x[0])
    merged = []
    start, end = intervals[0]
    for i in range(1, len(intervals)):
        interval = intervals[i]
        if interval[0] <= end:
            end = max(end, interval[1])
        else:
            merged.append((start, end))
            start, end = interval
    merged.append((start, end))
    return merged
```

### 6. Cyclic Sort

```pseudo
function cyclicSort(arr):
    i = 0
    while i < len(arr):
        correctIndex = arr[i] - 1
        if arr[i] != arr[correctIndex]:
            swap(arr, i, correctIndex)
        else:
            i += 1
```

### 7. In-place Reversal of Linked List

```pseudo
function reverseLinkedList(head):
    prev = null
    current = head
    while current != null:
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev
```

### 8. Tree Breadth-First Search (BFS)

```pseudo
function bfs(root):
    queue = []
    queue.append(root)
    while queue:
        node = queue.pop(0)
        process(node)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
```

### 9. Tree Depth-First Search (DFS)

```pseudo
function dfs(root):
    if root == null:
        return
    process(root)
    dfs(root.left)
    dfs(root.right)
```

### 10. Two Heaps

```pseudo
function findMedian(nums):
    minHeap = MinHeap()
    maxHeap = MaxHeap()
    
    for num in nums:
        if maxHeap.isEmpty() or num <= maxHeap.peek():
            maxHeap.add(num)
        else:
            minHeap.add(num)
        
        # Balance the heaps
        if maxHeap.size() > minHeap.size() + 1:
            minHeap.add(maxHeap.poll())
        elif minHeap.size() > maxHeap.size():
            maxHeap.add(minHeap.poll())

    if maxHeap.size() == minHeap.size():
        return (maxHeap.peek() + minHeap.peek()) / 2
    else:
        return maxHeap.peek()
```

### 11. Subsets

```pseudo
function findSubsets(nums):
    result = [[]]
    for num in nums:
        newSubsets = []
        for subset in result:
            newSubsets.append(subset + [num])
        result += newSubsets
    return result
```

### 12. Modified Binary Search

```pseudo
function searchRotatedArray(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) / 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:  # Left side is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right side is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

### 13. Top K Elements

```pseudo
function findTopKElements(arr, k):
    minHeap = MinHeap()
    for num in arr:
        minHeap.add(num)
        if minHeap.size() > k:
            minHeap.poll()
    return list(minHeap)
```

### 14. Bitwise XOR

```pseudo
function findSingleNumber(arr):
    result = 0
    for num in arr:
        result ^= num
    return result
```

### 15. Backtracking

```pseudo
function solveNQueens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def isValid(board, row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if isValid(board, row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result
```

### 16. 0/1 Knapsack (Dynamic Programming)

```pseudo
function knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
```

### 17. Topological Sort (Graph)

```pseudo
function topologicalSort(graph):
    def dfs(node):
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        visiting.remove(node)
        visited.add(node)
        result.append(node)
        return True
    
    visited = set()
    visiting = set()
    result = []
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Cycle detected
    return result[::-1]
```

### 18. K-way Merge

```pseudo
function mergeKSortedLists(lists):
    minHeap = MinHeap()
    for l in lists:
        if l:
            minHeap.add((l.val, l))
    dummy = ListNode(0)
    current = dummy
    while minHeap:
        val, node = minHeap.poll()
        current.next = node
        current = current.next
        if node.next:
            minHeap.add((node.next.val, node.next))
    return dummy.next
```

### 19. Monotonic Stack

```pseudo
function nextGreaterElement(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    return result
```

### 20. Multi-threaded

```pseudo
from threading import Thread

def process(data):
    # Process data here

def multiThreadedProcessing(data_chunks):
    threads = []
    for chunk in data_chunks:


        thread = Thread(target=process, args=(chunk,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
```

These pseudocode examples provide a high-level overview of each concept. Adapt them as needed for specific problems and implementations.



Sure! Here are sample problem statements and step-by-step solutions for each of the concepts:

### 1. Two Pointers
**Problem Statement:** Given a sorted array `arr` and a target sum `target`, find the indices of the two numbers that add up to `target`.

**Solution Explanation:**
1. Initialize two pointers: `left` at the beginning and `right` at the end of the array.
2. Calculate the sum of elements at `left` and `right` pointers.
3. If the sum is equal to the target, return the indices.
4. If the sum is less than the target, move the `left` pointer one step to the right.
5. If the sum is greater than the target, move the `right` pointer one step to the left.
6. Repeat steps 2-5 until you find the pair or the pointers cross each other.

```pseudo
function findPairWithSum(arr, target):
    left = 0
    right = len(arr) - 1
    while left < right:
        sum = arr[left] + arr[right]
        if sum == target:
            return (left, right)
        elif sum < target:
            left += 1
        else:
            right -= 1
    return (-1, -1)
```

### 2. Island (Matrix Traversal)
**Problem Statement:** Given a 2D grid of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.

**Solution Explanation:**
1. Initialize a counter for islands.
2. Traverse the grid.
3. When a '1' is found, increment the island counter and start a Depth-First Search (DFS) to mark all connected '1's as '0'.
4. Continue until all cells are visited.

```pseudo
function numIslands(grid):
    def dfs(i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] == '0':
            return
        grid[i][j] = '0'  # Mark as visited
        dfs(i + 1, j)
        dfs(i - 1, j)
        dfs(i, j + 1)
        dfs(i, j - 1)

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                dfs(i, j)
                count += 1
    return count
```

### 3. Fast and Slow Pointers
**Problem Statement:** Given a linked list, determine if it has a cycle in it.

**Solution Explanation:**
1. Initialize two pointers, `slow` and `fast`, both pointing to the head of the list.
2. Move `slow` by one step and `fast` by two steps in each iteration.
3. If `slow` and `fast` meet, a cycle is detected.
4. If `fast` reaches the end, no cycle is present.

```pseudo
function hasCycle(head):
    slow = head
    fast = head
    while fast != null and fast.next != null:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return true
    return false
```

### 4. Sliding Window
**Problem Statement:** Given an array of integers `arr` and an integer `k`, find the maximum sum of any contiguous subarray of size `k`.

**Solution Explanation:**
1. Calculate the sum of the first `k` elements and set it as the maximum sum.
2. Slide the window over the array by adding the next element and subtracting the element that is left behind.
3. Update the maximum sum if the current window sum is greater.
4. Continue until the end of the array.

```pseudo
function maxSumSubarray(arr, k):
    max_sum = 0
    window_sum = 0
    for i in range(k):
        window_sum += arr[i]
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum
```

### 5. Merge Intervals
**Problem Statement:** Given a collection of intervals, merge all overlapping intervals.

**Solution Explanation:**
1. Sort the intervals by their start times.
2. Initialize an empty list to hold the merged intervals.
3. Iterate through the sorted intervals and merge overlapping intervals.
4. Add the merged interval to the result list.

```pseudo
function mergeIntervals(intervals):
    if len(intervals) <= 1:
        return intervals
    intervals.sort(key=lambda x: x[0])
    merged = []
    start, end = intervals[0]
    for i in range(1, len(intervals)):
        interval = intervals[i]
        if interval[0] <= end:
            end = max(end, interval[1])
        else:
            merged.append((start, end))
            start, end = interval
    merged.append((start, end))
    return merged
```

### 6. Cyclic Sort
**Problem Statement:** Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the missing number.

**Solution Explanation:**
1. Iterate through the array and place each number at its correct index.
2. Swap elements until every number is at its correct index.
3. The first index that doesn't match the value is the missing number.

```pseudo
function findMissingNumber(arr):
    i = 0
    while i < len(arr):
        correctIndex = arr[i]
        if arr[i] < len(arr) and arr[i] != arr[correctIndex]:
            swap(arr, i, correctIndex)
        else:
            i += 1
    for i in range(len(arr)):
        if arr[i] != i:
            return i
    return len(arr)
```

### 7. In-place Reversal of Linked List
**Problem Statement:** Reverse a singly linked list.

**Solution Explanation:**
1. Initialize three pointers: `prev` as `null`, `current` as `head`, and `next` as `null`.
2. Iterate through the linked list and reverse the pointers.
3. Move `prev`, `current`, and `next` one step forward in each iteration.
4. Return `prev` as the new head of the reversed list.

```pseudo
function reverseLinkedList(head):
    prev = null
    current = head
    while current != null:
        next = current.next
        current.next = prev
        prev = current
        current = next
    return prev
```

### 8. Tree Breadth-First Search (BFS)
**Problem Statement:** Given a binary tree, return its level order traversal.

**Solution Explanation:**
1. Initialize a queue with the root node.
2. While the queue is not empty, process each node level by level.
3. For each node, enqueue its left and right children if they exist.
4. Collect nodes at the current level and add them to the result list.

```pseudo
function bfs(root):
    result = []
    if root == null:
        return result
    queue = []
    queue.append(root)
    while queue:
        level = []
        size = len(queue)
        for i in range(size):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### 9. Tree Depth-First Search (DFS)
**Problem Statement:** Given a binary tree, return all root-to-leaf paths.

**Solution Explanation:**
1. Initialize an empty list to hold the paths.
2. Perform a DFS traversal of the tree.
3. For each node, add the current node value to the path.
4. If a leaf node is reached, add the current path to the result list.
5. Backtrack by removing the current node from the path.

```pseudo
function findPaths(root):
    result = []
    def dfs(node, path):
        if node == null:
            return
        path.append(node.val)
        if node.left == null and node.right == null:
            result.append(list(path))
        else:
            dfs(node.left, path)
            dfs(node.right, path)
        path.pop()

    dfs(root, [])
    return result
```

### 10. Two Heaps
**Problem Statement:** Find the median of a data stream.

**Solution Explanation:**
1. Use two heaps: a max-heap for the lower half and a min-heap for the upper half of the data.
2. Add each new number to the appropriate heap.
3. Balance the heaps so that the difference in size is at most one.
4. The median is the root of the max-heap if it has more elements, otherwise it's the average of the roots of both heaps.

```pseudo
function findMedian(nums):
    minHeap = MinHeap()
    maxHeap = MaxHeap()
    
    for num in nums:
        if maxHeap.isEmpty() or num <= maxHeap.peek():
            maxHeap.add(num)
        else:
            minHeap.add(num)
        
        # Balance the heaps
        if maxHeap.size() > minHeap.size() + 1:
            minHeap.add(maxHeap.poll())
        elif minHeap.size() > maxHeap.size():
            maxHeap.add(minHeap.poll())

    if maxHeap.size() == minHeap.size():
        return (maxHeap.peek

() + minHeap.peek()) / 2
    else:
        return maxHeap.peek()
```

### 11. Subsets
**Problem Statement:** Given a set of distinct integers, return all possible subsets.

**Solution Explanation:**
1. Start with an empty subset.
2. For each number, add it to existing subsets to create new subsets.
3. Continue until all numbers are processed.

```pseudo
function findSubsets(nums):
    result = [[]]
    for num in nums:
        newSubsets = []
        for subset in result:
            newSubsets.append(subset + [num])
        result += newSubsets
    return result
```

### 12. Modified Binary Search
**Problem Statement:** Given a sorted array that has been rotated, find the index of a target value.

**Solution Explanation:**
1. Initialize two pointers: `left` at the beginning and `right` at the end of the array.
2. Find the middle element.
3. Determine which part (left or right) is sorted.
4. Adjust the pointers based on the target value's position relative to the sorted part.
5. Repeat until the target is found or the pointers cross each other.

```pseudo
function searchRotatedArray(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = left + (right - left) / 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:  # Left side is sorted
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right side is sorted
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

### 13. Top K Elements
**Problem Statement:** Given an array `arr` and an integer `k`, return the `k` largest elements in the array.

**Solution Explanation:**
1. Use a min-heap to keep track of the top `k` elements.
2. Add each element to the heap.
3. If the heap size exceeds `k`, remove the smallest element.
4. The remaining elements in the heap are the `k` largest elements.

```pseudo
function findTopKElements(arr, k):
    minHeap = MinHeap()
    for num in arr:
        minHeap.add(num)
        if minHeap.size() > k:
            minHeap.poll()
    return list(minHeap)
```

### 14. Bitwise XOR
**Problem Statement:** Given an array of integers where every element appears twice except for one, find that single one.

**Solution Explanation:**
1. Initialize a variable `result` to 0.
2. Iterate through the array, applying the XOR operation between `result` and each element.
3. The result will be the single element.

```pseudo
function findSingleNumber(arr):
    result = 0
    for num in arr:
        result ^= num
    return result
```

### 15. Backtracking
**Problem Statement:** Solve the N-Queens problem, which is to place `n` queens on an `n x n` chessboard such that no two queens threaten each other.

**Solution Explanation:**
1. Use a helper function to check if placing a queen is valid.
2. Use a recursive backtracking approach to try placing queens in each row.
3. If a valid configuration is found, add it to the result list.
4. Backtrack and try different placements.

```pseudo
function solveNQueens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def isValid(board, row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        for col in range(n):
            if isValid(board, row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result
```

### 16. 0/1 Knapsack (Dynamic Programming)
**Problem Statement:** Given weights and values of `n` items, put these items in a knapsack of capacity `W` to get the maximum total value in the knapsack.

**Solution Explanation:**
1. Use a 2D DP array to store the maximum value for each capacity up to `W`.
2. Initialize the DP array with 0.
3. Iterate through the items and capacities, updating the DP array based on whether the current item is included or excluded.
4. The value at `dp[n][W]` will be the maximum value that can be achieved with the given items and capacity.

```pseudo
function knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
```

### 17. Topological Sort (Graph)
**Problem Statement:** Given a directed acyclic graph (DAG), return a topological order of its vertices.

**Solution Explanation:**
1. Use a DFS approach to visit each node.
2. Keep track of visited nodes and the current path.
3. For each node, recursively visit its neighbors.
4. After visiting all neighbors, add the node to the result list.
5. Reverse the result list to get the topological order.

```pseudo
function topologicalSort(graph):
    def dfs(node):
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        visiting.remove(node)
        visited.add(node)
        result.append(node)
        return True
    
    visited = set()
    visiting = set()
    result = []
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []  # Cycle detected
    return result[::-1]
```

### 18. K-way Merge
**Problem Statement:** Merge `k` sorted linked lists and return it as one sorted list.

**Solution Explanation:**
1. Use a min-heap to keep track of the smallest elements from each list.
2. Add the first node of each list to the heap.
3. Extract the smallest element from the heap and add it to the merged list.
4. If the extracted node has a next node, add it to the heap.
5. Repeat until the heap is empty.

```pseudo
function mergeKSortedLists(lists):
    minHeap = MinHeap()
    for l in lists:
        if l:
            minHeap.add((l.val, l))
    dummy = ListNode(0)
    current = dummy
    while minHeap:
        val, node = minHeap.poll()
        current.next = node
        current = current.next
        if node.next:
            minHeap.add((node.next.val, node.next))
    return dummy.next
```

### 19. Monotonic Stack
**Problem Statement:** Given an array of integers, find the next greater element for each element in the array.

**Solution Explanation:**
1. Initialize an empty stack and a result array filled with -1.
2. Iterate through the array from right to left.
3. For each element, pop elements from the stack that are less than or equal to the current element.
4. If the stack is not empty, the top element is the next greater element.
5. Push the current element onto the stack.

```pseudo
function nextGreaterElement(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        while stack and stack[-1] <= nums[i]:
            stack.pop()
        if stack:
            result[i] = stack[-1]
        stack.append(nums[i])
    return result
```

### 20. Multi-threaded
**Problem Statement:** Process a large dataset in parallel using multiple threads.

**Solution Explanation:**
1. Split the data into chunks.
2. Create a thread for each chunk and assign it a processing function.
3. Start all threads.
4. Wait for all threads to complete before proceeding.

```pseudo
from threading import Thread

def process(data):
    # Process data here

def multiThreadedProcessing(data_chunks):
    threads = []
    for chunk in data_chunks:
        thread = Thread(target=process, args=(chunk,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
```

These problem statements and solutions cover a range of algorithmic concepts, providing a foundational understanding of each one. Adapt the pseudocode as needed

 for specific use cases or more detailed implementations.


 Recognizing if a problem can be solved using a specific algorithmic concept involves identifying certain patterns or characteristics in the problem statement. Here’s a guide on how to recognize if a problem is suited for each of the given concepts:

### 1. Two Pointers
**Recognition:**
- The problem involves a sorted array or a string.
- You're required to find pairs or triplets with certain properties (e.g., sum to a target).
- Problems involving subsequence or subarray in sorted structures.

### 2. Island (Matrix Traversal)
**Recognition:**
- The problem involves a grid or matrix.
- You need to count distinct regions, connected components, or perform some form of flood fill.
- Problems involving traversal of all connected nodes or regions.

### 3. Fast and Slow Pointers
**Recognition:**
- The problem involves linked lists or arrays.
- You're asked to detect cycles or find the middle of a structure.
- Problems requiring the identification of repetitive patterns or collisions.

### 4. Sliding Window
**Recognition:**
- The problem involves finding a subarray or substring with specific properties.
- You're asked for maximum, minimum, or fixed-length subarrays.
- Problems involving a moving or dynamic range of elements.

### 5. Merge Intervals
**Recognition:**
- The problem involves intervals or ranges.
- You're required to merge, insert, or find intersections of intervals.
- Problems dealing with scheduling, calendar events, or time ranges.

### 6. Cyclic Sort
**Recognition:**
- The problem involves an array with a range of integers from 1 to `n`.
- You're asked to sort, find missing numbers, or duplicate numbers.
- Problems requiring rearrangement of elements based on their values.

### 7. In-place Reversal of Linked List
**Recognition:**
- The problem involves linked lists.
- You're required to reverse the whole list or a part of it.
- Problems dealing with manipulation of node pointers.

### 8. Tree Breadth-First Search (BFS)
**Recognition:**
- The problem involves a tree or graph.
- You're asked for level order traversal, shortest path in an unweighted graph.
- Problems requiring visiting nodes level by level.

### 9. Tree Depth-First Search (DFS)
**Recognition:**
- The problem involves a tree or graph.
- You're asked for all paths, checking for a path, or traversal-based problems.
- Problems requiring visiting all nodes deeply before moving to the next node.

### 10. Two Heaps
**Recognition:**
- The problem involves continuous data streams.
- You're asked to find the median or Kth smallest/largest elements in real-time.
- Problems dealing with dynamic insertion and removal of elements.

### 11. Subsets
**Recognition:**
- The problem involves generating all possible combinations, subsets, or permutations.
- You're asked to explore all potential groups or selections from a set.
- Problems involving exhaustive search over combinations.

### 12. Modified Binary Search
**Recognition:**
- The problem involves a sorted array that might be rotated or includes some irregularity.
- You're asked to find a target element or identify a special element.
- Problems requiring efficient searching in altered sorted structures.

### 13. Top K Elements
**Recognition:**
- The problem involves finding the top K elements in a list or stream.
- You're asked for the largest, smallest, or most frequent elements.
- Problems dealing with large datasets where only a subset needs to be retrieved.

### 14. Bitwise XOR
**Recognition:**
- The problem involves finding a unique element in pairs or toggling states.
- You're asked to perform bit manipulation or identify unique elements.
- Problems dealing with parity or singular occurrences.

### 15. Backtracking
**Recognition:**
- The problem involves exploring all potential solutions to find a valid one.
- You're asked for all solutions, paths, or combinations with constraints.
- Problems requiring recursive exploration with undoing of steps.

### 16. 0/1 Knapsack (Dynamic Programming)
**Recognition:**
- The problem involves making decisions to maximize or minimize a value with constraints.
- You're asked for optimal selections based on weights and values.
- Problems involving bounded resources and choices.

### 17. Topological Sort (Graph)
**Recognition:**
- The problem involves a directed acyclic graph (DAG).
- You're asked for an order of tasks, courses, or dependencies.
- Problems dealing with precedence or ordering requirements.

### 18. K-way Merge
**Recognition:**
- The problem involves multiple sorted arrays or lists.
- You're asked to merge them into a single sorted list.
- Problems requiring the combination of multiple sorted sources.

### 19. Monotonic Stack
**Recognition:**
- The problem involves finding the next greater or smaller elements.
- You're asked for ranges, spans, or areas based on monotonic properties.
- Problems dealing with contiguous ranges and order.

### 20. Multi-threaded
**Recognition:**
- The problem involves large datasets or tasks that can be parallelized.
- You're asked to speed up processing by dividing work.
- Problems requiring concurrent execution and synchronization.

By identifying these patterns in the problem statements, you can determine which algorithmic concept is best suited for solving the problem.


## Practice question 
Sure! Here are some practice questions for each of the algorithmic concepts:

### 1. Two Pointers
1. **LeetCode 167. Two Sum II - Input array is sorted:** Given an array of integers `numbers` that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number.
2. **LeetCode 125. Valid Palindrome:** Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
3. **LeetCode 344. Reverse String:** Write a function that reverses a string. The input string is given as an array of characters.
4. **LeetCode 26. Remove Duplicates from Sorted Array:** Given a sorted array, remove the duplicates in-place such that each element appears only once and returns the new length.
5. **LeetCode 11. Container With Most Water:** Given `n` non-negative integers `a1, a2, ..., an`, where each represents a point at coordinate `(i, ai)`, find two lines that together with the x-axis form a container that holds the most water.

### 2. Island (Matrix Traversal)
1. **LeetCode 200. Number of Islands:** Given a 2D grid map of '1's (land) and '0's (water), count the number of islands.
2. **LeetCode 130. Surrounded Regions:** Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
3. **LeetCode 695. Max Area of Island:** Given a non-empty 2D array `grid` of 0's and 1's, an island is a group of 1's connected 4-directionally. Return the maximum area of an island.
4. **LeetCode 417. Pacific Atlantic Water Flow:** Given an `m x n` matrix of non-negative integers representing the height of each unit cell in a continent, find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.
5. **LeetCode 463. Island Perimeter:** You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. Return the perimeter of the island.

### 3. Fast and Slow Pointers
1. **LeetCode 141. Linked List Cycle:** Given a linked list, determine if it has a cycle in it.
2. **LeetCode 142. Linked List Cycle II:** Given a linked list, return the node where the cycle begins. If there is no cycle, return `null`.
3. **LeetCode 234. Palindrome Linked List:** Given a singly linked list, determine if it is a palindrome.
4. **LeetCode 876. Middle of the Linked List:** Given a non-empty, singly linked list with `head` node, return a middle node of the linked list.
5. **LeetCode 202. Happy Number:** Write an algorithm to determine if a number `n` is happy.

### 4. Sliding Window
1. **LeetCode 3. Longest Substring Without Repeating Characters:** Given a string, find the length of the longest substring without repeating characters.
2. **LeetCode 76. Minimum Window Substring:** Given two strings `s` and `t`, return the minimum window in `s` which will contain all the characters in `t`.
3. **LeetCode 567. Permutation in String:** Given two strings `s1` and `s2`, write a function to return true if `s2` contains the permutation of `s1`.
4. **LeetCode 438. Find All Anagrams in a String:** Given a string `s` and a non-empty string `p`, find all the start indices of `p`'s anagrams in `s`.
5. **LeetCode 239. Sliding Window Maximum:** Given an array `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. Find the maximum value in each window.

### 5. Merge Intervals
1. **LeetCode 56. Merge Intervals:** Given a collection of intervals, merge all overlapping intervals.
2. **LeetCode 57. Insert Interval:** Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
3. **LeetCode 252. Meeting Rooms:** Given an array of meeting time intervals consisting of start and end times, determine if a person could attend all meetings.
4. **LeetCode 253. Meeting Rooms II:** Given an array of meeting time intervals consisting of start and end times, find the minimum number of conference rooms required.
5. **LeetCode 986. Interval List Intersections:** Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order. Return the intersection of these two interval lists.

### 6. Cyclic Sort
1. **LeetCode 268. Missing Number:** Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.
2. **LeetCode 448. Find All Numbers Disappeared in an Array:** Given an array of integers where `1 ≤ a[i] ≤ n` (n = size of array), some elements appear twice and others appear once. Find all the elements that do not appear in this array.
3. **LeetCode 287. Find the Duplicate Number:** Given an array of integers `nums` containing `n + 1` integers where each integer is in the range `[1, n]` inclusive, there is only one repeated number. Find it.
4. **LeetCode 442. Find All Duplicates in an Array:** Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once. Find all the elements that appear twice in this array.
5. **LeetCode 41. First Missing Positive:** Given an unsorted integer array `nums`, find the smallest missing positive integer.

### 7. In-place Reversal of Linked List
1. **LeetCode 206. Reverse Linked List:** Reverse a singly linked list.
2. **LeetCode 92. Reverse Linked List II:** Reverse a linked list from position `m` to `n`.
3. **LeetCode 234. Palindrome Linked List:** Given a singly linked list, determine if it is a palindrome.
4. **LeetCode 25. Reverse Nodes in k-Group:** Given a linked list, reverse the nodes of a linked list `k` at a time and return its modified list.
5. **LeetCode 61. Rotate List:** Given a linked list, rotate the list to the right by `k` places.

### 8. Tree Breadth-First Search (BFS)
1. **LeetCode 102. Binary Tree Level Order Traversal:** Given a binary tree, return the level order traversal of its nodes' values.
2. **LeetCode 103. Binary Tree Zigzag Level Order Traversal:** Given a binary tree, return the zigzag level order traversal of its nodes' values.
3. **LeetCode 107. Binary Tree Level Order Traversal II:** Given a binary tree, return the bottom-up level order traversal of its nodes' values.
4. **LeetCode 111. Minimum Depth of Binary Tree:** Given a binary tree, find its minimum depth.
5. **LeetCode 637. Average of Levels in Binary Tree:** Given a non-empty binary tree, return the average value of the nodes on each level.

### 9. Tree Depth-First Search (DFS)
1. **LeetCode 104. Maximum Depth of Binary Tree:** Given a binary tree, find its maximum depth.
2. **LeetCode 124. Binary Tree Maximum Path Sum:** Given a non-empty binary tree, find the maximum path sum.
3. **LeetCode 543. Diameter of Binary Tree:** Given a binary tree, you need to compute the length of the diameter of the tree.
4. **LeetCode 98. Validate Binary Search Tree:** Given a binary tree, determine if it is a valid binary search tree.
5. **LeetCode 437. Path Sum III:** Given the root of a binary tree and an integer `sum`, return the number of paths where the sum of the values along the path equals `sum`.

### 10. Two Heaps
1. **LeetCode 295. Find Median from Data Stream:** The `MedianFinder` class is given that supports the following operations: addNum and findMedian.
2. **LeetCode 480. Sliding Window Median:** Given an array `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. Return the median array for each window in the original array.
3. **LeetCode 215. Kth Largest Element in an Array:** Find the `k`th largest element in an unsorted array.
4. **LeetCode 239. Sliding Window Maximum:** Given an array `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. Find the maximum value in each window.
5. **LeetCode 23. Merge k Sorted Lists:** Merge `k` sorted linked lists and return it as one sorted list.

### 11. Subsets
1. **LeetCode 78. Subsets:** Given a set of distinct integers

, return all possible subsets (the power set).
2. **LeetCode 90. Subsets II:** Given a collection of integers that might contain duplicates, return all possible subsets.
3. **LeetCode 46. Permutations:** Given an array of distinct integers, return all possible permutations.
4. **LeetCode 47. Permutations II:** Given a collection of numbers, possibly containing duplicates, return all unique permutations.
5. **LeetCode 77. Combinations:** Given two integers `n` and `k`, return all possible combinations of `k` numbers out of `1 ... n`.

### 12. Modified Binary Search
1. **LeetCode 33. Search in Rotated Sorted Array:** Given the array `nums` sorted in ascending order, rotated at some pivot, find the target element.
2. **LeetCode 34. Find First and Last Position of Element in Sorted Array:** Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.
3. **LeetCode 81. Search in Rotated Sorted Array II:** Suppose an array is sorted in ascending order and then rotated, and duplicates are allowed. Find the target element.
4. **LeetCode 153. Find Minimum in Rotated Sorted Array:** Suppose an array of distinct integers is sorted in ascending order and then rotated at some pivot. Find the minimum element.
5. **LeetCode 154. Find Minimum in Rotated Sorted Array II:** Suppose an array is sorted in ascending order and then rotated, and duplicates are allowed. Find the minimum element.

### 13. Top K Elements
1. **LeetCode 215. Kth Largest Element in an Array:** Find the `k`th largest element in an unsorted array.
2. **LeetCode 347. Top K Frequent Elements:** Given a non-empty array of integers, return the `k` most frequent elements.
3. **LeetCode 692. Top K Frequent Words:** Given an array of strings `words`, return the `k` most frequent words.
4. **LeetCode 973. K Closest Points to Origin:** Given an array of points in the `x`-y plane, find the `k` closest points to the origin.
5. **LeetCode 378. Kth Smallest Element in a Sorted Matrix:** Given a `n x n` matrix where each row and each column are sorted in ascending order, find the `k`th smallest element.

### 14. Bitwise XOR
1. **LeetCode 136. Single Number:** Given a non-empty array of integers, every element appears twice except for one. Find that single one.
2. **LeetCode 268. Missing Number:** Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.
3. **LeetCode 260. Single Number III:** Given an array of numbers `nums`, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.
4. **LeetCode 421. Maximum XOR of Two Numbers in an Array:** Given a non-empty array of numbers, find the maximum result of `nums[i] XOR nums[j]`.
5. **LeetCode 190. Reverse Bits:** Reverse bits of a given 32 bits unsigned integer.

### 15. Backtracking
1. **LeetCode 46. Permutations:** Given an array of distinct integers, return all possible permutations.
2. **LeetCode 39. Combination Sum:** Given an array of distinct integers `candidates` and a target integer `target`, return all unique combinations of `candidates` where the chosen numbers sum to `target`.
3. **LeetCode 40. Combination Sum II:** Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in `candidates` where the candidate numbers sum to `target`.
4. **LeetCode 22. Generate Parentheses:** Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
5. **LeetCode 51. N-Queens:** The `n-queens` puzzle is the problem of placing `n` queens on an `n x n` chessboard such that no two queens attack each other.

### 16. 0/1 Knapsack (Dynamic Programming)
1. **LeetCode 416. Partition Equal Subset Sum:** Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.
2. **LeetCode 494. Target Sum:** You are given a list of non-negative integers and a target `sum`. Find the number of ways to assign symbols to make the sum of numbers equal to the target `sum`.
3. **LeetCode 1049. Last Stone Weight II:** You are given an array of integers `stones` where `stones[i]` is the weight of the `i`th stone. We want to choose a subset of stones and smash them.
4. **LeetCode 322. Coin Change:** You are given coins of different denominations and a total amount of money `amount`. Find the fewest number of coins that you need to make up that amount.
5. **LeetCode 474. Ones and Zeroes:** You are given an array of binary strings `strs` and two integers `m` and `n`. Find the maximum number of strings that you can form with `m` 0's and `n` 1's.

### 17. Topological Sort (Graph)
1. **LeetCode 207. Course Schedule:** There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses-1`. Some courses may have prerequisites.
2. **LeetCode 210. Course Schedule II:** There are a total of `numCourses` courses you have to take, labeled from `0` to `numCourses-1`. Some courses may have prerequisites. Return the order of courses.
3. **LeetCode 269. Alien Dictionary:** Given a list of words from the alien language's dictionary, derive the order of letters in this language.
4. **LeetCode 444. Sequence Reconstruction:** Check whether the original sequence can be uniquely reconstructed from the sequences in `seqs`.
5. **LeetCode 329. Longest Increasing Path in a Matrix:** Given an `m x n` integers matrix, return the length of the longest increasing path in the matrix.

### 18. K-way Merge
1. **LeetCode 23. Merge k Sorted Lists:** Merge `k` sorted linked lists and return it as one sorted list.
2. **LeetCode 378. Kth Smallest Element in a Sorted Matrix:** Given a `n x n` matrix where each row and each column are sorted in ascending order, find the `k`th smallest element.
3. **LeetCode 632. Smallest Range Covering Elements from K Lists:** You have `k` lists of sorted integers. Find the smallest range that includes at least one number from each of the `k` lists.
4. **LeetCode 373. Find K Pairs with Smallest Sums:** You are given two integer arrays `nums1` and `nums2` sorted in ascending order and an integer `k`. Find the k pairs (u1, v1), (u2, v2), ... (uk, vk) with the smallest sums.
5. **LeetCode 502. IPO:** Suppose LeetCode will start its IPO soon. In order to sell a good price of its shares to Venture Capital, LeetCode would like to work on some projects to increase its capital.

### 19. Monotonic Stack
1. **LeetCode 739. Daily Temperatures:** Given a list of daily temperatures `T`, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature.
2. **LeetCode 496. Next Greater Element I:** You are given two arrays (without duplicates) `nums1` and `nums2` where `nums1` is a subset of `nums2`. Find all the next greater numbers for `nums1`'s elements in the corresponding places of `nums2`.
3. **LeetCode 503. Next Greater Element II:** Given a circular array `nums` (i.e., the next element of the last element is the first element of the array), find the Next Greater Number for every element.
4. **LeetCode 84. Largest Rectangle in Histogram:** Given an array of integers representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
5. **LeetCode 42. Trapping Rain Water:** Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

### 20. Multi-threaded
1. **LeetCode 1114. Print in Order:** Suppose we have a class with three functions: `first`, `second`, and `third`. The three functions will be called in a fixed order. You have to ensure that the `second` function is executed after the `first` function, and the `third` function is executed after the `second` function.

2. **LeetCode 1115. Print FooBar Alternately:** Suppose you have two threads. One will call `foo()` and the other will call `bar()`. You must output "foobar" `n` times.

3. **LeetCode 1116. Print Zero Even Odd:** Suppose you have a class with three functions: `zero()`, `even()`, and `odd()`. The same instance of `ZeroEvenOdd` will be passed to three different threads.

4. **LeetCode 1195. Fizz Buzz Multithreaded:** You have the four functions `fizz()`, `buzz()`, `fizzbuzz()`, and `number()` that print the numbers from `1` to `n` in a multithreaded environment.

5. **LeetCode 1242. Web Crawler Multithreaded:** Given a URL `startUrl` and an interface `HtmlParser`, implement a multi-threaded web crawler to crawl all links that are reachable from `startUrl`.

These questions should help you practice and understand each concept thoroughly.
