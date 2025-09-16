# 1. Two Pointers

The two pointers technique is a strategy for iterating over linear data structures with two stateful indices instead of one. By moving the indices in a coordinated way we can avoid redundant work, reduce nested loops, and reason about relationships between elements that are far apart in the input. This pattern shows up in arrays, strings, linked lists, and even custom iterators whenever you need to maintain a relationship between two positions in the data.

---

## Core Idea

1. **Maintain two references** (indices, iterators, or actual node pointers) into the same collection.
2. **Advance one or both pointers** based on the property you are trying to enforce (e.g., the sum of elements, substring containment, or distance between pointers).
3. **Stop when the pointers meet** a terminating condition, such as crossing each other, reaching the end of the structure, or satisfying a constraint.

Because the pointers typically move in a single pass, most two-pointer solutions run in \(O(n)\) time and \(O(1)\) extra space.

---

## Common Pointer Arrangements

| Arrangement | Description | Typical Use Cases |
|-------------|-------------|-------------------|
| **Opposite ends** | Start one pointer at the beginning and the other at the end, moving them toward each other. | Sorted arrays (pair sum, difference problems), palindrome checks. |
| **Same direction** | Start both pointers near the beginning, moving forward together while keeping an invariant (e.g., one pointer lags behind). | Removing duplicates, partitioning by condition, linked list manipulations. |
| **Sliding window** | Treat the two pointers as the bounds of a window that expands and contracts while keeping track of aggregated state. | Longest/shortest subarray with a property, substring problems, streaming analytics. |
| **Meet-in-the-middle** | Move pointers based on values they point to in order to converge to a desired relationship. | Merging sorted collections, scheduling conflicts. |

> ⚠️ Sliding-window and fast/slow pointer techniques are sometimes treated as separate patterns, but they are variations of the same "maintain two positions" idea.

---

## When to Reach for Two Pointers

Look for these signals while reading a problem statement:

- The input is already sorted, or can be sorted without breaking the constraints.
- You are asked to find or count pairs/triplets with a property (sum, difference, matching characters).
- The question involves contiguous regions (subarrays, substrings) with constraints like "at most \(k\) distinct characters".
- You need to merge or compare multiple ordered sequences in a streaming manner.
- The problem hints that a brute-force nested loop would be too slow and invites optimization.

When these clues appear, consider how two coordinated pointers could reduce the search space.

---

## Problem-Solving Blueprint

1. **Define what each pointer represents.** For example, the left pointer might mark the start of a candidate subarray, and the right pointer marks its end.
2. **Initialize pointers and any required state.** State might include a running sum, counts, or a data structure to maintain the invariant.
3. **Establish pointer movement rules.** Decide what condition causes each pointer to move. Often one pointer moves to expand the search space and the other shrinks it when constraints break.
4. **Update the state as pointers move.** Ensure the invariant stays true (e.g., the window contains at most \(k\) distinct characters).
5. **Record answers while traversing.** Update the best/first/last solution when conditions are met.
6. **Terminate gracefully.** Stop when a pointer reaches the end, the pointers cross, or the invariant can no longer be satisfied.

Pseudo-code outline for a typical opposite-ends example:

```text
left = 0
right = n - 1
while left < right:
    if nums[left] + nums[right] == target:
        return (left, right)
    elif nums[left] + nums[right] < target:
        left += 1   # Need a bigger sum
    else:
        right -= 1  # Need a smaller sum
return "no pair"
```

---

## Example Walkthrough – LeetCode 167 (Two Sum II)

- **Problem:** Given a sorted array, find two numbers that add up to a target and return their 1-indexed positions.
- **Reasoning:** Sorting is already done, and we seek a pair whose sum equals a constant. Starting at both ends lets us adjust the sum without re-checking pairs.
- **Algorithm:**
  1. Place one pointer `lo` at index `0` and another pointer `hi` at index `n - 1`.
  2. Compute `current = numbers[lo] + numbers[hi]`.
  3. If `current == target`, we are finished.
  4. If `current < target`, increment `lo` to increase the sum.
  5. If `current > target`, decrement `hi` to decrease the sum.
  6. Continue until the pair is found (the problem guarantees exactly one solution).
- **Complexity:** The pointers move at most `n` steps in total, so the runtime is \(O(n)\) with \(O(1)\) extra space.

---

## Variations in the Practice Problems

1. **LeetCode 167. Two Sum II – Input array is sorted**
   - Classic opposite-ends pointers, as described above.
2. **LeetCode 633. Sum of Square Numbers**
   - Treat the numbers \(a\) and \(b\) as pointers over integer values. Since squares grow monotonically, you can start `a = 0` and `b = floor(sqrt(c))` and adjust them depending on \(a^2 + b^2\) relative to \(c\).
3. **LeetCode 524. Longest Word in Dictionary through Deleting**
   - Use two pointers traveling in the *same direction* to check if a candidate dictionary word is a subsequence of the given string. Increment the dictionary pointer only when characters match, and walk through the main string with the other pointer.

Additional practice ideas:
- Container With Most Water (maximize area using opposite ends).
- Valid Palindrome / Reverse Vowels of a String (move inward skipping non-alphanumeric characters).
- Minimum Size Subarray Sum (sliding window two pointers).

---

## Real-World Use Cases

- **Log/Event Reconciliation:** Align two chronologically sorted logs to find matching or overlapping events without scanning the entire datasets repeatedly.
- **Multimedia Buffering:** Maintain start/end pointers to manage a playback buffer, ensuring latency constraints while streaming data.
- **Text Editors and IDEs:** Implement substring search or highlight matching brackets by walking from both sides toward the center.
- **Network Packet Windows:** Sliding windows control how many packets can be in flight by moving window boundaries as acknowledgments arrive.
- **Sensor Fusion:** Merge readings from time-ordered sensors (e.g., GPS and accelerometer) by advancing the pointer of whichever sensor lags behind.

---

## Tips and Common Pitfalls

- **Watch for off-by-one errors.** Decide whether pointers are inclusive or exclusive bounds and stay consistent.
- **Maintain invariants carefully.** When the invariant breaks (e.g., too many distinct characters), adjust the correct pointer before recording results.
- **Avoid unnecessary data structures.** The strength of two pointers is constant extra space; think twice before adding hash maps or sets unless required.
- **Consider sorting trade-offs.** If the input is unsorted, you might need to sort it first, which costs \(O(n \log n)\). Ensure sorting is allowed by the problem constraints.
- **Think about duplicates.** Decide whether to skip duplicates (e.g., while `left < right and nums[left] == nums[left-1]`, increment `left`).

---

Understanding the two pointers pattern equips you with a versatile tool for tackling array and string problems efficiently. Practice identifying when the input structure and constraints hint at this approach, and the optimal solution will often become clear.
