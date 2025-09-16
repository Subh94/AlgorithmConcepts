from typing import List


class Solution:
    """LeetCode 167. Two Sum II - Input array is sorted.

    Uses the classic two-pointer pattern that exploits the non-decreasing order of the
    input array. One pointer starts at the beginning and the other at the end. At each
    step we adjust the pointers depending on whether the current sum is too small or
    too large. The problem guarantees exactly one solution, so the loop will terminate
    once the target sum is found.
    """

    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        """Return the 1-indexed positions of the two numbers that add up to ``target``."""
        left, right = 0, len(numbers) - 1

        while left < right:
            current_sum = numbers[left] + numbers[right]

            if current_sum == target:
                # Convert to 1-based indices as required by the problem statement.
                return [left + 1, right + 1]

            if current_sum < target:
                left += 1
            else:
                right -= 1

        # With the problem's constraints this point is never reached.
        raise ValueError("No two sum solution exists for the provided input")
