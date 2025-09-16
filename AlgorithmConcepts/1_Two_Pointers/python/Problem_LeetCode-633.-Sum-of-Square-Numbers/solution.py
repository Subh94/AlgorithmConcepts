from math import isqrt


class Solution:
    """LeetCode 633. Sum of Square Numbers.

    The two-pointer approach considers candidate pairs ``(a, b)`` starting from the
    extremes: ``a`` at 0 and ``b`` at ``sqrt(c)``. Depending on how the sum of squares
    compares to ``c`` we move the pointers inward until we either find a match or
    exhaust the search space.
    """

    def judgeSquareSum(self, c: int) -> bool:
        """Return True if integers ``a`` and ``b`` exist such that ``a^2 + b^2 == c``."""
        left, right = 0, isqrt(c)

        while left <= right:
            total = left * left + right * right

            if total == c:
                return True

            if total < c:
                left += 1
            else:
                right -= 1

        return False
