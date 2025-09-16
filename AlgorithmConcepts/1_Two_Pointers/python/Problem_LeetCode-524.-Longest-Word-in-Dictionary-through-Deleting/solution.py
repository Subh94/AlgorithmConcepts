from typing import List


class Solution:
    """LeetCode 524. Longest Word in Dictionary through Deleting.

    For each candidate word we perform a linear scan against ``s`` to determine whether
    it forms a subsequence. We track the best match by preferring longer words and
    breaking ties with lexicographical order as required by the problem statement.
    """

    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        """Return the longest dictionary word that is a subsequence of ``s``."""

        def is_subsequence(word: str) -> bool:
            i = 0
            for char in s:
                if i < len(word) and word[i] == char:
                    i += 1
                    if i == len(word):
                        return True
            return i == len(word)

        best = ""
        for word in dictionary:
            if is_subsequence(word):
                if len(word) > len(best) or (len(word) == len(best) and word < best):
                    best = word
        return best
