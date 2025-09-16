#include <string>
#include <vector>

class Solution {
public:
    // LeetCode 524. Longest Word in Dictionary through Deleting
    std::string findLongestWord(const std::string& s, std::vector<std::string>& dictionary) {
        std::string best;
        for (const std::string& word : dictionary) {
            if (isSubsequence(word, s)) {
                if (word.size() > best.size() || (word.size() == best.size() && word < best)) {
                    best = word;
                }
            }
        }
        return best;
    }

private:
    static bool isSubsequence(const std::string& word, const std::string& source) {
        std::size_t i = 0;
        for (char ch : source) {
            if (i < word.size() && word[i] == ch) {
                ++i;
                if (i == word.size()) {
                    return true;
                }
            }
        }
        return i == word.size();
    }
};
