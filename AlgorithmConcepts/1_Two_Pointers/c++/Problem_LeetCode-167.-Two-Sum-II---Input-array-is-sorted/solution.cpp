#include <vector>

class Solution {
public:
    // LeetCode 167. Two Sum II - Input array is sorted
    std::vector<int> twoSum(std::vector<int>& numbers, int target) {
        int left = 0;
        int right = static_cast<int>(numbers.size()) - 1;

        while (left < right) {
            const int current_sum = numbers[left] + numbers[right];

            if (current_sum == target) {
                // Convert to 1-based indices before returning.
                return {left + 1, right + 1};
            }

            if (current_sum < target) {
                ++left;
            } else {
                --right;
            }
        }

        // Problem constraints guarantee that a solution always exists.
        return {};
    }
};
