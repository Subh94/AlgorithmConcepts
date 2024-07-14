import os

# Define the list of algorithmic concepts with numbered prefixes
concepts = [
    {
        "prefix": "1",
        "name": "Two Pointers",
        "description": "Two pointers is an efficient technique to solve problems involving iterating through elements with two pointers.",
        "practice_problems": [
            "LeetCode 167. Two Sum II - Input array is sorted",
            "LeetCode 633. Sum of Square Numbers",
            "LeetCode 524. Longest Word in Dictionary through Deleting"
        ]
    },
    {
        "prefix": "2",
        "name": "Island (Matrix Traversal)",
        "description": "Island traversal involves navigating through a matrix to find connected components (islands) of 1s.",
        "practice_problems": [
            "LeetCode 200. Number of Islands",
            "LeetCode 695. Max Area of Island",
            "LeetCode 130. Surrounded Regions"
        ]
    },
    {
        "prefix": "3",
        "name": "Fast and Slow Pointers",
        "description": "Fast and slow pointers are used to detect cycles in linked lists or find a position within the list efficiently.",
        "practice_problems": [
            "LeetCode 141. Linked List Cycle",
            "LeetCode 142. Linked List Cycle II",
            "LeetCode 876. Middle of the Linked List"
        ]
    },
    {
        "prefix": "4",
        "name": "Sliding Windows",
        "description": "Sliding window technique is used to find the maximum or minimum value in a subarray of fixed size k.",
        "practice_problems": [
            "LeetCode 239. Sliding Window Maximum",
            "LeetCode 76. Minimum Window Substring",
            "LeetCode 567. Permutation in String"
        ]
    },
    {
        "prefix": "5",
        "name": "Merge Intervals",
        "description": "Merge intervals is used to combine overlapping intervals into one or find overlapping intervals.",
        "practice_problems": [
            "LeetCode 56. Merge Intervals",
            "LeetCode 57. Insert Interval",
            "LeetCode 986. Interval List Intersections"
        ]
    },
    {
        "prefix": "6",
        "name": "Cyclic Sort",
        "description": "Cyclic sort is used to sort numbers from 1 to n in an array of size n with O(n) time complexity.",
        "practice_problems": [
            "LeetCode 41. First Missing Positive",
            "LeetCode 287. Find the Duplicate Number",
            "LeetCode 442. Find All Duplicates in an Array"
        ]
    },
    {
        "prefix": "7",
        "name": "In-place Reversal of Linked List",
        "description": "In-place reversal of linked list involves reversing the links between nodes of a singly linked list.",
        "practice_problems": [
            "LeetCode 206. Reverse Linked List",
            "LeetCode 92. Reverse Linked List II",
            "LeetCode 25. Reverse Nodes in k-Group"
        ]
    },
    {
        "prefix": "8",
        "name": "Tree Breadth First Search (BFS)",
        "description": "BFS is used to traverse or search a tree or graph level by level.",
        "practice_problems": [
            "LeetCode 102. Binary Tree Level Order Traversal",
            "LeetCode 107. Binary Tree Level Order Traversal II",
            "LeetCode 199. Binary Tree Right Side View"
        ]
    },
    {
        "prefix": "9",
        "name": "Tree Depth First Search (DFS)",
        "description": "DFS is used to traverse or search a tree or graph by exploring as far as possible along each branch before backtracking.",
        "practice_problems": [
            "LeetCode 104. Maximum Depth of Binary Tree",
            "LeetCode 111. Minimum Depth of Binary Tree",
            "LeetCode 112. Path Sum"
        ]
    },
    {
        "prefix": "10",
        "name": "Two Heaps",
        "description": "Two heaps are used to efficiently find the median of a stream of numbers or manage smallest/largest elements.",
        "practice_problems": [
            "LeetCode 295. Find Median from Data Stream",
            "LeetCode 480. Sliding Window Median",
            "LeetCode 378. Kth Smallest Element in a Sorted Matrix"
        ]
    },
    {
        "prefix": "11",
        "name": "Subsets",
        "description": "Generate all possible subsets or combinations of elements.",
        "practice_problems": [
            "LeetCode 78. Subsets",
            "LeetCode 90. Subsets II",
            "LeetCode 46. Permutations"
        ]
    },
    {
        "prefix": "12",
        "name": "Modified Binary Search",
        "description": "Efficiently search in sorted arrays that may be rotated or contain irregularities.",
        "practice_problems": [
            "LeetCode 33. Search in Rotated Sorted Array",
            "LeetCode 81. Search in Rotated Sorted Array II",
            "LeetCode 153. Find Minimum in Rotated Sorted Array"
        ]
    },
    {
        "prefix": "13",
        "name": "Top K Elements",
        "description": "Retrieve top K elements from data using various approaches like heaps or sorting.",
        "practice_problems": [
            "LeetCode 215. Kth Largest Element in an Array",
            "LeetCode 347. Top K Frequent Elements",
            "LeetCode 973. K Closest Points to Origin"
        ]
    },
    {
        "prefix": "14",
        "name": "Bitwise XOR",
        "description": "Use bitwise operations to solve problems involving unique elements or toggling states.",
        "practice_problems": [
            "LeetCode 136. Single Number",
            "LeetCode 268. Missing Number",
            "LeetCode 421. Maximum XOR of Two Numbers in an Array"
        ]
    },
    {
        "prefix": "15",
        "name": "Backtracking",
        "description": "Explore all potential solutions by building candidates step-by-step and backtracking when necessary.",
        "practice_problems": [
            "LeetCode 46. Permutations",
            "LeetCode 39. Combination Sum",
            "LeetCode 22. Generate Parentheses"
        ]
    },
    {
        "prefix": "16",
        "name": "0 or 1 Knapsack (Dynamic Programming)",
        "description": "Solve optimization problems by breaking them down into simpler subproblems and storing results.",
        "practice_problems": [
            "LeetCode 416. Partition Equal Subset Sum",
            "LeetCode 494. Target Sum",
            "LeetCode 322. Coin Change"
        ]
    },
    {
        "prefix": "17",
        "name": "Topological Sort (Graph)",
        "description": "Sort nodes in a directed graph such that for every directed edge from node u to node v, u comes before v.",
        "practice_problems": [
            "LeetCode 207. Course Schedule",
            "LeetCode 210. Course Schedule II",
            "LeetCode 269. Alien Dictionary"
        ]
    },
    {
        "prefix": "18",
        "name": "K-way Merge",
        "description": "Merge multiple sorted lists or arrays into a single sorted list efficiently.",
        "practice_problems": [
            "LeetCode 23. Merge k Sorted Lists",
            "LeetCode 632. Smallest Range Covering Elements from K Lists",
            "LeetCode 373. Find K Pairs with Smallest Sums"
        ]
    },
    {
        "prefix": "19",
        "name": "Monotonic Stack",
        "description": "Use a stack to maintain elements in a monotonic order (increasing or decreasing) and solve problems efficiently.",
        "practice_problems": [
            "LeetCode 739. Daily Temperatures",
            "LeetCode 496. Next Greater Element I",
            "LeetCode 84. Largest Rectangle in Histogram"
        ]
    },
    {
        "prefix": "20",
        "name": "Multi-threaded Programming",
        "description": "Develop solutions that use multiple threads or processes to solve problems concurrently.",
        "practice_problems": [
            "LeetCode 1114. Print in Order",
            "LeetCode 1115. Print FooBar Alternately",
            "LeetCode 1195. Fizz Buzz Multithreaded"
        ]
    }
]

# Function to create directory if not exists
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to create README.md for each concept
def create_readme(directory, concept):
    filename = os.path.join(directory, "README.md")
    with open(filename, 'w') as f:
        f.write(f"# {concept['prefix']}. {concept['name']}\n\n")
        f.write(f"{concept['description']}\n\n")
        f.write("### Practice Problems:\n")
        for i, problem in enumerate(concept['practice_problems'], 1):
            f.write(f"{i}. {problem}\n")

# Main function to organize concepts and create structure
def organize_concepts(concepts):
    # Create main directory for concepts
    create_directory("AlgorithmConcepts")
    
    # Create root README.md
    with open("README.md", 'w') as root_readme:
        root_readme.write("# Algorithm Practice Repository\n\n")
        root_readme.write("This repository contains practice problems and solutions for various algorithmic concepts.\n\n")
        root_readme.write("Each algorithmic concept is organized into its own directory with practice problems listed in corresponding README files.\n\n")
        root_readme.write("## Concepts:\n\n")
        for concept in concepts:
            root_readme.write(f"- [{concept['prefix']}. {concept['name']}](AlgorithmConcepts/{concept['name'].replace(' ', '_')}/README.md)\n")
    
    # Iterate through each concept
    for concept in concepts:
        concept_dir = os.path.join("AlgorithmConcepts", f"{concept['prefix']}_{concept['name'].replace(' ', '_')}"
                                   )
        # Create directory for each concept
        create_directory(concept_dir)
        # Create README.md for each concept
        create_readme(concept_dir, concept)
        
        # Create subfolders for Python and C++ solutions
        python_dir = os.path.join(concept_dir, "python")
        create_directory(python_dir)
        cpp_dir = os.path.join(concept_dir, "c++")
        create_directory(cpp_dir)
        
        # Create nested folders and solution files for practice problems
        for i, problem in enumerate(concept['practice_problems'], 1):
            problem_folder_name = problem.replace(" ", "-")
            problem_folder_python = os.path.join(python_dir, f"Problem_{problem_folder_name}")
            create_directory(problem_folder_python)
            with open(os.path.join(problem_folder_python, "solution.py"), 'w') as python_solution:
                python_solution.write(f"# Python solution for {problem}\n\n")
            
            problem_folder_cpp = os.path.join(cpp_dir, f"Problem_{problem_folder_name}")
            create_directory(problem_folder_cpp)
            with open(os.path.join(problem_folder_cpp, "solution.cpp"), 'w') as cpp_solution:
                cpp_solution.write(f"// C++ solution for {problem}\n\n")

if __name__ == "__main__":
    organize_concepts(concepts)
    print("Concepts organized successfully!")

