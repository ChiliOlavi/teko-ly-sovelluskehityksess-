"""Template for student's solution file `ratkaisu.py`.
Copy this to `ratkaisu.py` and implement faster versions of the functions.
"""
from typing import List, Any, Callable, Tuple

def add(a: float, b: float) -> float:
    return a + b

def multiply_list(nums: List[float]) -> float:
    prod = 1.0
    for x in nums:
        prod *= x
    return prod

def is_prime(n: int) -> bool:
    # implement faster primality if desired
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

# Implement remaining functions: factorial, fibonacci, sum_of_squares, reverse_string,
# count_vowels, normalize_text, compute_stats, matrix_multiply, evaluate_expression,
# quadratic_roots, approximate_root, permutations, longest_common_subsequence,
# top_k_frequent, flatten_nested
