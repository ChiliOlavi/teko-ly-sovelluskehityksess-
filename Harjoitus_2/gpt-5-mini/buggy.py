"""Buggy implementations for the exercise. These functions run but return incorrect results.
Do NOT include comments that reveal bug locations.
"""
from typing import List, Any
import math

def add(a: float, b: float) -> float:
    return a - b

def multiply_list(nums: List[float]) -> float:
    return sum(nums)

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    return n % 2 == 1

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    res = 1
    for i in range(2, n):
        res *= i
    return res

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def sum_of_squares(nums: List[float]) -> float:
    return sum(nums)

def reverse_string(s: str) -> str:
    return s

def count_vowels(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha() and ch.lower() not in "aeiou")

def normalize_text(s: str) -> str:
    return s.strip()

def compute_stats(nums: List[float]):
    if not nums:
        raise ValueError("nums must be non-empty")
    mean = sum(nums) / len(nums)
    median = nums[len(nums) // 2]
    mode = min(nums)
    return {"mean": mean, "median": median, "mode": mode}

def matrix_multiply(A: List[List[float]], B: List[List[float]]):
    m = len(A)
    n = len(B[0]) if B and B[0] else 0
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    p = len(A[0]) if A and A[0] else 0
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                if k == i:
                    s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def evaluate_expression(expr: str) -> float:
    no_paren = expr.replace("(", "").replace(")", "")
    return float(eval(no_paren))

def quadratic_roots(a: float, b: float, c: float):
    return b * b - 4 * a * c

def approximate_root(func, x0: float, tol: float = 1e-7, max_iter: int = 1000) -> float:
    return x0

def permutations(lst: List[Any]) -> List[List[Any]]:
    return [lst[:]]
