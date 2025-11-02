"""Fast reference implementations for the optimization exercise.
Pure Python 3.12, no external dependencies.
"""
from typing import List, Any, Callable, Tuple
import math
import itertools
from collections import Counter

def add(a: float, b: float) -> float:
    return a + b

def multiply_list(nums: List[float]) -> float:
    return math.prod(nums) if nums else 1.0

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    i = 3
    while i <= r:
        if n % i == 0:
            return False
        i += 2
    return True

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    return math.factorial(n)

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def sum_of_squares(nums: List[float]) -> float:
    return sum(x * x for x in nums)

def reverse_string(s: str) -> str:
    return s[::-1]

def count_vowels(s: str) -> int:
    return sum(1 for ch in s.lower() if ch in "aeiou")

def normalize_text(s: str) -> str:
    # strip and collapse whitespace, lower-case
    return " ".join(s.split()).strip()

def compute_stats(nums: List[float]) -> dict:
    if not nums:
        raise ValueError("nums must be non-empty")
    n = len(nums)
    mean = sum(nums) / n
    sorted_nums = sorted(nums)
    mid = n // 2
    if n % 2 == 1:
        median = sorted_nums[mid]
    else:
        median = (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
    counts = Counter(sorted_nums)
    mode = max(counts.keys(), key=lambda k: (counts[k], -sorted_nums.index(k)))
    return {"mean": mean, "median": median, "mode": mode}

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not A or not B:
        return []
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    # assume shapes are compatible
    # compute columns of B for faster inner loops
    B_cols = list(zip(*B))
    C = [[sum(A[i][k] * B_cols[j][k] for k in range(p)) for j in range(n)] for i in range(m)]
    return C

def evaluate_expression(expr: str) -> float:
    # evaluate arithmetic expressions; used for simple exercises
    return float(eval(expr, {}, {}))

def quadratic_roots(a: float, b: float, c: float) -> Tuple[complex, complex]:
    if a == 0:
        if b == 0:
            raise ValueError("Not a quadratic")
        return (-c / b, )
    disc = b * b - 4 * a * c
    sqrt = math.sqrt(disc) if disc >= 0 else complex(0, math.sqrt(-disc))
    r1 = (-b + sqrt) / (2 * a)
    r2 = (-b - sqrt) / (2 * a)
    return (r1, r2)

def approximate_root(func: Callable[[float], float], x0: float, tol: float = 1e-7, max_iter: int = 1000) -> float:
    # Newton-Raphson with numeric derivative
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        # numerical derivative
        h = 1e-8 if abs(x) < 1 else 1e-8 * abs(x)
        d = (func(x + h) - fx) / h
        if d == 0:
            break
        x_new = x - fx / d
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def permutations(lst: List[Any]) -> List[List[Any]]:
    return [list(p) for p in itertools.permutations(lst)]

def longest_common_subsequence(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    counts = Counter(nums)
    return [val for val, _ in counts.most_common(k)]

def flatten_nested(nested: List[Any]) -> List[Any]:
    out = []
    stack = list(reversed(nested))
    while stack:
        x = stack.pop()
        if isinstance(x, list):
            stack.extend(reversed(x))
        else:
            out.append(x)
    return out
