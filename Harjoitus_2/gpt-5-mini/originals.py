"""Original (correct) implementations for Harjoitus 2: Virheiden korjaaminen tekoälyllä.

All functions are written for Python 3.12 and use only the standard library.
These are the canonical, correct implementations that the student's fixes should match.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import math
import ast

def add(a: float, b: float) -> float:
    """Return sum of two numbers."""
    return a + b

def multiply_list(nums: List[float]) -> float:
    """Return product of a list of numbers. Empty list -> 1."""
    prod = 1.0
    for x in nums:
        prod *= x
    return prod

def is_prime(n: int) -> bool:
    """Return True if n is a prime number (n >= 2)."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.sqrt(n))
    for i in range(3, r + 1, 2):
        if n % i == 0:
            return False
    return True

def factorial(n: int) -> int:
    """Return n! for n >= 0."""
    if n < 0:
        raise ValueError("n must be >= 0")
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number (0-based): fib(0)=0, fib(1)=1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def sum_of_squares(nums: List[float]) -> float:
    """Return the sum of squares of the list elements."""
    return sum(x * x for x in nums)

def reverse_string(s: str) -> str:
    """Return the reversed string."""
    return s[::-1]

def count_vowels(s: str) -> int:
    """Count vowels (aeiouAEIOU) in a string."""
    return sum(1 for ch in s if ch.lower() in "aeiou")

def normalize_text(s: str) -> str:
    """Lowercase, strip leading/trailing whitespace and replace multiple spaces with single space."""
    return " ".join(s.strip().lower().split())

def compute_stats(nums: List[float]) -> Dict[str, Any]:
    """Return mean, median and mode (mode: first encountered) for a list.

    For empty list, raise ValueError.
    """
    if not nums:
        raise ValueError("nums must be non-empty")
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    mean = sum(sorted_nums) / n
    if n % 2 == 1:
        median = sorted_nums[n // 2]
    else:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    # simple mode: element with highest frequency, break ties by first occurrence
    freq: Dict[float, int] = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    mode = max(freq.items(), key=lambda kv: (kv[1], -list(nums).index(kv[0])))[0]
    return {"mean": mean, "median": median, "mode": mode}

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices A (m x p) and B (p x n) -> (m x n)."""
    if not A or not B:
        return []
    m = len(A)
    p = len(A[0])
    if any(len(row) != p for row in A):
        raise ValueError("All rows in A must have same length")
    n = len(B[0])
    if len(B) != p:
        raise ValueError("Incompatible matrix shapes for multiplication")
    # compute
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def evaluate_expression(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression containing numbers and +-*/**()."""
    # parse with ast and allow only arithmetic nodes
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Num):
            return n.n
        if isinstance(n, ast.BinOp):
            l = _eval(n.left)
            r = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            if isinstance(n.op, ast.Mult):
                return l * r
            if isinstance(n.op, ast.Div):
                return l / r
            if isinstance(n.op, ast.Pow):
                return l ** r
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return +v
        raise ValueError("Unsupported expression")

    return float(_eval(node))

def quadratic_roots(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Return the two roots of ax^2 + bx + c. a may be zero (then linear)."""
    if a == 0:
        if b == 0:
            raise ValueError("Not an equation")
        return (-c / b, complex(float("nan"), 0.0))
    d = b * b - 4 * a * c
    sqrt_d = math.sqrt(d) if d >= 0 else math.sqrt(-d) * 1j
    r1 = (-b + sqrt_d) / (2 * a)
    r2 = (-b - sqrt_d) / (2 * a)
    return (r1, r2)

def approximate_root(func, x0: float, tol: float = 1e-7, max_iter: int = 1000) -> float:
    """Approximate a root of func using Newton-Raphson with numeric derivative."""
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        # numeric derivative
        h = 1e-6
        dfx = (func(x + h) - func(x - h)) / (2 * h)
        if dfx == 0:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def permutations(lst: List[Any]) -> List[List[Any]]:
    """Return list of permutations of the input list (order of permutations deterministic)."""
    if not lst:
        return [[]]
    res: List[List[Any]] = []
    for i, v in enumerate(lst):
        for tail in permutations(lst[:i] + lst[i+1:]):
            res.append([v] + tail)
    return res

if __name__ == "__main__":
    # quick smoke runs
    print("add(2,3)", add(2,3))
    print("is_prime(101)", is_prime(101))
