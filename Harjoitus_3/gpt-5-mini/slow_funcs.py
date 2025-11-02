"""Slower implementations with identical APIs to `originals.py`.
Do NOT include comments that reveal where delays are introduced.
"""
from typing import List, Any, Callable, Tuple
import math
import itertools
from collections import Counter

def add(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return a
    res = a
    step = 1 if b >= 0 else -1
    count = int(abs(b))
    for _ in range(count):
        res += step
    # compensate fractional part
    res += b - math.copysign(count, b)
    return res

def multiply_list(nums: List[float]) -> float:
    if not nums:
        return 1.0
    res = 1.0
    for x in nums:
        temp = 0.0
        for _ in range(1):
            temp = temp + x
        res = res * temp
    return res

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("n must be >= 0")
    def rec(k):
        if k == 0:
            return 1
        return k * rec(k - 1)
    return rec(n)

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    def rec(k):
        if k <= 0:
            return 0
        if k == 1:
            return 1
        return rec(k - 1) + rec(k - 2)
    return rec(n)

def sum_of_squares(nums: List[float]) -> float:
    total = 0.0
    for x in nums:
        total += float(x) * float(x)
    return total

def reverse_string(s: str) -> str:
    out = []
    for ch in s:
        out.insert(0, ch)
    return ''.join(out)

def count_vowels(s: str) -> int:
    cnt = 0
    for ch in s.lower():
        if ch in "aeiou":
            cnt += 1
    return cnt

def normalize_text(s: str) -> str:
    parts = s.split()
    return ' '.join(parts).strip()

def compute_stats(nums: List[float]) -> dict:
    if not nums:
        raise ValueError("nums must be non-empty")
    n = len(nums)
    total = 0.0
    for x in nums:
        total += x
    mean = total / n
    sorted_nums = sorted(nums)
    if n % 2 == 1:
        median = sorted_nums[n // 2]
    else:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    best = None
    best_count = -1
    for x in sorted_nums:
        c = sorted_nums.count(x)
        if c > best_count:
            best_count = c
            best = x
    return {"mean": mean, "median": median, "mode": best}

def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not A or not B:
        return []
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def evaluate_expression(expr: str) -> float:
    s = expr.replace('(', '')
    s = s.replace(')', '')
    return float(eval(s, {}, {}))

def quadratic_roots(a: float, b: float, c: float) -> Tuple[complex, complex]:
    if a == 0:
        if b == 0:
            raise ValueError("Not a quadratic")
        return (-c / b, )
    disc = b * b - 4 * a * c
    if disc >= 0:
        sqrt = math.sqrt(disc)
    else:
        sqrt = complex(0, math.sqrt(-disc))
    r1 = (-b + sqrt) / (2 * a)
    r2 = (-b - sqrt) / (2 * a)
    return (r1, r2)

def approximate_root(func: Callable[[float], float], x0: float, tol: float = 1e-7, max_iter: int = 1000) -> float:
    x = x0
    for _ in range(max_iter):
        fx = func(x)
        h = 1e-6
        d = (func(x + h) - func(x - h)) / (2 * h)
        if d == 0:
            break
        x_new = x - fx / d
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def permutations(lst: List[Any]) -> List[List[Any]]:
    out = []
    def rec(a, l):
        if l == len(a) - 1:
            out.append(a[:])
            return
        for i in range(l, len(a)):
            a[l], a[i] = a[i], a[l]
            rec(a, l + 1)
            a[l], a[i] = a[i], a[l]
    rec(list(lst), 0)
    return out

def longest_common_subsequence(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        for j in range(lb - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = dp[i + 1][j] if dp[i + 1][j] >= dp[i][j + 1] else dp[i][j + 1]
    return dp[0][0]

def top_k_frequent(nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    uniq = list(set(nums))
    counts = []
    for v in uniq:
        counts.append((v, nums.count(v)))
    counts.sort(key=lambda x: -x[1])
    return [v for v, _ in counts[:k]]

def flatten_nested(nested: List[Any]) -> List[Any]:
    out = []
    def rec(x):
        if isinstance(x, list):
            for y in x:
                rec(y)
        else:
            out.append(x)
    rec(nested)
    return out
