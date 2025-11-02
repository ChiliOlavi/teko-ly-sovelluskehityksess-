"""Buggy implementations for the advanced algorithm exercise.
These functions run but return incorrect results.
Do NOT include comments that reveal bug locations.
"""
from typing import List, Any, Tuple, Dict, Optional, Callable
import math
import cmath

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm - returns (gcd, x, y) where gcd = ax + by."""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    gcd, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (gcd, y, x)

def matrix_determinant(matrix: List[List[float]]) -> float:
    """Calculate determinant using cofactor expansion."""
    n = len(matrix)
    if n == 0:
        return 1.0
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0.0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += matrix[0][j] * matrix_determinant(minor)
    return det

def fast_inverse_sqrt(x: float) -> float:
    """Implement Quake III fast inverse square root algorithm."""
    import struct
    threehalfs = 1.5
    x2 = x * 0.5
    i = struct.unpack('>l', struct.pack('>f', x))[0]
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('>f', struct.pack('>l', i))[0]
    y = y * (threehalfs - (x2 * y * y))
    return y

def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """Solve system of congruences using CRT."""
    if len(remainders) != len(moduli):
        raise ValueError("Remainders and moduli must have same length")
    
    M = 1
    for m in moduli:
        M *= m
    
    x = 0
    for i in range(len(remainders)):
        Mi = M // moduli[i]
        yi = Mi // moduli[i]
        x += remainders[i] * Mi * yi
    
    return x % M

def pollard_rho(n: int, max_iter: int = 100000) -> Optional[int]:
    """Pollard's rho algorithm for integer factorization."""
    if n % 2 == 0:
        return 2
    
    x, y, d = 2, 2, 1
    f = lambda x: (x * x + 2) % n
    
    for _ in range(max_iter):
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
        if 1 < d < n:
            return d
    return None

def fft(signal: List[complex]) -> List[complex]:
    """Cooley-Tukey FFT algorithm."""
    n = len(signal)
    if n <= 1:
        return signal
    if n % 2 != 0:
        raise ValueError("Signal length must be power of 2")
    
    even = fft([signal[i] for i in range(0, n, 2)])
    odd = fft([signal[i] for i in range(1, n, 2)])
    
    result = [0j] * n
    for k in range(n // 2):
        t = cmath.exp(2j * cmath.pi * k / n) * odd[k]
        result[k] = even[k] + t
        result[k + n // 2] = even[k] - t
    
    return result

def karatsuba_multiply(x: int, y: int) -> int:
    """Karatsuba algorithm for fast multiplication."""
    if x < 10 or y < 10:
        return x * y
    
    n = max(len(str(abs(x))), len(str(abs(y))))
    m = n // 2
    
    high1, low1 = divmod(x, 10**m)
    high2, low2 = divmod(y, 10**m)
    
    z0 = karatsuba_multiply(low1, low2)
    z1 = karatsuba_multiply(low1 + high1, low2 + high2)
    z2 = karatsuba_multiply(high1, high2)
    
    return z2 * 10**(2*m) + (z1 - z2) * 10**m + z0

def dijkstra_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[float, List[int]]:
    """Dijkstra's algorithm with path reconstruction."""
    import heapq
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current == end:
            break
        
        if current_dist > distances[current]:
            continue
        
        for neighbor, weight in graph.get(current, []):
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    return (distances[end], path if path[0] == start else [])

def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Graham scan algorithm for convex hull."""
    if len(points) < 3:
        return points
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) + (a[1] - o[1]) * (b[0] - o[0])
    
    points = sorted(set(points))
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]

def longest_increasing_subsequence(arr: List[int]) -> int:
    """Find length of LIS using binary search optimization."""
    if not arr:
        return 0
    
    tails = []
    
    for num in arr:
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] <= num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)

def knuth_morris_pratt(text: str, pattern: str) -> List[int]:
    """KMP string matching algorithm."""
    if not pattern:
        return []
    
    lps = [0] * len(pattern)
    length = 0
    i = 1
    
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    matches = []
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        
        if j == len(pattern):
            matches.append(i - j + 1)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return matches

def suffix_array(s: str) -> List[int]:
    """Build suffix array using prefix doubling."""
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n
    
    k = 1
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i-1]]
            if rank[sa[i]] != rank[sa[i-1]] or rank[sa[i] + k] != rank[sa[i-1] + k]:
                tmp[sa[i]] += 1
        
        rank = tmp[:]
        k *= 2
    
    return sa

def simplex_method(c: List[float], A: List[List[float]], b: List[float]) -> Optional[Tuple[float, List[float]]]:
    """Simplex algorithm for linear programming (maximization)."""
    m, n = len(A), len(c)
    
    tableau = [A[i] + [1 if j == i else 0 for j in range(m)] + [b[i]] for i in range(m)]
    tableau.append([-x for x in c] + [0] * (m + 1))
    
    while True:
        pivot_col = min(range(n + m), key=lambda j: tableau[-1][j])
        if tableau[-1][pivot_col] >= 0:
            break
        
        ratios = []
        for i in range(m):
            if tableau[i][pivot_col] > 0:
                ratios.append((tableau[i][-1] / tableau[i][pivot_col], i))
        
        if not ratios:
            return None
        
        pivot_row = min(ratios)[1]
        
        pivot = tableau[pivot_row][pivot_col]
        tableau[pivot_row] = [x / pivot for x in tableau[pivot_row]]
        
        for i in range(m + 1):
            if i != pivot_row:
                factor = tableau[i][pivot_col]
                tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] 
                             for j in range(n + m + 1)]
    
    solution = [0.0] * n
    for i in range(m):
        for j in range(n):
            if abs(tableau[i][j] - 1) < 1e-9:
                if all(abs(tableau[k][j]) < 1e-9 for k in range(m) if k != i):
                    solution[j] = tableau[i][-1]
    
    return (-tableau[-1][-1], solution)

def miller_rabin(n: int, k: int = 5) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2:
        return False
    if n % 2 == 0:
        return False
    
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    import random
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        
        if x == 1 or x == n - 1:
            continue
        
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    
    return True

def aho_corasick_search(text: str, patterns: List[str]) -> Dict[str, List[int]]:
    """Aho-Corasick algorithm for multiple pattern matching."""
    
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.output = []
            self.fail = None
    
    root = TrieNode()
    
    for pattern in patterns:
        node = root
        for char in pattern:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.output.append(pattern)
    
    from collections import deque
    queue = deque()
    
    for child in root.children.values():
        child.fail = root
        queue.append(child)
    
    while queue:
        current = queue.popleft()
        
        for char, child in current.children.items():
            queue.append(child)
            
            fail_node = current.fail
            while fail_node != root and char not in fail_node.children:
                fail_node = fail_node.fail
            
            if char in fail_node.children and fail_node.children[char] != child:
                child.fail = fail_node.children[char]
            else:
                child.fail = root
    
    results = {pattern: [] for pattern in patterns}
    node = root
    
    for i, char in enumerate(text):
        while node != root and char not in node.children:
            node = node.fail
        
        if char in node.children:
            node = node.children[char]
        
        for pattern in node.output:
            results[pattern].append(i - len(pattern) + 1)
    
    return results
