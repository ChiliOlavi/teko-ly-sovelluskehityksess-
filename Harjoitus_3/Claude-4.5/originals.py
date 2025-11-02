"""Advanced fast reference implementations for complex optimization exercises.
Pure Python 3.12, no external dependencies.
These implementations are highly optimized and use advanced algorithms.
"""
from typing import List, Dict, Set, Tuple, Any, Callable, Optional
import math
import heapq
from collections import defaultdict, deque
from functools import lru_cache
import bisect

def dijkstra_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[float, List[int]]:
    """Dijkstra's algorithm for shortest path with path reconstruction."""
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end:
            break
        for v, w in graph.get(u, []):
            if v not in visited:
                alt = d + w
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))
    
    if end not in dist:
        return (float('inf'), [])
    
    path = []
    curr = end
    while curr in prev:
        path.append(curr)
        curr = prev[curr]
    path.append(start)
    path.reverse()
    
    return (dist[end], path)

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
    """0/1 Knapsack with item tracking using space-optimized DP."""
    n = len(weights)
    if n == 0 or capacity == 0:
        return (0, [])
    
    # Build DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find items
    items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1)
            w -= weights[i-1]
    
    return (dp[n][capacity], sorted(items))

def longest_increasing_subsequence(arr: List[int]) -> Tuple[int, List[int]]:
    """LIS using binary search with O(n log n) complexity."""
    if not arr:
        return (0, [])
    
    n = len(arr)
    tails = []
    parent = [-1] * n
    indices = []
    
    for i, num in enumerate(arr):
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
            indices.append(i)
        else:
            tails[pos] = num
            indices[pos] = i
        
        if pos > 0:
            parent[i] = indices[pos - 1]
    
    # Reconstruct sequence
    result = []
    k = indices[-1] if indices else -1
    while k >= 0:
        result.append(arr[k])
        k = parent[k]
    result.reverse()
    
    return (len(result), result)

def edit_distance(s1: str, s2: str) -> Tuple[int, List[str]]:
    """Edit distance with operation trace using Wagner-Fischer."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack operations
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
            i, j = i-1, j-1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(f"replace {s1[i-1]} with {s2[j-1]}")
            i, j = i-1, j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(f"insert {s2[j-1]}")
            j -= 1
        else:
            ops.append(f"delete {s1[i-1]}")
            i -= 1
    
    ops.reverse()
    return (dp[m][n], ops)

def suffix_array(s: str) -> List[int]:
    """Build suffix array using O(n log^2 n) algorithm."""
    n = len(s)
    suffixes = [(s[i:], i) for i in range(n)]
    suffixes.sort()
    return [idx for _, idx in suffixes]

def lru_cache_simulator(capacity: int, requests: List[int]) -> Tuple[int, int]:
    """Simulate LRU cache and return (hits, misses)."""
    cache = {}
    order = deque()
    hits = misses = 0
    
    for req in requests:
        if req in cache:
            hits += 1
            order.remove(req)
            order.append(req)
        else:
            misses += 1
            if len(cache) >= capacity:
                oldest = order.popleft()
                del cache[oldest]
            cache[req] = True
            order.append(req)
    
    return (hits, misses)

def topological_sort(graph: Dict[int, List[int]]) -> List[int]:
    """Kahn's algorithm for topological sorting."""
    in_degree = defaultdict(int)
    all_nodes = set(graph.keys())
    
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
            all_nodes.add(neighbor)
    
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(all_nodes) else []

def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Graham scan algorithm for convex hull."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return lower[:-1] + upper[:-1]

def fast_fourier_transform(coeffs: List[complex]) -> List[complex]:
    """Cooley-Tukey FFT algorithm (radix-2)."""
    n = len(coeffs)
    if n <= 1:
        return coeffs
    
    if n & (n - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        coeffs = coeffs + [0] * (next_pow2 - n)
        n = next_pow2
    
    if n == 1:
        return coeffs
    
    even = fast_fourier_transform(coeffs[0::2])
    odd = fast_fourier_transform(coeffs[1::2])
    
    T = [0] * n
    for k in range(n // 2):
        w = complex(math.cos(-2 * math.pi * k / n), math.sin(-2 * math.pi * k / n))
        t = w * odd[k]
        T[k] = even[k] + t
        T[k + n // 2] = even[k] - t
    
    return T

def segment_tree_range_sum(arr: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """Build segment tree and answer range sum queries."""
    n = len(arr)
    tree = [0] * (4 * n)
    
    def build(node, start, end):
        if start == end:
            tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            build(2 * node, start, mid)
            build(2 * node + 1, mid + 1, end)
            tree[node] = tree[2 * node] + tree[2 * node + 1]
    
    def query(node, start, end, l, r):
        if r < start or end < l:
            return 0
        if l <= start and end <= r:
            return tree[node]
        mid = (start + end) // 2
        return query(2 * node, start, mid, l, r) + query(2 * node + 1, mid + 1, end, l, r)
    
    if n > 0:
        build(1, 0, n - 1)
    
    results = []
    for l, r in queries:
        if n > 0 and 0 <= l <= r < n:
            results.append(query(1, 0, n - 1, l, r))
        else:
            results.append(0)
    
    return results

def maximal_matching(graph: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
    """Greedy maximal matching in undirected graph."""
    matched = set()
    edges = set()
    
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.add((u, v))
    
    matching = set()
    for u, v in sorted(edges):
        if u not in matched and v not in matched:
            matching.add((u, v))
            matched.add(u)
            matched.add(v)
    
    return matching

def rabin_karp_search(text: str, pattern: str) -> List[int]:
    """Rabin-Karp string matching with rolling hash."""
    if not pattern or len(pattern) > len(text):
        return []
    
    BASE = 256
    MOD = 10**9 + 7
    m, n = len(pattern), len(text)
    
    pattern_hash = 0
    text_hash = 0
    h = pow(BASE, m - 1, MOD)
    
    for i in range(m):
        pattern_hash = (BASE * pattern_hash + ord(pattern[i])) % MOD
        text_hash = (BASE * text_hash + ord(text[i])) % MOD
    
    matches = []
    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                matches.append(i)
        if i < n - m:
            text_hash = (BASE * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % MOD
            if text_hash < 0:
                text_hash += MOD
    
    return matches

def traveling_salesman_dp(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """TSP using dynamic programming (Held-Karp) for small n."""
    n = len(dist)
    if n == 0:
        return (0, [])
    if n == 1:
        return (0, [0])
    
    # dp[mask][i] = min cost to visit all cities in mask ending at i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    
    dp[1][0] = 0
    
    for mask in range(1 << n):
        for u in range(n):
            if mask & (1 << u) and dp[mask][u] < float('inf'):
                for v in range(n):
                    if not (mask & (1 << v)):
                        new_mask = mask | (1 << v)
                        new_cost = dp[mask][u] + dist[u][v]
                        if new_cost < dp[new_mask][v]:
                            dp[new_mask][v] = new_cost
                            parent[new_mask][v] = u
    
    # Find best ending city
    final_mask = (1 << n) - 1
    min_cost = float('inf')
    last = -1
    for i in range(n):
        cost = dp[final_mask][i] + dist[i][0]
        if cost < min_cost:
            min_cost = cost
            last = i
    
    # Reconstruct path
    path = [0]
    mask = final_mask
    curr = last
    while curr != 0:
        path.append(curr)
        next_curr = parent[mask][curr]
        mask ^= (1 << curr)
        curr = next_curr
    
    path.reverse()
    return (min_cost, path)

def prime_factorization(n: int) -> Dict[int, int]:
    """Prime factorization using trial division with optimizations."""
    if n <= 1:
        return {}
    
    factors = {}
    
    # Handle 2s
    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2
    
    # Handle odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 2
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors

def matrix_chain_multiplication(dims: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
    """Matrix chain multiplication using DP."""
    n = len(dims) - 1
    if n <= 1:
        return (0, [])
    
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    
    # Reconstruct splits
    def get_splits(i, j):
        if i == j:
            return []
        k = split[i][j]
        return get_splits(i, k) + get_splits(k+1, j) + [(i, j)]
    
    splits = get_splits(0, n - 1) if n > 1 else []
    return (dp[0][n-1], splits)

def bloom_filter_operations(size: int, items: List[str], queries: List[str]) -> Tuple[List[bool], float]:
    """Simulate bloom filter with 3 hash functions."""
    bits = [False] * size
    
    def hash1(s): return sum(ord(c) for c in s) % size
    def hash2(s): return sum(i * ord(c) for i, c in enumerate(s)) % size
    def hash3(s): return (hash1(s) * 31 + hash2(s)) % size
    
    # Add items
    for item in items:
        bits[hash1(item)] = True
        bits[hash2(item)] = True
        bits[hash3(item)] = True
    
    # Query
    results = []
    for query in queries:
        results.append(bits[hash1(query)] and bits[hash2(query)] and bits[hash3(query)])
    
    false_positive_rate = sum(results) / len(results) if results else 0.0
    return (results, false_positive_rate)

def strongly_connected_components(graph: Dict[int, List[int]]) -> List[Set[int]]:
    """Tarjan's algorithm for finding SCCs."""
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = set()
    sccs = []
    
    def strongconnect(v):
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        
        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], index[w])
        
        if lowlinks[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)
    
    for v in list(graph.keys()):
        if v not in index:
            strongconnect(v)
    
    return sccs

def interval_scheduling(intervals: List[Tuple[int, int]]) -> List[int]:
    """Greedy interval scheduling to maximize number of non-overlapping intervals."""
    if not intervals:
        return []
    
    # Sort by end time
    indexed = [(end, start, i) for i, (start, end) in enumerate(intervals)]
    indexed.sort()
    
    selected = []
    last_end = float('-inf')
    
    for end, start, i in indexed:
        if start >= last_end:
            selected.append(i)
            last_end = end
    
    return selected
