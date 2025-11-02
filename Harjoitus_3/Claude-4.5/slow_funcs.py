"""Intentionally slower implementations with identical APIs to `originals.py`.
These contain algorithmic inefficiencies and anti-patterns for optimization practice.
"""
from typing import List, Dict, Set, Tuple, Any, Callable, Optional
import math
from collections import defaultdict

def dijkstra_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[float, List[int]]:
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {}
    unvisited = set(graph.keys())
    
    for neighbor in graph.values():
        for n, _ in neighbor:
            unvisited.add(n)
    
    while unvisited:
        u = None
        min_dist = float('inf')
        for node in unvisited:
            if dist.get(node, float('inf')) < min_dist:
                min_dist = dist.get(node, float('inf'))
                u = node
        
        if u is None or u == end:
            break
        
        unvisited.remove(u)
        
        for v, w in graph.get(u, []):
            alt = dist[u] + w
            if alt < dist.get(v, float('inf')):
                dist[v] = alt
                prev[v] = u
    
    if end not in dist or dist[end] == float('inf'):
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
    n = len(weights)
    
    def solve(i, w):
        if i >= n or w <= 0:
            return 0
        
        if weights[i] > w:
            return solve(i + 1, w)
        
        include = values[i] + solve(i + 1, w - weights[i])
        exclude = solve(i + 1, w)
        
        return max(include, exclude)
    
    max_val = solve(0, capacity)
    
    items = []
    for mask in range(1 << n):
        total_weight = 0
        total_value = 0
        subset = []
        for i in range(n):
            if mask & (1 << i):
                total_weight += weights[i]
                total_value += values[i]
                subset.append(i)
        if total_weight <= capacity and total_value == max_val:
            items = subset
            break
    
    return (max_val, sorted(items))

def longest_increasing_subsequence(arr: List[int]) -> Tuple[int, List[int]]:
    if not arr:
        return (0, [])
    
    n = len(arr)
    dp = [1] * n
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
    
    max_len = max(dp)
    max_idx = dp.index(max_len)
    
    result = []
    idx = max_idx
    while idx >= 0:
        result.append(arr[idx])
        idx = parent[idx]
    result.reverse()
    
    return (max_len, result)

def edit_distance(s1: str, s2: str) -> Tuple[int, List[str]]:
    def rec(i, j, ops):
        if i == 0:
            return j, ops + [f"insert {s2[k]}" for k in range(j)]
        if j == 0:
            return i, ops + [f"delete {s1[k]}" for k in range(i)]
        
        if s1[i-1] == s2[j-1]:
            return rec(i-1, j-1, ops)
        
        del_cost, del_ops = rec(i-1, j, ops + [f"delete {s1[i-1]}"])
        ins_cost, ins_ops = rec(i, j-1, ops + [f"insert {s2[j-1]}"])
        rep_cost, rep_ops = rec(i-1, j-1, ops + [f"replace {s1[i-1]} with {s2[j-1]}"])
        
        min_cost = min(del_cost, ins_cost, rep_cost)
        if min_cost == del_cost:
            return del_cost, del_ops
        elif min_cost == ins_cost:
            return ins_cost, ins_ops
        else:
            return rep_cost, rep_ops
    
    m, n = len(s1), len(s2)
    if m > 10 or n > 10:
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
        
        return (dp[m][n], [])
    
    cost, ops = rec(m, n, [])
    return (cost, ops)

def suffix_array(s: str) -> List[int]:
    """Naive suffix array construction."""
    n = len(s)
    suffixes = []
    for i in range(n):
        suffixes.append((s[i:], i))
    
    for i in range(n):
        for j in range(n - 1 - i):
            if suffixes[j][0] > suffixes[j+1][0]:
                suffixes[j], suffixes[j+1] = suffixes[j+1], suffixes[j]
    
    return [idx for _, idx in suffixes]

def lru_cache_simulator(capacity: int, requests: List[int]) -> Tuple[int, int]:
    """LRU cache with list-based tracking."""
    cache = []
    hits = misses = 0
    
    for req in requests:
        if req in cache:
            hits += 1
            cache.remove(req)
            cache.append(req)
        else:
            misses += 1
            if len(cache) >= capacity:
                cache.pop(0)
            cache.append(req)
    
    return (hits, misses)

def topological_sort(graph: Dict[int, List[int]]) -> List[int]:
    """DFS-based topological sort without optimizations."""
    visited = set()
    rec_stack = set()
    result = []
    has_cycle = [False]
    
    def dfs(node):
        if node in rec_stack:
            has_cycle[0] = True
            return
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            dfs(neighbor)
        
        rec_stack.remove(node)
        result.insert(0, node)
    
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    for node in sorted(all_nodes):
        if node not in visited:
            dfs(node)
    
    if has_cycle[0]:
        return []
    
    return result

def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Naive convex hull using gift wrapping."""
    points = list(set(points))
    if len(points) <= 1:
        return points
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2
    
    hull = []
    l = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    p = l
    
    while True:
        hull.append(points[p])
        q = (p + 1) % len(points)
        
        for i in range(len(points)):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        
        p = q
        if p == l:
            break
    
    return hull

def fast_fourier_transform(coeffs: List[complex]) -> List[complex]:
    """Naive DFT instead of FFT."""
    n = len(coeffs)
    if n <= 1:
        return coeffs
    
    result = []
    for k in range(n):
        sum_val = 0
        for j in range(n):
            angle = -2 * math.pi * k * j / n
            sum_val += coeffs[j] * complex(math.cos(angle), math.sin(angle))
        result.append(sum_val)
    
    return result

def segment_tree_range_sum(arr: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """Brute force range sum without segment tree."""
    results = []
    for l, r in queries:
        if 0 <= l <= r < len(arr):
            total = 0
            for i in range(l, r + 1):
                total += arr[i]
            results.append(total)
        else:
            results.append(0)
    
    return results

def maximal_matching(graph: Dict[int, List[int]]) -> Set[Tuple[int, int]]:
    """Inefficient matching using brute force."""
    edges = []
    for u in graph:
        for v in graph[u]:
            if u < v:
                edges.append((u, v))
    
    best_matching = set()
    
    for i in range(len(edges)):
        matched = set()
        current = set()
        for j in range(i, len(edges)):
            u, v = edges[j]
            if u not in matched and v not in matched:
                current.add((u, v))
                matched.add(u)
                matched.add(v)
        
        if len(current) > len(best_matching):
            best_matching = current
    
    return best_matching

def rabin_karp_search(text: str, pattern: str) -> List[int]:
    """Naive string matching without rolling hash."""
    if not pattern or len(pattern) > len(text):
        return []
    
    matches = []
    m, n = len(pattern), len(text)
    
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            matches.append(i)
    
    return matches

def traveling_salesman_dp(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """Brute force TSP by checking all permutations."""
    n = len(dist)
    if n == 0:
        return (0, [])
    if n == 1:
        return (0, [0])
    
    def generate_perms(arr, l, r, perms):
        if l == r:
            perms.append(arr[:])
        else:
            for i in range(l, r + 1):
                arr[l], arr[i] = arr[i], arr[l]
                generate_perms(arr, l + 1, r, perms)
                arr[l], arr[i] = arr[i], arr[l]
    
    cities = list(range(1, n))
    perms = []
    generate_perms(cities, 0, len(cities) - 1, perms)
    
    min_cost = float('inf')
    best_path = []
    
    for perm in perms:
        path = [0] + perm
        cost = 0
        for i in range(len(path) - 1):
            cost += dist[path[i]][path[i+1]]
        cost += dist[path[-1]][0]
        
        if cost < min_cost:
            min_cost = cost
            best_path = path
    
    return (min_cost, best_path)

def prime_factorization(n: int) -> Dict[int, int]:
    """Naive trial division."""
    if n <= 1:
        return {}
    
    factors = {}
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    return factors

def matrix_chain_multiplication(dims: List[int]) -> Tuple[int, List[Tuple[int, int]]]:
    """Recursive matrix chain without memoization."""
    n = len(dims) - 1
    if n <= 1:
        return (0, [])
    
    def solve(i, j):
        if i == j:
            return 0
        
        min_cost = float('inf')
        for k in range(i, j):
            cost = solve(i, k) + solve(k + 1, j) + dims[i] * dims[k + 1] * dims[j + 1]
            min_cost = min(min_cost, cost)
        
        return min_cost
    
    min_ops = solve(0, n - 1)
    return (min_ops, [])

def bloom_filter_operations(size: int, items: List[str], queries: List[str]) -> Tuple[List[bool], float]:
    """Simulate bloom filter with redundant hash computations."""
    bits = [False] * size
    
    def hash_func(s, seed):
        h = seed
        for c in s:
            h = (h * 31 + ord(c)) % size
        return h
    
    for item in items:
        h1 = hash_func(item, 1)
        h2 = hash_func(item, 2)
        h3 = hash_func(item, 3)
        bits[h1] = True
        bits[h2] = True
        bits[h3] = True
    
    results = []
    for query in queries:
        h1 = hash_func(query, 1)
        h2 = hash_func(query, 2)
        h3 = hash_func(query, 3)
        results.append(bits[h1] and bits[h2] and bits[h3])
    
    false_positive_rate = sum(results) / len(results) if results else 0.0
    return (results, false_positive_rate)

def strongly_connected_components(graph: Dict[int, List[int]]) -> List[Set[int]]:
    """Kosaraju's algorithm with inefficient transpose."""
    def dfs1(v, visited, stack):
        visited.add(v)
        for w in graph.get(v, []):
            if w not in visited:
                dfs1(w, visited, stack)
        stack.append(v)
    
    def dfs2(v, visited, component):
        visited.add(v)
        component.add(v)
        for w in transpose.get(v, []):
            if w not in visited:
                dfs2(w, visited, component)
    
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)
    
    transpose = {node: [] for node in all_nodes}
    for u in graph:
        for v in graph[u]:
            transpose[v].append(u)
    
    visited = set()
    stack = []
    
    for node in all_nodes:
        if node not in visited:
            dfs1(node, visited, stack)
    
    visited = set()
    sccs = []
    
    while stack:
        node = stack.pop()
        if node not in visited:
            component = set()
            dfs2(node, visited, component)
            sccs.append(component)
    
    return sccs

def interval_scheduling(intervals: List[Tuple[int, int]]) -> List[int]:
    """Brute force interval scheduling."""
    if not intervals:
        return []
    
    n = len(intervals)
    best = []
    
    for mask in range(1 << n):
        selected = []
        for i in range(n):
            if mask & (1 << i):
                selected.append(i)
        
        # Check if valid
        valid = True
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                s1, e1 = intervals[selected[i]]
                s2, e2 = intervals[selected[j]]
                if not (e1 <= s2 or e2 <= s1):
                    valid = False
                    break
            if not valid:
                break
        
        if valid and len(selected) > len(best):
            best = selected
    
    return best
