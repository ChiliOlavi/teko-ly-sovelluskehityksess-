"""Performance comparison script for advanced optimization exercises.
Compares slow_funcs.py implementations against your ratkaisu.py solutions.
"""
import time
import sys
from typing import Callable, Any, List, Tuple
import traceback

try:
    import slow_funcs
    import originals
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

try:
    import ratkaisu
    has_solution = True
except ImportError:
    print("Warning: ratkaisu.py not found. Create it to test your solutions.")
    has_solution = False
    sys.exit(1)

# Test data generators
def generate_graph(n: int, density: float = 0.3):
    """Generate random weighted graph."""
    import random
    graph = {}
    for i in range(n):
        graph[i] = []
        for j in range(n):
            if i != j and random.random() < density:
                graph[i].append((j, random.uniform(1, 10)))
    return graph

def generate_dag(n: int):
    """Generate directed acyclic graph."""
    import random
    graph = {}
    for i in range(n):
        graph[i] = []
        for j in range(i + 1, min(i + 4, n)):
            if random.random() < 0.5:
                graph[i].append(j)
    return graph

def generate_distance_matrix(n: int):
    """Generate symmetric distance matrix."""
    import random
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = random.uniform(1, 100)
            dist[i][j] = d
            dist[j][i] = d
    return dist

# Test cases
TEST_CASES = [
    # Graph Algorithms
    {
        'name': 'dijkstra_shortest_path',
        'tests': [
            ({0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}, 0, 3),
            (generate_graph(20, 0.2), 0, 10),
        ]
    },
    {
        'name': 'topological_sort',
        'tests': [
            ({0: [1, 2], 1: [3], 2: [3], 3: []},),
            (generate_dag(15),),
        ]
    },
    {
        'name': 'strongly_connected_components',
        'tests': [
            ({0: [1], 1: [2], 2: [0, 3], 3: [4], 4: [5], 5: [3]},),
            ({i: [(i + 1) % 20, (i + 2) % 20] for i in range(20)},),
        ]
    },
    {
        'name': 'maximal_matching',
        'tests': [
            ({0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]},),
            ({i: [i+1, i+2] for i in range(0, 18, 2)},),
        ]
    },
    
    # Dynamic Programming
    {
        'name': 'knapsack_01',
        'tests': [
            ([2, 3, 4, 5], [3, 4, 5, 6], 8),
            ([1, 2, 3, 4, 5] * 3, [5, 4, 3, 2, 1] * 3, 20),
        ]
    },
    {
        'name': 'longest_increasing_subsequence',
        'tests': [
            ([10, 9, 2, 5, 3, 7, 101, 18],),
            (list(range(50, 0, -1)) + list(range(1, 51)),),
        ]
    },
    {
        'name': 'edit_distance',
        'tests': [
            ('kitten', 'sitting'),
            ('algorithm', 'altruistic'),
        ]
    },
    {
        'name': 'traveling_salesman_dp',
        'tests': [
            ([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],),
            (generate_distance_matrix(8),),
        ]
    },
    {
        'name': 'matrix_chain_multiplication',
        'tests': [
            ([10, 20, 30, 40, 30],),
            ([5, 10, 3, 12, 5, 50, 6],),
        ]
    },
    
    # String Algorithms
    {
        'name': 'suffix_array',
        'tests': [
            ('banana',),
            ('the quick brown fox jumps',),
        ]
    },
    {
        'name': 'rabin_karp_search',
        'tests': [
            ('ababcababa', 'aba'),
            ('a' * 100 + 'b' + 'a' * 100, 'b'),
        ]
    },
    
    # Data Structures
    {
        'name': 'lru_cache_simulator',
        'tests': [
            (3, [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]),
            (5, list(range(10)) * 5),
        ]
    },
    {
        'name': 'segment_tree_range_sum',
        'tests': [
            ([1, 3, 5, 7, 9, 11], [(0, 2), (1, 4), (2, 5)]),
            (list(range(100)), [(i, i+10) for i in range(0, 90, 10)]),
        ]
    },
    {
        'name': 'bloom_filter_operations',
        'tests': [
            (100, ['apple', 'banana', 'cherry'], ['apple', 'date', 'banana']),
            (1000, [f'item{i}' for i in range(100)], [f'item{i}' for i in range(50, 150)]),
        ]
    },
    
    # Geometry and Math
    {
        'name': 'convex_hull',
        'tests': [
            ([(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)],),
            ([(i/10, i*i/100) for i in range(-20, 21)],),
        ]
    },
    {
        'name': 'fast_fourier_transform',
        'tests': [
            ([complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)],),
            ([complex(i, 0) for i in range(32)],),
        ]
    },
    {
        'name': 'prime_factorization',
        'tests': [
            (60,),
            (123456,),
            (2**10 * 3**5 * 5**3,),
        ]
    },
    {
        'name': 'interval_scheduling',
        'tests': [
            ([(1, 3), (2, 5), (4, 7), (6, 9), (8, 10)],),
            ([(i, i+3) for i in range(0, 50, 2)],),
        ]
    },
]

def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call and return (result, time_in_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def results_match(r1: Any, r2: Any, tolerance: float = 1e-6) -> bool:
    """Check if two results match (with tolerance for floats)."""
    if type(r1) != type(r2):
        return False
    
    if isinstance(r1, (list, tuple)):
        if len(r1) != len(r2):
            return False
        return all(results_match(a, b, tolerance) for a, b in zip(r1, r2))
    
    if isinstance(r1, set):
        return r1 == r2
    
    if isinstance(r1, dict):
        if set(r1.keys()) != set(r2.keys()):
            return False
        return all(results_match(r1[k], r2[k], tolerance) for k in r1)
    
    if isinstance(r1, float):
        return abs(r1 - r2) < tolerance
    
    if isinstance(r1, complex):
        return abs(r1 - r2) < tolerance
    
    return r1 == r2

def run_tests():
    """Run all test cases and display results."""
    print("=" * 80)
    print("Advanced Algorithm Optimization - Performance Comparison")
    print("=" * 80)
    print()
    
    total_tests = 0
    passed_tests = 0
    total_speedup = 0
    speedup_count = 0
    
    for test_case in TEST_CASES:
        func_name = test_case['name']
        
        # Check if functions exist
        if not hasattr(slow_funcs, func_name):
            print(f"⚠️  {func_name}: Not found in slow_funcs")
            continue
        
        if not hasattr(ratkaisu, func_name):
            print(f"⚠️  {func_name}: Not found in ratkaisu")
            continue
        
        slow_func = getattr(slow_funcs, func_name)
        fast_func = getattr(ratkaisu, func_name)
        orig_func = getattr(originals, func_name) if hasattr(originals, func_name) else None
        
        print(f"\n📊 Testing {func_name}")
        print("-" * 80)
        
        for i, test_args in enumerate(test_case['tests'], 1):
            total_tests += 1
            
            try:
                # Run slow version
                slow_result, slow_time = time_function(slow_func, *test_args)
                
                # Run fast version
                fast_result, fast_time = time_function(fast_func, *test_args)
                
                # Check correctness
                if orig_func:
                    orig_result, _ = time_function(orig_func, *test_args)
                    correct = results_match(fast_result, orig_result)
                else:
                    correct = results_match(fast_result, slow_result)
                
                if correct:
                    passed_tests += 1
                    speedup = slow_time / fast_time if fast_time > 0 else float('inf')
                    total_speedup += speedup
                    speedup_count += 1
                    
                    status = "✅" if speedup > 1 else "⚠️"
                    print(f"  Test {i}: {status} Correct")
                    print(f"    Slow:  {slow_time*1000:8.3f} ms")
                    print(f"    Fast:  {fast_time*1000:8.3f} ms")
                    print(f"    Speedup: {speedup:6.2f}x {'🚀' if speedup > 5 else '⚡' if speedup > 2 else ''}")
                else:
                    print(f"  Test {i}: ❌ INCORRECT - Results don't match!")
                    print(f"    Expected: {slow_result}")
                    print(f"    Got:      {fast_result}")
            
            except Exception as e:
                print(f"  Test {i}: ❌ ERROR - {str(e)}")
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if speedup_count > 0:
        avg_speedup = total_speedup / speedup_count
        print(f"Average speedup: {avg_speedup:.2f}x")
        
        if avg_speedup > 10:
            print("🏆 Outstanding performance!")
        elif avg_speedup > 5:
            print("🌟 Excellent optimizations!")
        elif avg_speedup > 2:
            print("👍 Good improvements!")
        else:
            print("💡 Room for more optimization")
    
    print()

if __name__ == '__main__':
    run_tests()
