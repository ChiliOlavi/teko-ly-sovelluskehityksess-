"""Compare execution time of functions in slow_funcs.py vs student's ratkaisu.py.
Run with Python 3.12 in the Harjoitus_3 folder.
"""
from __future__ import annotations
import importlib.util
import importlib
import os
import sys
import time
from typing import Any, Callable

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

def load_module_by_name_or_path(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        # try loading by file path; handle import-time errors gracefully
        path = os.path.join(HERE, f"{name}.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"Failed to import module '{name}' from path {path}: {e}")
            return None
        return mod

def approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        import numbers
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return abs(complex(a) - complex(b)) <= tol
    except Exception:
        pass
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(approx_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(approx_equal(a[k], b[k], tol) for k in a.keys())
    return a == b


def time_function(fn: Callable, args: tuple, kwargs: dict, iterations: int) -> float:
    # run a quick warmup
    for _ in range(2):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn(*args, **kwargs)
    t1 = time.perf_counter()
    return (t1 - t0) / iterations


def main():
    slow = load_module_by_name_or_path('slow_funcs')
    if slow is None:
        print("Could not find slow_funcs.py in this folder.")
        return

    stud = load_module_by_name_or_path('ratkaisu')
    if stud is None:
        print("Could not find ratkaisu.py — create it from ratkaisu_template.py and rerun.")
        return

    # tests and default iteration counts (lower for expensive functions)
    tests = {
        'add': (((2, 3), {}), 100000),
        'multiply_list': ((([2, 3, 4, 5],), {}), 10000),
        'is_prime': (((10007,), {}), 1000),
        'factorial': (((10,), {}), 1000),
        'fibonacci': (((20,), {}), 1000),
        'sum_of_squares': (((list(range(50)),), {}), 2000),
        'reverse_string': ((("hello world",), {}), 10000),
        'count_vowels': ((("Banana",
                           ), {}), 10000),
        'normalize_text': ((("  Hello   WORLD  ",), {}), 10000),
        'compute_stats': ((([1,2,2,3,5,8],), {}), 2000),
        'matrix_multiply': ((([[1,2],[3,4]], [[5,6],[7,8]]), {}), 500),
        'evaluate_expression': ((("2*(3+4)",), {}), 5000),
        'quadratic_roots': (((1,0,-1), {}), 2000),
        'approximate_root': (((lambda x: x*x - 2, 1.0), {}), 200),
        'permutations': ((([1,2,3,4],), {}), 200),
        'longest_common_subsequence': ((("AGGTAB","GXTXAYB"), {}), 500),
        'top_k_frequent': ((([1,2,2,3,3,3,4], 2), {}), 2000),
        'flatten_nested': ((([1,[2,[3,4],5],6],), {}), 2000),
    }

    print("Timing comparison: slow_funcs.py  vs  ratkaisu.py\n")
    results = []
    for fn_name, ((args, kwargs), iters) in tests.items():
        orig_fn = getattr(slow, fn_name, None)
        stud_fn = getattr(stud, fn_name, None)
        if orig_fn is None:
            print(f"{fn_name:25s}: missing in slow_funcs.py — skipping")
            continue
        if stud_fn is None:
            print(f"{fn_name:25s}: missing in ratkaisu.py — student must implement")
            continue
        # ensure args is tuple
        if not isinstance(args, tuple):
            args = (args,)
        # correctness check
        try:
            expected = orig_fn(*args, **kwargs)
        except Exception as e:
            print(f"{fn_name:25s}: error when running slow implementation: {e}")
            continue
        try:
            got = stud_fn(*args, **kwargs)
        except Exception as e:
            print(f"{fn_name:25s}: student's implementation raised an exception: {e}")
            continue
        if not approx_equal(got, expected):
            print(f"{fn_name:25s}: WRONG RESULT — student's output differs from slow_funcs")
            continue

        # timing
        try:
            t_slow = time_function(orig_fn, args, kwargs, max(1, iters))
            t_stud = time_function(stud_fn, args, kwargs, max(1, iters))
        except Exception as e:
            print(f"{fn_name:25s}: error during timing: {e}")
            continue

        speedup = t_slow / t_stud if t_stud > 0 else float('inf')
        results.append((fn_name, t_slow, t_stud, speedup))

    # print summary
    print("\nSummary:\n")
    print(f"{'Function':25}  slow(s)   stud(s)   speedup")
    for fn_name, t_slow, t_stud, speedup in results:
        print(f"{fn_name:25s}: {t_slow:8.6f}  {t_stud:8.6f}  {speedup:7.2f}x")

    faster = sum(1 for r in results if r[3] > 1.0)
    total = len(results)
    print(f"\nFunctions faster in ratkaisu.py: {faster} / {total}")


if __name__ == '__main__':
    main()
