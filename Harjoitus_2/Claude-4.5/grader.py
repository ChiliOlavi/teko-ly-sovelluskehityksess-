"""Grader for Advanced Algorithm Challenge (Claude-4.5 level).

Tests state-of-the-art LLM capabilities across:
- Number theory (Extended GCD, CRT, Pollard Rho, Miller-Rabin)
- Numerical algorithms (Fast inverse sqrt, FFT, Karatsuba)
- Graph algorithms (Dijkstra, Convex Hull)
- String algorithms (KMP, Suffix Array, Aho-Corasick)
- Optimization (Simplex method)
- Dynamic programming (LIS)

Each test includes edge cases, numerical precision challenges, and algorithmic correctness.
"""
from __future__ import annotations
import importlib
import importlib.util
import os
import sys
from typing import Any
import math
import cmath


HERE = os.path.dirname(__file__)

def load_student_module(name: str = "ratkaisu"):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        path = os.path.join(HERE, f"{name}.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def approx_equal(a: Any, b: Any, tol: float = 1e-5) -> bool:
    """Enhanced comparison for complex numbers, lists, dicts, tuples."""
    import numbers
    
    # Handle complex numbers
    if isinstance(a, complex) or isinstance(b, complex):
        try:
            return abs(complex(a) - complex(b)) <= tol
        except:
            return False
    
    # Handle regular numbers
    if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
        if math.isnan(a) and math.isnan(b):
            return True
        if math.isinf(a) and math.isinf(b):
            return a == b
        return abs(float(a) - float(b)) <= tol
    
    # Handle lists/tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(approx_equal(x, y, tol) for x, y in zip(a, b))
    
    # Handle dicts
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(approx_equal(a[k], b[k], tol) for k in a.keys())
    
    # Handle sets (convert to sorted lists)
    if isinstance(a, set) and isinstance(b, set):
        return approx_equal(sorted(list(a)), sorted(list(b)), tol)
    
    return a == b


def run_test(fn_name: str, args: tuple, kwargs: dict, orig_mod, stud_mod) -> tuple[bool, str]:
    """Run a single test and return (passed, error_msg)."""
    orig_fn = getattr(orig_mod, fn_name)
    expected = None
    
    try:
        expected = orig_fn(*args, **kwargs)
    except Exception as e:
        return (False, f"Reference implementation error: {e}")

    stud_fn = getattr(stud_mod, fn_name, None)
    if stud_fn is None:
        return (False, f"Function '{fn_name}' not found in student solution")
    
    try:
        got = stud_fn(*args, **kwargs)
    except Exception as e:
        return (False, f"Runtime error: {e}")
    
    if approx_equal(got, expected):
        return (True, "")
    else:
        return (False, f"Expected {expected}, got {got}")


def main():
    sys.path.insert(0, HERE)
    
    try:
        import originals
    except Exception as e:
        print(f"Failed to import originals.py: {e}")
        return

    stud = load_student_module("ratkaisu")
    if stud is None:
        print("Could not find 'ratkaisu.py'. Create it and rerun grader.")
        return

    # Comprehensive test suite
    tests = {
        "extended_gcd": [
            ((48, 18), {}),
            ((100, 35), {}),
            ((17, 0), {}),
            ((-48, 18), {}),
            ((1071, 462), {}),
        ],
        "matrix_determinant": [
            (([[2, 3], [1, 4]],), {}),
            (([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), {}),
            (([[6, 1, 1], [4, -2, 5], [2, 8, 7]],), {}),
            (([[1]],), {}),
            (([[2, -1, 0], [1, 3, 2], [-1, 0, 4]],), {}),
        ],
        "fast_inverse_sqrt": [
            ((4.0,), {}),
            ((9.0,), {}),
            ((1.0,), {}),
            ((100.0,), {}),
            ((0.25,), {}),
        ],
        "chinese_remainder_theorem": [
            (([2, 3, 2], [3, 5, 7]), {}),
            (([0, 3, 4], [3, 4, 5]), {}),
            (([1, 4, 6], [3, 5, 7]), {}),
        ],
        "pollard_rho": [
            ((15,), {}),
            ((77,), {}),
            ((1234567,), {}),
            ((8051,), {}),
        ],
        "fft": [
            (([1+0j, 1+0j, 1+0j, 1+0j],), {}),
            (([1+0j, 0+0j, 0+0j, 0+0j],), {}),
            (([1+1j, 2-1j, 0+0j, 0+0j],), {}),
            (([complex(i, 0) for i in range(8)],), {}),
        ],
        "karatsuba_multiply": [
            ((123, 456), {}),
            ((1234, 5678), {}),
            ((7, 8), {}),
            ((9999, 9999), {}),
            ((12345678, 87654321), {}),
        ],
        "dijkstra_shortest_path": [
            (({0: [(1, 4), (2, 1)], 1: [(3, 1)], 2: [(1, 2), (3, 5)], 3: []}, 0, 3), {}),
            (({0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 5)], 2: [(3, 1)], 3: []}, 0, 3), {}),
            (({0: [(1, 2)], 1: [(2, 3)], 2: [(3, 1)], 3: []}, 0, 3), {}),
        ],
        "convex_hull": [
            (([(0, 0), (1, 1), (0, 1), (1, 0), (0.5, 0.5)],), {}),
            (([(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)],), {}),
            (([(0, 0), (1, 0), (0, 1)],), {}),
            (([(0, 0), (3, 0), (3, 3), (0, 3), (1, 1), (2, 2)],), {}),
        ],
        "longest_increasing_subsequence": [
            (([10, 9, 2, 5, 3, 7, 101, 18],), {}),
            (([0, 1, 0, 3, 2, 3],), {}),
            (([7, 7, 7, 7, 7, 7, 7],), {}),
            (([],), {}),
            (([1, 3, 6, 7, 9, 4, 10, 5, 6],), {}),
        ],
        "knuth_morris_pratt": [
            (("ABABDABACDABABCABAB", "ABABCABAB"), {}),
            (("AABAACAADAABAABA", "AABA"), {}),
            (("abcdefgh", "xyz"), {}),
            (("aaaaa", "aa"), {}),
        ],
        "suffix_array": [
            (("banana",), {}),
            (("mississippi",), {}),
            (("abracadabra",), {}),
            (("aaa",), {}),
        ],
        "simplex_method": [
            (([3, 5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18]), {}),
            (([1, 1], [[1, 1], [-1, 1]], [2, 1]), {}),
        ],
        "miller_rabin": [
            ((2,), {}),
            ((3,), {}),
            ((4,), {}),
            ((17,), {}),
            ((561,), {}),  # Carmichael number
            ((1009,), {}),
            ((1000000007,), {}),
        ],
        "aho_corasick_search": [
            (("ushers", ["he", "she", "his", "hers"]), {}),
            (("abcdefgh", ["abc", "def", "gh"]), {}),
            (("aaaa", ["aa"]), {}),
            (("the quick brown fox", ["quick", "fox", "the"]), {}),
        ],
    }

    total = 0
    max_score = len(tests)
    details = []
    
    for fn_name, cases in tests.items():
        all_passed = True
        error_msgs = []
        
        for args, kwargs in cases:
            if not isinstance(args, tuple):
                args = (args,)
            
            passed, msg = run_test(fn_name, args, kwargs, originals, stud)
            if not passed:
                all_passed = False
                error_msgs.append(f"  Args {args}: {msg}")
        
        details.append((fn_name, all_passed, error_msgs))
        if all_passed:
            total += 1

    print("\n" + "="*80)
    print("ADVANCED ALGORITHM CHALLENGE - GRADER RESULTS")
    print("="*80)
    print("\nThis test suite challenges state-of-the-art LLMs with:")
    print("  • Advanced number theory algorithms")
    print("  • Numerical methods with precision requirements")
    print("  • Complex graph and geometry algorithms")
    print("  • Sophisticated string matching algorithms")
    print("  • Linear programming optimization")
    print("="*80 + "\n")
    
    for fn_name, ok, msgs in details:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:8s} {fn_name:35s}")
        if not ok and msgs:
            for msg in msgs[:2]:  # Show first 2 errors
                print(f"         {msg}")
    
    print("\n" + "="*80)
    percentage = (total / max_score * 100) if max_score > 0 else 0
    print(f"FINAL SCORE: {total}/{max_score} ({percentage:.1f}%)")
    print("="*80)
    
    if total == max_score:
        print("\n🎉 PERFECT SCORE! You've mastered advanced algorithms!")
    elif total >= max_score * 0.8:
        print("\n🌟 EXCELLENT! You're performing at state-of-the-art level!")
    elif total >= max_score * 0.6:
        print("\n👍 GOOD! Keep refining those tricky algorithms!")
    else:
        print("\n💪 Keep practicing! These are challenging problems!")


if __name__ == "__main__":
    main()
