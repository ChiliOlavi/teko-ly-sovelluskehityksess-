"""Grader for Harjoitus 2.

Compare functions in student's `ratkaisu.py` to the originals in `originals.py`.
Run with Python 3.12. Outputs a per-function pass/fail and total score.
"""
from __future__ import annotations
import importlib
import importlib.util
import os
import sys
from typing import Any


HERE = os.path.dirname(__file__)

def load_student_module(name: str = "ratkaisu"):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        # try loading by file path
        path = os.path.join(HERE, f"{name}.py")
        if not os.path.exists(path):
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def approx_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    # numbers (int, float, complex)
    try:
        import numbers
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            return abs(complex(a) - complex(b)) <= tol
    except Exception:
        pass
    # lists/tuples
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(approx_equal(x, y, tol) for x, y in zip(a, b))
    # dicts
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(approx_equal(a[k], b[k], tol) for k in a.keys())
    return a == b


def run_test(fn_name: str, args: tuple, kwargs: dict, orig_mod, stud_mod) -> bool:
    orig_fn = getattr(orig_mod, fn_name)
    expected = None
    try:
        expected = orig_fn(*args, **kwargs)
    except Exception as e:
        print(f"Error when computing expected for {fn_name} with args {args}, {e}")
        return False

    stud_fn = getattr(stud_mod, fn_name, None)
    if stud_fn is None:
        return False
    try:
        got = stud_fn(*args, **kwargs)
    except Exception:
        return False
    return approx_equal(got, expected)


def main():
    # ensure originals importable
    sys.path.insert(0, HERE)
    try:
        import originals as originals
    except Exception as e:
        print("Failed to import originals.py:", e)
        return

    stud = load_student_module("ratkaisu")
    if stud is None:
        print("Could not find 'ratkaisu.py' in the exercise folder. Create it and rerun grader.")
        return

    # tests per function
    tests = {
        "add": [((2,3), {}), ((-1,5), {})],
        "multiply_list": [(([2,3,4],), {}), (([],), {})],
        "is_prime": [((2,), {}), ((15,), {}), ((17,), {})],
        "factorial": [((5,), {}), ((0,), {})],
        "fibonacci": [((0,), {}), ((1,), {}), ((10,), {})],
        "sum_of_squares": [(([1,2,3],), {})],
        "reverse_string": [(("hello",), {})],
        "count_vowels": [(("Banana",), {})],
        "normalize_text": [(("  Hello   WORLD  ",), {})],
        "compute_stats": [(([1,2,2,3],), {})],
        "matrix_multiply": [((((1,2),(3,4)), ((5,6),(7,8))), {})],
        "evaluate_expression": [(("2*(3+4)",), {}), (("2**3",), {})],
    "quadratic_roots": [((1,0,-1), {})],
        "approximate_root": [((lambda x: x*x - 2, 1.0), {})],
        "permutations": [(([1,2,3],), {})],
    }

    total = 0
    max_score = len(tests)
    details = []
    for fn_name, cases in tests.items():
        okay = True
        for (args, kwargs) in cases:
            # ensure args is a flat tuple
            if not isinstance(args, tuple):
                args = (args,)
            passed = run_test(fn_name, args, kwargs, originals, stud)
            if not passed:
                okay = False
                break
        details.append((fn_name, okay))
        if okay:
            total += 1

    print("\nGrader results:")
    for fn_name, ok in details:
        print(f"  {fn_name:20s}: {'OK' if ok else 'FAIL'}")
    print(f"\nScore: {total} / {max_score}")


if __name__ == "__main__":
    main()
