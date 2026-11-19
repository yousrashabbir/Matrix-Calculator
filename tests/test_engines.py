"""
Tests for Matrix Studio's two engines.

    python -m unittest discover -s tests -v

* Pure-Python engine: known answers (always runs).
* NumPy engine: the same answers, plus a randomised cross-check against the
  pure engine (skipped automatically when NumPy isn't installed).
* Every step-by-step explanation is generated for many matrices, in both
  decimal and fraction modes, to make sure none of them crash.
"""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matrix_common import NumberFormat, parse_number, poly_text  # noqa: E402
from matrix_calculator_no_numpy import PureEngine  # noqa: E402

try:
    from matrix_calculator_numpy import NumpyEngine
    HAVE_NUMPY = True
except SystemExit:  # numpy edition exits politely when NumPy is missing
    HAVE_NUMPY = False

A = [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]]          # det = -1
B = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]         # det = -3
SING = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]       # det = 0, rank 2
ROT = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]       # eigen 1, ±i
DEFECT = [[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]     # λ=2 twice, 1 vector
I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
Z3 = [[0.0] * 3 for _ in range(3)]
SYM = [[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 5.0]]
NIL = [[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]
FRAC = [[0.5, 1 / 3, 0.25], [2.0, -1.5, 0.2], [1.0, 1.0, 1.0]]

SAMPLES = [A, B, SING, ROT, DEFECT, I3, Z3, SYM, NIL, FRAC]
TWO_OPS = ["add", "sub", "mul", "mul_ba", "hadamard", "div", "solve", "commutator"]
ONE_OPS = ["det", "inv", "transpose", "trace", "rank", "rref", "cofactor", "adjugate",
           "eigen", "charpoly", "norm", "props"]
K_OPS = ["scale", "power"]


def close(x, y, tol=1e-7):
    if isinstance(x, list):
        return all(close(a, b, tol) for a, b in zip(x, y))
    return abs(complex(x) - complex(y)) <= tol * max(1.0, abs(complex(x)))


def call(engine, op, M=None, N=None, k=None):
    if op == "mul_ba":
        return engine.mul(N, M)
    if op in TWO_OPS:
        return getattr(engine, op)(M, N)
    if op in K_OPS:
        return getattr(engine, op)(M, k)
    return getattr(engine, op)(M)


class KnownAnswers:
    engine = None

    def test_basic(self):
        e = self.engine
        self.assertTrue(close(e.add(A, B), [[3, 3, 4], [5, 8, 8], [8, 8, 10]]))
        self.assertTrue(close(e.mul(A, B), [[13, 17, 22], [27, 33, 41], [1, 2, 3]]))
        self.assertTrue(close(e.det(A), -1))
        self.assertTrue(close(e.det(SING), 0, 1e-9))
        self.assertEqual(e.rank(SING), 2)
        self.assertTrue(close(e.trace(B), 16))
        self.assertTrue(close(e.inv(A), [[0, 0, 1], [-2, 1, 3], [3, -1, -5]]))
        self.assertTrue(close(e.norm(I3), math.sqrt(3)))
        self.assertTrue(close(e.charpoly(I3), [1, -3, 3, -1]))
        self.assertTrue(close(e.power(A, 0), I3))
        self.assertTrue(close(e.power(A, -1), e.inv(A)))
        self.assertTrue(close(e.power(A, 5), e.mul(e.mul(e.mul(e.mul(A, A), A), A), A)))

    def test_solve(self):
        X = self.engine.solve(A, B)
        self.assertTrue(close(self.engine.mul(A, X), B))

    def test_singular_errors(self):
        for fn in (lambda: self.engine.inv(SING), lambda: self.engine.solve(SING, B),
                   lambda: self.engine.div(A, SING), lambda: self.engine.power(SING, -2)):
            with self.assertRaises(ValueError):
                fn()
        with self.assertRaises(ValueError):
            self.engine.power(A, 1.5)

    def test_eigen(self):
        e = self.engine
        vals = sorted((complex(g["value"]) for g in e.eigen(ROT)), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
        self.assertTrue(close(vals, [-1j, 1j, 1]) or close(vals, [complex(0, -1), complex(0, 1), 1]))
        d = e.eigen(DEFECT)
        self.assertEqual([(g["value"], g["mult"], len(g["vectors"])) for g in d], [(3, 1, 1), (2, 2, 1)])
        i3 = e.eigen(I3)
        self.assertEqual((i3[0]["mult"], len(i3[0]["vectors"])), (3, 3))
        for M in SAMPLES:
            for g in e.eigen(M):
                for v in g["vectors"]:
                    Av = [sum(M[i][k] * v[k] for k in range(3)) for i in range(3)]
                    self.assertTrue(close(Av, [g["value"] * x for x in v], 1e-6), (M, g))

    def test_props(self):
        p = self.engine.props(SYM)
        self.assertTrue(p["symmetric"] and p["invertible"] and not p["diagonal"])
        p = self.engine.props(ROT)
        self.assertTrue(p["orthogonal"] and not p["symmetric"])
        self.assertTrue(self.engine.props(NIL)["nilpotent"])
        self.assertTrue(self.engine.props(I3)["identity"])

    def test_explanations_never_crash(self):
        for mode in ("decimal", "fraction"):
            f = NumberFormat(mode, 4)
            for M in SAMPLES:
                for Nm in (A, SING):
                    ctx = {"A": M, "B": Nm, "M": M, "name": "A", "k": 3.0}
                    for op in TWO_OPS + ONE_OPS + K_OPS:
                        for k in ((3.0, -2.0, 0.0) if op == "power" else (3.0,)):
                            ctx["k"] = k
                            try:
                                res = call(self.engine, op, M, Nm, k)
                            except ValueError:
                                continue
                            steps = self.engine.explain(op, ctx, res, f)
                            self.assertTrue(steps, op)


class TestPure(KnownAnswers, unittest.TestCase):
    engine = PureEngine()


@unittest.skipUnless(HAVE_NUMPY, "NumPy not installed")
class TestNumpy(KnownAnswers, unittest.TestCase):
    engine = NumpyEngine() if HAVE_NUMPY else None


@unittest.skipUnless(HAVE_NUMPY, "NumPy not installed")
class TestCrossCheck(unittest.TestCase):
    """The two engines must agree on random integer and decimal matrices."""

    def test_random(self):
        pure, npe = PureEngine(), NumpyEngine()
        rng = random.Random(42)
        for trial in range(300):
            gen = (lambda: rng.randint(-9, 9)) if trial % 2 else (lambda: round(rng.uniform(-5, 5), 2))
            M = [[float(gen()) for _ in range(3)] for _ in range(3)]
            Nm = [[float(gen()) for _ in range(3)] for _ in range(3)]
            for op in TWO_OPS + ONE_OPS + K_OPS:
                k = rng.choice([-2.0, 0.0, 1.0, 3.0]) if op == "power" else 2.5
                try:
                    r1 = call(pure, op, M, Nm, k)
                except ValueError:
                    with self.assertRaises(ValueError):
                        call(npe, op, M, Nm, k)
                    continue
                r2 = call(npe, op, M, Nm, k)
                if op == "eigen":
                    v1 = sorted((complex(g["value"]) for g in r1), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
                    v2 = sorted((complex(g["value"]) for g in r2), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
                    self.assertTrue(close(v1, v2, 1e-6), (M, v1, v2))
                elif op == "props":
                    for key in r1:
                        if isinstance(r1[key], bool):
                            self.assertEqual(r1[key], r2[key], (op, key, M))
                        else:
                            self.assertTrue(close(r1[key], r2[key], 1e-6), (op, key, M))
                elif op == "rank":
                    self.assertEqual(r1, r2, M)
                else:
                    self.assertTrue(close(r1, r2, 1e-6), (op, M, r1, r2))


class TestCommon(unittest.TestCase):
    def test_parse(self):
        cases = {"3": 3, "-2.5": -2.5, "1/3": 1 / 3, "2^3": 8, "sqrt(4)": 2, "√9": 3,
                 "pi": math.pi, "": 0, " 1e-3 ": 0.001, "−4": -4, "2*pi": 2 * math.pi, "(1+2)*3": 9}
        for text, val in cases.items():
            self.assertAlmostEqual(parse_number(text), val, msg=text)
        for bad in ("abc", "1/0", "sqrt(-1)", "__import__('os')", "2**999", "nan", "inf", "1,5"):
            with self.assertRaises(ValueError, msg=bad):
                parse_number(bad)

    def test_format(self):
        f = NumberFormat("decimal", 4)
        self.assertEqual(f(2.9999999999), "3")
        self.assertEqual(f(-0.5), "−0.5")
        self.assertEqual(f(-0.5, pretty=False), "-0.5")
        self.assertEqual(f(1 / 3), "0.3333")
        self.assertEqual(f(1e-15), "0")
        self.assertEqual(f(complex(1, -2)), "1 − 2i")
        self.assertEqual(f(complex(0, 1)), "i")
        fr = NumberFormat("fraction", 4)
        self.assertEqual(fr(1 / 3), "1/3")
        self.assertEqual(fr(-0.75), "−3/4")
        self.assertEqual(poly_text([1, -5, 3, -1], f), "λ³ − 5λ² + 3λ − 1")
        self.assertEqual(poly_text([1, 0, -1, 0], f), "λ³ − λ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
