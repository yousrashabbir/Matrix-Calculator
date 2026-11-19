"""
Matrix Studio · 3×3 Matrix Calculator — NUMPY edition
=====================================================

Same app and operations as the pure-Python edition, but every calculation is
done with NumPy (np.linalg).  The step-by-step panel shows the exact NumPy calls
in a Python-REPL style, plus independent cross-checks (eigenvalue products,
singular values, residuals) so you can see why NumPy's answer is right.

Install:  pip install numpy
Run:      python matrix_calculator_numpy.py
"""
from __future__ import annotations

import sys

try:
    import numpy as np
except ImportError:  # friendly message instead of a traceback
    msg = ("NumPy is not installed.\n\nInstall it with:\n    pip install numpy\n\n"
           "Or run the dependency-free edition:\n    python matrix_calculator_no_numpy.py")
    print(msg)
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
        _r = _tk.Tk()
        _r.withdraw()
        _mb.showerror("Matrix Studio — NumPy edition", msg)
    except Exception:
        pass
    sys.exit(1)

from matrix_common import (
    N, Steps, column, describe_ops, par, poly_text, sup_text, text_matrix, text_row,
)

EYE = np.eye(N)


# ==========================================================================
#  Helpers
# ==========================================================================
def arr(M):
    return np.array(M, dtype=float)


def out(X):
    """NumPy array -> plain nested lists of Python floats (what the UI expects)."""
    return [[float(x) for x in row] for row in np.asarray(X, dtype=float)]


def scale_of(*Xs):
    return max([1.0] + [float(np.abs(X).max()) for X in Xs])


def close(X, Y, rel=1e-9):
    return bool(np.allclose(X, Y, rtol=0, atol=rel * scale_of(X, Y)))


def singular(X):
    return abs(np.linalg.det(X)) <= 1e-12 * scale_of(X) ** 3


def a2s(X):
    """NumPy's own array printing (the way it looks in a Python REPL)."""
    return np.array2string(np.asarray(X), precision=4, suppress_small=True, separator=", ")


def repl(*pairs):
    """[('A @ B', '<output>'), ...] -> '>>> A @ B\\n<output>' text."""
    lines = []
    for cmd, result in pairs:
        lines.append(">>> " + cmd)
        if result != "":
            lines.append(str(result))
    return "\n".join(lines)


def literal(M, f, var="A"):
    cells = [[f(x, pretty=False) for x in row] for row in M]
    w = max(len(c) for row in cells for c in row)
    rows = ["[" + ", ".join(c.rjust(w) for c in row) + "]" for row in cells]
    head = f"{var} = np.array(["
    pad = " " * len(head)
    return head + (",\n" + pad).join(rows) + "])"


def py(x):
    """Python scalar from a NumPy scalar, tidying -0.0 and tiny imaginary parts."""
    z = complex(x)
    if abs(z.imag) <= 1e-12 * max(1.0, abs(z)):
        v = float(z.real)
        return 0.0 if v == 0 else v
    return z


def clean_number(z, scale=1.0):
    z = complex(z)
    tol = 1e-9 * max(1.0, scale)
    re_ = 0.0 if abs(z.real) <= tol else z.real
    im_ = 0.0 if abs(z.imag) <= tol else z.imag
    if abs(re_ - round(re_)) <= 1e-9 * max(1.0, abs(re_)):
        re_ = float(round(re_))
    if abs(im_ - round(im_)) <= 1e-9 * max(1.0, abs(im_)):
        im_ = float(round(im_))
    return re_ if im_ == 0 else complex(re_, im_)


def normalise(v):
    v = np.asarray(v, dtype=complex)
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) == 0:
        return [py(x) for x in v]
    v = v / v[k]
    s = float(np.abs(v).max())
    return [clean_number(x, s) for x in v]


def rref_np(M, pivot_cols=None, log=False, tol=None):
    """Gauss–Jordan with vectorised NumPy row operations (NumPy has no built-in rref)."""
    Rm = np.array(M, dtype=complex if np.iscomplexobj(M) else float)
    rows, cols = Rm.shape
    pc = cols if pivot_cols is None else pivot_cols
    tol = 1e-10 * scale_of(Rm) if tol is None else tol
    pivots, steps, r = [], [], 0

    def record(ops):
        if log and ops:
            snap = Rm.real if not np.iscomplexobj(Rm) else Rm
            steps.append((ops, snap.tolist()))

    for c in range(pc):
        if r >= rows:
            break
        col = np.abs(Rm[r:, c])
        if col.max() <= tol:
            Rm[r:, c] = 0
            continue
        units = np.flatnonzero(np.abs(col - 1) < 1e-12)
        p = r if abs(col[0] - 1) < 1e-12 else r + int(units[0] if units.size else np.argmax(col))
        if p != r:
            Rm[[r, p]] = Rm[[p, r]]
            record([("swap", r, p)])
        piv = Rm[r, c]
        if abs(piv - 1) > 1e-15:
            Rm[r] = Rm[r] / piv
            Rm[r, c] = 1
            Rm[r][np.abs(Rm[r]) <= tol] = 0
            record([("div", r, py(piv))])
        factors = Rm[:, c].copy()
        factors[r] = 0
        factors[np.abs(factors) <= tol] = 0
        if np.any(factors):
            Rm -= np.outer(factors, Rm[r])          # all rows in one vectorised step
            Rm[np.abs(Rm) <= tol] = 0
            record([("sub", int(i), py(factors[i]), r) for i in np.flatnonzero(factors)])
        pivots.append(c)
        r += 1
    return Rm, pivots, steps


def cofactor_np(A):
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            C[i, j] = (-1) ** (i + j) * np.linalg.det(minor)
    return C


# ==========================================================================
#  The engine
# ==========================================================================
class NumpyEngine:
    name = "NumPy"
    badge = f"NumPy {np.__version__}"
    tagline = f"NumPy {np.__version__} engine"

    # ---------------- two matrices ----------------
    def add(self, A, B):
        return out(arr(A) + arr(B))

    def sub(self, A, B):
        return out(arr(A) - arr(B))

    def mul(self, A, B):
        return out(arr(A) @ arr(B))

    def hadamard(self, A, B):
        return out(arr(A) * arr(B))

    def div(self, A, B):
        if singular(arr(B)):
            raise ValueError("B is singular (det B = 0), so B⁻¹ — and A × B⁻¹ — does not exist.")
        return out(arr(A) @ np.linalg.inv(arr(B)))

    def solve(self, A, B):
        if singular(arr(A)):
            raise ValueError("A is singular (det A = 0), so A·X = B has no unique solution. "
                             "Try Rank or RREF on A to see why.")
        return out(np.linalg.solve(arr(A), arr(B)))

    def commutator(self, A, B):
        A, B = arr(A), arr(B)
        return out(A @ B - B @ A)

    # ---------------- one matrix ----------------
    def det(self, A):
        return float(np.linalg.det(arr(A)))

    def inv(self, A):
        if singular(arr(A)):
            raise ValueError("This matrix is singular (determinant = 0), so it has no inverse.")
        return out(np.linalg.inv(arr(A)))

    def transpose(self, A):
        return out(arr(A).T)

    def trace(self, A):
        return float(np.trace(arr(A)))

    def rank(self, A):
        return int(np.linalg.matrix_rank(arr(A)))

    def rref(self, A):
        return out(rref_np(arr(A))[0])

    def cofactor(self, A):
        return out(cofactor_np(arr(A)))

    def adjugate(self, A):
        return out(cofactor_np(arr(A)).T)

    def norm(self, A):
        return float(np.linalg.norm(arr(A)))

    def charpoly(self, A):
        return [float(c) for c in np.real(np.poly(arr(A)))]

    def eigen(self, A):
        A = arr(A)
        w, V = np.linalg.eig(A)
        scale = max(1.0, float(np.abs(w).max()), scale_of(A))
        groups = []
        for idx, lam in enumerate(w):
            for g in groups:
                if abs(w[g[0]] - lam) <= 1e-4 * scale:
                    g.append(idx)
                    break
            else:
                groups.append([idx])
        result = []
        for g in groups:
            lam = clean_number(np.mean(w[g]), scale)
            if len(g) == 1:
                vecs = [V[:, g[0]]]
            else:  # repeated eigenvalue: null space of (A − λI) via the SVD
                _, s, vh = np.linalg.svd(A - lam * EYE)
                basis = vh[int(np.sum(s > 1e-6 * scale)):].conj()
                if len(basis) == 0:
                    vecs = [V[:, g[0]]]
                else:  # row-reduce the basis so it reads nicely
                    Rb, piv, _ = rref_np(basis, tol=1e-9)
                    vecs = [Rb[k] for k in range(len(piv))]
            result.append({"value": lam, "mult": len(g), "vectors": [normalise(v) for v in vecs]})
        result.sort(key=lambda g: (-complex(g["value"]).real, -complex(g["value"]).imag))
        return result

    def scale(self, A, k):
        return out(k * arr(A))

    def power(self, A, n):
        n = self._int_exponent(n)
        if n < 0 and singular(arr(A)):
            raise ValueError("The matrix is singular, so negative powers (which need the inverse) don't exist.")
        with np.errstate(over="ignore", invalid="ignore"):  # overflow is reported by the UI
            return out(np.linalg.matrix_power(arr(A), n))

    @staticmethod
    def _int_exponent(n):
        if abs(n - round(n)) > 1e-9:
            raise ValueError("The power n must be a whole number (for example 3 or −2).")
        n = int(round(n))
        if abs(n) > 1000:
            raise ValueError("Please keep the power between −1000 and 1000.")
        return n

    def props(self, A):
        A = arr(A)
        tol = 1e-9 * scale_of(A)
        A2 = A @ A
        return {
            "det": float(np.linalg.det(A)), "trace": float(np.trace(A)),
            "rank": int(np.linalg.matrix_rank(A)), "norm": float(np.linalg.norm(A)),
            "invertible": not singular(A),
            "symmetric": close(A, A.T),
            "skew": close(A, -A.T),
            "diagonal": bool(np.all(np.abs(A - np.diag(np.diag(A))) <= tol)),
            "upper": bool(np.all(np.abs(np.tril(A, -1)) <= tol)),
            "lower": bool(np.all(np.abs(np.triu(A, 1)) <= tol)),
            "identity": close(A, EYE),
            "zero": bool(np.abs(A).max() <= 1e-12),
            "orthogonal": close(A @ A.T, EYE),
            "idempotent": close(A2, A),
            "involutory": close(A2, EYE),
            "nilpotent": bool(np.abs(A2 @ A).max() <= 1e-9 * scale_of(A) ** 3),
        }

    # ======================================================================
    #  Explanations: the NumPy calls + independent cross-checks
    # ======================================================================
    def explain(self, op, ctx, res, f):
        S = Steps()
        fn = getattr(self, "_x_" + op, None)
        if fn is None:
            return S.p("No step-by-step explanation is available for this operation.")
        fn(S, ctx, res, f)
        return S

    def _setup(self, S, ctx, f, names):
        S.h("NumPy setup")
        S.code("import numpy as np\n" + "\n".join(literal(ctx[k], f, v) for k, v in names))

    def _setup_m(self, S, ctx, f):
        self._setup(S, ctx, f, [("M", ctx["name"])])
        return arr(ctx["M"]), ctx["name"]

    # ---------- two matrices ----------
    def _x_add(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        S.h("Element-wise addition")
        S.code(repl(("A + B", a2s(res))))
        S.p("+ between two arrays is applied element by element (np.add) — no Python loops, "
            "the work runs in compiled C.")

    def _x_sub(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        S.h("Element-wise subtraction")
        S.code(repl(("A - B", a2s(res))))

    def _x_hadamard(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        S.h("Element-wise (Hadamard) product")
        S.code(repl(("A * B", a2s(res))))
        S.warn("In NumPy, * is element-wise. The matrix product is the @ operator.")

    def _product(self, S, ctx, res, f, x, y):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        X, Y = arr(ctx[x]), arr(ctx[y])
        S.h("Matrix product")
        S.code(repl((f"{x} @ {y}", a2s(res))))
        S.p("@ is matrix multiplication (np.matmul): row i of the left matrix · column j of the right one.")
        S.h("Cross-check with Einstein summation")
        ok = np.allclose(X @ Y, np.einsum("ik,kj->ij", X, Y))
        S.code(repl((f"np.allclose({x} @ {y}, np.einsum('ik,kj->ij', {x}, {y}))", ok)))
        comm = np.allclose(X @ Y, Y @ X)
        S.code(repl((f"np.allclose({x} @ {y}, {y} @ {x})", comm)))
        if comm:
            S.ok("A and B commute here, so A @ B equals B @ A.")
        else:
            S.warn("A @ B ≠ B @ A — matrix multiplication is not commutative.")

    def _x_mul(self, S, ctx, res, f):
        self._product(S, ctx, res, f, "A", "B")

    def _x_mul_ba(self, S, ctx, res, f):
        self._product(S, ctx, res, f, "B", "A")

    def _x_div(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        A, B = arr(ctx["A"]), arr(ctx["B"])
        S.h("A × B⁻¹")
        S.code(repl(("A @ np.linalg.inv(B)", a2s(res))))
        alt = np.linalg.solve(B.T, A.T).T
        S.h("A better way (no explicit inverse)")
        S.code(repl(("np.allclose(A @ np.linalg.inv(B), np.linalg.solve(B.T, A.T).T)",
                     np.allclose(arr(res), alt))))
        S.p("Solving Bᵀ·Xᵀ = Aᵀ gives the same X = A·B⁻¹ but is faster and more accurate.")

    def _x_solve(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        A, B, X = arr(ctx["A"]), arr(ctx["B"]), arr(res)
        S.h("Solve A·X = B")
        S.code(repl(("X = np.linalg.solve(A, B)", ""), ("X", a2s(X))))
        S.p("solve() LU-factorises A once (LAPACK gesv) and back-substitutes for every column of B — "
            "quicker and more accurate than inv(A) @ B.")
        S.h("Residual check")
        S.code(repl(("np.linalg.norm(A @ X - B)", repr(float(np.linalg.norm(A @ X - B))))))
        S.ok("A residual near 1e−15 means X solves the system to machine precision.")

    def _x_commutator(self, S, ctx, res, f):
        self._setup(S, ctx, f, [("A", "A"), ("B", "B")])
        A, B = arr(ctx["A"]), arr(ctx["B"])
        S.h("Commutator [A, B]")
        S.code(repl(("A @ B - B @ A", a2s(res)), ("np.allclose(A @ B, B @ A)", np.allclose(A @ B, B @ A))))
        if np.allclose(A @ B, B @ A):
            S.ok("[A, B] = 0, so A and B commute.")
        else:
            S.warn("[A, B] ≠ 0, so A and B do not commute.")

    # ---------- one matrix ----------
    def _x_det(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Determinant")
        S.code(repl((f"np.linalg.det({n})", repr(res))))
        S.p("NumPy LU-factorises the matrix (P·A = L·U, LAPACK getrf). det is the product of U's "
            "diagonal, with the sign flipped once per row swap.")
        if res != 0 and abs(res - round(res)) > 0 and abs(res - round(res)) < 1e-9 * max(1, abs(res)):
            S.p(f"The raw float {res!r} isn't exactly {round(res)}: binary floating-point rounding "
                f"leaves an error around 1e−16. Matrix Studio tidies it to {f(res)} for display.")
        S.h("Cross-check: det = product of the eigenvalues")
        S.code(repl((f"np.prod(np.linalg.eigvals({n})).real", repr(float(np.prod(np.linalg.eigvals(A)).real)))))
        if singular(A):
            S.warn(f"det({n}) ≈ 0, so {n} is singular (no inverse, rank < 3).")
        else:
            S.ok(f"det({n}) ≠ 0, so {n} is invertible.")

    def _x_inv(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        Ai = arr(res)
        S.h("Inverse")
        S.code(repl((f"{n}_inv = np.linalg.inv({n})", ""), (f"{n}_inv", a2s(Ai))))
        S.h("Verify")
        S.code(repl((f"np.allclose({n} @ {n}_inv, np.eye(3))", np.allclose(A @ Ai, EYE))))
        cond = float(np.linalg.cond(A))
        S.code(repl((f"np.linalg.cond({n})", f"{cond:.6g}")))
        if cond < 1e3:
            S.ok("Condition number is small: the inverse is numerically reliable.")
        else:
            S.warn("Large condition number: the matrix is close to singular, so rounding errors are amplified.")

    def _x_transpose(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Transpose")
        S.code(repl((f"{n}.T", a2s(res)), (f"{n}.strides, {n}.T.strides", f"({A.strides}, {A.T.strides})")))
        S.p(".T is a view: NumPy just swaps the strides (bytes to step per row/column) — no data is copied.")

    def _x_trace(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Trace")
        S.code(repl((f"np.trace({n})", repr(res)),
                    (f"np.linalg.eigvals({n}).sum().real", repr(float(np.linalg.eigvals(A).sum().real)))))
        S.ok("The trace equals the sum of the eigenvalues.")

    def _x_rank(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        s = np.linalg.svd(A, compute_uv=False)
        tol = s.max() * max(A.shape) * np.finfo(float).eps
        S.h("Rank via the SVD")
        S.code(repl((f"s = np.linalg.svd({n}, compute_uv=False)", ""), ("s", a2s(s)),
                    ("s.max() * 3 * np.finfo(float).eps   # tolerance", f"{tol:.3e}"),
                    (f"np.linalg.matrix_rank({n})", res)))
        S.p(f"matrix_rank counts singular values above the tolerance: {res} of 3 here.")
        if res < N:
            S.warn(f"Rank-deficient: {N - res} singular value(s) are effectively zero, so det({n}) = 0.")
        else:
            S.ok("Full rank.")

    def _x_rref(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Gauss–Jordan with NumPy")
        S.p("NumPy has no rref(), so this edition implements it with vectorised row operations — "
            "each column is cleared in one shot:")
        S.code("factors = R[:, c].copy(); factors[r] = 0\nR -= np.outer(factors, R[r])")
        _, piv, steps = rref_np(A, log=True)
        if steps:
            descs = [describe_ops(ops, f) for ops, _ in steps]
            w = max(len(d) for d in descs)
            for k, ((_, snap), d) in enumerate(zip(steps, descs), 1):
                S.m(text_row(f"{k:>2}. {d}".ljust(w + 6), text_matrix(snap, f)))
        S.m(text_row(f"rref({n}) = ", text_matrix(res, f)))
        S.p(f"Pivot columns: {', '.join(str(c + 1) for c in piv) or 'none'}  →  rank {len(piv)}.")

    def _x_cofactor(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Cofactors with np.delete")
        S.code("def cofactor(M):\n    C = np.zeros((3, 3))\n    for i in range(3):\n        for j in range(3):\n"
               "            minor = np.delete(np.delete(M, i, axis=0), j, axis=1)\n"
               "            C[i, j] = (-1) ** (i + j) * np.linalg.det(minor)\n    return C")
        S.code(repl((f"cofactor({n})", a2s(res))))
        if not singular(A):
            S.code(repl((f"np.allclose(cofactor({n}), np.linalg.det({n}) * np.linalg.inv({n}).T)",
                         np.allclose(arr(res), np.linalg.det(A) * np.linalg.inv(A).T))))
            S.ok("Shortcut identity: cof(M) = det(M) · (M⁻¹)ᵀ.")

    def _x_adjugate(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Adjugate = cofactor matrix transposed")
        S.code(repl((f"adj = cofactor({n}).T", ""), ("adj", a2s(res)),
                    (f"np.allclose({n} @ adj, np.linalg.det({n}) * np.eye(3))",
                     np.allclose(A @ arr(res), np.linalg.det(A) * EYE))))
        S.ok("M · adj(M) = det(M) · I, so M⁻¹ = adj(M) / det(M).")

    def _x_norm(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Frobenius norm")
        sv = np.linalg.svd(A, compute_uv=False)
        S.code(repl((f"np.linalg.norm({n})", repr(res)),
                    (f"np.sqrt(({n} ** 2).sum())", repr(float(np.sqrt((A ** 2).sum())))),
                    (f"np.sqrt((np.linalg.svd({n}, compute_uv=False) ** 2).sum())", repr(float(np.sqrt((sv ** 2).sum()))))))
        S.ok("Three routes, same answer: ‖M‖F = √Σaᵢⱼ² = √Σσᵢ².")

    def _x_charpoly(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        c = np.array(res)
        S.h("Characteristic polynomial")
        S.code(repl((f"c = np.poly({n})", ""), ("c", a2s(c)), ("np.roots(c)", a2s(np.roots(c)))))
        S.m(f"p(λ) = {poly_text(res, f)}")
        S.h("Cayley–Hamilton check")
        PA = sum(ci * np.linalg.matrix_power(A, N - i) for i, ci in enumerate(c))
        S.code(repl((f"pA = sum(ci * np.linalg.matrix_power({n}, 3 - i) for i, ci in enumerate(c))", ""),
                    ("np.allclose(pA, 0)", np.allclose(PA, 0, atol=1e-7 * scale_of(A) ** 3))))
        S.ok("Every square matrix satisfies its own characteristic polynomial.")

    def _x_eigen(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        w, V = np.linalg.eig(A)
        S.h("Eigen-decomposition (LAPACK geev)")
        S.code(repl((f"w, V = np.linalg.eig({n})", ""), ("w", a2s(w)), ("V", a2s(V))))
        S.p("w holds the eigenvalues; column i of V is a unit-length eigenvector for w[i]. "
            "Matrix Studio rescales each vector so its largest entry is 1, which is easier to read.")
        S.h("Residual check:  ‖A·v − λ·v‖ for every column")
        S.code(repl((f"np.linalg.norm({n} @ V - V * w, axis=0)", a2s(np.linalg.norm(A @ V - V * w, axis=0)))))
        S.h("Result")
        for g in res:
            blocks = [f"λ = {f(g['value'])}" + (f" (×{g['mult']})" if g["mult"] > 1 else "") + "   v = "]
            for k, v in enumerate(g["vectors"]):
                blocks += ([",  "] if k else []) + [text_matrix(column(v), f)]
            S.m(text_row(*blocks))
            if g["mult"] > 1:
                S.p("Repeated eigenvalue: its eigenvectors come from the null space of (A − λI), "
                    "found with np.linalg.svd.")
                if len(g["vectors"]) < g["mult"]:
                    S.warn(f"Only {len(g['vectors'])} independent eigenvector: {n} is defective (not diagonalisable).")

    def _x_props(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        S.h("Checks used")
        checks = [
            (f"np.allclose({n}, {n}.T)", res["symmetric"], "symmetric"),
            (f"np.allclose({n}, -{n}.T)", res["skew"], "skew-symmetric"),
            (f"np.allclose({n}, np.diag(np.diag({n})))", res["diagonal"], "diagonal"),
            (f"np.allclose(np.tril({n}, -1), 0)", res["upper"], "upper triangular"),
            (f"np.allclose(np.triu({n}, 1), 0)", res["lower"], "lower triangular"),
            (f"np.allclose({n} @ {n}.T, np.eye(3))", res["orthogonal"], "orthogonal"),
            (f"np.allclose({n} @ {n}, {n})", res["idempotent"], "idempotent"),
            (f"np.allclose({n} @ {n}, np.eye(3))", res["involutory"], "involutory"),
            (f"np.allclose(np.linalg.matrix_power({n}, 3), 0)", res["nilpotent"], "nilpotent"),
        ]
        S.code("\n".join(f">>> {cmd}   # {what}\n{val}" for cmd, val, what in checks))
        S.code(repl((f"np.linalg.det({n}), np.trace({n}), np.linalg.matrix_rank({n})",
                     f"({res['det']!r}, {res['trace']!r}, {res['rank']})")))

    def _x_scale(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        k = f(ctx["k"], pretty=False)
        S.h("Scalar multiple (broadcasting)")
        S.code(repl((f"{k} * {n}", a2s(res))))
        S.p("NumPy broadcasts the scalar across every entry of the array.")

    def _x_power(self, S, ctx, res, f):
        A, n = self._setup_m(S, ctx, f)
        p = self._int_exponent(ctx["k"])
        S.h("Matrix power")
        S.code(repl((f"np.linalg.matrix_power({n}, {p})", a2s(res))))
        m = abs(p)
        if p < 0:
            S.p(f"For a negative power NumPy first inverts {n}, then raises the inverse to {m}.")
        if m > 1:
            mults = (m.bit_length() - 1) + (bin(m).count("1") - 1)
            S.ok(f"matrix_power uses repeated squaring: {mults} multiplication(s) instead of {m - 1}.")
        if p == 0:
            S.p(f"{n}{sup_text(0)} is the identity for every square matrix.")


ENGINE = NumpyEngine()

if __name__ == "__main__":
    from matrix_ui import run_app
    run_app(ENGINE)
