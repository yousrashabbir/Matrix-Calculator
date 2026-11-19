"""
Matrix Studio · 3×3 Matrix Calculator — PURE PYTHON edition (no NumPy)
======================================================================

Every operation below is written by hand with plain Python lists, so this
edition needs nothing but a standard Python 3.8+ install (tkinter ships with it).

Operations
    Two matrices : A + B, A − B, A × B, B × A, A ∘ B (Hadamard), A × B⁻¹,
                   solve A·X = B, commutator AB − BA
    One matrix   : determinant, inverse, transpose, trace, rank, RREF, cofactor,
                   adjugate, eigenvalues + eigenvectors, characteristic polynomial,
                   Frobenius norm, property analysis, k·M, Mⁿ

Each operation also produces a step-by-step explanation of how it was computed.

Run:   python matrix_calculator_no_numpy.py
"""
from __future__ import annotations

import cmath
import math

from matrix_common import (
    N, Steps, column, describe_ops, ij, par, poly_text, sub_text, sup_text,
    text_matrix, text_row,
)

R3 = range(N)


# ==========================================================================
#  Low-level helpers (plain lists)
# ==========================================================================
def identity():
    return [[1.0 if i == j else 0.0 for j in R3] for i in R3]


def zeros():
    return [[0.0] * N for _ in R3]


def max_abs(M) -> float:
    return max((abs(x) for row in M for x in row), default=0.0)


def approx_equal(X, Y, rel=1e-9) -> bool:
    tol = rel * max(1.0, max_abs(X), max_abs(Y))
    return all(abs(X[i][j] - Y[i][j]) <= tol for i in range(len(X)) for j in range(len(X[0])))


def m_add(A, B):
    return [[A[i][j] + B[i][j] for j in R3] for i in R3]


def m_sub(A, B):
    return [[A[i][j] - B[i][j] for j in R3] for i in R3]


def m_mul(A, B):
    """General product (rows of A) × (columns of B)."""
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))]
            for i in range(len(A))]


def m_scale(A, k):
    return [[k * A[i][j] for j in R3] for i in R3]


def m_transpose(A):
    return [[A[j][i] for j in R3] for i in R3]


def m_hadamard(A, B):
    return [[A[i][j] * B[i][j] for j in R3] for i in R3]


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in R3) for i in R3]


def minor_matrix(A, i, j):
    return [[A[r][c] for c in R3 if c != j] for r in R3 if r != i]


def minor(A, i, j):
    (a, b), (c, d) = minor_matrix(A, i, j)
    return a * d - b * c


def det3(A):
    """Cofactor (Laplace) expansion along the first row."""
    return sum(((-1) ** j) * A[0][j] * minor(A, 0, j) for j in R3)


def trace3(A):
    return A[0][0] + A[1][1] + A[2][2]


def cofactor_matrix(A):
    return [[((-1) ** (i + j)) * minor(A, i, j) for j in R3] for i in R3]


def is_singular(A, d=None) -> bool:
    d = det3(A) if d is None else d
    return abs(d) <= 1e-12 * max(1.0, max_abs(A)) ** 3


def rref(M, pivot_cols=None, log=False, tol=None):
    """Gauss–Jordan elimination.  Works for any r×c matrix (real or complex).

    Returns (R, pivot_columns, steps) where steps is a list of
    ([row-operation], snapshot) pairs when log=True."""
    R = [list(row) for row in M]
    rows, cols = len(R), len(R[0])
    pc = cols if pivot_cols is None else pivot_cols
    tol = 1e-10 * max(1.0, max_abs(M)) if tol is None else tol
    pivots, steps = [], []

    def clean(row):
        for j in range(cols):
            if abs(row[j]) <= tol:
                row[j] = 0.0

    def record(op):
        if log:
            steps.append(([op], [row[:] for row in R]))

    r = 0
    for c in range(pc):
        if r >= rows:
            break
        cand = [i for i in range(r, rows) if abs(R[i][c]) > tol]
        if not cand:
            continue
        unit = lambda i: abs(abs(R[i][c]) - 1) < 1e-12  # noqa: E731
        if unit(r):
            p = r
        else:  # prefer a ±1 pivot (nicer numbers), else the largest one (more stable)
            p = next((i for i in cand if unit(i)), max(cand, key=lambda i: abs(R[i][c])))
        if p != r:
            R[r], R[p] = R[p], R[r]
            record(("swap", r, p))
        piv = R[r][c]
        if abs(piv - 1) > 1e-15:
            R[r] = [x / piv for x in R[r]]
            R[r][c] = 1.0
            clean(R[r])
            record(("div", r, piv))
        for i in range(rows):
            if i != r and abs(R[i][c]) > tol:
                fct = R[i][c]
                R[i] = [R[i][j] - fct * R[r][j] for j in range(cols)]
                R[i][c] = 0.0
                clean(R[i])
                record(("sub", i, fct, r))
        for i in range(rows):
            if i != r:
                R[i][c] = 0.0 if abs(R[i][c]) <= tol else R[i][c]
        pivots.append(c)
        r += 1
    return R, pivots, steps


# ---------- eigen helpers -------------------------------------------------
def char_coeffs(A):
    """Coefficients of p(λ) = det(λI − A) = λ³ − tr·λ² + c₂·λ − det."""
    c2 = minor(A, 0, 0) + minor(A, 1, 1) + minor(A, 2, 2)
    return [1.0, -trace3(A), c2, -det3(A)]


def _polish(z, b, c, d):
    """A few Newton steps on z³ + b z² + c z + d."""
    for _ in range(6):
        fz = ((z + b) * z + c) * z + d
        dz = (3 * z + 2 * b) * z + c
        if abs(dz) < 1e-14:
            break
        step = fz / dz
        z -= step
        if abs(step) <= 1e-16 * max(1.0, abs(z)):
            break
    return z


def cubic_roots(b, c, d):
    """Roots of t³ + b t² + c t + d via Cardano's formula (complex-safe) + Newton polish."""
    p = c - b * b / 3
    q = 2 * b ** 3 / 27 - b * c / 3 + d
    s = cmath.sqrt((q / 2) ** 2 + (p / 3) ** 3)
    w1, w2 = -q / 2 + s, -q / 2 - s
    w = w1 if abs(w1) >= abs(w2) else w2
    u = w ** (1 / 3) if abs(w) > 1e-300 else 0
    omega = complex(-0.5, math.sqrt(3) / 2)
    roots = []
    for k in range(3):
        if u == 0:
            t = 0j
        else:
            uk = u * omega ** k
            t = uk - p / (3 * uk)
        roots.append(_polish(t - b / 3, b, c, d))
    return roots


def clean_number(z, scale=1.0):
    """Tidy a (possibly complex) number: drop tiny imaginary parts, snap integers."""
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
    """Scale an eigenvector so its largest-magnitude entry is exactly 1."""
    k = max(R3, key=lambda i: abs(v[i]))
    if abs(v[k]) == 0:
        return v
    s = v[k]
    scale = max(abs(x / s) for x in v)
    return [clean_number(x / s, scale) for x in v]


def _cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _norm(v):
    return math.sqrt(sum(abs(x) ** 2 for x in v))


def null_space(M, tol):
    Rm, piv, _ = rref(M, tol=tol)
    basis = []
    for free in (c for c in R3 if c not in piv):
        v = [0.0] * N
        v[free] = 1.0
        for row, pc in enumerate(piv):
            v[pc] = -Rm[row][free]
        basis.append(v)
    return basis


def eigenvectors_for(A, lam, mult):
    scale = max(1.0, max_abs(A), abs(lam))
    Ml = [[A[i][j] - (lam if i == j else 0) for j in R3] for i in R3]
    if mult == 1:
        # For a simple eigenvalue (A − λI) has rank 2: the cross product of two
        # independent rows is orthogonal to every row, i.e. it is the eigenvector.
        best = max((_cross(Ml[a], Ml[b]) for a, b in ((0, 1), (0, 2), (1, 2))), key=_norm)
        if _norm(best) > 1e-8 * scale ** 2:
            return [normalise(best)]
    basis = null_space(Ml, tol=1e-6 * scale)
    return [normalise(v) for v in basis]


def _refine_repeated(z, mult, b, c):
    """Cardano loses precision on repeated roots (error ≈ √eps).  A triple root is
    exactly −b/3; a double root is also a root of p′(λ) = 3λ² + 2bλ + c."""
    if mult >= 3:
        return -b / 3
    disc = cmath.sqrt(4 * b * b - 12 * c)
    candidates = [(-2 * b + disc) / 6, (-2 * b - disc) / 6]
    best = min(candidates, key=lambda r: abs(r - z))
    return best if abs(best - z) <= 1e-3 * max(1.0, abs(z)) else z


def group_eigenvalues(roots, scale, b=None, c=None):
    groups = []
    for z in roots:
        for g in groups:
            if abs(g[0] - z) <= 1e-4 * scale:
                g[1].append(z)
                break
        else:
            groups.append([z, [z]])
    out = []
    for _, members in groups:
        z = sum(members) / len(members)
        if len(members) > 1 and b is not None:
            z = _refine_repeated(z, len(members), b, c)
        out.append((clean_number(z, scale), len(members)))
    out.sort(key=lambda t: (-complex(t[0]).real, -complex(t[0]).imag))
    return out


# ==========================================================================
#  The engine (what the UI talks to)
# ==========================================================================
class PureEngine:
    name = "Pure Python"
    badge = "Pure Python · 0 dependencies"
    tagline = "pure-Python engine, no NumPy"

    # ---------------- two matrices ----------------
    def add(self, A, B):
        return m_add(A, B)

    def sub(self, A, B):
        return m_sub(A, B)

    def mul(self, A, B):
        return m_mul(A, B)

    def hadamard(self, A, B):
        return m_hadamard(A, B)

    def div(self, A, B):
        if is_singular(B):
            raise ValueError("B is singular (det B = 0), so B⁻¹ — and A × B⁻¹ — does not exist.")
        return m_mul(A, self.inv(B))

    def solve(self, A, B):
        aug = [list(A[i]) + list(B[i]) for i in R3]
        Rm, piv, _ = rref(aug, pivot_cols=N)
        if len(piv) < N:
            raise ValueError("A is singular (det A = 0), so A·X = B has no unique solution. "
                             "Try Rank or RREF on A to see why.")
        return [row[N:] for row in Rm]

    def commutator(self, A, B):
        return m_sub(m_mul(A, B), m_mul(B, A))

    # ---------------- one matrix ----------------
    def det(self, A):
        return det3(A)

    def inv(self, A):
        d = det3(A)
        if is_singular(A, d):
            raise ValueError("This matrix is singular (determinant = 0), so it has no inverse.")
        return m_scale(m_transpose(cofactor_matrix(A)), 1.0 / d)

    def transpose(self, A):
        return m_transpose(A)

    def trace(self, A):
        return trace3(A)

    def rank(self, A):
        return len(rref(A)[1])

    def rref(self, A):
        return rref(A)[0]

    def cofactor(self, A):
        return cofactor_matrix(A)

    def adjugate(self, A):
        return m_transpose(cofactor_matrix(A))

    def norm(self, A):
        return math.sqrt(sum(x * x for row in A for x in row))

    def charpoly(self, A):
        return char_coeffs(A)

    def eigen(self, A):
        _, b, c, d = char_coeffs(A)
        roots = cubic_roots(b, c, d)
        scale = max(1.0, max(abs(z) for z in roots))
        return [{"value": lam, "mult": m, "vectors": eigenvectors_for(A, lam, m)}
                for lam, m in group_eigenvalues(roots, scale, b, c)]

    def scale(self, A, k):
        return m_scale(A, k)

    def power(self, A, n):
        n = self._int_exponent(n)
        if n < 0:
            if is_singular(A):
                raise ValueError("The matrix is singular, so negative powers (which need the inverse) don't exist.")
            A, n = self.inv(A), -n
        result, base = identity(), A
        while n:
            if n & 1:
                result = m_mul(result, base)
            n >>= 1
            if n:
                base = m_mul(base, base)
        return result

    @staticmethod
    def _int_exponent(n):
        if abs(n - round(n)) > 1e-9:
            raise ValueError("The power n must be a whole number (for example 3 or −2).")
        n = int(round(n))
        if abs(n) > 1000:
            raise ValueError("Please keep the power between −1000 and 1000.")
        return n

    def props(self, A):
        I, T = identity(), m_transpose(A)
        A2 = m_mul(A, A)
        tol = 1e-9 * max(1.0, max_abs(A))
        off = lambda cond: all(abs(A[i][j]) <= tol for i in R3 for j in R3 if cond(i, j))  # noqa: E731
        d = det3(A)
        return {
            "det": d, "trace": trace3(A), "rank": self.rank(A), "norm": self.norm(A),
            "invertible": not is_singular(A, d),
            "symmetric": approx_equal(A, T),
            "skew": approx_equal(A, m_scale(T, -1)),
            "diagonal": off(lambda i, j: i != j),
            "upper": off(lambda i, j: i > j),
            "lower": off(lambda i, j: i < j),
            "identity": approx_equal(A, I),
            "zero": max_abs(A) <= 1e-12,
            "orthogonal": approx_equal(m_mul(A, T), I),
            "idempotent": approx_equal(A2, A),
            "involutory": approx_equal(A2, I),
            "nilpotent": max_abs(m_mul(A2, A)) <= 1e-9 * max(1.0, max_abs(A)) ** 3,
        }

    # ======================================================================
    #  Step-by-step explanations
    # ======================================================================
    def explain(self, op, ctx, res, f):
        S = Steps()
        fn = getattr(self, "_x_" + op, None)
        if fn is None:
            return S.p("No step-by-step explanation is available for this operation.")
        fn(S, ctx, res, f)
        return S

    # ---------- element-wise ----------
    def _elementwise(self, S, ctx, res, f, sym, entry_sym, rule):
        A, B = ctx["A"], ctx["B"]
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        S.h("Rule")
        S.p(rule)
        S.m(text_row(tm(A), f"  {sym}  ", tm(B), "  =  ", tm(res)))
        S.h("Entry by entry")
        S.m([f"c{ij(i, j)} = {par(f(A[i][j]))} {entry_sym} {par(f(B[i][j]))} = {f(res[i][j])}"
             for i in R3 for j in R3])

    def _x_add(self, S, ctx, res, f):
        self._elementwise(S, ctx, res, f, "+", "+", "Matrices of the same size are added entry by entry: cᵢⱼ = aᵢⱼ + bᵢⱼ.")
        S.ok("Addition is commutative: A + B = B + A.")

    def _x_sub(self, S, ctx, res, f):
        self._elementwise(S, ctx, res, f, "−", "−", "Subtract entry by entry: cᵢⱼ = aᵢⱼ − bᵢⱼ.")
        S.p("Note: A − B = −(B − A), so the order matters.")

    def _x_hadamard(self, S, ctx, res, f):
        self._elementwise(S, ctx, res, f, "∘", "·",
                          "The Hadamard (element-wise) product multiplies matching entries: cᵢⱼ = aᵢⱼ · bᵢⱼ. "
                          "It is NOT the usual matrix product.")

    # ---------- products ----------
    def _product(self, S, X, Y, res, f, nx, ny):
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        S.h("Row × column")
        S.p(f"Each entry cᵢⱼ is the dot product of row i of {nx} with column j of {ny}.")
        S.m(text_row(tm(X), "  ×  ", tm(Y), "  =  ", tm(res)))
        S.h("All nine dot products")
        lines = []
        for i in R3:
            for j in R3:
                terms = " + ".join(f"{par(f(X[i][k]))}·{par(f(Y[k][j]))}" for k in R3)
                lines.append(f"c{ij(i, j)} = {terms} = {f(res[i][j])}")
        S.m(lines)
        S.p("27 multiplications and 18 additions in total.")

    def _x_mul(self, S, ctx, res, f):
        self._product(S, ctx["A"], ctx["B"], res, f, "A", "B")
        if approx_equal(res, m_mul(ctx["B"], ctx["A"])):
            S.ok("Here A × B = B × A — these two matrices commute.")
        else:
            S.warn("In general A × B ≠ B × A — try the B × A button to compare.")

    def _x_mul_ba(self, S, ctx, res, f):
        self._product(S, ctx["B"], ctx["A"], res, f, "B", "A")
        if approx_equal(res, m_mul(ctx["A"], ctx["B"])):
            S.ok("Here B × A = A × B — these two matrices commute.")
        else:
            S.warn("B × A ≠ A × B — matrix multiplication is not commutative.")

    def _x_div(self, S, ctx, res, f):
        A, B = ctx["A"], ctx["B"]
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        Binv = self.inv(B)
        S.p("Matrices can't be divided directly; “A ÷ B” means A × B⁻¹ (B must be invertible).")
        S.h("1 · Invert B")
        S.m(f"det(B) = {f(det3(B))}  ≠ 0")
        S.m(text_row("B⁻¹ = adj(B) ÷ det(B) = ", tm(Binv)))
        S.h("2 · Multiply")
        S.m(text_row(tm(A), "  ×  ", tm(Binv), "  =  ", tm(res)))
        if approx_equal(m_mul(res, B), A):
            S.ok("Check: (A × B⁻¹) × B = A")

    def _x_solve(self, S, ctx, res, f):
        A, B = ctx["A"], ctx["B"]
        aug = [list(A[i]) + list(B[i]) for i in R3]
        S.h("Gauss–Jordan elimination on [A | B]")
        S.p("Row-reduce the augmented matrix until the left block becomes I. "
            "The right block is then the solution X = A⁻¹B.")
        S.m(text_row("[A | B] = ", text_matrix(aug, f, split=N)))
        self._log_steps(S, rref(aug, pivot_cols=N, log=True)[2], f, split=N)
        S.h("Solution")
        S.m(text_row("X = ", text_matrix(res, f)))
        AX = m_mul(A, res)
        S.h("Check")
        S.m(text_row("A × X = ", text_matrix(AX, f)))
        if approx_equal(AX, B):
            S.ok("A × X = B")

    def _x_commutator(self, S, ctx, res, f):
        A, B = ctx["A"], ctx["B"]
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        S.p("The commutator [A, B] = AB − BA measures how far A and B are from commuting.")
        S.m(text_row("AB = ", tm(m_mul(A, B)), "     BA = ", tm(m_mul(B, A))))
        S.m(text_row("AB − BA = ", tm(res)))
        if max_abs(res) <= 1e-9 * max(1.0, max_abs(A), max_abs(B)) ** 2:
            S.ok("[A, B] = 0, so A and B commute (AB = BA).")
        else:
            S.warn("[A, B] ≠ 0, so A and B do not commute.")

    # ---------- determinant & friends ----------
    def _x_det(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.h("Cofactor expansion along row 1")
        S.m(f"det({n}) = a₁₁·M₁₁ − a₁₂·M₁₂ + a₁₃·M₁₃")
        S.p("Mᵢⱼ is the 2×2 determinant left after deleting row i and column j.")
        mins = []
        for j in R3:
            (a, b), (c, d) = sm = minor_matrix(A, 0, j)
            mval = minor(A, 0, j)
            mins.append(mval)
            S.m(text_row(f"M₁{sub_text(j + 1)} = det ", text_matrix(sm, f),
                         f"  = {par(f(a))}·{par(f(d))} − {par(f(b))}·{par(f(c))} = {f(mval)}"))
        terms = [f"{par(f(A[0][j]))}·{par(f(mins[j]))}" for j in R3]
        S.m([f"det({n}) = {terms[0]} − {terms[1]} + {terms[2]}",
             f"       = {f(res)}"])
        S.h("Cross-check · rule of Sarrus")
        (a, b, c), (d, e, g6), (g, h, i) = A
        plus = a * e * i + b * g6 * g + c * d * h
        minus = c * e * g + b * d * i + a * g6 * h
        S.m(["(aei + bfg + cdh) − (ceg + bdi + afh)",
             f"= ({f(plus)}) − ({f(minus)}) = {f(plus - minus)}"])
        if is_singular(A, res):
            S.warn(f"det({n}) = 0, so {n} is singular: no inverse, and rank < 3.")
        else:
            S.ok(f"det({n}) ≠ 0, so {n} is invertible.")

    def _x_inv(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        d = det3(A)
        C = cofactor_matrix(A)
        S.p(f"Formula:  {n}⁻¹ = adj({n}) ÷ det({n})")
        S.h("1 · Determinant")
        S.m(f"det({n}) = {f(d)}  ≠ 0  → the inverse exists")
        S.h("2 · Cofactor matrix   Cᵢⱼ = (−1)ⁱ⁺ʲ · Mᵢⱼ")
        S.m(text_row("C = ", tm(C)))
        S.h("3 · Adjugate = transpose of C")
        S.m(text_row(f"adj({n}) = ", tm(m_transpose(C))))
        S.h("4 · Divide by the determinant")
        S.m(text_row(f"{n}⁻¹ = adj({n}) ÷ {par(f(d))} = ", tm(res)))
        check = m_mul(A, res)
        S.h("5 · Check")
        S.m(text_row(f"{n} × {n}⁻¹ = ", tm(check)))
        if approx_equal(check, identity()):
            S.ok("The product is the identity matrix I.")

    def _x_cofactor(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.h("Cᵢⱼ = (−1)ⁱ⁺ʲ · Mᵢⱼ")
        S.p("Delete row i and column j, take the 2×2 determinant, then apply the checkerboard of signs:")
        S.m(["┌ +  −  + ┐", "│ −  +  − │", "└ +  −  + ┘"])
        lines = []
        for i in R3:
            for j in R3:
                (a, b), (c, d) = minor_matrix(A, i, j)
                sign = "+" if (i + j) % 2 == 0 else "−"
                lines.append(f"C{ij(i, j)} = {sign}({par(f(a))}·{par(f(d))} − {par(f(b))}·{par(f(c))}) = {f(res[i][j])}")
        S.m(lines)
        S.m(text_row(f"cof({n}) = ", text_matrix(res, f)))

    def _x_adjugate(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        C = cofactor_matrix(A)
        S.p("The adjugate (classical adjoint) is the transpose of the cofactor matrix.")
        S.m(text_row("C = ", text_matrix(C, f), f"   →   adj({n}) = Cᵀ = ", text_matrix(res, f)))
        S.h("Key identity")
        d = det3(A)
        S.m(text_row(f"{n} × adj({n}) = ", text_matrix(m_mul(A, res), f), f"  = det({n})·I = {f(d)}·I"))
        S.ok(f"So whenever det({n}) ≠ 0:  {n}⁻¹ = adj({n}) ÷ det({n}).")

    def _x_transpose(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.p(f"Rows become columns: ({n}ᵀ)ᵢⱼ = {n.lower()}ⱼᵢ.")
        S.m(text_row(text_matrix(A, f), "   →   ", text_matrix(res, f)))
        if approx_equal(A, res):
            S.ok(f"{n}ᵀ = {n}, so {n} is symmetric.")

    def _x_trace(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.p("The trace is the sum of the main-diagonal entries.")
        S.m(f"tr({n}) = a₁₁ + a₂₂ + a₃₃ = {f(A[0][0])} + {par(f(A[1][1]))} + {par(f(A[2][2]))} = {f(res)}")
        S.p("Fun fact: the trace also equals the sum of the eigenvalues.")

    def _x_norm(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        sq = [x * x for row in A for x in row]
        S.p("The Frobenius norm is the square root of the sum of the squares of all entries.")
        S.m([f"Σ aᵢⱼ² = {' + '.join(f(x) for x in sq)}", f"       = {f(sum(sq))}",
             f"‖{n}‖F = √{f(sum(sq))} = {f(res)}"])

    # ---------- elimination ----------
    def _log_steps(self, S, steps, f, split=None):
        if not steps:
            S.p("Already in reduced row-echelon form — no row operations needed.")
            return
        S.h("Row operations")
        descs = [describe_ops(ops, f) for ops, _ in steps]
        w = max(len(d) for d in descs) + 3
        for k, ((ops, snap), d) in enumerate(zip(steps, descs), 1):
            S.m(text_row(f"{k:>2}. {d}".ljust(w + 4), text_matrix(snap, f, split=split)))

    def _x_rref(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.h("Gauss–Jordan elimination")
        S.p("Goal: a leading 1 in each pivot row, with zeros everywhere else in that column.")
        S.m(text_row(f"{n} = ", text_matrix(A, f)))
        _, piv, steps = rref(A, log=True)
        self._log_steps(S, steps, f)
        S.h("Result")
        S.m(text_row(f"rref({n}) = ", text_matrix(res, f)))
        S.p(f"Pivot columns: {', '.join(str(c + 1) for c in piv) or 'none'}  →  rank {len(piv)}.")

    def _x_rank(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        S.p("Rank = number of pivots (non-zero rows) after row reduction.")
        Rm, piv, steps = rref(A, log=True)
        self._log_steps(S, steps, f)
        S.m(text_row(f"rref({n}) = ", text_matrix(Rm, f)))
        S.m(f"rank({n}) = {res}      nullity = 3 − {res} = {N - res}")
        if res == N:
            S.ok("Full rank: the rows (and columns) are linearly independent.")
        else:
            S.warn(f"Rank-deficient: {N - res} row(s) depend on the others, so det({n}) = 0.")

    # ---------- eigen ----------
    def _char_poly_steps(self, S, A, n, f):
        coeffs = char_coeffs(A)
        m = [minor(A, k, k) for k in R3]
        S.h("1 · Characteristic polynomial")
        S.p(f"Eigenvalues are the roots of p(λ) = det(λI − {n}). For any 3×3 matrix:")
        S.m("p(λ) = λ³ − tr·λ² + c₂·λ − det")
        S.m([f"tr  = {f(trace3(A))}",
             f"c₂  = M₁₁ + M₂₂ + M₃₃ = {par(f(m[0]))} + {par(f(m[1]))} + {par(f(m[2]))} = {f(coeffs[2])}",
             f"det = {f(det3(A))}"])
        S.m(f"p(λ) = {poly_text(coeffs, f)}")
        return coeffs

    def _x_charpoly(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        c = self._char_poly_steps(S, A, n, f)
        S.h("2 · Cayley–Hamilton check")
        S.p("Every square matrix satisfies its own characteristic polynomial:")
        A2 = m_mul(A, A)
        A3 = m_mul(A2, A)
        PA = [[A3[i][j] + c[1] * A2[i][j] + c[2] * A[i][j] + (c[3] if i == j else 0)
               for j in R3] for i in R3]
        S.m(text_row(f"p({n}) = {n}³ {'−' if c[1] > 0 else '+'} {f(abs(c[1]))}{n}² … = ", text_matrix(PA, f)))
        if max_abs(PA) <= 1e-7 * max(1.0, max_abs(A)) ** 3:
            S.ok(f"p({n}) = 0 — Cayley–Hamilton holds.")
        S.p("The roots of p(λ) are the eigenvalues — press “Eigen λ, v” to find them.")

    def _x_eigen(self, S, ctx, res, f):
        A, n = ctx["M"], ctx["name"]
        self._char_poly_steps(S, A, n, f)
        S.h("2 · Solve p(λ) = 0")
        S.p("Cardano's formula gives the three roots of the cubic; Newton's method polishes them.")
        S.m([f"λ = {f(g['value'])}" + (f"    (multiplicity {g['mult']})" if g["mult"] > 1 else "")
             for g in res])
        S.h("3 · Eigenvectors: solve (A − λI)·v = 0")
        for g in res:
            lam = g["value"]
            Ml = [[A[i][j] - (lam if i == j else 0) for j in R3] for i in R3]
            blocks = [f"λ = {f(lam)}:   {n} − λI = ", text_matrix(Ml, f), "   →   v = "]
            for k, v in enumerate(g["vectors"]):
                blocks += ([",  "] if k else []) + [text_matrix(column(v), f)]
            S.m(text_row(*blocks))
            if len(g["vectors"]) < g["mult"]:
                S.warn(f"Only {len(g['vectors'])} independent eigenvector for a multiplicity-{g['mult']} "
                       f"eigenvalue: {n} is defective (not diagonalisable).")
        S.h("4 · Check  A·v = λ·v")
        lines = []
        for g in res:
            for v in g["vectors"]:
                Av = mat_vec(A, v)
                err = math.sqrt(sum(abs(Av[i] - g["value"] * v[i]) ** 2 for i in R3))
                lines.append(f"λ = {f(g['value'])}:   ‖Av − λv‖ = {err:.1e}")
        S.m(lines)
        S.ok("Each vector is scaled so its largest entry is 1 (any non-zero multiple also works).")

    # ---------- props / scalar / power ----------
    def _x_props(self, S, ctx, res, f):
        from matrix_common import PROPERTY_FLAGS
        n = ctx["name"]
        S.h("Numbers")
        S.m([f"det  = {f(res['det'])}", f"tr   = {f(res['trace'])}",
             f"rank = {res['rank']}", f"‖{n}‖F = {f(res['norm'])}"])
        S.h("Tests (with a small tolerance for rounding)")
        S.m([f"{'✓' if res[key] else '✗'}  {label:<17} {rule.replace('M', n)}" for key, label, rule in PROPERTY_FLAGS])

    def _x_scale(self, S, ctx, res, f):
        A, n, k = ctx["M"], ctx["name"], ctx["k"]
        S.p("Scalar multiplication multiplies every entry by k.")
        S.m(text_row(f"{par(f(k))} × ", text_matrix(A, f), "  =  ", text_matrix(res, f)))
        S.p(f"det(k·{n}) = k³·det({n}) = {f(k ** 3 * det3(A))}")

    def _x_power(self, S, ctx, res, f):
        A, name = ctx["M"], ctx["name"]
        n = self._int_exponent(ctx["k"])
        tm = lambda M: text_matrix(M, f)  # noqa: E731
        if n == 0:
            S.p(f"{name}⁰ = I for every square matrix.")
            S.m(text_row(f"{name}⁰ = ", tm(res)))
            return
        base, label = A, name
        if n < 0:
            base, label = self.inv(A), f"{name}⁻¹"
            S.p(f"Negative power: {name}{sup_text(n)} = ({name}⁻¹){sup_text(-n)}, so first invert {name}:")
            S.m(text_row(f"{name}⁻¹ = ", tm(base)))
        m = abs(n)
        S.h("Exponentiation by squaring")
        S.p(f"{m} = {bin(m)[2:]}₂ — square repeatedly, then multiply the powers you need.")
        powers, k, B = [], 1, base
        while k <= m:
            if m & k:
                powers.append(k)
            if k <= 8:
                S.m(text_row(f"{label}{sup_text(k)} = ", tm(B)))
            k <<= 1
            if k <= m:
                B = m_mul(B, B)
        if powers:
            S.m(f"{label}{sup_text(m)} = " + " × ".join(f"{label}{sup_text(p)}" for p in powers))
        S.m(text_row(f"{name}{sup_text(n)} = ", tm(res)))
        mults = (m.bit_length() - 1) + (bin(m).count("1") - 1)
        S.ok(f"{mults} matrix multiplication(s) instead of {m - 1}.")


ENGINE = PureEngine()

if __name__ == "__main__":
    from matrix_ui import run_app
    run_app(ENGINE)
