"""
matrix_common.py — helpers shared by both Matrix Studio engines and the UI.

Standard library only, so the "no NumPy" edition keeps zero dependencies.

Contents
--------
* NumberFormat   – turns numbers into tidy text (decimal or fraction mode)
* parse_number   – safe parser for what users type into a cell ("1/3", "sqrt(2)", "2^3", ...)
* text_matrix / text_row – draw matrices with box brackets for the step-by-step panel
* Steps          – small container the engines fill with an explanation
* describe_ops   – readable row-operation labels (R₂ ← R₂ − 4·R₁)
* poly_text      – pretty characteristic polynomial
"""
from __future__ import annotations

import ast
import math
import operator
import re
from fractions import Fraction
from typing import Iterable, List, Sequence, Union

N = 3

# --------------------------------------------------------------------------
#  Sub / superscripts
# --------------------------------------------------------------------------
_SUB = str.maketrans("0123456789-−+", "₀₁₂₃₄₅₆₇₈₉₋₋₊")
_SUP = str.maketrans("0123456789-−+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺")


def sub_text(value) -> str:
    """'12' -> '₁₂'"""
    return str(value).translate(_SUB)


def sup_text(value) -> str:
    """'-2' -> '⁻²'"""
    return str(value).translate(_SUP)


def ij(i: int, j: int) -> str:
    """Zero-based (i, j) -> subscript '₁₂' style index."""
    return sub_text(f"{i + 1}{j + 1}")


# --------------------------------------------------------------------------
#  Number formatting
# --------------------------------------------------------------------------
class NumberFormat:
    """Callable formatter.  mode = 'decimal' | 'fraction', digits = decimals shown."""

    def __init__(self, mode: str = "decimal", digits: int = 4):
        self.mode = mode if mode in ("decimal", "fraction") else "decimal"
        self.digits = max(0, min(12, int(digits)))

    # pretty=True uses a real minus sign (−) for display;
    # pretty=False gives plain ASCII that can be typed back into a cell.
    def __call__(self, x, pretty: bool = True) -> str:
        if isinstance(x, complex):
            s = self._complex(x)
        else:
            s = self._real(float(x))
        return s.replace("-", "−") if pretty else s

    def _complex(self, z: complex) -> str:
        scale = max(1.0, abs(z))
        re_, im_ = z.real, z.imag
        if abs(im_) <= 1e-9 * scale:
            return self._real(re_)
        im_txt = self._real(abs(im_))
        im_txt = "i" if im_txt == "1" else im_txt + "i"
        if abs(re_) <= 1e-9 * scale:
            return ("-" if im_ < 0 else "") + im_txt
        return f"{self._real(re_)} {'-' if im_ < 0 else '+'} {im_txt}"

    def _real(self, x: float) -> str:
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "∞" if x > 0 else "-∞"
        if abs(x) < 1e-12:
            return "0"
        d = self.digits
        if abs(x) >= 1e12:
            return f"{x:.{max(d, 3)}g}"
        r = round(x)
        if abs(x - r) <= 1e-9 * max(1.0, abs(x)):
            return str(int(r))
        if self.mode == "fraction":
            fr = Fraction(x).limit_denominator(10_000)
            if abs(fr.numerator / fr.denominator - x) <= 1e-9 * max(1.0, abs(x)):
                return f"{fr.numerator}/{fr.denominator}"
        too_small = abs(x) < (10 ** (-d) if d else 0.5)
        if too_small:  # would round to 0 -> show significant digits instead
            return f"{x:.{max(d, 2)}g}"
        s = f"{x:.{d}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return "0" if s in ("-0", "") else s


def par(s: str) -> str:
    """Wrap negative numbers in brackets: '−2' -> '(−2)'."""
    return f"({s})" if s.startswith(("−", "-")) or " " in s else s


# --------------------------------------------------------------------------
#  Safe number parser for cell input
# --------------------------------------------------------------------------
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log10, "ln": math.log, "exp": math.exp, "abs": abs,
}
_CONSTS = {"pi": math.pi, "π": math.pi, "e": math.e, "tau": math.tau}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
            raise ValueError("power too large")
        return _BINOPS[type(node.op)](a, b)
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and len(node.args) == 1 and not node.keywords):
        return _FUNCS[node.func.id](_eval(node.args[0]))
    raise ValueError("unsupported expression")


def parse_number(text) -> float:
    """Parse a cell.  Accepts 3, -2.5, 1e-3, 1/3, 2^3, sqrt(2), √2, pi/4, 2*pi ...
    Empty text counts as 0.  Raises ValueError with a friendly message."""
    raw = str(text)
    s = raw.strip()
    if not s:
        return 0.0
    s = (s.replace("−", "-").replace("×", "*").replace("·", "*")
          .replace("÷", "/").replace("^", "**"))
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|\([^()]*\)|pi|π|e)", r"sqrt(\1)", s)
    try:
        value = float(s)
    except ValueError:
        try:
            value = _eval(ast.parse(s, mode="eval").body)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, RecursionError):
            raise ValueError(f"“{raw.strip()}” isn't a valid number") from None
    if isinstance(value, complex) or not math.isfinite(value):
        raise ValueError(f"“{raw.strip()}” isn't a valid real number")
    return float(value)


# --------------------------------------------------------------------------
#  Text rendering of matrices (monospace, for the step-by-step panel)
# --------------------------------------------------------------------------
Block = Union[str, Sequence[str]]


def text_matrix(M, f, split: int = None) -> List[str]:
    """Matrix -> list of lines with box brackets.  `split` draws a │ before that column
    (used for augmented matrices [A | B])."""
    cells = [[f(x) for x in row] for row in M]
    rows, cols = len(cells), len(cells[0])
    w = [max(len(cells[i][j]) for i in range(rows)) for j in range(cols)]
    out = []
    for i, row in enumerate(cells):
        parts = []
        for j, c in enumerate(row):
            if split is not None and j == split:
                parts.append("│")
            parts.append(c.rjust(w[j]))
        body = "  ".join(parts)
        if rows == 1:
            l, r = "[", "]"
        elif i == 0:
            l, r = "┌", "┐"
        elif i == rows - 1:
            l, r = "└", "┘"
        else:
            l, r = "│", "│"
        out.append(f"{l} {body} {r}")
    return out


def text_row(*blocks: Block) -> List[str]:
    """Place text blocks side by side, vertically centred:
    text_row(A_lines, "  +  ", B_lines)."""
    bl = [[b] if isinstance(b, str) else list(b) for b in blocks]
    height = max(len(b) for b in bl)
    columns = []
    for b in bl:
        width = max(len(s) for s in b)
        top = (height - len(b)) // 2
        columns.append([" " * width] * top + [s.ljust(width) for s in b]
                       + [" " * width] * (height - len(b) - top))
    return ["".join(col[i] for col in columns).rstrip() for i in range(height)]


def column(v) -> List[List]:
    """[a, b, c] -> [[a], [b], [c]] so a vector can be drawn as a column."""
    return [[x] for x in v]


# --------------------------------------------------------------------------
#  Step-by-step container
# --------------------------------------------------------------------------
class Steps(list):
    """A list of (kind, text).  kinds: h = heading, p = paragraph, m = maths block,
    code = code / REPL block, ok = success line, warn = warning line."""

    def h(self, text):
        self.append(("h", text))
        return self

    def p(self, text):
        self.append(("p", text))
        return self

    def m(self, block: Block):
        self.append(("m", block if isinstance(block, str) else "\n".join(block)))
        return self

    def code(self, text):
        self.append(("code", text))
        return self

    def ok(self, text):
        self.append(("ok", "✓  " + text))
        return self

    def warn(self, text):
        self.append(("warn", "!  " + text))
        return self


# --------------------------------------------------------------------------
#  Row operations (Gauss–Jordan) as readable text
# --------------------------------------------------------------------------
def _rn(i):
    return "R" + sub_text(i + 1)


def describe_op(op, f) -> str:
    kind = op[0]
    if kind == "swap":
        return f"{_rn(op[1])} ↔ {_rn(op[2])}"
    if kind == "div":
        return f"{_rn(op[1])} ← {_rn(op[1])} ÷ {par(f(op[2]))}"
    if kind == "sub":
        _, target, factor, source = op
        neg = (factor.real if isinstance(factor, complex) else factor) < 0 \
            and not (isinstance(factor, complex) and abs(factor.imag) > 1e-12)
        mag = f(-factor if neg else factor)
        coef = "" if mag == "1" else f"{par(mag)}·"
        return f"{_rn(target)} ← {_rn(target)} {'+' if neg else '−'} {coef}{_rn(source)}"
    return str(op)


def describe_ops(ops: Iterable, f) -> str:
    return ",  ".join(describe_op(op, f) for op in ops)


# --------------------------------------------------------------------------
#  Polynomials
# --------------------------------------------------------------------------
def poly_text(coeffs, f, var: str = "λ") -> str:
    """[1, -5, 3, -1] -> 'λ³ − 5λ² + 3λ − 1'"""
    deg = len(coeffs) - 1
    terms = []
    for k, c in enumerate(coeffs):
        power = deg - k
        c = c.real if isinstance(c, complex) else c
        if abs(c) < 1e-12:
            continue
        mag = f(abs(c))
        v = var + (sup_text(power) if power > 1 else "") if power else ""
        coef = "" if (mag == "1" and power) else mag
        if "/" in coef and v:
            coef = f"({coef})"
        terms.append(("−" if c < 0 else "+", coef + v))
    if not terms:
        return "0"
    first_sign, first = terms[0]
    out = ("−" if first_sign == "−" else "") + first
    for sign, t in terms[1:]:
        out += f" {sign} {t}"
    return out


# --------------------------------------------------------------------------
#  Display names of the boolean matrix properties (shared by engines and UI)
# --------------------------------------------------------------------------
PROPERTY_FLAGS = [
    ("invertible", "Invertible", "det ≠ 0"),
    ("symmetric", "Symmetric", "M = Mᵀ"),
    ("skew", "Skew-symmetric", "M = −Mᵀ"),
    ("diagonal", "Diagonal", "zeros off the diagonal"),
    ("upper", "Upper triangular", "zeros below the diagonal"),
    ("lower", "Lower triangular", "zeros above the diagonal"),
    ("orthogonal", "Orthogonal", "M·Mᵀ = I"),
    ("idempotent", "Idempotent", "M² = M"),
    ("involutory", "Involutory", "M² = I"),
    ("nilpotent", "Nilpotent", "M³ = 0"),
]
