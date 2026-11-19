# ▦ Matrix Studio — 3×3 Matrix Calculator

A polished desktop app for 3×3 matrix operations, in **two editions**:

| Edition | File | Maths done by | Needs |
|---|---|---|---|
| **NumPy** | `matrix_calculator_numpy.py` | `numpy` / `np.linalg` | Python 3.8+, `pip install numpy` |
| **Pure Python** | `matrix_calculator_no_numpy.py` | hand-written Python (lists, Gauss–Jordan, Cardano) | Python 3.8+ only |

Both editions share the same interface (`matrix_ui.py`). The interface does no maths itself: every number you see comes from the edition's own engine.

## Run it

```bash
python matrix_calculator_no_numpy.py      # zero dependencies
pip install numpy
python matrix_calculator_numpy.py         # NumPy edition
```

## Operations (22)

**Two matrices:** A + B · A − B · A × B · B × A · A ∘ B (Hadamard) · A × B⁻¹ · solve A·X = B · commutator AB − BA

**One matrix** (A or B): determinant · inverse · transpose · trace · rank · RREF · cofactor · adjugate · eigenvalues **and** eigenvectors (complex too) · characteristic polynomial · Frobenius norm · full property report · k·M · Mⁿ (negative n uses the inverse)

## What makes it different

- **Step-by-step working for every operation.**
  - *Pure Python:* the working done by hand: cofactor expansion plus a Sarrus cross-check, every Gauss–Jordan row operation (`R₂ ← R₂ − 4·R₁`) with the matrix after each step, all nine dot products of a product, Cardano + eigenvector derivation, and a Cayley–Hamilton check.
  - *NumPy:* the exact NumPy calls in Python-REPL style (`>>> np.linalg.det(A)`), plus independent cross-checks: det = product of the eigenvalues, rank from singular values, residuals ‖Av − λv‖, the condition number, and `einsum` against `@`.
- **Live insights** under each matrix as you type: det, trace, rank, plus *Invertible / Singular*, *Symmetric*, *Orthogonal*, *Triangular* and similar badges.
- **Smart cells:** type `1/3`, `-2.5`, `2^3`, `sqrt(2)`, `√2`, `pi/4`, `sin(pi/6)`. Invalid cells turn red instantly.
- **Exact fractions:** switch between `0.75` and `3/4` display, with adjustable precision.
- **Chain calculations:** *Use as A / Use as B* feeds a result back in.
- **History:** click any earlier calculation to see it and its working again.
- **Copy anywhere:** results copy as tab-separated text (pastes into Excel) or as LaTeX `bmatrix`. You can paste a 3×3 block from Excel or text straight into a matrix.
- **Dark and light themes** with rounded cards, hover states and an animated result view. The layout adapts to high-DPI and small screens, and your matrices, theme and settings are remembered between runs.
- **Friendly errors:** singular matrices, bad input and overflow are explained in the app instead of failing.

## Keyboard

| Keys | Action |
|---|---|
| `Ctrl+Enter` / `F5` | Repeat the last operation |
| `Enter`, arrow keys | Move between cells (A → B → A) |
| Mouse wheel on a cell | Nudge the value ±1 (`Shift` = ±0.1) |
| `Ctrl+R` / `Ctrl+Shift+R` | Random A / random B |
| `Ctrl+D` | Decimals ⇄ fractions |
| `Ctrl+T` | Dark ⇄ light theme |
| `Ctrl+L` | Clear both matrices |
| `Ctrl+1` / `Ctrl+2` | Jump to matrix A / B |
| `F1` | Help |

## Project layout

```
matrix_calculator_numpy.py      NumPy engine + explanations   (entry point)
matrix_calculator_no_numpy.py   Pure-Python engine + explanations (entry point)
matrix_ui.py                    Shared Tkinter interface
matrix_common.py                Number formatting, safe input parser, text helpers
tests/test_engines.py           Unit tests + 300 random NumPy-vs-pure cross-checks
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The pure-Python engine is checked against known answers. When NumPy is installed, both engines are also checked against each other on 300 random matrices for every operation, and every explanation is generated in both display modes.
