"""
matrix_ui.py — the Matrix Studio desktop interface (shared by both editions).

Pure tkinter (standard library only).  The maths lives in the engine that is
passed to run_app(); the UI never computes anything itself, so the pure-Python
edition stays NumPy-free and the NumPy edition really uses NumPy.

Highlights
    • Dark / light themes, rounded cards, hover states, smooth result animation
    • Live insight badges under each matrix (det, trace, rank, symmetric, …)
    • Cells accept expressions: 1/3, -2.5, 2^3, sqrt(2), pi/4 …
    • Step-by-step working for every operation + clickable history
    • Decimal ⇄ fraction display, adjustable precision, copy as text / LaTeX
    • Keyboard-first: arrows / Enter move between cells, Ctrl+Enter repeats
"""
from __future__ import annotations

import json
import random
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional

from matrix_common import (
    N, PROPERTY_FLAGS, NumberFormat, parse_number, poly_text, sub_text, sup_text,
)

APP_NAME = "Matrix Studio"
APP_VERSION = "2.0"
SETTINGS_PATH = Path.home() / ".matrix_studio.json"

DEFAULT_A = ["2", "1", "1", "1", "3", "2", "1", "0", "0"]
DEFAULT_B = ["1", "2", "3", "4", "5", "6", "7", "8", "10"]

# ==========================================================================
#  Themes
# ==========================================================================
THEMES = {
    "dark": dict(
        bg="#0c0f16", surface="#151a25", surface2="#1d2331", surface3="#272e40",
        border="#232a3a", text="#e8eaf2", muted="#8b92a8", faint="#555d72",
        accent="#8b5cf6", accent_hover="#9b73ff", accent_press="#7646e8", accent_text="#b49cff",
        on_accent="#ffffff", accent2="#22d3ee", on_accent2="#04202a",
        success="#34d399", danger="#f87171", warn="#fbbf24",
        cell="#0f131d", cell_text="#f3f4f9", select="#3d3170", code_bg="#10141e",
        tip_bg="#2a3144", tip_fg="#eef0f6", toast_bg="#2a3144", toast_fg="#ffffff",
    ),
    "light": dict(
        bg="#f2f3f8", surface="#ffffff", surface2="#eef0f6", surface3="#e2e5ef",
        border="#e2e5ed", text="#161a24", muted="#5f6679", faint="#a5abbb",
        accent="#6d4aff", accent_hover="#7d60ff", accent_press="#5935f0", accent_text="#5a36e8",
        on_accent="#ffffff", accent2="#0891b2", on_accent2="#ffffff",
        success="#059669", danger="#dc2626", warn="#b45309",
        cell="#f6f7fb", cell_text="#161a24", select="#e3dbff", code_bg="#f5f6fa",
        tip_bg="#1b1f2b", tip_fg="#ffffff", toast_bg="#1b1f2b", toast_fg="#ffffff",
    ),
}


def _rgb(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1: str, c2: str, t: float) -> str:
    a, b = _rgb(c1), _rgb(c2)
    return "#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def make_theme(name: str) -> Dict[str, str]:
    t = dict(THEMES[name], name=name)
    soft = 0.18 if name == "dark" else 0.10
    t["accent_soft"] = mix(t["surface2"], t["accent"], soft)
    t["accent_soft2"] = mix(t["surface2"], t["accent"], soft * 1.8)
    for key in ("accent2", "success", "danger", "warn"):
        t[key + "_soft"] = mix(t["surface"], t[key], 0.14 if name == "dark" else 0.10)
    return t


# ==========================================================================
#  Operation catalogue
# ==========================================================================
#  group: AB = needs A and B · M = one matrix · Mk = matrix + number
OPS: List[Dict[str, str]] = [
    dict(key="add", group="AB", label="A + B", title="A + B", kind="matrix",
         tip="Add matching entries:  cᵢⱼ = aᵢⱼ + bᵢⱼ"),
    dict(key="sub", group="AB", label="A − B", title="A − B", kind="matrix",
         tip="Subtract matching entries:  cᵢⱼ = aᵢⱼ − bᵢⱼ"),
    dict(key="mul", group="AB", label="A × B", title="A × B", kind="matrix",
         tip="Matrix product — each entry is a row of A · a column of B"),
    dict(key="mul_ba", group="AB", label="B × A", title="B × A", kind="matrix",
         tip="Product in the other order. Usually different from A × B!"),
    dict(key="hadamard", group="AB", label="A ∘ B", title="A ∘ B", kind="matrix",
         tip="Hadamard product — multiply matching entries"),
    dict(key="div", group="AB", label="A × B⁻¹", title="A × B⁻¹", kind="matrix",
         tip="“Divide” A by B: multiply A by the inverse of B"),
    dict(key="solve", group="AB", label="Solve AX = B", title="Solve A·X = B", label_eq="X", kind="matrix",
         tip="Find the matrix X with A·X = B (Gauss–Jordan elimination)"),
    dict(key="commutator", group="AB", label="AB − BA", title="[A, B] = AB − BA", label_eq="AB − BA",
         kind="matrix", tip="Commutator — zero exactly when A and B commute"),

    dict(key="det", group="M", label="Determinant", title="det({M})", kind="scalar",
         tip="Determinant — non-zero means the matrix is invertible"),
    dict(key="inv", group="M", label="Inverse", title="{M}⁻¹", kind="matrix",
         tip="Inverse matrix: {M} × {M}⁻¹ = I"),
    dict(key="transpose", group="M", label="Transpose", title="{M}ᵀ", kind="matrix",
         tip="Swap rows and columns"),
    dict(key="trace", group="M", label="Trace", title="tr({M})", kind="scalar",
         tip="Sum of the main diagonal (= sum of the eigenvalues)"),
    dict(key="rank", group="M", label="Rank", title="rank({M})", kind="scalar",
         tip="Number of linearly independent rows"),
    dict(key="rref", group="M", label="RREF", title="rref({M})", kind="matrix",
         tip="Reduced row-echelon form, with every row operation shown"),
    dict(key="cofactor", group="M", label="Cofactor", title="cof({M})", kind="matrix",
         tip="Matrix of cofactors Cᵢⱼ = (−1)ⁱ⁺ʲ · Mᵢⱼ"),
    dict(key="adjugate", group="M", label="Adjugate", title="adj({M})", kind="matrix",
         tip="Adjugate (classical adjoint) = transpose of the cofactor matrix"),
    dict(key="eigen", group="M", label="Eigen λ, v", title="Eigenvalues & eigenvectors of {M}", kind="eigen",
         tip="Eigenvalues λ and eigenvectors v with {M}·v = λ·v"),
    dict(key="charpoly", group="M", label="Char. poly", title="Characteristic polynomial of {M}", kind="poly",
         tip="p(λ) = det(λI − {M}); its roots are the eigenvalues"),
    dict(key="norm", group="M", label="Norm ‖{M}‖", title="‖{M}‖F", kind="scalar",
         tip="Frobenius norm √(Σ aᵢⱼ²) — the matrix's “length”"),
    dict(key="props", group="M", label="Properties", title="Properties of {M}", kind="report",
         tip="Full analysis: symmetric, orthogonal, triangular, nilpotent …"),

    dict(key="scale", group="Mk", label="k · {M}", title="{k} · {M}", kind="matrix",
         tip="Multiply every entry by the number k"),
    dict(key="power", group="Mk", label="{M}ⁿ", title="{M}{n}", kind="matrix",
         tip="Matrix power using the value n (negative n uses the inverse)"),
]
OP_BY_KEY = {op["key"]: op for op in OPS}


@dataclass
class Result:
    op: str
    title: str
    label: str            # left of "=" on the result canvas
    kind: str             # matrix | scalar | eigen | report | poly | error
    value: Any
    ctx: Dict[str, Any]
    elapsed: float = 0.0
    when: str = ""
    caption: str = ""
    tone: str = "neutral"   # neutral | ok | bad | info


# ==========================================================================
#  Drawing helpers
# ==========================================================================
def round_rect(c: tk.Canvas, x1, y1, x2, y2, r, **kw):
    r = max(0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    pts = [x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r,
           x2, y2 - r, x2, y2 - r, x2, y2, x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2,
           x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1]
    return c.create_polygon(pts, smooth=True, **kw)


def draw_logo(c: tk.Canvas, x, y, size, t, alpha=1.0, bg=None):
    """3 × 3 grid of rounded tiles fading from accent to accent2."""
    gap = size * 0.08
    tile = (size - 2 * gap) / 3
    for i in range(3):
        for j in range(3):
            col = mix(t["accent"], t["accent2"], (i + j) / 4)
            if alpha < 1 and bg:
                col = mix(bg, col, alpha)
            x0 = x + j * (tile + gap)
            y0 = y + i * (tile + gap)
            round_rect(c, x0, y0, x0 + tile, y0 + tile, tile * 0.3, fill=col, outline=col)


# ==========================================================================
#  Widgets
# ==========================================================================
class Tooltip:
    def __init__(self, widget, app, text):
        self.w, self.app, self.text = widget, app, text
        self.tip = None
        self._job = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._job = self.w.after(550, self._show)

    def _cancel(self):
        if self._job:
            try:
                self.w.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _show(self):
        self._job = None
        if self.tip or not self.w.winfo_exists():
            return
        t, px = self.app.t, self.app.px
        self.tip = tw = tk.Toplevel(self.w)
        tw.wm_overrideredirect(True)
        try:
            tw.attributes("-topmost", True)
        except tk.TclError:
            pass
        tw.configure(bg=t["border"])
        tk.Label(tw, text=self.text, bg=t["tip_bg"], fg=t["tip_fg"], font=self.app.fonts["small"],
                 padx=px(10), pady=px(6), justify="left", wraplength=px(300)).pack(padx=1, pady=1)
        x = self.w.winfo_rootx()
        y = self.w.winfo_rooty() + self.w.winfo_height() + px(6)
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, _=None):
        self._cancel()
        if self.tip is not None:
            try:
                self.tip.destroy()
            except tk.TclError:
                pass
            self.tip = None


class RoundButton(tk.Canvas):
    """Flat rounded button drawn on a canvas, with hover / press / focus states."""

    def __init__(self, parent, app, text, command=None, variant="ghost", font=None,
                 height=34, width=None, padx=14, radius=9, tip=None, takefocus=True):
        self.app, self.text, self.command, self.variant = app, text, command, variant
        self.font = font or app.fonts["btn"]
        self._padx = app.px(padx)
        self._auto_w = width is None
        w = self.font.measure(text) + 2 * self._padx if width is None else app.px(width)
        super().__init__(parent, width=w, height=app.px(height), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=1 if takefocus else 0)
        self.radius = app.px(radius)
        self._hover = self._press = self._focus = False
        self.selected = False
        self.enabled = True
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, press=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(press=True))
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<KeyPress-space>", lambda e: self.invoke())
        self.bind("<KeyPress-Return>", lambda e: self.invoke())
        self.bind("<FocusIn>", lambda e: self._set(focus=True))
        self.bind("<FocusOut>", lambda e: self._set(focus=False))
        self.bind("<Configure>", lambda e: self._draw())
        if tip:
            Tooltip(self, app, tip)
        self._draw()

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, "_" + k, v)
        self._draw()

    def _release(self, e):
        fire = self._press and 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height()
        self._set(press=False)
        if fire:
            self.invoke()

    def invoke(self):
        if self.enabled and self.command:
            self.command()

    def set_text(self, text):
        self.text = text
        if self._auto_w:
            self.configure(width=self.font.measure(text) + 2 * self._padx)
        self._draw()

    def set_selected(self, flag):
        if self.selected != flag:
            self.selected = flag
            self._draw()

    def set_enabled(self, flag):
        self.enabled = flag
        self.configure(cursor="hand2" if flag else "arrow")
        self._draw()

    def _palette(self):
        t, v = self.app.t, self.variant
        outline = None
        if v == "primary":
            base, hov, prs, fg = t["accent"], t["accent_hover"], t["accent_press"], t["on_accent"]
        elif v == "op":
            base, hov, prs, fg = t["surface2"], t["accent_soft"], t["accent_soft2"], t["text"]
            outline = t["accent"] if self._hover else None
        elif v == "chip":
            base, hov, prs, fg = t["surface2"], t["surface3"], t["border"], (t["text"] if self._hover else t["muted"])
        elif v == "subtle":
            bg = self.cget("bg")
            base, hov, prs, fg = bg, t["surface2"], t["surface3"], t["muted"]
        else:  # ghost
            base, hov, prs, fg = t["surface2"], t["surface3"], t["border"], t["text"]
        if self.selected:
            base = hov = t["accent_soft2"]
            fg, outline = t["accent_text"], t["accent"]
        if not self.enabled:
            base = hov = prs = t["surface2"]
            fg, outline = t["faint"], None
        return base, hov, prs, fg, outline

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 1:
            w, h = int(self.cget("width")), int(self.cget("height"))
        base, hov, prs, fg, outline = self._palette()
        fill = prs if (self._press and self.enabled) else hov if (self._hover and self.enabled) else base
        round_rect(self, 1, 1, w - 1, h - 1, self.radius, fill=fill, outline=outline or fill)
        if self._focus:
            round_rect(self, 1, 1, w - 1, h - 1, self.radius, fill="", outline=self.app.t["accent"], width=2)
        self.create_text(w / 2, h / 2 + (1 if self._press else 0), text=self.text, fill=fg, font=self.font)


class Segmented(tk.Canvas):
    """Pill-shaped segmented control (like iOS / macOS)."""

    def __init__(self, parent, app, options, value, command=None, height=30, font=None, padx=12):
        self.app, self.options, self.value, self.command = app, list(options), value, command
        self.font = font or app.fonts["small_bold"]
        self.inset = app.px(3)
        self.seg_w = max(self.font.measure(lbl) for _, lbl in options) + 2 * app.px(padx)
        w = self.seg_w * len(options) + 2 * self.inset
        super().__init__(parent, width=w, height=app.px(height), bg=parent.cget("bg"),
                         highlightthickness=0, bd=0, cursor="hand2")
        self._hover = None
        self.bind("<Motion>", lambda e: self._set_hover(self._index(e.x)))
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self.bind("<Button-1>", self._click)
        self._draw()

    def _index(self, x):
        i = int((x - self.inset) // self.seg_w)
        return i if 0 <= i < len(self.options) else None

    def _set_hover(self, i):
        if i != self._hover:
            self._hover = i
            self._draw()

    def _click(self, e):
        i = self._index(e.x)
        if i is None:
            return
        v = self.options[i][0]
        if v != self.value:
            self.value = v
            self._draw()
            if self.command:
                self.command(v)

    def set(self, value):
        self.value = value
        self._draw()

    def _draw(self):
        t = self.app.t
        self.delete("all")
        w, h = int(self.cget("width")), int(self.cget("height"))
        r = self.app.px(9)
        round_rect(self, 0, 0, w - 1, h - 1, r, fill=t["surface2"], outline=t["border"])
        for i, (v, lbl) in enumerate(self.options):
            x0 = self.inset + i * self.seg_w
            x1 = x0 + self.seg_w
            if v == self.value:
                round_rect(self, x0, self.inset, x1, h - 1 - self.inset, r - 2, fill=t["accent"], outline=t["accent"])
                fg = t["on_accent"]
            else:
                fg = t["text"] if i == self._hover else t["muted"]
            self.create_text((x0 + x1) / 2, h / 2, text=lbl, fill=fg, font=self.font)


class RoundCard(tk.Canvas):
    """A rounded 'card' surface.  Put child widgets inside `.body`."""

    def __init__(self, parent, app, pad=18, radius=16, expand=False):
        self.app = app
        super().__init__(parent, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.pad, self.r, self.expand = app.px(pad), app.px(radius), expand
        self.body = tk.Frame(self, bg=app.t["surface"])
        self._win = self.create_window(self.pad, self.pad, window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body)
        self.bind("<Configure>", self._on_canvas)

    def _on_body(self, _):
        w = self.body.winfo_reqwidth() + 2 * self.pad
        h = self.body.winfo_reqheight() + 2 * self.pad
        if int(self.cget("width")) != w or int(self.cget("height")) != h:
            self.configure(width=w, height=h)

    def _on_canvas(self, e):
        t = self.app.t
        self.delete("shape")
        round_rect(self, 1, 1, e.width - 2, e.height - 2, self.r, fill=t["surface"],
                   outline=t["border"], tags="shape")
        self.tag_lower("shape")
        kw = {"width": max(1, e.width - 2 * self.pad)}
        if self.expand:
            kw["height"] = max(1, e.height - 2 * self.pad)
        self.itemconfigure(self._win, **kw)


# ==========================================================================
#  Result canvas
# ==========================================================================
class ResultView(tk.Canvas):
    def __init__(self, parent, app):
        super().__init__(parent, bg=app.t["surface"], highlightthickness=0, bd=0,
                         width=app.px(320), height=app.px(180))
        self.app = app
        self.payload: Optional[Result] = None
        self._gen = 0
        self.bind("<Configure>", lambda e: self.redraw(False))

    def show(self, payload, animate=True):
        self.payload = payload
        self.redraw(animate)

    # ---------- main dispatcher ----------
    def redraw(self, animate=False):
        self._gen += 1
        self.delete("all")
        W, H = self.winfo_width(), self.winfo_height()
        if W < 40 or H < 40:
            return
        p = self.payload
        try:
            if p is None:
                self._draw_empty(W, H)
            else:
                getattr(self, "_draw_" + p.kind)(p, W, H)
        except Exception as exc:  # never let a drawing bug kill the app
            self.delete("all")
            self.create_text(W / 2, H / 2, text=f"Couldn't draw this result:\n{exc}",
                             fill=self.app.t["danger"], font=self.app.fonts["body"], justify="center")
        if animate:
            self._animate()

    # ---------- helpers ----------
    def _font(self, kind, size, weight="normal"):
        return self.app.font(kind, size, weight)

    def _hover(self, item, text):
        self.tag_bind(item, "<Enter>", lambda e: self.app.status(text))

    def _pill(self, cx, y, text, tone):
        t, px = self.app.t, self.app.px
        fg, bg = {
            "ok": (t["success"], t["success_soft"]), "bad": (t["danger"], t["danger_soft"]),
            "info": (t["accent2"], t["accent2_soft"]), "neutral": (t["muted"], t["surface2"]),
        }.get(tone, (t["muted"], t["surface2"]))
        f = self.app.fonts["small_bold"]
        w = f.measure(text) + px(22)
        h = f.metrics("linespace") + px(10)
        round_rect(self, cx - w / 2, y, cx + w / 2, y + h, h / 2, fill=bg, outline=bg)
        self.create_text(cx, y + h / 2, text=text, fill=fg, font=f, tags=("anim",))
        return h

    def _layout_matrix(self, cells, font):
        px = self.app.px
        rows, cols = len(cells), len(cells[0])
        colw = [max(font.measure(cells[i][j]) for i in range(rows)) for j in range(cols)]
        rowh = int(font.metrics("linespace") * 1.5)
        gap = max(px(16), int(font.measure("0") * 2.0))
        inner = px(12)
        w = sum(colw) + gap * (cols - 1) + 2 * inner
        h = rowh * rows + px(4)
        return dict(colw=colw, rowh=rowh, gap=gap, inner=inner, w=w, h=h)

    def _matrix_at(self, cells, raw, x, y, font, L, color=None):
        t, px = self.app.t, self.app.px
        tick, lw = px(7), max(2, px(2))
        x0, x1, y0, y1 = x, x + L["w"], y, y + L["h"]
        for pts in ((x0 + tick, y0, x0, y0, x0, y1, x0 + tick, y1),
                    (x1 - tick, y0, x1, y0, x1, y1, x1 - tick, y1)):
            self.create_line(*pts, fill=t["faint"], width=lw, capstyle="round", joinstyle="round", tags=("anim",))
        cx = x0 + L["inner"]
        for j in range(len(cells[0])):
            right = cx + L["colw"][j]
            for i in range(len(cells)):
                cy = y + px(2) + L["rowh"] * i + L["rowh"] / 2
                v = raw[i][j]
                tiny = abs(complex(v)) < 1e-12
                fill = color or (t["faint"] if tiny else t["text"])
                item = self.create_text(right, cy, text=cells[i][j], anchor="e", font=font, fill=fill,
                                        tags=("anim",))
                self._hover(item, f"Row {i + 1}, column {j + 1}  =  {v!r}")
            cx = right + L["gap"]

    # ---------- kinds ----------
    def _draw_empty(self, W, H):
        t, px = self.app.t, self.app.px
        size = px(64)
        draw_logo(self, W / 2 - size / 2, H / 2 - size - px(14), size, t, alpha=0.35, bg=t["surface"])
        self.create_text(W / 2, H / 2 + px(12), text="Your result will appear here",
                         font=self.app.fonts["h2"], fill=t["text"])
        self.create_text(W / 2, H / 2 + px(38),
                         text="Enter two matrices, then pick any operation on the left.",
                         font=self.app.fonts["body"], fill=t["muted"])

    def _draw_matrix(self, p, W, H):
        t, px, f = self.app.t, self.app.px, self.app.fmt
        cells = [[f(x) for x in row] for row in p.value]
        label = p.label + "  ="
        choice = None
        for size in (22, 20, 18, 16, 14, 13, 12, 11, 10, 9):
            mf, lf = self._font("mono", size), self._font("ui", max(10, size - 4))
            L = self._layout_matrix(cells, mf)
            lw = lf.measure(label) + px(18)
            if lw + L["w"] <= W - px(30) and L["h"] <= H - px(56):
                choice = ("inline", mf, lf, L, lw)
                break
            stack_h = L["h"] + lf.metrics("linespace") + px(14)
            if L["w"] <= W - px(30) and stack_h <= H - px(50):
                choice = ("stack", mf, lf, L, lw)
                break
        if choice is None:
            choice = ("stack", mf, lf, L, lw)
        mode, mf, lf, L, lw = choice
        cap_h = px(34) if p.caption else 0
        if mode == "inline":
            x = (W - lw - L["w"]) / 2
            y = (H - L["h"] - cap_h) / 2
            self.create_text(x, y + L["h"] / 2, text=label, anchor="w", font=lf, fill=t["muted"], tags=("anim",))
            self._matrix_at(cells, p.value, x + lw, y, mf, L)
            bottom = y + L["h"]
        else:
            lh = lf.metrics("linespace")
            y = (H - lh - px(12) - L["h"] - cap_h) / 2
            self.create_text(W / 2, y, text=label, anchor="n", font=lf, fill=t["muted"], tags=("anim",))
            self._matrix_at(cells, p.value, (W - L["w"]) / 2, y + lh + px(12), mf, L)
            bottom = y + lh + px(12) + L["h"]
        if p.caption:
            self._pill(W / 2, bottom + px(14), p.caption, p.tone)

    def _draw_scalar(self, p, W, H):
        t, px = self.app.t, self.app.px
        val = self.app.fmt(p.value)
        for size in (44, 38, 32, 26, 22, 18, 14):
            vf = self._font("ui", size, "bold")
            if vf.measure(val) <= W - px(40):
                break
        lf = self._font("ui", 13)
        lh, vh = lf.metrics("linespace"), vf.metrics("linespace")
        cap_h = px(38) if p.caption else 0
        y = (H - lh - px(4) - vh - cap_h) / 2
        self.create_text(W / 2, y, text=p.label + "  =", anchor="n", font=lf, fill=t["muted"], tags=("anim",))
        item = self.create_text(W / 2, y + lh + px(4), text=val, anchor="n", font=vf, fill=t["accent_text"],
                                tags=("anim",))
        self._hover(item, f"{p.label} = {p.value!r}")
        if p.caption:
            self._pill(W / 2, y + lh + px(4) + vh + px(12), p.caption, p.tone)

    def _draw_eigen(self, p, W, H):
        t, px, f = self.app.t, self.app.px, self.app.fmt
        groups = p.value
        for size in (18, 16, 15, 14, 13, 12, 11, 10):
            vf = self._font("mono", size, "bold")
            cf = self._font("mono", max(9, size - 3))
            lf = self._font("ui", 10)
            cols = []
            for g in groups:
                vtxt = f(g["value"])
                vcells = [[[f(x)] for x in v] for v in g["vectors"]]
                Ls = [self._layout_matrix(vc, cf) for vc in vcells]
                vw = sum(L["w"] for L in Ls) + px(8) * max(0, len(Ls) - 1)
                cw = max(vf.measure(vtxt), vw, lf.measure("λ₁  ·  multiplicity 3")) + px(28)
                cols.append((g, vtxt, vcells, Ls, vw, cw))
            total = sum(c[-1] for c in cols)
            if total <= W - px(20):
                break
        lh, vh = lf.metrics("linespace"), vf.metrics("linespace")
        vec_h = max((L["h"] for c in cols for L in c[3]), default=0)
        block_h = lh + px(2) + vh + px(16) + lh + px(4) + vec_h
        cap_h = px(36) if p.caption else 0
        y0 = (H - block_h - cap_h) / 2
        x = (W - total) / 2
        for k, (g, vtxt, vcells, Ls, vw, cw) in enumerate(cols):
            cx = x + cw / 2
            head = f"λ{sub_text(k + 1)}" + (f"  ·  multiplicity {g['mult']}" if g["mult"] > 1 else "")
            self.create_text(cx, y0, text=head, anchor="n", font=lf, fill=t["muted"], tags=("anim",))
            item = self.create_text(cx, y0 + lh + px(2), text=vtxt, anchor="n", font=vf,
                                    fill=t["accent_text"], tags=("anim",))
            self._hover(item, f"λ{sub_text(k + 1)} = {g['value']!r}")
            vy = y0 + lh + px(2) + vh + px(16)
            vlabel = "eigenvector" if len(vcells) == 1 else "eigenvectors"
            self.create_text(cx, vy, text=vlabel, anchor="n", font=lf, fill=t["faint"], tags=("anim",))
            vx = cx - vw / 2
            for vc, L, vec in zip(vcells, Ls, g["vectors"]):
                self._matrix_at(vc, [[z] for z in vec], vx, vy + lh + px(4), cf, L)
                vx += L["w"] + px(8)
            if k:
                self.create_line(x, y0, x, y0 + block_h, fill=t["border"], width=1)
            x += cw
        if p.caption:
            self._pill(W / 2, y0 + block_h + px(14), p.caption, p.tone)

    def _draw_report(self, p, W, H):
        t, px, f = self.app.t, self.app.px, self.app.fmt
        P = p.value
        stats = [("det", f(P["det"])), ("trace", f(P["trace"])), ("rank", str(P["rank"])), ("‖·‖F", f(P["norm"]))]
        pad = px(12)
        tile_w = (W - pad * 5) / 4
        vf, lf = self._font("mono", 15, "bold"), self._font("ui", 9)
        tile_h = vf.metrics("linespace") + lf.metrics("linespace") + px(18)
        rows = (len(PROPERTY_FLAGS) + 1) // 2
        ff = self._font("ui", 10)
        row_h = ff.metrics("linespace") + px(8)
        total_h = tile_h + px(16) + rows * row_h
        y = max(px(4), (H - total_h) / 2)
        for k, (name, val) in enumerate(stats):
            x0 = pad + k * (tile_w + pad)
            round_rect(self, x0, y, x0 + tile_w, y + tile_h, px(10), fill=t["surface2"], outline=t["surface2"])
            self.create_text(x0 + tile_w / 2, y + px(8), text=val, anchor="n", font=vf,
                             fill=t["accent_text"], tags=("anim",))
            self.create_text(x0 + tile_w / 2, y + tile_h - px(7), text=name, anchor="s", font=lf,
                             fill=t["muted"], tags=("anim",))
        y += tile_h + px(16)
        col_w = (W - 3 * pad) / 2
        for idx, (key, label, rule) in enumerate(PROPERTY_FLAGS):
            col, row = divmod(idx, rows)
            x0 = pad + col * (col_w + pad) + px(6)
            cy = y + row * row_h + row_h / 2
            yes = bool(P[key])
            self.create_text(x0, cy, text="✓" if yes else "✗", anchor="w", font=self._font("ui", 11, "bold"),
                             fill=t["success"] if yes else t["faint"], tags=("anim",))
            self.create_text(x0 + px(22), cy, text=label, anchor="w", font=ff,
                             fill=t["text"] if yes else t["muted"], tags=("anim",))
            self.create_text(x0 + col_w - px(10), cy, text=rule.replace("M", p.ctx.get("name", "M")), anchor="e",
                             font=self._font("ui", 9), fill=t["faint"], tags=("anim",))

    def _draw_poly(self, p, W, H):
        t, px = self.app.t, self.app.px
        text = "p(λ) = " + poly_text(p.value, self.app.fmt)
        for size in (26, 22, 19, 16, 14, 12):
            pf = self._font("ui", size, "bold")
            if pf.measure(text) <= W - px(40):
                break
        lf = self._font("ui", 12)
        ph = pf.metrics("linespace")
        y = (H - ph - px(44)) / 2
        self.create_text(W / 2, y, text=text, anchor="n", font=pf, fill=t["accent_text"], tags=("anim",))
        self.create_text(W / 2, y + ph + px(10), text=f"= det(λI − {p.ctx.get('name', 'M')})", anchor="n",
                         font=lf, fill=t["muted"], tags=("anim",))
        if p.caption:
            self._pill(W / 2, y + ph + px(40), p.caption, p.tone)

    def _draw_error(self, p, W, H):
        t, px = self.app.t, self.app.px
        r = px(26)
        cy = H / 2 - px(34)
        self.create_oval(W / 2 - r, cy - r, W / 2 + r, cy + r, fill=t["danger_soft"], outline=t["danger_soft"])
        self.create_text(W / 2, cy, text="!", font=self._font("ui", 22, "bold"), fill=t["danger"], tags=("anim",))
        self.create_text(W / 2, cy + r + px(16), text=p.value, anchor="n", width=W - px(80), justify="center",
                         font=self.app.fonts["body_bold"], fill=t["text"], tags=("anim",))

    # ---------- entrance animation ----------
    def _animate(self):
        gen = self._gen
        items = list(self.find_withtag("anim"))
        if not items:
            return
        bg = self.app.t["surface"]
        finals = {}
        for i in items:
            try:
                finals[i] = self.itemcget(i, "fill")
                self.itemconfigure(i, fill=mix(bg, finals[i], 0.0))
            except (tk.TclError, ValueError):
                finals.pop(i, None)
        frames, stagger = 9, 0.6

        def step(k):
            if gen != self._gen:
                return
            try:
                for n, i in enumerate(items):
                    if i in finals:
                        u = min(1.0, max(0.0, (k - n * stagger) / frames))
                        u = 1 - (1 - u) ** 3
                        self.itemconfigure(i, fill=mix(bg, finals[i], u))
            except tk.TclError:
                return
            if k < frames + len(items) * stagger:
                self.after(16, step, k + 1)

        step(0)


# ==========================================================================
#  The application
# ==========================================================================
class App:
    def __init__(self, engine):
        self.engine = engine
        _enable_windows_dpi()
        self.root = tk.Tk()
        self.root.withdraw()
        s = self.settings = _load_settings()
        self.theme_name = s.get("theme") if s.get("theme") in THEMES else "dark"
        self.t = make_theme(self.theme_name)
        self._fit_scaling()
        self._font_cache: Dict[tuple, tkfont.Font] = {}
        self.fonts = self._make_fonts()

        self.cells_text = {"A": _nine(s.get("A")) or DEFAULT_A[:], "B": _nine(s.get("B")) or DEFAULT_B[:]}
        self.target = s.get("target") if s.get("target") in ("A", "B") else "A"
        self.k_text = str(s.get("k", "2"))[:24]
        self.fmt = NumberFormat(s.get("mode", "decimal"), s.get("digits", 4) if isinstance(s.get("digits"), int) else 4)
        self.history: List[Result] = []
        self.current: Optional[Result] = None
        self.last_op: Optional[str] = None
        self.tab = "working"
        self._toast_w = None
        self._toast_job = None
        self._help = None
        self._badge_jobs: Dict[str, str] = {}

        self.root.title(f"{APP_NAME} — 3×3 Matrix Calculator ({engine.name} edition)")
        self._set_icon()
        self._build()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._place_window()
        self.root.deiconify()
        self.entries["A"][0][0].focus_set()

    def mainloop(self):
        self.root.mainloop()

    # ------------------------------------------------------------------ setup
    def px(self, n) -> int:
        return int(round(n * self.S))

    def _fit_scaling(self):
        """Shrink everything proportionally on small screens (e.g. 1366×768 laptops)."""
        r = self.root
        S = r.winfo_fpixels("1i") / 96.0
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        fit = min(1.0, (sw - 40) / (1280 * S), (sh - 90) / (820 * S))
        fit = max(fit, 0.7)
        if fit < 0.999:
            try:
                r.tk.call("tk", "scaling", float(r.tk.call("tk", "scaling")) * fit)
            except tk.TclError:
                pass
        self.S = r.winfo_fpixels("1i") / 96.0

    def _make_fonts(self):
        fams = {f.lower(): f for f in tkfont.families(self.root)}

        def pick(cands, fallback):
            for c in cands:
                if c.lower() in fams:
                    return fams[c.lower()]
            return tkfont.nametofont(fallback).actual("family")

        self.ui_family = pick(["Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Ubuntu",
                               "Cantarell", "Noto Sans", "DejaVu Sans", "Arial"], "TkDefaultFont")
        self.mono_family = pick(["Cascadia Mono", "Cascadia Code", "JetBrains Mono", "Consolas", "SF Mono",
                                 "Menlo", "DejaVu Sans Mono", "Courier New"], "TkFixedFont")
        F = self.font
        return dict(
            title=F("ui", 17, "bold"), subtitle=F("ui", 9), h2=F("ui", 12, "bold"), body=F("ui", 10),
            body_bold=F("ui", 10, "bold"), small=F("ui", 9), small_bold=F("ui", 9, "bold"),
            caps=F("ui", 8, "bold"), btn=F("ui", 10), cell=F("mono", 13), mono=F("mono", 10),
            badge=F("ui", 8, "bold"), steps_h=F("ui", 10, "bold"),
        )

    def font(self, kind, size, weight="normal"):
        key = (kind, size, weight)
        if key not in self._font_cache:
            fam = self.mono_family if kind == "mono" else self.ui_family
            self._font_cache[key] = tkfont.Font(root=self.root, family=fam, size=size, weight=weight)
        return self._font_cache[key]

    def _set_icon(self):
        try:
            size = 32
            img = tk.PhotoImage(width=size, height=size)
            tile, gap, off = 8, 2, 3
            for i in range(3):
                for j in range(3):
                    col = mix(self.t["accent"], self.t["accent2"], (i + j) / 4)
                    x0, y0 = off + j * (tile + gap), off + i * (tile + gap)
                    img.put(col, to=(x0, y0, x0 + tile, y0 + tile))
            self._icon = img
            self.root.iconphoto(True, img)
        except tk.TclError:
            pass

    def _place_window(self):
        r = self.root
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        w, h = min(self.px(1280), sw - 40), min(self.px(820), sh - 80)
        r.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        r.minsize(min(self.px(1100), w), min(self.px(700), h))

    def _style_ttk(self):
        t = self.t
        st = ttk.Style(self.root)
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        st.configure("Studio.Vertical.TScrollbar", gripcount=0, background=t["surface3"], darkcolor=t["surface3"],
                     lightcolor=t["surface3"], troughcolor=t["surface"], bordercolor=t["surface"],
                     arrowcolor=t["muted"], relief="flat", arrowsize=self.px(11))
        st.map("Studio.Vertical.TScrollbar", background=[("active", t["accent"]), ("pressed", t["accent"])])

    # ------------------------------------------------------------------ build
    def _build(self):
        t, px, root = self.t, self.px, self.root
        root.configure(bg=t["bg"])
        self._style_ttk()
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self._build_header().grid(row=0, column=0, sticky="ew", padx=px(22), pady=(px(16), px(12)))

        body = tk.Frame(root, bg=t["bg"])
        body.grid(row=1, column=0, sticky="nsew", padx=px(22))
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=t["bg"])
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        right = tk.Frame(body, bg=t["bg"])
        right.grid(row=0, column=1, sticky="nsew", padx=(px(16), 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=5)
        right.rowconfigure(1, weight=4)

        self._build_inputs(left)
        self._build_ops(left)
        self._build_result(right)
        self._build_bottom(right)
        self._build_status().grid(row=2, column=0, sticky="ew", padx=px(22), pady=(px(8), px(10)))

        self._refresh_op_labels()
        self._refresh_history()
        for name in ("A", "B"):
            self._validate_all(name)
        if self.current is not None:
            self.show_result(self.current, record=False, animate=False)
        else:
            self._render_steps(None)

    def _build_header(self):
        t, px = self.t, self.px
        h = tk.Frame(self.root, bg=t["bg"])
        logo = tk.Canvas(h, width=px(42), height=px(42), bg=t["bg"], highlightthickness=0)
        draw_logo(logo, px(1), px(1), px(40), t)
        logo.grid(row=0, column=0, rowspan=2, padx=(0, px(12)))
        tk.Label(h, text=APP_NAME, font=self.fonts["title"], bg=t["bg"], fg=t["text"]).grid(
            row=0, column=1, sticky="sw")
        tk.Label(h, text=f"3 × 3 matrix calculator  ·  {self.engine.tagline}", font=self.fonts["subtitle"],
                 bg=t["bg"], fg=t["muted"]).grid(row=1, column=1, sticky="nw")
        h.columnconfigure(2, weight=1)
        right = tk.Frame(h, bg=t["bg"])
        right.grid(row=0, column=3, rowspan=2, sticky="e")

        # engine badge
        f = self.fonts["small_bold"]
        text = self.engine.badge
        bw, bh = f.measure(text) + px(40), px(30)
        badge = tk.Canvas(right, width=bw, height=bh, bg=t["bg"], highlightthickness=0)
        round_rect(badge, 1, 1, bw - 1, bh - 1, bh / 2, fill=t["accent2_soft"], outline=t["accent2_soft"])
        badge.create_oval(px(13), bh / 2 - px(4), px(21), bh / 2 + px(4), fill=t["accent2"], outline=t["accent2"])
        badge.create_text(px(29), bh / 2, text=text, anchor="w", font=f, fill=t["accent2"])
        badge.pack(side="left", padx=(0, px(10)))
        Tooltip(badge, self, f"Every result in this window is computed by the {self.engine.name} engine.")

        dark = self.theme_name == "dark"
        RoundButton(right, self, "☀  Light" if dark else "☾  Dark", self.toggle_theme, height=32,
                    tip="Switch theme  (Ctrl+T)", takefocus=False).pack(side="left", padx=(0, px(8)))
        RoundButton(right, self, "?", self.show_help, height=32, width=32, padx=0,
                    tip="Help & keyboard shortcuts  (F1)", takefocus=False).pack(side="left")
        return h

    # ---------- matrix inputs ----------
    def _build_inputs(self, left):
        t, px = self.t, self.px
        row = tk.Frame(left, bg=t["bg"])
        row.grid(row=0, column=0, sticky="ew")
        self.entries: Dict[str, List[List[tk.Entry]]] = {}
        self.badge_canvas: Dict[str, tk.Canvas] = {}
        self._matrix_card(row, "A").grid(row=0, column=0, sticky="nsew")
        mid = tk.Frame(row, bg=t["bg"])
        mid.grid(row=0, column=1, padx=px(8))
        for text, cmd, tip in (("⇄", self.swap, "Swap A and B"),
                               ("→", lambda: self.copy_matrix("A", "B"), "Copy A into B"),
                               ("←", lambda: self.copy_matrix("B", "A"), "Copy B into A")):
            RoundButton(mid, self, text, cmd, width=36, height=36, padx=0, tip=tip,
                        font=self.font("ui", 12, "bold"), takefocus=False).pack(pady=px(4))
        self._matrix_card(row, "B").grid(row=0, column=2, sticky="nsew")

    def _matrix_card(self, parent, name):
        t, px = self.t, self.px
        card = RoundCard(parent, self, pad=16)
        b = card.body
        b.columnconfigure(0, weight=1)
        color = t["accent"] if name == "A" else t["accent2"]
        on = t["on_accent"] if name == "A" else t["on_accent2"]

        head = tk.Frame(b, bg=t["surface"])
        head.grid(row=0, column=0, sticky="ew", pady=(0, px(12)))
        d = px(26)
        dot = tk.Canvas(head, width=d, height=d, bg=t["surface"], highlightthickness=0)
        dot.create_oval(1, 1, d - 1, d - 1, fill=color, outline=color)
        dot.create_text(d / 2, d / 2, text=name, fill=on, font=self.fonts["body_bold"])
        dot.grid(row=0, column=0)
        tk.Label(head, text=f"Matrix {name}", font=self.fonts["h2"], bg=t["surface"], fg=t["text"]).grid(
            row=0, column=1, padx=(px(8), 0))
        head.columnconfigure(2, weight=1)
        tk.Label(head, text="3 × 3", font=self.fonts["small"], bg=t["surface"], fg=t["faint"]).grid(
            row=0, column=2, sticky="e")

        grid = tk.Frame(b, bg=t["surface"])
        grid.grid(row=1, column=0)
        entries = []
        for i in range(N):
            row = []
            for j in range(N):
                e = tk.Entry(grid, width=7, justify="center", font=self.fonts["cell"], relief="flat", bd=0,
                             bg=t["cell"], fg=t["cell_text"], insertbackground=t["text"],
                             highlightthickness=max(1, px(2)), highlightbackground=t["border"],
                             highlightcolor=color, selectbackground=t["select"], selectforeground=t["text"])
                e.grid(row=i, column=j, padx=px(3), pady=px(3), ipady=px(6))
                e.insert(0, self.cells_text[name][i * N + j])
                self._bind_cell(e, name, i, j)
                row.append(e)
            entries.append(row)
        self.entries[name] = entries

        tools = tk.Frame(b, bg=t["surface"])
        tools.grid(row=2, column=0, pady=(px(12), 0))
        for label, cmd, tip in (
                ("Random", lambda: self.randomize(name), "Random integers from −9 to 9  (Ctrl+R / Ctrl+Shift+R)"),
                ("Identity", lambda: self.load(name, "identity"), "Load the identity matrix I"),
                ("Zero", lambda: self.load(name, "zero"), "Fill with zeros"),
                ("Paste", lambda: self.paste(name), "Paste 9 numbers from the clipboard — Excel, CSV, "
                                                    "[[1,2,3],…] or plain text all work"),
                ("Copy", lambda: self.copy_input(name), "Copy as tab-separated text (pastes into Excel)")):
            RoundButton(tools, self, label, cmd, variant="chip", font=self.fonts["small"], height=26,
                        padx=9, radius=7, tip=tip, takefocus=False).pack(side="left", padx=(0, px(4)))

        bc = tk.Canvas(b, height=px(22), width=1, bg=t["surface"], highlightthickness=0)
        bc.grid(row=3, column=0, sticky="ew", pady=(px(12), 0))
        bc.bind("<Configure>", lambda e, n=name: self._draw_badges(n))
        self.badge_canvas[name] = bc
        return card

    def _bind_cell(self, e, name, i, j):
        e.bind("<FocusIn>", lambda ev: ev.widget.after_idle(lambda: _select_all(ev.widget)))
        e.bind("<KeyRelease>", lambda ev: self._on_cell_edit(name, ev.widget))
        e.bind("<Up>", lambda ev: self._move(name, i, j, -1, 0))
        e.bind("<Down>", lambda ev: self._move(name, i, j, 1, 0))
        e.bind("<Left>", lambda ev: self._move(name, i, j, 0, -1)
               if ev.widget.index("insert") == 0 and not ev.widget.selection_present() else None)
        e.bind("<Right>", lambda ev: self._move(name, i, j, 0, 1)
               if ev.widget.index("insert") == len(ev.widget.get()) and not ev.widget.selection_present() else None)
        e.bind("<Return>", lambda ev: self._next_cell(name, i, j))
        e.bind("<KP_Enter>", lambda ev: self._next_cell(name, i, j))
        e.bind("<MouseWheel>", lambda ev: self._nudge(ev.widget, name, 1 if ev.delta > 0 else -1, ev))
        e.bind("<Button-4>", lambda ev: self._nudge(ev.widget, name, 1, ev))
        e.bind("<Button-5>", lambda ev: self._nudge(ev.widget, name, -1, ev))
        self._entry_shortcuts(e)

    def _move(self, name, i, j, di, dj):
        if dj and not (0 <= j + dj < N):          # hop across to the other matrix
            if name == "A" and j + dj >= N:
                name, j = "B", -1
            elif name == "B" and j + dj < 0:
                name, j = "A", N
        self.entries[name][(i + di) % N][(j + dj) % N].focus_set()
        return "break"

    def _next_cell(self, name, i, j):
        k = i * N + j + 1
        if k == N * N:
            name, k = ("B" if name == "A" else "A"), 0
        self.entries[name][k // N][k % N].focus_set()
        return "break"

    def _nudge(self, w, name, direction, ev):
        if self.root.focus_get() is not w:
            return None
        try:
            v = parse_number(w.get())
        except ValueError:
            return "break"
        step = 0.1 if (ev.state & 0x0001) else 1.0     # Shift = fine steps
        w.delete(0, "end")
        w.insert(0, NumberFormat("decimal", 6)(v + direction * step, pretty=False))
        self._on_cell_edit(name, w)
        return "break"

    def _on_cell_edit(self, name, w):
        self._validate_cell(w)
        self._schedule_badges(name)

    def _validate_cell(self, w):
        t = self.t
        try:
            parse_number(w.get())
            w.configure(fg=t["cell_text"], highlightbackground=t["border"])
            return True
        except ValueError:
            w.configure(fg=t["danger"], highlightbackground=t["danger"])
            return False

    def _validate_all(self, name):
        for row in self.entries[name]:
            for e in row:
                self._validate_cell(e)
        self._schedule_badges(name)

    # ---------- live badges ----------
    def _schedule_badges(self, name):
        job = self._badge_jobs.get(name)
        if job:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self._badge_jobs[name] = self.root.after(120, lambda: self._draw_badges(name))

    def _badge_items(self, name):
        try:
            M = self.read_matrix(name, quiet=True)
        except ValueError:
            return [("Fix the red cell", "bad")]
        try:
            P = self.engine.props(M)
        except Exception:
            return []
        f = self.fmt
        items = [(f"det {f(P['det'])}", "neutral"), (f"tr {f(P['trace'])}", "neutral"),
                 (f"rank {P['rank']}", "neutral"),
                 ("Invertible", "ok") if P["invertible"] else ("Singular", "bad")]
        if P["zero"]:
            items.append(("Zero matrix", "info"))
        elif P["identity"]:
            items.append(("Identity", "info"))
        elif P["diagonal"]:
            items.append(("Diagonal", "info"))
        else:
            if P["symmetric"]:
                items.append(("Symmetric", "info"))
            elif P["skew"]:
                items.append(("Skew-symmetric", "info"))
            if P["upper"]:
                items.append(("Upper △", "info"))
            elif P["lower"]:
                items.append(("Lower △", "info"))
        if P["orthogonal"] and not P["identity"]:
            items.append(("Orthogonal", "info"))
        return items

    def _draw_badges(self, name):
        c = self.badge_canvas.get(name)
        try:
            if c is None or not c.winfo_exists():
                return
        except tk.TclError:
            return
        t, px = self.t, self.px
        c.delete("all")
        W = c.winfo_width()
        if W <= 1:
            return
        f = self.fonts["badge"]
        h, gap, x, y = px(22), px(6), 0, 0
        for text, style in self._badge_items(name):
            fg, bg = {"neutral": (t["muted"], t["surface2"]), "ok": (t["success"], t["success_soft"]),
                      "bad": (t["danger"], t["danger_soft"]), "info": (t["accent2"], t["accent2_soft"])}[style]
            w = f.measure(text) + px(18)
            if x + w > W and x > 0:
                x, y = 0, y + h + gap
            round_rect(c, x, y, x + w, y + h, h / 2, fill=bg, outline=bg)
            c.create_text(x + w / 2, y + h / 2, text=text, fill=fg, font=f)
            x += w + gap
        need = y + h + 1
        if int(c.cget("height")) != need:
            c.configure(height=need)

    # ---------- operations panel ----------
    def _build_ops(self, left):
        t, px = self.t, self.px
        card = RoundCard(left, self, pad=18, expand=True)
        card.grid(row=1, column=0, sticky="nsew", pady=(px(16), 0))
        b = card.body
        b.columnconfigure(0, weight=1)
        self.op_buttons: Dict[str, RoundButton] = {}

        def section(row, title, hint):
            f = tk.Frame(b, bg=t["surface"])
            f.grid(row=row, column=0, sticky="ew")
            tk.Label(f, text=title, font=self.fonts["caps"], bg=t["surface"], fg=t["muted"]).pack(side="left")
            tk.Label(f, text="  " + hint, font=self.fonts["small"], bg=t["surface"], fg=t["faint"]).pack(side="left")
            return f

        def grid_of(row, ops, pady):
            g = tk.Frame(b, bg=t["surface"])
            g.grid(row=row, column=0, sticky="ew", pady=pady)
            for c in range(4):
                g.columnconfigure(c, weight=1, uniform="ops")
            for idx, op in enumerate(ops):
                btn = RoundButton(g, self, op["label"], lambda k=op["key"]: self.run(k), variant="op",
                                  height=38, padx=10, tip=op["tip"].replace("{M}", "M"))
                btn.grid(row=idx // 4, column=idx % 4, sticky="ew", padx=px(3), pady=px(3))
                self.op_buttons[op["key"]] = btn
            return g

        section(0, "TWO MATRICES", "use A and B")
        grid_of(1, [op for op in OPS if op["group"] == "AB"], (px(8), px(16)))

        head = section(2, "ONE MATRIX", "")
        self.seg_target = Segmented(head, self, [("A", "Matrix A"), ("B", "Matrix B")], self.target,
                                    self._set_target, height=28)
        self.seg_target.pack(side="right")
        tk.Label(head, text="apply to  ", font=self.fonts["small"], bg=t["surface"], fg=t["faint"]).pack(side="right")

        g = grid_of(3, [op for op in OPS if op["group"] == "M"], (px(8), 0))
        # final row: the k / n field and the two parameterised operations
        param = tk.Frame(g, bg=t["surface"])
        param.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=px(3), pady=px(3))
        tk.Label(param, text="k / n  =", font=self.fonts["small_bold"], bg=t["surface"], fg=t["muted"]).pack(
            side="left", padx=(px(4), px(8)))
        self.k_entry = tk.Entry(param, width=7, justify="center", font=self.fonts["mono"], relief="flat", bd=0,
                                bg=t["cell"], fg=t["cell_text"], insertbackground=t["text"],
                                highlightthickness=1, highlightbackground=t["border"], highlightcolor=t["accent"],
                                selectbackground=t["select"], selectforeground=t["text"])
        self.k_entry.pack(side="left", fill="x", expand=True, padx=(0, px(8)), pady=px(6), ipady=px(3))
        self.k_entry.insert(0, self.k_text)
        self.k_entry.bind("<KeyRelease>", lambda e: self._validate_cell(self.k_entry))
        self.k_entry.bind("<Return>", lambda e: self.run(self.last_op if self.last_op in ("scale", "power") else "scale"))
        self._entry_shortcuts(self.k_entry)
        Tooltip(self.k_entry, self, "The number used by k·M (any number) and Mⁿ (whole number; negative uses the inverse)")
        for col, key in ((2, "scale"), (3, "power")):
            op = OP_BY_KEY[key]
            btn = RoundButton(g, self, op["label"], lambda k=key: self.run(k), variant="op", height=38,
                              padx=10, tip=op["tip"])
            btn.grid(row=3, column=col, sticky="ew", padx=px(3), pady=px(3))
            self.op_buttons[key] = btn

        b.rowconfigure(10, weight=1)
        tk.Label(b, text="Hover any button for a description  ·  Ctrl+Enter repeats the last operation",
                 font=self.fonts["small"], bg=t["surface"], fg=t["faint"]).grid(row=11, column=0, sticky="w",
                                                                              pady=(px(10), 0))

    def _refresh_op_labels(self):
        for op in OPS:
            if "{M}" in op["label"]:
                self.op_buttons[op["key"]].set_text(op["label"].replace("{M}", self.target))
        for key, btn in self.op_buttons.items():
            btn.set_selected(key == self.last_op)

    def _set_target(self, value):
        self.target = value
        self._refresh_op_labels()
        self.status(f"One-matrix operations now apply to matrix {value}")

    # ---------- result card ----------
    def _build_result(self, right):
        t, px = self.t, self.px
        card = RoundCard(right, self, pad=18, expand=True)
        card.grid(row=0, column=0, sticky="nsew")
        b = card.body
        b.columnconfigure(0, weight=1)
        b.rowconfigure(1, weight=1)

        head = tk.Frame(b, bg=t["surface"])
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(0, weight=1)
        self.res_title = tk.Label(head, text="Result", font=self.fonts["h2"], bg=t["surface"], fg=t["text"],
                                  anchor="w")
        self.res_title.grid(row=0, column=0, sticky="w")
        self.res_sub = tk.Label(head, text="Nothing calculated yet", font=self.fonts["small"], bg=t["surface"],
                                fg=t["muted"], anchor="w")
        self.res_sub.grid(row=1, column=0, sticky="w")

        ctrl = tk.Frame(head, bg=t["surface"])
        ctrl.grid(row=0, column=1, rowspan=2, sticky="e")
        self.seg_mode = Segmented(ctrl, self, [("decimal", "0.75"), ("fraction", "3/4")], self.fmt.mode,
                                  self._set_mode, height=28)
        self.seg_mode.pack(side="left", padx=(0, px(12)))
        Tooltip(self.seg_mode, self, "Show numbers as decimals or exact fractions  (Ctrl+D)")
        tk.Label(ctrl, text="digits", font=self.fonts["small"], bg=t["surface"], fg=t["faint"]).pack(side="left")
        RoundButton(ctrl, self, "−", lambda: self._set_digits(-1), variant="chip", height=26, width=26, padx=0,
                    radius=7, takefocus=False, tip="Fewer decimal places").pack(side="left", padx=(px(6), 0))
        self.lbl_digits = tk.Label(ctrl, text=str(self.fmt.digits), width=2, font=self.fonts["body_bold"],
                                   bg=t["surface"], fg=t["text"])
        self.lbl_digits.pack(side="left")
        RoundButton(ctrl, self, "+", lambda: self._set_digits(1), variant="chip", height=26, width=26, padx=0,
                    radius=7, takefocus=False, tip="More decimal places").pack(side="left")

        self.view = ResultView(b, self)
        self.view.grid(row=1, column=0, sticky="nsew", pady=(px(8), px(8)))

        foot = tk.Frame(b, bg=t["surface"])
        foot.grid(row=2, column=0, sticky="ew")
        self.btn_use_a = RoundButton(foot, self, "Use as A", lambda: self.use_result("A"), variant="primary",
                                     height=32, tip="Load this result into matrix A to keep calculating")
        self.btn_use_a.pack(side="left")
        self.btn_use_b = RoundButton(foot, self, "Use as B", lambda: self.use_result("B"), height=32,
                                     tip="Load this result into matrix B")
        self.btn_use_b.pack(side="left", padx=(px(8), 0))
        self.btn_latex = RoundButton(foot, self, "LaTeX", lambda: self.copy_result(latex=True), height=32,
                                     tip="Copy the result as LaTeX (bmatrix)")
        self.btn_latex.pack(side="right")
        self.btn_copy = RoundButton(foot, self, "Copy", self.copy_result, height=32,
                                    tip="Copy the result as plain text (pastes into Excel)")
        self.btn_copy.pack(side="right", padx=(0, px(8)))
        self._update_result_buttons()

    def _build_bottom(self, right):
        t, px = self.t, self.px
        card = RoundCard(right, self, pad=16, expand=True)
        card.grid(row=1, column=0, sticky="nsew", pady=(px(16), 0))
        b = card.body
        b.columnconfigure(0, weight=1)
        b.rowconfigure(1, weight=1)

        head = tk.Frame(b, bg=t["surface"])
        head.grid(row=0, column=0, sticky="ew")
        self.seg_tab = Segmented(head, self, [("working", "Step-by-step"), ("history", "History")], self.tab,
                                 self._set_tab, height=30)
        self.seg_tab.pack(side="left")
        self.btn_clear_hist = RoundButton(head, self, "Clear history", self.clear_history, variant="chip",
                                          font=self.fonts["small"], height=26, padx=10, radius=7, takefocus=False)
        self.lbl_engine = tk.Label(head, text=f"computed with {self.engine.name}", font=self.fonts["small"],
                                   bg=t["surface"], fg=t["faint"])

        stack = tk.Frame(b, bg=t["surface"])
        stack.grid(row=1, column=0, sticky="nsew", pady=(px(10), 0))
        stack.rowconfigure(0, weight=1)
        stack.columnconfigure(0, weight=1)

        self.work_frame = tk.Frame(stack, bg=t["surface"])
        self.work_frame.rowconfigure(0, weight=1)
        self.work_frame.columnconfigure(0, weight=1)
        self.work = tk.Text(self.work_frame, wrap="word", bg=t["surface"], fg=t["text"], bd=0, highlightthickness=0,
                            font=self.fonts["body"], padx=px(4), pady=px(2), width=40, height=6, cursor="arrow",
                            insertwidth=0, selectbackground=t["select"], selectforeground=t["text"],
                            spacing1=px(1), spacing3=px(1))
        sb = ttk.Scrollbar(self.work_frame, orient="vertical", command=self.work.yview,
                           style="Studio.Vertical.TScrollbar")
        self.work.configure(yscrollcommand=sb.set)
        self.work.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.work.bind("<Shift-MouseWheel>", lambda e: (self.work.xview_scroll(-1 if e.delta > 0 else 1, "units"),
                                                       "break")[1])
        mono, code_bg = self.fonts["mono"], t["code_bg"]
        self.work.tag_configure("h", font=self.fonts["steps_h"], foreground=t["text"], spacing1=px(10), spacing3=px(3))
        self.work.tag_configure("p", foreground=t["muted"], lmargin1=px(2), lmargin2=px(2))
        self.work.tag_configure("m", font=mono, foreground=t["text"], background=code_bg, lmargin1=px(10),
                                lmargin2=px(10), wrap="none")
        self.work.tag_configure("code", font=mono, foreground=t["accent_text"], background=code_bg,
                                lmargin1=px(10), lmargin2=px(10), wrap="none")
        self.work.tag_configure("ok", foreground=t["success"], font=self.fonts["body_bold"], spacing1=px(4))
        self.work.tag_configure("warn", foreground=t["warn"], font=self.fonts["body_bold"], spacing1=px(4))
        self.work.tag_configure("gap", font=self.font("ui", 4))
        self.work.tag_configure("empty", foreground=t["faint"], justify="center", spacing1=px(30))

        self.hist_frame = tk.Frame(stack, bg=t["surface"])
        self.hist_frame.rowconfigure(0, weight=1)
        self.hist_frame.columnconfigure(0, weight=1)
        self.hist = tk.Listbox(self.hist_frame, bg=t["surface"], fg=t["text"], bd=0, highlightthickness=0,
                               selectbackground=t["accent_soft2"], selectforeground=t["text"], activestyle="none",
                               font=self.fonts["mono"], height=6, exportselection=False)
        sb2 = ttk.Scrollbar(self.hist_frame, orient="vertical", command=self.hist.yview,
                            style="Studio.Vertical.TScrollbar")
        self.hist.configure(yscrollcommand=sb2.set)
        self.hist.grid(row=0, column=0, sticky="nsew")
        sb2.grid(row=0, column=1, sticky="ns")
        self.hist.bind("<<ListboxSelect>>", self._on_history_select)
        self._show_tab()

    def _build_status(self):
        t, px = self.t, self.px
        f = tk.Frame(self.root, bg=t["bg"])
        d = px(8)
        dot = tk.Canvas(f, width=d, height=d, bg=t["bg"], highlightthickness=0)
        dot.create_oval(0, 0, d, d, fill=t["success"], outline=t["success"])
        dot.pack(side="left", padx=(0, px(8)))
        self.status_lbl = tk.Label(f, text=f"Ready  ·  {self.engine.badge}", font=self.fonts["small"],
                                   bg=t["bg"], fg=t["muted"])
        self.status_lbl.pack(side="left")
        tk.Label(f, text="Ctrl+Enter repeat  ·  Ctrl+D fractions  ·  Ctrl+T theme  ·  F1 help",
                 font=self.fonts["small"], bg=t["bg"], fg=t["faint"]).pack(side="right")
        return f

    # ------------------------------------------------------------------ reading / writing matrices
    def read_matrix(self, name, quiet=False):
        M = []
        for i in range(N):
            row = []
            for j in range(N):
                e = self.entries[name][i][j]
                try:
                    row.append(parse_number(e.get()))
                except ValueError as err:
                    if not quiet:
                        self._flash_error_cell(e)
                        raise ValueError(f"Matrix {name}, row {i + 1}, column {j + 1}: {err}.") from None
                    raise
            M.append(row)
        return M

    def _flash_error_cell(self, e):
        t = self.t
        e.focus_set()
        e.configure(bg=t["danger_soft"])
        self.root.after(600, lambda: _safe(lambda: e.configure(bg=t["cell"])))

    def set_texts(self, name, texts):
        for i in range(N):
            for j in range(N):
                e = self.entries[name][i][j]
                e.delete(0, "end")
                e.insert(0, texts[i * N + j])
        self._validate_all(name)
        self._flash_cells(name)

    def set_matrix(self, name, M):
        self.set_texts(name, [self.fmt(M[i][j], pretty=False) for i in range(N) for j in range(N)])

    def texts(self, name):
        return [self.entries[name][i][j].get() for i in range(N) for j in range(N)]

    def _flash_cells(self, name):
        t = self.t
        color = t["accent"] if name == "A" else t["accent2"]
        start = mix(t["cell"], color, 0.35)
        cells = [e for row in self.entries[name] for e in row]

        def step(k, frames=10):
            u = k / frames
            for e in cells:
                _safe(lambda e=e: e.configure(bg=mix(start, t["cell"], u)))
            if k < frames:
                self.root.after(30, step, k + 1)

        step(0)

    # ---------- toolbar actions ----------
    def randomize(self, name):
        self.set_texts(name, [str(random.randint(-9, 9)) for _ in range(N * N)])
        self.status(f"Matrix {name} filled with random integers")

    def load(self, name, kind):
        vals = ["1" if (kind == "identity" and k % (N + 1) == 0) else "0" for k in range(N * N)]
        self.set_texts(name, vals)
        self.status(f"Matrix {name} set to {'I' if kind == 'identity' else 'zero'}")

    def swap(self):
        a, b = self.texts("A"), self.texts("B")
        self.set_texts("A", b)
        self.set_texts("B", a)
        self.status("Swapped A and B")

    def copy_matrix(self, src, dst):
        self.set_texts(dst, self.texts(src))
        self.status(f"Copied {src} into {dst}")

    def clear_all(self):
        self.load("A", "zero")
        self.load("B", "zero")
        self.current = None
        self.res_title.configure(text="Result", fg=self.t["text"])
        self.res_sub.configure(text="Nothing calculated yet")
        self.view.show(None)
        self._render_steps(None)
        self._update_result_buttons()
        self.status("Cleared both matrices")

    def paste(self, name):
        import re
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            self.toast("The clipboard is empty", "error")
            return
        tokens = [tok for tok in re.split(r"[\s,;]+", re.sub(r"[\[\](){}|]", " ", text)) if tok]
        if len(tokens) != N * N:
            self.toast(f"Need exactly 9 numbers in the clipboard — found {len(tokens)}", "error")
            return
        bad = [tok for tok in tokens if not _is_number(tok)]
        if bad:
            self.toast(f"“{bad[0][:20]}” isn't a number", "error")
            return
        self.set_texts(name, tokens)
        self.toast(f"Pasted into matrix {name}", "ok")

    def copy_input(self, name):
        rows = self.texts(name)
        self._to_clipboard("\n".join("\t".join(rows[i * N:(i + 1) * N]) for i in range(N)))
        self.toast(f"Matrix {name} copied", "ok")

    # ------------------------------------------------------------------ running operations
    def run(self, key):
        spec = OP_BY_KEY[key]
        self.last_op = key
        self._refresh_op_labels()
        name = self.target
        fmt_title = lambda tpl, ctx: tpl.format(  # noqa: E731
            M=name, k=ctx.get("k_text", "k"), n=sup_text(ctx.get("k_text", "n")))
        ctx: Dict[str, Any] = {"name": name}
        when = time.strftime("%H:%M:%S")
        try:
            if spec["group"] == "AB":
                ctx["A"], ctx["B"] = self.read_matrix("A"), self.read_matrix("B")
            else:
                ctx["M"] = self.read_matrix(name)
                if spec["group"] == "Mk":
                    try:
                        ctx["k"] = parse_number(self.k_entry.get())
                    except ValueError as err:
                        self._flash_error_cell(self.k_entry)
                        raise ValueError(f"The k / n field: {err}.") from None
                    ctx["k_text"] = self.fmt(ctx["k"])
            t0 = time.perf_counter()
            value = self._call_engine(key, ctx)
            elapsed = (time.perf_counter() - t0) * 1000
            _check_finite(value)
            title = fmt_title(spec["title"], ctx)
            res = Result(op=key, title=title, label=fmt_title(spec.get("label_eq", spec["title"]), ctx),
                         kind=spec["kind"], value=value, ctx=ctx, elapsed=elapsed, when=when)
            res.caption, res.tone = self._caption(res)
            self.show_result(res, record=True)
        except (ValueError, ZeroDivisionError, OverflowError, ArithmeticError) as err:
            title = fmt_title(spec["title"], ctx) if "{k}" not in spec["title"] or "k_text" in ctx else spec["label"]
            res = Result(op=key, title=f"Can't compute {title}", label="", kind="error", value=str(err),
                         ctx=ctx, when=when)
            self.show_result(res, record=False)

    def _call_engine(self, key, ctx):
        e = self.engine
        if key == "mul_ba":
            return e.mul(ctx["B"], ctx["A"])
        group = OP_BY_KEY[key]["group"]
        if group == "AB":
            return getattr(e, key)(ctx["A"], ctx["B"])
        if group == "Mk":
            return getattr(e, key)(ctx["M"], ctx["k"])
        return getattr(e, key)(ctx["M"])

    def _caption(self, r: Result):
        v, op = r.value, r.op
        if op == "det":
            return ("Invertible", "ok") if abs(v) > 1e-12 * max(1, max(abs(x) for row in r.ctx["M"] for x in row)) ** 3 \
                else ("Singular — no inverse", "bad")
        if op == "rank":
            return ("Full rank", "ok") if v == N else (f"Rank-deficient · nullity {N - v}", "bad")
        if op == "trace":
            return "Sum of the diagonal = sum of the eigenvalues", "neutral"
        if op == "norm":
            return "Frobenius norm  √(Σ aᵢⱼ²)", "neutral"
        if op == "eigen":
            if any(isinstance(g["value"], complex) for g in v):
                return "Complex eigenvalues — the matrix rotates space", "info"
            if any(len(g["vectors"]) < g["mult"] for g in v):
                return "Defective — not diagonalisable", "bad"
            return "Vectors scaled so the largest entry is 1", "neutral"
        if op == "charpoly":
            return "Its roots are the eigenvalues", "neutral"
        if op == "commutator":
            zero = all(abs(x) < 1e-9 for row in v for x in row)
            return ("Zero — A and B commute", "ok") if zero else ("Non-zero — A and B don't commute", "info")
        if op == "solve":
            return "Unique solution of A·X = B", "ok"
        if op == "inv":
            return "Check:  M × M⁻¹ = I", "ok"
        return "", "neutral"

    def show_result(self, res: Result, record=True, animate=True):
        t = self.t
        self.current = res
        if res.kind == "error":
            self.res_title.configure(text=res.title, fg=t["danger"])
            self.res_sub.configure(text="Nothing was changed — fix the input and try again")
            self.status("⚠  " + res.value)
        else:
            self.res_title.configure(text=res.title, fg=t["text"])
            kind = {"matrix": "3 × 3 matrix", "scalar": "number", "eigen": "eigen-decomposition",
                    "report": "property report", "poly": "polynomial"}[res.kind]
            self.res_sub.configure(text=f"{kind}  ·  {self.engine.name}  ·  {res.elapsed:.2f} ms  ·  {res.when}")
            self.status(f"{res.title}  ✓  computed in {res.elapsed:.2f} ms with {self.engine.name}")
        self.view.show(res, animate=animate)
        self._render_steps(res)
        self._update_result_buttons()
        if record:
            self.history.insert(0, res)
            del self.history[150:]
            self._refresh_history()
            self._save_settings()

    def _update_result_buttons(self):
        cur = self.current
        is_matrix = cur is not None and cur.kind == "matrix"
        has = cur is not None and cur.kind != "error"
        for b in (self.btn_use_a, self.btn_use_b):
            b.set_enabled(is_matrix)
        for b in (self.btn_copy, self.btn_latex):
            b.set_enabled(has)

    def _render_steps(self, res: Optional[Result]):
        w = self.work
        w.configure(state="normal")
        w.delete("1.0", "end")
        if res is None:
            w.insert("end", "Step-by-step working appears here after each calculation.\n", "empty")
        elif res.kind == "error":
            w.insert("end", "What went wrong\n", "h")
            w.insert("end", res.value + "\n", "p")
            w.insert("end", "Tip: singular matrices (det = 0) have no inverse. Try Rank or RREF to see why.\n", "p")
        else:
            try:
                steps = self.engine.explain(res.op, res.ctx, res.value, self.fmt)
            except Exception as exc:  # an explanation must never break the result
                steps = [("p", f"(No step-by-step explanation available: {exc})")]
            for kind, text in steps:
                if kind in ("m", "code"):
                    # pad every line to the same width so the shaded block is a neat rectangle
                    lines = text.split("\n")
                    width = max(len(s) for s in lines)
                    blank = " " * (width + 4)
                    text = "\n".join([blank] + ["  " + s.ljust(width) + "  " for s in lines] + [blank])
                    w.insert("end", text + "\n", kind)
                    w.insert("end", "\n", "gap")
                else:
                    w.insert("end", text + "\n", kind)
        w.configure(state="disabled")
        w.yview_moveto(0)

    # ---------- result actions ----------
    def use_result(self, name):
        if self.current is None or self.current.kind != "matrix":
            return
        self.set_matrix(name, self.current.value)
        self.toast(f"Result loaded into matrix {name}", "ok")

    def copy_result(self, latex=False):
        cur = self.current
        if cur is None or cur.kind == "error":
            return
        self._to_clipboard(self._result_text(cur, latex))
        self.toast("LaTeX copied" if latex else "Result copied", "ok")

    def _result_text(self, r: Result, latex: bool) -> str:
        f = lambda x: self.fmt(x, pretty=False)  # noqa: E731
        lx = _latex_num
        if r.kind == "matrix":
            if latex:
                rows = [" & ".join(lx(f(x)) for x in row) for row in r.value]
                return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"
            return "\n".join("\t".join(f(x) for x in row) for row in r.value)
        if r.kind == "scalar":
            return lx(f(r.value)) if latex else f(r.value)
        if r.kind == "poly":
            p = poly_text(r.value, f)
            if latex:
                p = p.replace("λ³", "\\lambda^3").replace("λ²", "\\lambda^2").replace("λ", "\\lambda").replace("−", "-")
                return "p(\\lambda) = " + p
            return "p(λ) = " + p.replace("−", "-")
        if r.kind == "eigen":
            lines = []
            for k, g in enumerate(r.value, 1):
                vecs = "; ".join("(" + ", ".join(f(x) for x in v) + ")" for v in g["vectors"])
                if latex:
                    lines.append(f"\\lambda_{k} = {lx(f(g['value']))}")
                else:
                    lines.append(f"λ{k} = {f(g['value'])}   (×{g['mult']})   v = {vecs}")
            return "\n".join(lines)
        if r.kind == "report":
            P = r.value
            lines = [f"det = {f(P['det'])}", f"trace = {f(P['trace'])}", f"rank = {P['rank']}",
                     f"Frobenius norm = {f(P['norm'])}"]
            lines += [f"{label}: {'yes' if P[key] else 'no'}" for key, label, _ in PROPERTY_FLAGS]
            return "\n".join(lines)
        return str(r.value)

    # ---------- display settings ----------
    def _set_mode(self, mode):
        self.fmt.mode = mode
        self._rerender()
        self.status("Showing exact fractions where possible" if mode == "fraction" else "Showing decimals")

    def toggle_mode(self):
        new = "fraction" if self.fmt.mode == "decimal" else "decimal"
        self.seg_mode.set(new)
        self._set_mode(new)

    def _set_digits(self, delta):
        self.fmt.digits = max(0, min(10, self.fmt.digits + delta))
        self.lbl_digits.configure(text=str(self.fmt.digits))
        self._rerender()

    def _rerender(self):
        if self.current is not None:
            self.view.redraw(False)
            self._render_steps(self.current)
        self._refresh_history()
        for name in ("A", "B"):
            self._schedule_badges(name)
        self._save_settings()

    # ---------- tabs / history ----------
    def _set_tab(self, tab):
        self.tab = tab
        self._show_tab()

    def _show_tab(self):
        px = self.px
        if self.tab == "working":
            self.hist_frame.grid_remove()
            self.work_frame.grid(row=0, column=0, sticky="nsew")
            self.btn_clear_hist.pack_forget()
            self.lbl_engine.pack(side="right")
        else:
            self.work_frame.grid_remove()
            self.hist_frame.grid(row=0, column=0, sticky="nsew")
            self.lbl_engine.pack_forget()
            self.btn_clear_hist.pack(side="right", pady=px(2))

    def _summary(self, r: Result) -> str:
        f = self.fmt
        if r.kind == "scalar":
            return f(r.value)
        if r.kind == "eigen":
            return "λ = " + ", ".join(f(g["value"]) for g in r.value)
        if r.kind == "poly":
            return poly_text(r.value, f)
        if r.kind == "report":
            return "invertible" if r.value["invertible"] else "singular"
        first = "  ".join(f(x) for x in r.value[0])
        return f"[{first} …]"

    def _refresh_history(self):
        h = self.hist
        h.delete(0, "end")
        if not self.history:
            h.insert("end", "  No calculations yet")
            h.itemconfigure(0, foreground=self.t["faint"])
            return
        for r in self.history:
            title = r.title if len(r.title) <= 24 else r.title[:23] + "…"
            h.insert("end", f" {r.when}   {title:<24}  {self._summary(r)}")

    def _on_history_select(self, _=None):
        sel = self.hist.curselection()
        if not sel or not self.history or sel[0] >= len(self.history):
            return
        self.show_result(self.history[sel[0]], record=False)
        self.status(f"Showing history item from {self.history[sel[0]].when}")

    def clear_history(self):
        self.history.clear()
        self._refresh_history()
        self.toast("History cleared", "ok")

    # ------------------------------------------------------------------ theme
    def toggle_theme(self):
        self._capture_state()
        self.theme_name = "light" if self.theme_name == "dark" else "dark"
        self.t = make_theme(self.theme_name)
        self._close_help()
        for w in self.root.winfo_children():
            w.destroy()
        self._toast_w = None
        self._badge_jobs.clear()
        self._build()
        self._set_icon()
        self._save_settings()
        self.status(f"{self.theme_name.title()} theme")

    def _capture_state(self):
        try:
            self.cells_text = {n: self.texts(n) for n in ("A", "B")}
            self.k_text = self.k_entry.get()
        except (tk.TclError, AttributeError):
            pass

    # ------------------------------------------------------------------ feedback
    def status(self, text):
        _safe(lambda: self.status_lbl.configure(text=text))

    def toast(self, msg, kind="info"):
        t, px = self.t, self.px
        if self._toast_w is not None:
            _safe(self._toast_w.destroy)
        if self._toast_job:
            _safe(lambda: self.root.after_cancel(self._toast_job))
        icon = {"ok": "✓   ", "error": "⚠   "}.get(kind, "")
        lbl = tk.Label(self.root, text=icon + msg, bg=t["toast_bg"], fg=t["toast_fg"],
                       font=self.fonts["body_bold"], padx=px(18), pady=px(9))
        lbl.place(relx=0.5, rely=1.0, y=-px(48), anchor="s")
        lbl.lift()
        self._toast_w = lbl
        self._toast_job = self.root.after(1900, lambda: _safe(lbl.destroy))
        if kind == "error":
            self.root.bell()

    def _to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    # ------------------------------------------------------------------ help
    def show_help(self):
        if self._help is not None and _safe(self._help.winfo_exists):
            self._help.lift()
            return
        t, px = self.t, self.px
        win = self._help = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} — help")
        win.configure(bg=t["bg"])
        win.transient(self.root)
        win.resizable(False, False)
        card = RoundCard(win, self, pad=24)
        card.pack(padx=px(18), pady=px(18))
        b = card.body
        row = 0

        def heading(text):
            nonlocal row
            tk.Label(b, text=text, font=self.fonts["h2"], bg=t["surface"], fg=t["text"]).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(px(12) if row else 0, px(6)))
            row += 1

        def line(key, desc):
            nonlocal row
            tk.Label(b, text=f" {key} ", font=self.fonts["mono"], bg=t["surface2"], fg=t["accent_text"]).grid(
                row=row, column=0, sticky="w", pady=px(2), padx=(0, px(14)))
            tk.Label(b, text=desc, font=self.fonts["body"], bg=t["surface"], fg=t["muted"]).grid(
                row=row, column=1, sticky="w")
            row += 1

        heading("Keyboard")
        for k, d in (("Ctrl+Enter / F5", "Repeat the last operation"), ("Enter", "Next cell (A → B → A)"),
                     ("Arrow keys", "Move between cells"), ("Mouse wheel", "Nudge a focused cell by ±1 (Shift: ±0.1)"),
                     ("Ctrl+R", "Random A   (Ctrl+Shift+R: random B)"), ("Ctrl+D", "Toggle decimals / fractions"),
                     ("Ctrl+T", "Toggle dark / light theme"), ("Ctrl+L", "Clear both matrices"),
                     ("Ctrl+1 / Ctrl+2", "Jump to matrix A / B"), ("F1", "This help")):
            line(k, d)
        heading("What you can type in a cell")
        for k, d in (("3   -2.5   1e-3", "Plain numbers"), ("1/3   3/4", "Fractions"),
                     ("2^3   (1+2)*4", "Arithmetic"), ("sqrt(2)   √2", "Square roots"),
                     ("pi/4   sin(pi/6)   e", "Constants & functions: sin cos tan log ln exp abs")):
            line(k, d)
        heading("Tips")
        for k, d in (("Use as A / B", "Chain calculations by feeding a result back in"),
                     ("Hover a result", "The status bar shows the full-precision value"),
                     ("Paste", "Copy a 3×3 block from Excel and paste it straight into a matrix"),
                     ("History", "Click any past calculation to see it — and its working — again")):
            line(k, d)
        tk.Label(b, text=f"{APP_NAME} {APP_VERSION}  ·  {self.engine.badge}", font=self.fonts["small"],
                 bg=t["surface"], fg=t["faint"]).grid(row=row, column=0, sticky="w", pady=(px(16), 0))
        RoundButton(b, self, "Got it", self._close_help, variant="primary", height=32).grid(
            row=row, column=1, sticky="e", pady=(px(16), 0))
        win.bind("<Escape>", lambda e: self._close_help())
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_reqheight()) // 3
        win.geometry(f"+{max(0, x)}+{max(0, y)}")
        win.focus_set()

    def _close_help(self):
        if self._help is not None:
            _safe(self._help.destroy)
            self._help = None

    # ------------------------------------------------------------------ shortcuts
    def _shortcut_table(self):
        repeat = lambda: self.run(self.last_op) if self.last_op else self.toast("Pick an operation first")  # noqa
        return {
            "<Control-Return>": repeat, "<F5>": repeat,
            "<Control-t>": self.toggle_theme, "<Control-T>": self.toggle_theme,
            "<Control-d>": self.toggle_mode, "<Control-D>": self.toggle_mode,
            "<Control-r>": lambda: self.randomize("A"), "<Control-R>": lambda: self.randomize("B"),
            "<Control-l>": self.clear_all, "<Control-L>": self.clear_all,
            "<Control-Key-1>": lambda: self.entries["A"][0][0].focus_set(),
            "<Control-Key-2>": lambda: self.entries["B"][0][0].focus_set(),
            "<F1>": self.show_help,
        }

    def _bind_shortcuts(self):
        for seq, fn in self._shortcut_table().items():
            self.root.bind_all(seq, lambda e, fn=fn: (fn(), "break")[1])
        self.root.bind_all("<Escape>", lambda e: self._close_help())

    def _entry_shortcuts(self, e):
        # Entry widgets have their own Ctrl+T / Ctrl+D bindings (transpose / delete char);
        # binding on the widget itself and returning "break" keeps those from also firing.
        for seq, fn in self._shortcut_table().items():
            e.bind(seq, lambda ev, fn=fn: (fn(), "break")[1])

    # ------------------------------------------------------------------ persistence
    def _save_settings(self):
        self._capture_state()
        data = {"theme": self.theme_name, "A": self.cells_text["A"], "B": self.cells_text["B"],
                "target": self.target, "k": self.k_text, "mode": self.fmt.mode, "digits": self.fmt.digits}
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _on_close(self):
        self._save_settings()
        self.root.destroy()


# ==========================================================================
#  Small utilities
# ==========================================================================
def _safe(fn):
    try:
        return fn()
    except tk.TclError:
        return None


def _select_all(w):
    try:
        w.select_range(0, "end")
        w.icursor("end")
    except tk.TclError:
        pass


def _is_number(text):
    try:
        parse_number(text)
        return True
    except ValueError:
        return False


def _nine(value):
    if isinstance(value, list) and len(value) == N * N and all(isinstance(v, str) for v in value):
        return [v[:24] for v in value]
    return None


def _latex_num(s: str) -> str:
    if "/" in s and " " not in s:
        num, den = s.split("/", 1)
        sign = "-" if num.startswith("-") else ""
        return f"{sign}\\frac{{{num.lstrip('-')}}}{{{den}}}"
    return s


def _check_finite(value):
    import math

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, (int, float, complex)) and not isinstance(v, bool):
            z = complex(v)
            if not (math.isfinite(z.real) and math.isfinite(z.imag)):
                raise OverflowError("The result is too large to represent (it overflowed). Try a smaller power or smaller numbers.")

    walk(value)


def _load_settings() -> Dict[str, Any]:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _enable_windows_dpi():
    """Crisp text on high-DPI Windows displays."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def run_app(engine):
    App(engine).mainloop()
