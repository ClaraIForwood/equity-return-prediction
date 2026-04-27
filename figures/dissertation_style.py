"""
Shared matplotlib style for dissertation figures.

Usage (one line):
    from figures.dissertation_style import apply_style, FIG_W, FIG_H, SAVEFIG_KW, SVG_KW, VLINE_KW
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H['timeline']))
    fig.savefig('out.png', **SAVEFIG_KW)   # raster
    fig.savefig('out.svg', **SVG_KW)       # vector
"""

import matplotlib as mpl

# ── Physical dimensions ────────────────────────────────────────────────────────
FIG_W = 17 / 2.54          # 17 cm → inches  (full A4 text column, 2.5 cm margins)
FIG_DPI = 150

FIG_H = {
    'timeline': 4  / 2.54, # 4 cm
    'macd':     10 / 2.54, # 10 cm
}

# ── Colours ────────────────────────────────────────────────────────────────────
TEXT_COLOR = '#222222'
DASH_COLOR = '#444444'
BG_COLOR   = '#ffffff'

# ── Reusable kwargs ────────────────────────────────────────────────────────────
VLINE_KW   = dict(color=DASH_COLOR, lw=0.8, ls='--')
SAVEFIG_KW = dict(dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.1)
SVG_KW     = dict(bbox_inches='tight', pad_inches=0.1)

# ── rcParams block ─────────────────────────────────────────────────────────────
_RC = {
    'font.family':        'DejaVu Sans',
    'font.size':          9,
    'text.color':         TEXT_COLOR,

    'axes.titlesize':     10,
    'axes.labelsize':     10,
    'axes.labelcolor':    TEXT_COLOR,
    'axes.edgecolor':     TEXT_COLOR,
    'axes.facecolor':     BG_COLOR,
    'axes.spines.top':    False,
    'axes.spines.right':  False,

    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'xtick.color':        TEXT_COLOR,
    'ytick.color':        TEXT_COLOR,

    'legend.fontsize':    9,
    'legend.frameon':     False,

    'figure.facecolor':   BG_COLOR,
    'figure.dpi':         FIG_DPI,

    'grid.color':         '#cccccc',
    'grid.linewidth':     0.5,
    'lines.linewidth':    1.2,

    'svg.fonttype':       'none',   # keep text as text in SVG (not paths)
}


def apply_style():
    """Apply dissertation rcParams globally. Call once per script, before any figure."""
    mpl.rcParams.update(_RC)
