"""
Paired slope plot: multimodal versus image-only across the 20 splits.

One panel per primary outcome. Each line joins the two models evaluated on the
same split, so the within-split comparison is visible directly. Lines are
coloured by direction (multimodal higher or lower) and the number favouring
each model is annotated.

Add to paper_figures.py, or import from it:

    from src.paper_figures import MODEL_COLOURS, PRIMARY, load_results, _save
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# reuse the visual system from paper_figures.py
from paper_figures import (DOUBLE_COL, MODEL_COLOURS, PRIMARY,
                           _save, load_results)

def _panel_label(ax, letter):
    ax.text(0.5, 0.95, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')

UP   = '#009E73'   # multimodal higher  (same green as the multimodal model)
DOWN = '#D55E00'   # multimodal lower
TIE  = '#999999'


def paired_slope(run_a, run_b, out, label_a='Multimodal', label_b='Image-only',
                 metrics=PRIMARY):
    """
    run_a, run_b: run directories. Lines run from b (left) to a (right), so
    an upward line means the first-named model performed better on that split.
    """
    A, B = load_results(run_a), load_results(run_b)
    seeds = sorted(set(A.index) & set(B.index))
    assert len(seeds) == len(A) == len(B), "runs do not share an identical seed set"

    fig, axes = plt.subplots(1, len(metrics), figsize=(DOUBLE_COL, 2.6))

    for k, (ax, (col, name)) in enumerate(zip(axes, metrics)):
        b = B.loc[seeds, col].values
        a = A.loc[seeds, col].values

        n_up   = int((a > b).sum())
        n_down = int((a < b).sum())
        n_tie  = int((a == b).sum())

        for bi, ai in zip(b, a):
            c = UP if ai > bi else (DOWN if ai < bi else TIE)
            ax.plot([0, 1], [bi, ai], color=c, lw=0.7, alpha=0.65, zorder=1)

        ax.scatter(np.zeros(len(b)), b, s=11, color=MODEL_COLOURS[label_b],
                   edgecolor='none', zorder=3)
        ax.scatter(np.ones(len(a)), a, s=11, color=MODEL_COLOURS[label_a],
                   edgecolor='none', zorder=3)

        # group means, drawn as short horizontal markers
        for x, v, lab in ((0, b, label_b), (1, a, label_a)):
            ax.plot([x - 0.10, x + 0.10], [v.mean()] * 2,
                    color=MODEL_COLOURS[lab], lw=1.8, solid_capstyle='butt',
                    zorder=4)

        tie_txt = f", {n_tie} tied" if n_tie else ""
        ax.set_title(f"{n_up}/{len(seeds)} splits favour {label_a.lower()}{tie_txt}",
                     fontsize=7, color='0.3', pad=6)

        ax.set_xlim(-0.28, 1.28)
        ax.set_xticks([0, 1])
        #ax.set_xticklabels([label_b.replace('-', '-\n'),
        #                    label_a.replace('-', '-\n')])
        
        ax.set_xticklabels([label_b, label_a])
        ax.set_ylabel(name)
        _panel_label(ax, 'ABC'[k])

    fig.tight_layout(w_pad=2.0)
    _save(fig, out)


if __name__ == '__main__':
    from src.config import RUNS_DIR
    paired_slope(RUNS_DIR / 'multi_dens', RUNS_DIR / 'image_only_dens',
                 RUNS_DIR / 'figures' / 'fig_paired_slope')
