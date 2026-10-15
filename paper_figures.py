"""
Publication figures for the BRAR revision.

One visual system for every figure: Okabe-Ito colour-blind-safe palette with a
fixed colour per model, no titles, light axes, vector PDF output. Every figure
is generated from saved run files -- no retraining.

    from src.paper_figures import make_all_figures
    make_all_figures()

Or call the individual functions.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

# ── visual system ─────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          8,
    'axes.titlesize':     8,
    'axes.labelsize':     8,
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          False,
    'savefig.bbox':       'tight',
    'savefig.dpi':        600,
    'pdf.fonttype':       42,      # editable text in the PDF
})

SINGLE_COL = 3.5    # inches -- BMC single column
DOUBLE_COL = 7.0    # inches -- full width

# fixed semantic colour per model, used in every figure
MODEL_COLOURS = {
    'Image-only':      '#0072B2',
    'Metadata-only':   '#E69F00',
    'Multimodal':      '#009E73',
    'Image + age':     '#56B4E9',
    'Image + non-age': '#D55E00',
    'Late fusion':     '#999999',
    'Ensemble':        '#CC79A7',
}
LEVEL_COLOURS = {1: '#0072B2', 2: '#E69F00', 3: '#D55E00'}
LEVEL_NAMES   = {1: 'Level 1', 2: 'Level 2', 3: 'Level 3'}

PRIMARY = [('macro_f1', 'Macro F1'),
           ('macro_auc', 'Macro AUC-ROC'),
           ('l3_recall', 'Level 3 recall')]


def _panel_label(ax, letter):
    ax.text( 0.2, 1.04, letter, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom', ha='left')


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'))
    plt.close(fig)
    print(f"Saved → {out.with_suffix('.pdf')}")


def load_results(run_dir):
    return pd.read_csv(Path(run_dir) / 'results.csv').set_index('seed').sort_index()


# ── Figure 2: paired distributions across the 20 splits ───────────────────────
def paired_distributions(runs, out, metrics=PRIMARY):
    """
    One panel per metric; one x position per model; each grey line joins the
    same split across models, so the paired structure is visible.

    runs: ordered dict {display name: run_dir}. Names must be MODEL_COLOURS keys.
    """
    data  = {name: load_results(d) for name, d in runs.items()}
    names = list(runs)
    seeds = sorted(set.intersection(*(set(v.index) for v in data.values())))
    assert all(len(v) == len(seeds) for v in data.values()), \
        "runs do not share an identical seed set"

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(DOUBLE_COL, 2.4), sharey=False)
    x = np.arange(len(names))

    for k, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        M = np.array([[data[n].loc[s, col] for n in names] for s in seeds])

        #for row in M:                                    # paired lines
        #    ax.plot(x, row, color='0.75', lw=0.5, zorder=1)

        bp = ax.boxplot(M, positions=x, widths=0.45, showfliers=False,
                        patch_artist=True, zorder=2)
        for patch, n in zip(bp['boxes'], names):
            patch.set_facecolor('none')
            patch.set_edgecolor(MODEL_COLOURS[n])
            patch.set_linewidth(0.7) # was 1.0
        for part in ('whiskers', 'caps'):
            for i, line in enumerate(bp[part]):
                line.set_color(MODEL_COLOURS[names[i // 2]])
                line.set_linewidth(0.6) # was 0.6
        for i, line in enumerate(bp['medians']):
            line.set_color(MODEL_COLOURS[names[i]])
            line.set_linewidth(1.1) #1.4

        rng = np.random.default_rng(0)
        for j, n in enumerate(names):                    # points
            jitter = rng.uniform(-0.06, 0.06, len(seeds))
            ax.scatter(np.full(len(seeds), j) + jitter, M[:, j], s=7,
                       color=MODEL_COLOURS[n], edgecolor='none',
                       alpha=0.85, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0)
        #ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel(label)
        ax.set_xticklabels([n.replace('-', '\n') for n in names])
        _panel_label(ax, 'ABCDEFG'[k])

    fig.tight_layout(w_pad=2.0)
    _save(fig, out)


# ── Figure 3: paired-difference forest plot ───────────────────────────────────
def _paired_ci(a, b, stat='median', n_boot=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    f = np.median if stat == 'median' else np.mean
    rng = np.random.default_rng(seed)
    boots = f(rng.choice(d, size=(n_boot, len(d)), replace=True), axis=1)
    return f(d), np.percentile(boots, 2.5), np.percentile(boots, 97.5), d


def _holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def effect_size_forest(comparisons, runs, out, metrics=PRIMARY,
                       stat='median', save_table=True):
    """
    Forest plot of paired differences (A minus B) with bootstrap 95% CIs.
    Filled markers: Holm-adjusted p < 0.05 across the three primary metrics
    for that comparison (Wilcoxon signed-rank, zero_method='zsplit').

    comparisons: list of (group, label, model_A, model_B)
    runs:        {display name: run_dir}
    """
    data = {n: load_results(d) for n, d in runs.items()}
    rows = []
    for group, label, A, B in comparisons:
        seeds = sorted(set(data[A].index) & set(data[B].index))
        assert len(seeds) == len(data[A]) == len(data[B]), \
            f"{A} vs {B}: seed sets differ"
        pvals, per_metric = [], []
        for col, mname in metrics:
            a = data[A].loc[seeds, col].values
            b = data[B].loc[seeds, col].values
            est, lo, hi, d = _paired_ci(a, b, stat=stat)
            p = 1.0 if np.all(d == 0) else \
                stats.wilcoxon(a, b, zero_method='zsplit').pvalue
            pvals.append(p)
            per_metric.append((col, mname, est, lo, hi, p))
        for (col, mname, est, lo, hi, p), padj in zip(per_metric, _holm(pvals)):
            rows.append(dict(group=group, comparison=label, metric=mname,
                             estimate=est, ci_low=lo, ci_high=hi,
                             p_raw=p, p_holm=padj))
    df = pd.DataFrame(rows)

    labels = list(dict.fromkeys(df.comparison))
    groups = list(dict.fromkeys(df.group))

    # y positions with a gap between groups
    ypos, y, group_spans = {}, 0.0, []
    for g in groups:
        start = y
        for lab in [l for l in labels if df[df.comparison == l].group.iloc[0] == g]:
            ypos[lab] = y
            y += 0.55
        group_spans.append((g, start, y - 1))
        y += 0.8

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(DOUBLE_COL, 0.18 * y + 0.9), sharey=True)
    for k, (ax, (col, mname)) in enumerate(zip(axes, metrics)):
        sub = df[df.metric == mname]
        for _, r in sub.iterrows():
            yy = ypos[r.comparison]
            sig = r.p_holm < 0.05
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color='0.25', lw=0.9)
            ax.plot(r.estimate, yy, 'o', ms=4.2,
                    mfc='0.15' if sig else 'white', mec='0.15', mew=0.9)
        ax.axvline(0, color='0.55', lw=0.7, ls='--', zorder=0)
        ax.set_xlabel(f"Δ {mname}")
        _panel_label(ax, 'ABC'[k])
        ax.invert_yaxis() if k == 0 else None

    axes[0].set_yticks([ypos[l] for l in labels])
    axes[0].set_yticklabels(labels)
    for g, y0, y1 in group_spans:
        axes[0].text(-0.02, y0 - 0.65, g, transform=axes[0].get_yaxis_transform(),
                     ha='right', va='center', fontsize=7, fontweight='bold')
    axes[0].tick_params(axis='y', length=0)

    fig.tight_layout(w_pad=1.6)
    _save(fig, out)

    if save_table:
        tab = df.copy()
        tab['CI'] = tab.apply(lambda r: f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]", axis=1)
        tab.to_csv(Path(out).with_suffix('.csv'), index=False)
        print(f"Saved → {Path(out).with_suffix('.csv')}")
    return df


# ── ordinal confusion matrix (model and expert use the same function) ────────
def ordinal_confusion(ax, y_true, y_pred, row_label='Reference level',
                      col_label='Predicted level', show_ylabel=True):
    """
    Row-normalized percentages with counts. Cells are coloured by ordinal
    error distance: correct / one-grade / two-grade. Intensity = row %.
    Identical scale every time it is called, so matrices are comparable.
    """
    labels = [1, 2, 3]
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    base = {0: np.array([0.0, 0.62, 0.45]),   # correct   (green)
            1: np.array([0.90, 0.62, 0.0]),   # one-grade (amber)
            2: np.array([0.84, 0.37, 0.0])}   # two-grade (vermillion)
    img = np.ones((3, 3, 3))
    for i in range(3):
        for j in range(3):
            w = 0.15 + 0.85 * pct[i, j] / 100
            img[i, j] = 1 - w * (1 - base[abs(i - j)])
    ax.imshow(img, aspect='equal')

    for i in range(3):
        for j in range(3):
            dark = pct[i, j] > 55
            ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm[i, j]})", ha='center',
                    va='center', fontsize=7,
                    color='white' if dark else '0.1')

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"L{l}" for l in labels])
    ax.set_yticklabels([f"L{l}" for l in labels])
    ax.set_xlabel(col_label)
    if show_ylabel:
        ax.set_ylabel(row_label)
        
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return cm


def _ordinal_legend(fig, y=0.0):
    from matplotlib.patches import Patch
    h = [Patch(color=(0.0, 0.62, 0.45), label='Correct'),
         Patch(color=(0.90, 0.62, 0.0), label='One-grade error'),
         Patch(color=(0.84, 0.37, 0.0), label='Two-grade error')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, y))


# ── Figure 4: representative split -- ROC, PR, and confusion matrix ──────────
def representative_split(probs_csv, out, seed_label='seed 2024'):
    p = pd.read_csv(probs_csv)
    y  = p['true_level'].values
    yp = p['predicted_level'].values
    P  = p[['prob_l1', 'prob_l2', 'prob_l3']].values

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.35),
                             gridspec_kw={'width_ratios': [1, 1, 0.95]})

    ax = axes[0]
    for L in (1, 2, 3):
        fpr, tpr, _ = roc_curve(y == L, P[:, L - 1])
        auc = roc_auc_score(y == L, P[:, L - 1])
        ax.plot(fpr, tpr, color=LEVEL_COLOURS[L], lw=1.1,
                label=f"{LEVEL_NAMES[L]} ({auc:.2f})")
    ax.plot([0, 1], [0, 1], color='0.6', lw=0.6, ls='--')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(title='AUC', frameon=False, loc='lower right', title_fontsize=7)
    _panel_label(ax, 'A')

    ax = axes[1]
    for L in (1, 2, 3):
        prec, rec, _ = precision_recall_curve(y == L, P[:, L - 1])
        ap = average_precision_score(y == L, P[:, L - 1])
        ax.plot(rec, prec, color=LEVEL_COLOURS[L], lw=1.1,
                label=f"{LEVEL_NAMES[L]} ({ap:.2f})")
        ax.axhline((y == L).mean(), color=LEVEL_COLOURS[L],
                   lw=0.6, ls=':', alpha=0.8)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(title='AP', frameon=False, loc='lower left', title_fontsize=7)
    _panel_label(ax, 'B')

    ordinal_confusion(axes[2], y, yp)
    _panel_label(axes[2], 'C')

    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(bottom=0.26)
    _ordinal_legend(fig, y=-0.02)
    _save(fig, out)


# ── Figure 6: expert analysis -- matrices plus two aligned panels ────────────
def expert_analysis(comparison_csv, out):
    """
    Top row: three agreement matrices on identical scales.
    Bottom row: (D) dentist vs model accuracy per level, (E) dentist confidence.
    No dual axes.
    """
    c = pd.read_csv(comparison_csv)
    t, m, d = c['true_level'], c['predicted_level'], c['visual_level']
    n = len(c)

    fig = plt.figure(figsize=(DOUBLE_COL, 4.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.05, 1], hspace=0.7, wspace=1.2)

    pairs = [(t, d, 'Reference level', 'Dentist grade'),
             (t, m, 'Reference level', 'Model prediction'),
             (d, m, 'Dentist grade',   'Model prediction')]
    for k, (a, b, rl, cl) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ordinal_confusion(ax, a, b, row_label=rl, col_label=cl)
        _panel_label(ax, 'ABC'[k])

    levels = [1, 2, 3]
    x = np.arange(3)

    ax = fig.add_subplot(gs[1, 0:3])
    acc_d = [(d[t == L] == L).mean() for L in levels]
    acc_m = [(m[t == L] == L).mean() for L in levels]
    w = 0.36
    ax.bar(x - w / 2, acc_d, w, color='0.35', label='Dentist')
    ax.bar(x + w / 2, acc_m, w, color=MODEL_COLOURS['Multimodal'], label='Model')
    for i in range(3):
        ax.text(x[i] - w / 2, acc_d[i] + 0.02, f"{acc_d[i]:.2f}",
                ha='center', fontsize=6.5)
        ax.text(x[i] + w / 2, acc_m[i] + 0.02, f"{acc_m[i]:.2f}",
                ha='center', fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([LEVEL_NAMES[L] for L in levels])
    #ax.set_ylabel('Accuracy vs reference'); 
    ax.set_ylabel('Agreement with reference')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc='upper right', ncol=2)
    _panel_label(ax, 'D')

    ax = fig.add_subplot(gs[1, 3:6])
    conf = [c.loc[t == L, 'visual_confidence'] for L in levels]
    rng = np.random.default_rng(0)
    for i, v in enumerate(conf):
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v,
                   s=5, color='0.6', edgecolor='none', alpha=0.6)
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt='o', ms=4.5,
                    color='0.15', capsize=3, lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([LEVEL_NAMES[L] for L in levels])
    ax.set_ylabel('Dentist confidence (1–10)'); ax.set_ylim(0, 10.5)
    _panel_label(ax, 'E')

    #fig.text(0.5, 1.0,
    #         f"Purposively selected, error-enriched subset (n = {n}); "
    #         "not representative of overall performance",
    #         ha='center', va='bottom', fontsize=7, style='italic', color='0.3')
    
    fig.text(0.5, 0.985,
         f"Purposively selected, error-enriched subset (n = {n}); "
         "not representative of overall performance",
         ha='center', va='top', fontsize=8.5, style='italic', fontweight='bold', 
         color='0.25')
    
    fig.subplots_adjust(bottom=0.1)
    _ordinal_legend(fig, y=0.47)
    _save(fig, out)


# ── wiring to the project ─────────────────────────────────────────────────────
def make_all_figures(fig_dir=None):
    from src.config import RUNS_DIR
    fig_dir = Path(fig_dir) if fig_dir else RUNS_DIR / 'figures'

    R = {
        'Image-only':      RUNS_DIR / 'image_only_dens',
        'Metadata-only':   RUNS_DIR / 'meta_only_dens',
        'Multimodal':      RUNS_DIR / 'multi_dens',
        'Image + age':     RUNS_DIR / 'multi_dens_age_only',
        'Image + non-age': RUNS_DIR / 'multi_dens_no_age',
        'Late fusion':     RUNS_DIR / 'ensemble_late_fusion',
        'Ensemble':        RUNS_DIR / 'ensemble_tuned',
    }

    paired_distributions(
        #{k: R[k] for k in ('Image-only', 'Metadata-only', 'Multimodal')},
        {k: R[k] for k in ('Image-only', 'Metadata-only', 'Multimodal')},
        fig_dir / 'fig2_paired_distributions')


    effect_size_forest(
        comparisons=[
            ('Main comparisons', 'Multimodal − Image-only',      'Multimodal', 'Image-only'),
            ('Main comparisons', 'Multimodal − Metadata-only',   'Multimodal', 'Metadata-only'),
            ('Metadata ablation', 'Image + age − Image-only',    'Image + age', 'Image-only'),
            ('Metadata ablation', 'Image + non-age − Image-only', 'Image + non-age', 'Image-only'),
            ('Metadata ablation', 'Multimodal − Image + age',    'Multimodal', 'Image + age'),
            ('Metadata ablation', 'Multimodal − Image + non-age', 'Multimodal', 'Image + non-age'),
        ],
        runs=R, out=fig_dir / 'fig3_effect_sizes')

    '''
    effect_size_forest(
        comparisons=[
            ('Main comparisons', 'Multimodal − Image-only',      'Multimodal', 'Image-only'),
            ('Main comparisons', 'Multimodal − Metadata-only',   'Multimodal', 'Metadata-only'),
            ('Metadata ablation', 'Image + age − Image-only',    'Image + age', 'Image-only'),
            ('Metadata ablation', 'Image + non-age − Image-only', 'Image + non-age', 'Image-only'),
            ('Metadata ablation', 'Multimodal − Image + age',    'Multimodal', 'Image + age'),
            ('Metadata ablation', 'Multimodal − Image + non-age', 'Multimodal', 'Image + non-age'),
            ('Fusion strategy', 'Multimodal − Late fusion', 'Multimodal', 'Late fusion'),
            ('Fusion strategy', 'Multimodal − Ensemble',    'Multimodal', 'Ensemble'),

        ],
        runs=R, out=fig_dir / 'fig3_effect_sizes')
    '''
    
    representative_split(R['Multimodal'] / 'seed_2024_probs.csv',
                         fig_dir / 'fig4_representative_split')

    expert_analysis(RUNS_DIR / 'dentist_experiment' / 'dentist_comparison.csv',
                    fig_dir / 'fig6_expert_analysis')


if __name__ == '__main__':
    make_all_figures()
