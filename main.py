"""
Multimodal Deep Learning for Automated Assessment of 
Periodontal Bone Resorption Severity Using Panoramic 
Radiographs and Clinical Metadata

Main script to reproduce all results reported in the paper.
Run from project root: python main.py

Sections correspond to paper sections:
    Section 4.1 — Baseline Comparison
    Section 4.2 — Multimodal Fusion and Ablation Study  
    Section 4.5 — Model Interpretability (GradCAM)
    Section 4.7 — Inter-annotator Agreement Analysis
"""

import time

from src.config       import config, RUNS_DIR, DATA_DIR, PROJECT_ROOT
from src.experiment   import (
    run_experiment,
    run_ensemble_from_probs,
    run_gradcam,
    run_dentist_experiment,
    compare_experiments,
    find_best_ensemble_weights,
)
from src.evaluate import print_results_tables
from pathlib import Path
import pandas as pd
from scipy import stats
import re
import copy
import json
import shutil

# ── Run IDs ───────────────────────────────────────────────────────────────────
IMAGE_ONLY_ID  = 'image_only_dens'
META_ONLY_ID   = 'meta_only_dens'
MULTIMODAL_ID  = 'multi_dens'
NO_AGE_ID      = 'multi_dens_no_age'
ENSEMBLE_ID    = 'ensemble_tuned' #'ensemble_final'
DENTIST_ID     = 'dentist_experiment'
GRADCAM_ID     = 'gradcam_figures'
MULTIMODAL_AGE_ID      = 'multi_dens_age_only'

# ── Run directories ───────────────────────────────────────────────────────────
IMAGE_ONLY_RUN = RUNS_DIR / IMAGE_ONLY_ID
META_ONLY_RUN  = RUNS_DIR / META_ONLY_ID
MULTIMODAL_RUN = RUNS_DIR / MULTIMODAL_ID
NO_AGE_RUN     = RUNS_DIR / NO_AGE_ID
ENSEMBLE_RUN   = RUNS_DIR / ENSEMBLE_ID
DENTIST_RUN    = RUNS_DIR / DENTIST_ID
GRADCAM_RUN    = RUNS_DIR / GRADCAM_ID
MULTIMODAL_AGE_RUN     = RUNS_DIR / MULTIMODAL_AGE_ID

def dentist_file_org():
    dent_eva = DATA_DIR / 'dentist_eval_summary.csv'
    d = pd.read_csv(dent_eva)
    d['image_path'] = d.image_path.apply(lambda x: Path(str(x).replace('\\','/')).name)
    d[['eval_id','image_path','visual_level','visual_confidence']].to_csv(
        'data/dentist_grades.csv', index=False)

def get_ensemble_weights(force_retune=False):
    """
    Load ensemble weights from runs/statistical_analysis/ensemble_weights.json.
    Runs the tuning pipeline if the file is absent.

    Returns:
        (labels, weights) — parallel lists, weights in label order.
    """
    weights_path = RUNS_DIR / 'statistical_analysis' / 'ensemble_weights.json'

    if weights_path.exists() and not force_retune:
        chosen = json.loads(weights_path.read_text())
        print(f"Ensemble weights loaded from {weights_path}")
        print(f"  {dict(zip(chosen['labels'], chosen['weights']))}  "
              f"(tuned on {len(chosen['tuning_seeds'])} seeds, "
              f"{chosen['optimize_on']}={chosen['mean_score']:.4f})")
        return chosen['labels'], chosen['weights']

    print(f"No weights at {weights_path} — running tuning pipeline.")
    section_2_ensemble_weight_runs()      # trains the three base models
    section_3_ensemble_weight_tuning()    # grid search, writes the JSON

    if not weights_path.exists():
        raise RuntimeError(f"tuning finished but {weights_path} was not written")

    chosen = json.loads(weights_path.read_text())
    return chosen['labels'], chosen['weights']

# ─────────────────────────────────────────────────────────────────────────────
def section_1_baseline_and_ablation():
    image_only_config = {**config,
                         'fusion_mode': 'image_only',
                         'run_id':      IMAGE_ONLY_ID}
    meta_only_config  = {**config,
                         'fusion_mode': 'meta_only',
                         'run_id':      META_ONLY_ID}
    multimodal_config = {**config,
                         'fusion_mode': 'multimodal',
                         'run_id':      MULTIMODAL_ID}

    run_experiment(image_only_config)
    run_experiment(meta_only_config)
    run_experiment(multimodal_config)

# ─────────────────────────────────────────────────────────────────────────────
def section_2_ensemble_weight_runs():
    """Train the three base models on tuning seeds held out from evaluation."""
    tuning_seeds = [17, 37, 53, 71, 83, 97, 107, 131, 149, 163]

    eval_seeds = set(config['seed'] if isinstance(config['seed'], list)
                     else [config['seed']])
    overlap = eval_seeds & set(tuning_seeds)
    assert not overlap, f"tuning seeds overlap evaluation seeds: {sorted(overlap)}"

    for fusion_mode, run_id in [('multimodal', 'multi_dens_tuning'),
                                ('image_only', 'image_only_tuning'),
                                ('meta_only',  'meta_only_tuning')]:
        cfg = copy.deepcopy(config)
        cfg.update({'fusion_mode': fusion_mode,
                    'run_id':      run_id,
                    'seed':        list(tuning_seeds)})
        run_experiment(cfg)

# ─────────────────────────────────────────────────────────────────────────────
def section_3_ensemble_weight_tuning():
    tuning_seeds = [17, 37, 53, 71, 83, 97, 107, 131, 149, 163]

    best_weights, best_score, grid = find_best_ensemble_weights(
        run_dirs={
            'multimodal': RUNS_DIR / 'multi_dens_tuning',
            'image_only': RUNS_DIR / 'image_only_tuning',
            'meta_only':  RUNS_DIR / 'meta_only_tuning',
        },
        seeds=tuning_seeds,
        optimize_on='macro_f1',
    )
    return best_weights

# ─────────────────────────────────────────────────────────────────────────────
def section_4_ensemble():
    """Soft-probability ensemble using weights tuned on held-out seeds."""

    
    labels, weights = get_ensemble_weights()

    available = {
        'multimodal': MULTIMODAL_RUN,
        'image_only': IMAGE_ONLY_RUN,
        'meta_only':  META_ONLY_RUN,
    }
    missing = [l for l in labels if l not in available]
    if missing:
        raise KeyError(f"tuned weights reference unknown models: {missing}")

    run_ensemble_from_probs(
        run_dirs = {l: available[l] for l in labels},
        weights  = weights,
        seeds    = config['seed'],
        run_id   = ENSEMBLE_ID,
        save_dir = ENSEMBLE_RUN,
    )
    
# ─────────────────────────────────────────────────────────────────────────────
def section_5_statistical_comparison():
    stats_dir = RUNS_DIR / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)

    compare_experiments(
        csv_path1 = MULTIMODAL_RUN / 'results.csv',
        csv_path2 = IMAGE_ONLY_RUN / 'results.csv',
        label1    = 'multimodal',
        label2    = 'image_only',
        save_path = stats_dir / 'multi_vs_image.csv'
    )
    compare_experiments(
        csv_path1 = MULTIMODAL_RUN / 'results.csv',
        csv_path2 = META_ONLY_RUN  / 'results.csv',
        label1    = 'multimodal',
        label2    = 'meta_only',
        save_path = stats_dir / 'multi_vs_meta.csv'
    )
    compare_experiments(
        csv_path1 = ENSEMBLE_RUN   / 'results.csv',
        csv_path2 = MULTIMODAL_RUN / 'results.csv',
        label1    = 'ensemble',
        label2    = 'multimodal',
        save_path = stats_dir / 'ensemble_vs_multi.csv'
    )
    print_results_tables(
        results_paths = [
            IMAGE_ONLY_RUN / 'results.csv',
            META_ONLY_RUN  / 'results.csv',
            MULTIMODAL_RUN / 'results.csv',
            ENSEMBLE_RUN   / 'results.csv',
        ],
        labels    = ['Image-only', 'Meta-only', 'Multimodal', 'Ensemble'],
        title     = 'Performance Comparison — All Models',
        save_path = stats_dir / 'performance_table.csv'
    )

# ─────────────────────────────────────────────────────────────────────────────
def section_6_gradcam():
    multimodal_config = {**config, 'fusion_mode': 'multimodal'}
    run_gradcam(
        run_dir         = MULTIMODAL_RUN,
        seed            = 2024,
        config          = multimodal_config,
        n_correct       = 1,
        n_misclassified = 1,
        pool_all_seeds  = True,
        overlay_only    = False, 
        show_titles     = False,
        save_dir        = GRADCAM_RUN
    )

# ─────────────────────────────────────────────────────────────────────────────
def section_7_inter_annotator():
    """
    Inter-annotator agreement against independent dentist grading.

    Trains a model on all data except the dentist-graded images, predicts on
    those images, and compares model predictions, dentist grading, and the
    original annotations. Uses a single seed (the first in the config list),
    since the dentist graded one fixed set of radiographs.
    """
    from src.evaluate import (plot_interannotator_confusion,
                              plot_dentist_agreement_by_level)

    dentist_csv = DATA_DIR / 'dentist_grades.csv'
    if not dentist_csv.exists():
        raise FileNotFoundError(
            f"Dentist grading file not found: {dentist_csv}. "
            f"It must contain image_path, visual_level and visual_confidence."
        )
    
    multimodal_config = {**config,
                         'fusion_mode': 'multimodal',
                         'seed':        2024}
    
    comparison, results = run_dentist_experiment(
        dentist_csv = dentist_csv,
        config      = multimodal_config,
        save_dir    = DENTIST_RUN,
    )

    plot_interannotator_confusion(
        comparison,
        save_path=DENTIST_RUN / 'figure_interannotator_confusion.png',
    )
    plot_dentist_agreement_by_level(
        comparison,
        save_path=DENTIST_RUN / 'figure_dentist_accuracy_confidence.png',
    )

    return comparison, results

# ─────────────────────────────────────────────────────────────────────────────
def section_8_sensitivity_no_age():
    no_age_config = {**config,
                     'fusion_mode':    'multimodal',
                     'run_id':         NO_AGE_ID,
                     'metadata_cols':  [
                         "gender",
                         "number_of_missing_teeth",
                         "implant",
                         "residual_root",
                         "functional_tooth_logarithm"
                     ],
                     'normalize_cols': [
                         "number_of_missing_teeth",
                         "implant",
                         "residual_root",
                         "functional_tooth_logarithm"
                     ]}
    run_experiment(no_age_config)

# ─────────────────────────────────────────────────────────────────────────────
def section_9_sensitivity_comparison():
    stats_dir = RUNS_DIR / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Full multimodal vs no-age multimodal
    compare_experiments(
        csv_path1 = MULTIMODAL_RUN / 'results.csv',
        csv_path2 = NO_AGE_RUN     / 'results.csv',
        label1    = 'multimodal_full',
        label2    = 'multimodal_no_age',
        save_path = stats_dir / 'sensitivity_full_vs_no_age.csv'
    )

    # No-age multimodal vs image-only
    compare_experiments(
        csv_path1 = NO_AGE_RUN     / 'results.csv',
        csv_path2 = IMAGE_ONLY_RUN / 'results.csv',
        label1    = 'multimodal_no_age',
        label2    = 'image_only',
        save_path = stats_dir / 'sensitivity_no_age_vs_image.csv'
    )

# ─────────────────────────────────────────────────────────────────────────────
def section_10_late_fusion():
    """
    Late fusion of the unimodal models, with weights tuned on the held-out
    tuning seeds, compared against joint multimodal fusion.

    Tests whether the multimodal gain comes from learning a joint
    representation or merely from access to both modalities.
    """
    weights_path = RUNS_DIR / 'statistical_analysis' / 'ensemble_weights.json'
    backup_path  = weights_path.with_suffix('.json.bak')

    # the three-model weights live at the same path — preserve them
    if weights_path.exists():
        shutil.copy2(weights_path, backup_path)

    try:
        best_weights, _, _ = find_best_ensemble_weights(
            run_dirs={'image_only': RUNS_DIR / 'image_only_tuning',
                      'meta_only':  RUNS_DIR / 'meta_only_tuning'},
            seeds=[17, 37, 53, 71, 83, 97, 107, 131, 149, 163],
        )

        run_ensemble_from_probs(
            run_dirs={'image_only': IMAGE_ONLY_RUN, 'meta_only': META_ONLY_RUN},
            weights=best_weights,
            seeds=config['seed'],
            run_id='ensemble_late_fusion',
            save_dir=RUNS_DIR / 'ensemble_late_fusion',
        )

        compare_experiments(
            RUNS_DIR / 'ensemble_late_fusion' / 'results.csv',
            MULTIMODAL_RUN / 'results.csv',
            label1='late_fusion_tuned', label2='multimodal',
            save_path=RUNS_DIR / 'statistical_analysis' / 'latefusion_vs_multi.csv',
        )
    finally:
        if backup_path.exists():
            shutil.move(str(backup_path), str(weights_path))
            print(f"Restored three-model weights → {weights_path}")

# ─────────────────────────────────────────────────────────────────────────────
def section_11_sensitivity_only_age():
    image_age_config = {**config,
                     'fusion_mode':    'multimodal',
                     'run_id':         MULTIMODAL_AGE_ID,
                     'metadata_cols':  ["age"],
                     'normalize_cols': ["age"]}
    run_experiment(image_age_config)

# ─────────────────────────────────────────────────────────────────────────────
def section_12_sensitivity_only_age_comparison():
    stats_dir = RUNS_DIR / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Full multimodal vs age multimodal
    
    compare_experiments(MULTIMODAL_AGE_RUN /'results.csv',
                        IMAGE_ONLY_RUN/'results.csv',
                        label1='multimodal_age_only', label2='image_only',
                        save_path=RUNS_DIR/'statistical_analysis'/'ageonly_vs_image.csv')
    
    compare_experiments(MULTIMODAL_RUN/'results.csv',
                        MULTIMODAL_AGE_RUN/'results.csv',
                        label1='multimodal_full', label2='multimodal_age_only',
                        save_path=RUNS_DIR/'statistical_analysis'/'full_vs_ageonly.csv')
    
    mean_ci_metrics(runs={'image_only': IMAGE_ONLY_RUN,
                          'multimodal_age_only': MULTIMODAL_AGE_RUN,
                          'multimodal_no_age': NO_AGE_RUN,
                          'multimodal_full': MULTIMODAL_RUN}, name='mean_ci_ablation')
    
# ─────────────────────────────────────────────────────────────────────────────
def grade_error_analysis(runs=None, metrics_ci=True):
    """
    Mean Absolute Grade Error (MAGE) and the rate of errors exceeding one
    grade, computed per seed and summarized across seeds.

    MAGE treats severity as ordinal: predicting L1 for an L3 case is a worse
    error than predicting L2. Standard accuracy treats both identically.

    Args:
        runs: dict of {label: run_dir}. Defaults to the four main runs.

    Returns:
        (summary_df, per_seed_df)
    """
    if runs is None:
        runs = {
            'meta_only':  META_ONLY_RUN,
            'image_only': IMAGE_ONLY_RUN,
            'multimodal': MULTIMODAL_RUN,
            'ensemble':   ENSEMBLE_RUN,
        }

    stats_dir = RUNS_DIR / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)

    def load_probs(run_dir):
        """Load all per-seed prob files, handling both naming conventions."""
        run_dir = Path(run_dir)
        files = sorted(run_dir.glob('seed_*_probs.csv'))
        if not files:
            files = sorted(run_dir.glob('seed_*_ensemble_probs.csv'))
        if not files:
            raise FileNotFoundError(f"no per-seed prob files in {run_dir}")

        frames = []
        for f in files:
            m = re.match(r'seed_(-?\d+)_', f.stem)
            if not m:
                continue
            df = pd.read_csv(f)
            df['seed'] = int(m.group(1))
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    per_seed_all, summary_rows = [], []

    for label, run_dir in runs.items():
        all_probs = load_probs(run_dir)
        all_probs['abs_error'] = (all_probs['predicted_level']
                                  - all_probs['true_level']).abs()

        per_seed = (all_probs
                    .groupby('seed')
                    .apply(lambda x: pd.Series({
                        'mage':       x['abs_error'].mean(),
                        'exceed_one': (x['abs_error'] > 1).mean(),
                    }), include_groups=False)
                    .reset_index())
        per_seed.insert(0, 'model', label)
        per_seed_all.append(per_seed)

        print(f"\n{label}  (n={len(per_seed)} seeds)")
        row = {'model': label, 'n_seeds': len(per_seed)}
        for metric in ('mage', 'exceed_one'):
            v = per_seed[metric].values
            lo, hi = stats.t.interval(0.95, df=len(v) - 1,
                                      loc=v.mean(), scale=stats.sem(v))
            row.update({
                f'{metric}_mean':    round(float(v.mean()), 4),
                f'{metric}_std':     round(float(v.std(ddof=1)), 4),
                f'{metric}_ci_low':  round(float(lo), 4),
                f'{metric}_ci_high': round(float(hi), 4),
            })
            print(f"  {metric:12s} {v.mean():.4f} ± {v.std(ddof=1):.4f}  "
                  f"[{lo:.4f}, {hi:.4f}]")
        summary_rows.append(row)

    summary_df  = pd.DataFrame(summary_rows)
    per_seed_df = pd.concat(per_seed_all, ignore_index=True)

    summary_df.to_csv(stats_dir / 'grade_error.csv', index=False)
    per_seed_df.to_csv(stats_dir / 'grade_error_per_seed.csv', index=False)
    print(f"\nSaved → {stats_dir / 'grade_error.csv'}")
    print(f"Saved → {stats_dir / 'grade_error_per_seed.csv'}")

    return summary_df, per_seed_df

# ─────────────────────────────────────────────────────────────────────────────
def mean_ci_metrics(runs=None, metrics=('accuracy', 'macro_f1', 'macro_auc',
                                        'l3_recall'), name='mean_ci'):
    """
    Mean with 95% CI across seeds for each model, for the results table.
    Always writes to runs/statistical_analysis/.

    Returns:
        (summary_df, wide) — long format and manuscript-ready wide format.
    """
    if runs is None:
        runs = {
            'meta_only':  META_ONLY_RUN,
            'image_only': IMAGE_ONLY_RUN,
            'multimodal': MULTIMODAL_RUN,
            'ensemble':   ENSEMBLE_RUN,
        }

    stats_dir = RUNS_DIR / 'statistical_analysis'
    stats_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, run_dir in runs.items():
        df = pd.read_csv(Path(run_dir) / 'results.csv')
        print(f"\n{label}  (n={len(df)} seeds)")
        for m in metrics:
            if m not in df.columns:
                print(f"  {m:12s} — not in results.csv, skipping")
                continue
            v = df[m].values
            lo, hi = stats.t.interval(0.95, df=len(v) - 1,
                                      loc=v.mean(), scale=stats.sem(v))
            rows.append({
                'model':   label,
                'metric':  m,
                'mean':    round(float(v.mean()), 4),
                'std':     round(float(v.std(ddof=1)), 4),
                'ci_low':  round(float(lo), 4),
                'ci_high': round(float(hi), 4),
                'n_seeds': len(v),
            })
            print(f"  {m:12s} {v.mean():.4f} [{lo:.4f}, {hi:.4f}]")

    summary_df = pd.DataFrame(rows)

    wide = summary_df.assign(
        cell=summary_df.apply(
            lambda r: f"{r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
            axis=1)
    ).pivot(index='metric', columns='model', values='cell')
    wide = wide.reindex([m for m in metrics if m in wide.index])
    wide = wide.reindex(columns=[k for k in runs if k in wide.columns])

    print(f"\n{'='*80}")
    print("Mean [95% CI] across seeds")
    print(f"{'='*80}")
    print(wide.to_string())
    
    long_path = stats_dir / f'{name}.csv'
    wide_path = stats_dir / f'{name}_wide.csv'

    summary_df.to_csv(long_path, index=False)
    wide.to_csv(wide_path)
    print(f"\nSaved → {long_path}")
    print(f"Saved → {wide_path}")

    return summary_df, wide

# ─────────────────────────────────────────────────────────────────────────────
def save_run_provenance():
    """
    Record the git state producing results in runs/.

    Writes runs/provenance.json (current state) and appends to
    runs/provenance_log.jsonl (every invocation, one JSON object per line).
    A runs/ directory built across several commits will show all of them.
    """
    import subprocess, platform, sys
    from datetime import datetime

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    def git(*args):
        try:
            return subprocess.check_output(
                ['git', *args], cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    dirty = git('status', '--porcelain', '--untracked-files=no')

    info = {
        'commit':       git('rev-parse', 'HEAD'),
        'commit_short': git('rev-parse', '--short', 'HEAD'),
        'branch':       git('rev-parse', '--abbrev-ref', 'HEAD'),
        'tag':          git('describe', '--tags', '--always'),
        'dirty':        bool(dirty),
        'uncommitted':  dirty.splitlines() if dirty else [],
        'timestamp':    datetime.now().isoformat(),
        'python':       sys.version.split()[0],
        'platform':     platform.platform(),
    }

    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda']  = torch.version.cuda
        info['gpu']   = (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None)
    except Exception:
        pass

    (RUNS_DIR / 'provenance.json').write_text(json.dumps(info, indent=2))
    with open(RUNS_DIR / 'provenance_log.jsonl', 'a') as f:
        f.write(json.dumps(info) + '\n')

    # summarise every commit that has contributed to this runs/ directory
    log_path = RUNS_DIR / 'provenance_log.jsonl'
    entries = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    commits = {}
    for e in entries:
        c = e.get('commit_short') or 'unknown'
        commits.setdefault(c, []).append(e['timestamp'])

    print(f"Provenance → {RUNS_DIR / 'provenance.json'}")
    print(f"  commit {info['commit_short']} on {info['branch']}"
          + ("  ** WORKING TREE DIRTY **" if info['dirty'] else ""))
    if info['dirty']:
        print("  Uncommitted changes — this hash does not fully describe the code:")
        for line in info['uncommitted'][:10]:
            print(f"    {line}")

    if len(commits) > 1:
        print(f"\n  NOTE: this runs/ directory has been written by "
              f"{len(commits)} different commits:")
        for c, ts in commits.items():
            print(f"    {c}  ({len(ts)} run(s), first {ts[0][:19]})")
        print("  Results here may not all come from the same code version.")

    return info

if __name__ == "__main__":
    start = time.time()

    save_run_provenance()
    print("=" * 60)
    print("BRAR Classification — Reproducing Paper Results")
    print("=" * 60)
    print("\nChecking existing runs:")
    for name, path in [
        ("Image-only",    IMAGE_ONLY_RUN),
        ("Meta-only",     META_ONLY_RUN),
        ("Multimodal",    MULTIMODAL_RUN),
        ("Ensemble",      ENSEMBLE_RUN),
        ("GradCAM",       GRADCAM_RUN),
        ("Dentist",       DENTIST_RUN),
        ("No-age",        NO_AGE_RUN),
    ]:
        if (path / '.complete').exists():
            status = "complete"
        elif path.exists():
            status = "incomplete — will rerun"
        else:
            status = "missing"
        print(f"  {name:15s} {status}")
    print()

    steps = [
        #("Baseline and ablation experiments",    section_1_baseline_and_ablation),
        #("Training base models on tuning seeds", section_2_ensemble_weight_runs),
        #("Finding optimal ensemble weights",     section_3_ensemble_weight_tuning),
        #("Ensemble analysis",                    section_4_ensemble),
        #("Statistical comparison",               section_5_statistical_comparison),
        #("GradCAM visualizations",                section_6_gradcam),
        #("Inter-annotator evaluation",           section_7_inter_annotator),
        #("Sensitivity analysis (no age)",        section_8_sensitivity_no_age),
        #("Full vs no-age comparison",            section_9_sensitivity_comparison),
        #("Late fusion"              ,            section_10_late_fusion),
        #("Sensitivity analysis (only age)",      section_11_sensitivity_only_age),
        ("Full vs only age comparison",          section_12_sensitivity_only_age_comparison),
        ("Confidence intervals",                 mean_ci_metrics),
        #("Grade error analysis (MAGE)",          grade_error_analysis),
    ]

    print("=" * 60) 
    print("Reproducing paper results")
    print("=" * 60)

    for i, (label, fn) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {label}...")
        t0 = time.time()
        fn()
        print(f"    done in {time.time() - t0:.1f}s")

    elapsed = time.time() - start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n{'='*60}")
    print(f"Full execution time: {h}h {m}m {s}s")
    print(f"{'='*60}")

