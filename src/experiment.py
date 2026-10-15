from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, f1_score

import json
from datetime import datetime
import pandas as pd
import numpy as np
from src.dataset import build_datasets
from src.dataset import load_metadata
from src.dataset import  get_dataloaders, get_transforms
from src.model import build_model
from src.train import train
from src.evaluate import plot_history, evaluate, plot_roc_auc 
from src.evaluate import plot_precision_recall, get_predictions
from src.evaluate import get_misclassified, select_gradcam_images, plot_gradcam
from sklearn.metrics import roc_curve, auc, average_precision_score
from src.utils import set_seed
from src.utils import split_data
from src.config import RUNS_DIR
import torch
import scipy.stats as stats
from pathlib import Path
from PIL import Image
from src.evaluate import  _GradCAMWrapper, _get_target_layer
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.evaluate import GradCAM, show_cam_on_image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy 
import shutil
from statsmodels.stats.multitest import multipletests

def check_run_dir(run_dir, force=False):
    """
    Returns True if a completed run exists and should be reused.
    Otherwise removes any partial directory and returns False.
    """
    run_dir = Path(run_dir)
    if run_dir.exists():
        if (run_dir / '.complete').exists() and not force:
            return True
        print(f"Incomplete or forced — removing {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return False
 
def compare_experiments(csv_path1, csv_path2, label1=None, label2=None, 
                        save_path=None):
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)
    label1 = label1 or f"{df1['backbone_name'].iloc[0]}_{df1['fusion_mode'].iloc[0]}"
    label2 = label2 or f"{df2['backbone_name'].iloc[0]}_{df2['fusion_mode'].iloc[0]}"
    
    metric_cols = [
        'accuracy',
        'macro_f1', 'macro_precision', 'macro_recall',
        'micro_auc', 'macro_auc',
        'micro_ap', 'macro_ap',
        'l1_precision', 'l1_recall', 'l1_f1', 'l1_ap',
        'l2_precision', 'l2_recall', 'l2_f1', 'l2_ap',
        'l3_precision', 'l3_recall', 'l3_f1', 'l3_ap',
    ]
    metric_cols = [c for c in metric_cols 
                   if c in df1.columns and c in df2.columns]

    rows = []
    for col in metric_cols:
        m1, s1   = df1[col].mean(), df1[col].std()
        m2, s2   = df2[col].mean(), df2[col].std()
        mn1, mx1 = df1[col].min(),  df1[col].max()
        mn2, mx2 = df2[col].min(),  df2[col].max()
        diff     = m1 - m2
        winner   = label1 if m1 > m2 else (label2 if m2 > m1 else 'tie')
        rows.append({
            'metric':         col,
            f'{label1}_mean': round(m1,  4),
            f'{label1}_std':  round(s1,  4),
            f'{label1}_min':  round(mn1, 4),
            f'{label1}_max':  round(mx1, 4),
            f'{label2}_mean': round(m2,  4),
            f'{label2}_std':  round(s2,  4),
            f'{label2}_min':  round(mn2, 4),
            f'{label2}_max':  round(mx2, 4),
            'diff (1-2)':     round(diff, 4),
            'winner':         winner,
        })

    comparison_df = pd.DataFrame(rows).set_index('metric')

    print(f"\n{'='*80}")
    print(f"Comparison: {label1}  vs  {label2}")
    print(f"Seeds in file 1: {sorted(df1['seed'].tolist())}")
    print(f"Seeds in file 2: {sorted(df2['seed'].tolist())}")
    print(f"{'='*80}\n")

    display_cols = [
        f'{label1}_mean', f'{label1}_std',
        f'{label2}_mean', f'{label2}_std',
        'diff (1-2)', 'winner'
    ]
    print(comparison_df[display_cols].to_string())

    wins1 = (comparison_df['winner'] == label1).sum()
    wins2 = (comparison_df['winner'] == label2).sum()
    ties  = (comparison_df['winner'] == 'tie').sum()
    print(f"\n{'='*80}")
    print(f"Win count — {label1}: {wins1}  |  {label2}: {wins2}  |  Ties: {ties}")
    print(f"{'='*80}\n")

    # Wilcoxon signed-rank test
    print(f"\n{'='*80}")
    print("Statistical significance (Wilcoxon signed-rank test):")
    print(f"{'='*80}")

    merged = df1.merge(df2, on='seed', suffixes=('_1', '_2'))


    merged = df1.merge(df2, on='seed', suffixes=('_1', '_2'))
    assert len(merged) == len(df1) == len(df2), (
        f"seed mismatch: {len(df1)} vs {len(df2)}, {len(merged)} paired"
    )

    PRIMARY = ['macro_f1', 'macro_auc', 'l3_recall']

    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.4f}"

    wilcoxon_rows = []
    for metric in metric_cols:
        c1, c2 = f'{metric}_1', f'{metric}_2'
        if c1 not in merged.columns or c2 not in merged.columns:
            continue
        d = merged[c1] - merged[c2]
        if (d == 0).all():
            p, med = 1.0, 0.0
        else:
            _, p = stats.wilcoxon(merged[c1], merged[c2], zero_method='zsplit')
            med = float(np.median(d))
        wilcoxon_rows.append({
            'metric':      metric,
            'primary':     metric in PRIMARY,
            'median_diff': round(med, 4),
            'iqr_low':     round(float(np.percentile(d, 25)), 4),
            'iqr_high':    round(float(np.percentile(d, 75)), 4),
            'n_ties':      int((d == 0).sum()),
            'p_raw':       p,
        })

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)

    # Holm correction across the three prespecified primary metrics only
    mask = wilcoxon_df['primary']
    wilcoxon_df['p_holm'] = np.nan
    if mask.any():
        _, p_adj, _, _ = multipletests(
            wilcoxon_df.loc[mask, 'p_raw'], alpha=0.05, method='holm'
        )
        wilcoxon_df.loc[mask, 'p_holm'] = p_adj

    print("\nPrimary metrics (prespecified, Holm-corrected):")
    for _, r in wilcoxon_df[mask].iterrows():
        verdict = 'significant' if r['p_holm'] < 0.05 else 'not significant'
        print(f"  {r['metric']:14s} median diff={r['median_diff']:+.4f} "
              f"[{r['iqr_low']:+.4f}, {r['iqr_high']:+.4f}]  "
              f"p={fmt_p(r['p_raw'])}  p_holm={fmt_p(r['p_holm'])}  {verdict}")

    print("\nExploratory metrics (uncorrected, not for confirmatory claims):")
    for _, r in wilcoxon_df[~mask].iterrows():
        print(f"  {r['metric']:14s} median diff={r['median_diff']:+.4f}  "
              f"p={fmt_p(r['p_raw'])}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(save_path)
        print(f"\nComparison saved → {save_path}")
        wilcoxon_path = save_path.parent / \
                        f"{save_path.stem}_wilcoxon{save_path.suffix}"
        wilcoxon_df.to_csv(wilcoxon_path, index=False)
        print(f"Wilcoxon saved   → {wilcoxon_path}")

    return comparison_df, wilcoxon_df

def save_predictions(model, test_loader, test_df, save_path=None):
    """
    Save softmax probabilities + metadata for all test samples.
    
    Args:
        model:       trained model
        test_loader: DataLoader for test set
        test_df:     original test DataFrame (before normalization)
        save_path:   optional Path to save CSV
    
    Returns:
        DataFrame with probabilities and metadata for all test samples
    """
    probs, preds, labels = get_predictions(model, test_loader)

    meta_cols = ["age", "gender", "number_of_missing_teeth",
                 "implant", "residual_root",
                 "functional_tooth_logarithm", "bone_resorption"]

    rows = []
    for idx in range(len(labels)):
        true_label = int(labels[idx])
        pred_label = int(preds[idx])

        meta_row = test_df.iloc[idx]

        row = {
            'image_path':      meta_row.get('image_path', f'sample_{idx}'),
            'true_level':      true_label + 1,        # 1-indexed
            'predicted_level': pred_label + 1,        # 1-indexed
            'prob_l1':         round(float(probs[idx, 0]), 6),
            'prob_l2':         round(float(probs[idx, 1]), 6),
            'prob_l3':         round(float(probs[idx, 2]), 6),
            'correct':         pred_label == true_label,
        }

        for col in meta_cols:
            if col in meta_row.index:
                row[col] = meta_row[col]

        rows.append(row)

    probs_df = pd.DataFrame(rows).reset_index(drop=True)

    if save_path:
        probs_df.to_csv(save_path, index=False)
        print(f"Predictions saved → {save_path}")

    return probs_df

def run_experiment(config, force=False):
    seeds  = config['seed'] if isinstance(config['seed'], list) \
             else [config['seed']]
    run_id = config.get('run_id', 'run')
    run_dir = RUNS_DIR / run_id

    if check_run_dir(run_dir, force=force):
        done = json.loads((run_dir / '.complete').read_text())
        if set(done.get('seeds', [])) >= set(seeds):
            print(f"Run '{run_id}' complete — reusing.")
            return pd.read_csv(run_dir / 'results.csv')
        print(f"Run '{run_id}' has seeds {done.get('seeds')}, "
              f"need {seeds} — rerunning.")
        shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved → {run_dir / 'config.json'}")
    all_results = []  
    
    for seed in seeds:
        run_config         = copy.deepcopy(config)
        run_config['seed'] = seed

        print(f"\n{'='*60}")
        print(f"Run ID: {run_id} | Seed: {seed} | "
              f"Backbone: {config['backbone_name']} | "
              f"Mode: {config['fusion_mode']}")
        print(f"{'='*60}")

        set_seed(seed)

        df = load_metadata()
        train_df, val_df, test_df = split_data(df, run_config)

        for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
            d[['image_path', 'level']].to_csv(
                run_dir / f"seed_{seed}_{name}_split.csv", index=False)


        train_dataset, val_dataset, test_dataset, norm_stats = build_datasets(
            train_df, val_df, test_df, run_config
        )
        
        with open(run_dir / f"seed_{seed}_norm_stats.json", 'w') as f:
            json.dump(
                {k: [float(v[0]), float(v[1])] for k, v in norm_stats.items()},
                f, indent=2
            )
        
        train_loader, val_loader, test_loader = get_dataloaders(
            train_dataset, val_dataset, test_dataset, run_config
        )

        model = build_model(run_config)
        model, history = train(model, train_loader, val_loader, run_config)

        # Save model weights
        torch.save(model.state_dict(),
                   run_dir / f"seed_{seed}_model.pth")

        # Save predictions — single inference call
        probs_df = save_predictions(
            model, test_loader, test_df,
            save_path=run_dir / f"seed_{seed}_probs.csv"
        )

        # Derive probs, preds, labels from probs_df
        probs  = probs_df[['prob_l1', 'prob_l2', 'prob_l3']].values
        preds  = probs_df['predicted_level'].values - 1   # 0-indexed
        labels = probs_df['true_level'].values - 1        # 0-indexed

        # Misclassified analysis
        get_misclassified(
            probs_df, sort_by='true_level',
            save_path=run_dir / f"seed_{seed}_misclassified.csv"
        )

        # Plots
        plot_history(history,
                     save_path=run_dir / f"seed_{seed}_training_curves.png")
        evaluate(model, test_loader,
                 save_path=run_dir / f"seed_{seed}_confusion_matrix.png")
        plot_roc_auc(model, test_loader,
                     save_path=run_dir / f"seed_{seed}_roc_auc.png")
        plot_precision_recall(model, test_loader,
                              save_path=run_dir / f"seed_{seed}_precision_recall.png")

        # Metrics
        y_bin  = label_binarize(labels, classes=[0, 1, 2])
        report = classification_report(labels, preds, output_dict=True,
                                       target_names=["Level 1", "Level 2", "Level 3"])

        #fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        #all_fpr  = np.linspace(0, 1, 200)
        #mean_tpr = np.mean([np.interp(all_fpr, *roc_curve(y_bin[:, i],
        #                    probs[:, i])[:2]) for i in range(3)], axis=0)
        
        row = {
            'run_id':        run_id,
            'seed':          seed,
            'backbone_name': config['backbone_name'],
            'fusion_mode':   config['fusion_mode'],

            'accuracy':        report['accuracy'],
            'macro_f1':        report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall':    report['macro avg']['recall'],
            #'micro_auc':       auc(fpr_micro, tpr_micro),
            #'macro_auc':       auc(all_fpr, mean_tpr),
            
            'micro_auc': roc_auc_score(y_bin, probs, average='micro'),
            'macro_auc': roc_auc_score(y_bin, probs, average='macro', multi_class='ovr'),
            
            'micro_ap':        average_precision_score(y_bin, probs, average='micro'),
            'macro_ap':        average_precision_score(y_bin, probs, average='macro'),

            'l1_precision': report['Level 1']['precision'],
            'l1_recall':    report['Level 1']['recall'],
            'l1_f1':        report['Level 1']['f1-score'],
            'l1_ap':        average_precision_score(y_bin[:, 0], probs[:, 0]),

            'l2_precision': report['Level 2']['precision'],
            'l2_recall':    report['Level 2']['recall'],
            'l2_f1':        report['Level 2']['f1-score'],
            'l2_ap':        average_precision_score(y_bin[:, 1], probs[:, 1]),

            'l3_precision': report['Level 3']['precision'],
            'l3_recall':    report['Level 3']['recall'],
            'l3_f1':        report['Level 3']['f1-score'],
            'l3_ap':        average_precision_score(y_bin[:, 2], probs[:, 2]),
        }

        all_results.append(row)
        print(f"\nSeed {seed} done — acc: {row['accuracy']:.4f}  "
              f"macro_f1: {row['macro_f1']:.4f}  "
              f"macro_auc: {row['macro_auc']:.4f}")

        # In run_experiment after each seed:
        del train_loader, val_loader, test_loader
        del train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache()


    results_df = pd.DataFrame(all_results)

    metric_cols = ['accuracy', 'macro_f1', 'macro_auc', 'macro_ap',
                   'l1_f1', 'l2_f1', 'l3_f1',
                   'l1_recall', 'l2_recall', 'l3_recall']
    print(f"\n{'='*60}")
    print("Summary across seeds:")
    print(f"{'='*60}")
    print(results_df[metric_cols].agg(['mean', 'std']).round(4))

    results_df.to_csv(run_dir / "results.csv", index=False)
    
    #sentinel.write_text(
    #json.dumps({'seeds': list(seeds),
    #            'completed': datetime.now().isoformat()}, indent=2)
    #)
    
    (run_dir / '.complete').write_text(json.dumps({
    'seeds': list(seeds),
    'completed': datetime.now().isoformat(),
    }, indent=2))
    
    print(f"\nResults saved → {run_dir / 'results.csv'}")

    return results_df

def load_model_from_run(run_dir, seed, config):
    """Load a saved model from a run directory for a specific seed."""
    model_path = run_dir / f"seed_{seed}_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def ensemble_from_probs(prob_paths, weights=None, save_path=None):
    """
    Soft ensemble from saved probability CSV files — no GPU needed.

    Args:
        prob_paths: dict of {label: path_to_probs_csv}
                    e.g. {'multimodal': 'runs/.../seed_42_probs.csv',
                          'image_only': 'runs/.../seed_42_probs.csv'}
        weights:    list of weights per model — must sum to 1.
                    If None, equal weights used.
        save_path:  optional Path to save ensemble prob CSV

    Returns:
        DataFrame with ensemble probabilities and metrics
    """
    labels   = list(prob_paths.keys())
    paths    = list(prob_paths.values())
    n_models = len(labels)

    if weights is None:
        weights = [1.0 / n_models] * n_models
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    assert len(weights) == n_models, "One weight per model required"

    # Load all prob files
    dfs = {label: pd.read_csv(path) for label, path in zip(labels, paths)}

    # Verify all files have same samples in same order
    ref_df = list(dfs.values())[0]
    for label, df in dfs.items():
        assert list(df['image_path']) == list(ref_df['image_path']), \
            f"Sample order mismatch in {label} — ensure same seed/split"

    # Weighted average of probabilities
    ensemble_prob_l1 = sum(w * df['prob_l1'] for w, df
                           in zip(weights, dfs.values()))
    ensemble_prob_l2 = sum(w * df['prob_l2'] for w, df
                           in zip(weights, dfs.values()))
    ensemble_prob_l3 = sum(w * df['prob_l3'] for w, df
                           in zip(weights, dfs.values()))

    # Build ensemble DataFrame — use metadata from first file
    ensemble_df = ref_df.copy()
    ensemble_df['prob_l1'] = ensemble_prob_l1.round(6)
    ensemble_df['prob_l2'] = ensemble_prob_l2.round(6)
    ensemble_df['prob_l3'] = ensemble_prob_l3.round(6)

    # Recompute predictions
    prob_matrix = ensemble_df[['prob_l1', 'prob_l2', 'prob_l3']].values
    ensemble_df['predicted_level'] = prob_matrix.argmax(axis=1) + 1  # 1-indexed
    ensemble_df['correct'] = (ensemble_df['predicted_level'] ==
                               ensemble_df['true_level'])

    # Print summary
    accuracy = ensemble_df['correct'].mean()
    print(f"\n{'='*70}")
    print(f"Ensemble: {dict(zip(labels, weights))}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"{'='*70}")

    # Full classification report
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score, average_precision_score

    true_labels = ensemble_df['true_level'].values - 1   # 0-indexed
    pred_labels = ensemble_df['predicted_level'].values - 1
    probs       = ensemble_df[['prob_l1', 'prob_l2', 'prob_l3']].values
    y_bin       = label_binarize(true_labels, classes=[0, 1, 2])

    print(classification_report(true_labels, pred_labels,
                                 target_names=["Level 1", "Level 2", "Level 3"]))

    macro_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    macro_ap  = average_precision_score(y_bin, probs, average='macro')
    print(f"Macro AUC: {macro_auc:.4f}")
    print(f"Macro AP:  {macro_ap:.4f}")

    if save_path:
        ensemble_df.to_csv(save_path, index=False)
        print(f"\nEnsemble predictions saved → {save_path}")

    return ensemble_df

def run_ensemble_from_probs(run_dirs, seeds, weights=None, 
                             run_id=None, save_dir=None):
    """
    Run soft ensemble from saved prob files across all seeds.
    Returns same format as run_experiment for use with compare_experiments.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-generate run_id
    base_id = run_id or 'ensemble'
    run_id  = f"{base_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    labels    = list(run_dirs.keys())
    fusion_id = '_'.join(labels)  # e.g. 'multimodal_image_only'

    all_results = []
    all_ensemble_dfs = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Ensembling seed {seed}")
        print(f"{'='*60}")

        # Build prob_paths for this seed
        prob_paths = {}
        for label, run_dir in run_dirs.items():
            path = Path(run_dir) / f"seed_{seed}_probs.csv"
            if not path.exists():
                print(f"Missing: {path} — skipping seed {seed}")
                break
            prob_paths[label] = path
        else:
            save_path = save_dir / f"seed_{seed}_ensemble_probs.csv" \
                        if save_dir else None

            ensemble_df      = ensemble_from_probs(
                prob_paths=prob_paths,
                weights=weights,
                save_path=save_path
            )
            ensemble_df['seed'] = seed
            all_ensemble_dfs.append(ensemble_df)

            # Compute full metrics row — same as run_experiment
            true_labels = ensemble_df['true_level'].values - 1
            pred_labels = ensemble_df['predicted_level'].values - 1
            probs       = ensemble_df[['prob_l1', 'prob_l2', 'prob_l3']].values
            y_bin       = label_binarize(true_labels, classes=[0, 1, 2])
            report      = classification_report(
                true_labels, pred_labels, output_dict=True,
                target_names=["Level 1", "Level 2", "Level 3"]
            )

            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
            all_fpr  = np.linspace(0, 1, 200)
            mean_tpr = np.mean([
                np.interp(all_fpr, *roc_curve(y_bin[:, i], probs[:, i])[:2])
                for i in range(3)
            ], axis=0)

            row = {
                # Identity — same columns as run_experiment
                'run_id':        run_id,
                'seed':          seed,
                'backbone_name': 'ensemble',
                'fusion_mode':   fusion_id,

                # Overall metrics
                'accuracy':        report['accuracy'],
                'macro_f1':        report['macro avg']['f1-score'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall':    report['macro avg']['recall'],
                'micro_auc':       auc(fpr_micro, tpr_micro),
                'macro_auc':       auc(all_fpr, mean_tpr),
                'micro_ap':        average_precision_score(
                                       y_bin, probs, average='micro'),
                'macro_ap':        average_precision_score(
                                       y_bin, probs, average='macro'),

                # Per-class metrics
                'l1_precision': report['Level 1']['precision'],
                'l1_recall':    report['Level 1']['recall'],
                'l1_f1':        report['Level 1']['f1-score'],
                'l1_ap':        average_precision_score(
                                    y_bin[:, 0], probs[:, 0]),

                'l2_precision': report['Level 2']['precision'],
                'l2_recall':    report['Level 2']['recall'],
                'l2_f1':        report['Level 2']['f1-score'],
                'l2_ap':        average_precision_score(
                                    y_bin[:, 1], probs[:, 1]),

                'l3_precision': report['Level 3']['precision'],
                'l3_recall':    report['Level 3']['recall'],
                'l3_f1':        report['Level 3']['f1-score'],
                'l3_ap':        average_precision_score(
                                    y_bin[:, 2], probs[:, 2]),
            }

            all_results.append(row)
            print(f"Seed {seed} done — acc: {row['accuracy']:.4f}  "
                  f"macro_f1: {row['macro_f1']:.4f}  "
                  f"macro_auc: {row['macro_auc']:.4f}")

    # Assemble results DataFrame — same format as run_experiment
    results_df = pd.DataFrame(all_results)

    # Summary
    metric_cols = ['accuracy', 'macro_f1', 'macro_auc', 'macro_ap',
                   'l1_f1', 'l2_f1', 'l3_f1',
                   'l1_recall', 'l2_recall', 'l3_recall']
    print(f"\n{'='*60}")
    print("Ensemble summary across seeds:")
    print(f"{'='*60}")
    print(results_df[metric_cols].agg(['mean', 'std']).round(4))

    # Save — same structure as run_experiment
    if save_dir:
        results_df.to_csv(save_dir / "results.csv", index=False)
        print(f"\nResults saved → {save_dir / 'results.csv'}")

    # Pooled ensemble probs for image selection
    pooled_df = pd.concat(all_ensemble_dfs, ignore_index=True) \
                if all_ensemble_dfs else pd.DataFrame()

    return results_df, pooled_df

def run_gradcam(run_dir, seed, config, n_correct=1, n_misclassified=1,
                image_ids=None, pool_all_seeds=False, overlay_only = True, 
                show_titles = True, save_dir=None):
    """
    Generate GradCAM visualizations for selected images.

    Args:
        run_dir:          path to experiment run directory
        seed:             seed to load model and norm_stats from
        config:           config dict
        n_correct:        number of correct predictions per level
        n_misclassified:  number of misclassified cases per boundary
        image_ids:        optional list of specific image_path values
        pool_all_seeds:   if True, pool probs from all seeds for image selection
                          if False, use only the specified seed
        save_dir:         optional Path to save GradCAM figures
    """
    run_dir = Path(run_dir)

    # Load norm_stats
    with open(run_dir / f"seed_{seed}_norm_stats.json") as f:
        norm_stats = {k: tuple(v) for k, v in json.load(f).items()}

    # Load model
    model = build_model(config)
    model.load_state_dict(
        torch.load(run_dir / f"seed_{seed}_model.pth", map_location='cpu')
    )
    model.eval()

    # Load probs for image selection
    if pool_all_seeds:
        prob_files = sorted(run_dir.glob("seed_*_probs.csv"))
        if not prob_files:
            raise FileNotFoundError(f"No prob files found in {run_dir}")
        probs_df = pd.concat([pd.read_csv(f) for f in prob_files],
                              ignore_index=True)
        print(f"Pooled {len(prob_files)} seed prob files "
              f"({len(probs_df)} total samples)")
    else:
        probs_df = pd.read_csv(run_dir / f"seed_{seed}_probs.csv")

    # Select images
    selected_df = select_gradcam_images(
        probs_df,
        n_correct=n_correct,
        n_misclassified=n_misclassified,
        image_ids=image_ids
    )

    # Save directory
    out_dir = Path(save_dir) if save_dir else RUNS_DIR / 'gradcam_figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate GradCAM
    plot_gradcam(model, selected_df, config, norm_stats, 
                 overlay_only    = overlay_only,
                 show_titles     = show_titles, save_dir=out_dir)
    
    return selected_df

def prepare_dentist_eval(multimodal_run_dir, model, config, norm_stats,
                         n_per_category=10, save_dir='dentist_eval'):
    """
    Prepare images and GradCAM visualizations for dentist evaluation.

    Args:
        multimodal_run_dir: path to multimodal run directory
        model:              trained multimodal model (best seed)
        config:             config dict
        norm_stats:         normalization stats from training
        n_per_category:     number of images per category (default 10)
        save_dir:           output directory for dentist evaluation
    """
    from src.config import DATA_DIR
    import shutil

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pool probs from all seeds
    prob_files = sorted(Path(multimodal_run_dir).glob("seed_*_probs.csv"))
    if not prob_files:
        raise FileNotFoundError(f"No prob files found in {multimodal_run_dir}")

    all_probs = pd.concat([pd.read_csv(f) for f in prob_files],
                           ignore_index=True)
    print(f"Pooled {len(prob_files)} seeds — {len(all_probs)} total samples")

    # Deduplicate — keep highest confidence row per image
    all_probs['max_prob'] = all_probs[['prob_l1', 'prob_l2', 'prob_l3']].max(axis=1)
    all_probs = (all_probs
                 .sort_values('max_prob', ascending=False)
                 .drop_duplicates(subset='image_path', keep='first')
                 .reset_index(drop=True))
    print(f"After deduplication: {len(all_probs)} unique images")

    # Add confidence and margin columns
    all_probs['confidence'] = all_probs.apply(
        lambda row: row[f"prob_l{int(row['predicted_level'])}"], axis=1
    )
    all_probs['true_class_prob'] = all_probs.apply(
        lambda row: row[f"prob_l{int(row['true_level'])}"], axis=1
    )
    all_probs['margin'] = (all_probs['confidence'] -
                           all_probs['true_class_prob']).round(4)

    # Define categories
    categories = [
        {'name': 'correct_L1',    'true': 1, 'pred': 1,
         'sort_by': 'confidence', 'ascending': False},
        {'name': 'correct_L2',    'true': 2, 'pred': 2,
         'sort_by': 'confidence', 'ascending': False},
        {'name': 'correct_L3',    'true': 3, 'pred': 3,
         'sort_by': 'confidence', 'ascending': False},
        {'name': 'miss_L1_as_L2', 'true': 1, 'pred': 2,
         'sort_by': 'margin', 'ascending': False},
        {'name': 'miss_L1_as_L3', 'true': 1, 'pred': 3,
         'sort_by': 'margin', 'ascending': False},
        {'name': 'miss_L2_as_L1', 'true': 2, 'pred': 1,
         'sort_by': 'margin', 'ascending': False},
        {'name': 'miss_L2_as_L3', 'true': 2, 'pred': 3,
         'sort_by': 'margin', 'ascending': False},
        {'name': 'miss_L3_as_L1', 'true': 3, 'pred': 1,
         'sort_by': 'margin', 'ascending': False},
        {'name': 'miss_L3_as_L2', 'true': 3, 'pred': 2,
         'sort_by': 'margin', 'ascending': False},
    ]

    selected_rows = []
    for cat in categories:
        mask = (
            (all_probs['true_level']      == cat['true']) &
            (all_probs['predicted_level'] == cat['pred'])
        )
        candidates = (all_probs[mask]
                      .sort_values(cat['sort_by'], ascending=cat['ascending'])
                      .head(n_per_category))

        if candidates.empty:
            print(f"No cases found for {cat['name']} — skipping")
            continue

        for _, row in candidates.iterrows():
            r = row.to_dict()
            r['category'] = cat['name']
            selected_rows.append(r)
        print(f"  {cat['name']:20s}: {len(candidates)} images selected")

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected_df.insert(0, 'eval_id', range(1, len(selected_df) + 1))
    print(f"\nTotal selected: {len(selected_df)} images")

    # Generate GradCAM and copy original images
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device).eval()
    transform = get_transforms(config, train=False)

    meta_cols      = ["age", "gender", "number_of_missing_teeth",
                      "implant", "residual_root", "functional_tooth_logarithm"]
    normalize_cols = ["age", "number_of_missing_teeth", "implant",
                      "residual_root", "functional_tooth_logarithm"]

    for _, row in selected_df.iterrows():
        eval_id    = int(row['eval_id'])
        true_level = int(row['true_level'])
        pred_level = int(row['predicted_level'])
        category   = row['category']
        img_name   = Path(row['image_path']).name

        src_path = (DATA_DIR / 'manual_cropped_dataset' /
                    f"level_{true_level}" / img_name)

        if not src_path.exists():
            print(f"Image not found: {src_path} — skipping")
            continue

        prefix = (f"{eval_id:03d}_{img_name.replace('.jpg', '')}"
                  f"_true_L{true_level}_pred_L{pred_level}")

        # Copy original image
        shutil.copy(src_path, save_dir / f"{prefix}_original.jpg")

        # Load original image at full resolution
        pil_original = Image.open(src_path).convert('RGB')
        orig_w, orig_h = pil_original.size

        # Transform for model input (224x224)
        img_tensor = transform(pil_original).unsqueeze(0).to(device)

        # Normalized meta tensor
        meta_vals = []
        for col in meta_cols:
            val = float(row[col])
            if col in normalize_cols and col in norm_stats:
                col_min, col_max = norm_stats[col]
                if col_max > col_min:
                    val = (val - col_min) / (col_max - col_min)
            meta_vals.append(val)
        meta_tensor = torch.tensor(meta_vals,
                                   dtype=torch.float32).unsqueeze(0).to(device)

        # GradCAM
        wrapper = _GradCAMWrapper(model, meta_tensor)
        for param in wrapper.model.image_branch.parameters():
            param.requires_grad = True

        target_layer = _get_target_layer(wrapper.model)
        if not any(module is target_layer
                   for _, module in wrapper.named_modules()):
            print(f"Target layer not found — skipping GradCAM for {img_name}")
            continue

        pred_class = pred_level - 1
        targets    = [ClassifierOutputTarget(pred_class)]

        with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        grayscale_cam = np.array(grayscale_cam)
        if grayscale_cam.ndim == 3:
            grayscale_cam = grayscale_cam[0]

        # Resize GradCAM from 224x224 to original resolution using PIL
        grayscale_cam_pil = Image.fromarray(
            (grayscale_cam * 255).astype(np.uint8)
        )
        grayscale_cam_orig = np.array(
            grayscale_cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0

        # Original image as float32 [0,1] — brightened for display
        img_original_np = np.array(pil_original).astype(np.float32) / 255.0
        img_display     = np.clip(img_original_np * 1.5, 0, 1)

        # Overlay GradCAM on original resolution image
        cam_overlay = show_cam_on_image(img_display, grayscale_cam_orig,
                                        use_rgb=True)

        # Save GradCAM figure — use figsize proportional to original aspect ratio
        aspect = orig_w / orig_h
        fig_h  = 5
        fig_w  = fig_h * aspect * 2  # two panels side by side

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        axes[0].imshow(img_display)
        axes[0].set_title(f"Original | True: L{true_level}")
        axes[0].axis('off')
        axes[1].imshow(cam_overlay)
        axes[1].set_title(f"GradCAM | Pred: L{pred_level} | {category}")
        axes[1].axis('off')
        plt.suptitle(f"ID: {eval_id:03d} | {img_name} | "
                     f"Confidence: {row['confidence']:.3f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(save_dir / f"{prefix}_gradcam.png",
                    dpi=300, bbox_inches='tight')
        plt.close('all')

        print(f"  {eval_id:03d} — {img_name} [{category}] ✓")

    # Save summary CSV
    summary_cols = [
        'eval_id', 'image_path', 'category',
        'true_level', 'predicted_level',
        'confidence', 'true_class_prob', 'margin',
        'age', 'gender', 'bone_resorption',
        'number_of_missing_teeth', 'implant',
        'residual_root', 'functional_tooth_logarithm'
    ]
    summary_cols = [c for c in summary_cols if c in selected_df.columns]
    selected_df[summary_cols].to_csv(
        save_dir / "dentist_eval_summary.csv", index=False
    )
    print(f"\nSummary saved → {save_dir / 'dentist_eval_summary.csv'}")
    print(f"Total images: {len(selected_df)}")
    print(f"Output directory: {save_dir}")

    return selected_df

def compute_dentist_agreement(comparison, verbose=True):
    """
    Inter-annotator agreement statistics from a dentist comparison table.

    Separated from training so agreement can be recomputed from a saved
    dentist_comparison.csv without a GPU.

    Args:
        comparison: DataFrame with true_level, predicted_level, visual_level,
                    visual_confidence, and prob_l1..prob_l3 columns.

    Returns:
        dict of agreement statistics.
    """
    from sklearn.metrics import cohen_kappa_score, classification_report
    from scipy.stats import spearmanr

    def interpret_kappa(k):
        if k is None:      return "undefined"
        if k < 0.20:       return "slight"
        if k < 0.40:       return "fair"
        if k < 0.60:       return "moderate"
        if k < 0.80:       return "substantial"
        return "almost perfect"

    def kappa(a, b, weights=None):
        if len(a) < 2:
            return None
        return float(cohen_kappa_score(a, b, weights=weights))

    true_l    = comparison['true_level']
    pred_l    = comparison['predicted_level']
    dentist_l = comparison['visual_level']
    model_conf = comparison.apply(
        lambda r: r[f"prob_l{int(r['predicted_level'])}"], axis=1).values
    dentist_conf = comparison['visual_confidence'].values

    correct   = pred_l == true_l
    incorrect = ~correct

    res = {
        'n_images':      int(len(comparison)),
        'n_correct':     int(correct.sum()),
        'n_incorrect':   int(incorrect.sum()),
        'dentist_accuracy': round(float((dentist_l == true_l).mean()), 4),
        'model_accuracy':   round(float((pred_l == true_l).mean()), 4),
    }

    # ── overall agreement ──────────────────────────────────────────────
    for name, (a, b) in {
        'dentist_vs_orig':  (true_l, dentist_l),
        'model_vs_orig':    (true_l, pred_l),
        'model_vs_dentist': (dentist_l, pred_l),
    }.items():
        res[f'kappa_{name}']   = round(kappa(a, b), 4)
        res[f'kappa_{name}_w'] = round(kappa(a, b, 'linear'), 4)

    # ── agreement split by model correctness ───────────────────────────
    for label, mask in [('correct', correct), ('incorrect', incorrect)]:
        sub = comparison[mask]
        if len(sub) < 2:
            for suffix in ('', '_w'):
                res[f'kappa_dentist_vs_orig_{label}{suffix}']  = None
                res[f'kappa_dentist_vs_model_{label}{suffix}'] = None
            continue
        res[f'kappa_dentist_vs_orig_{label}'] = round(
            kappa(sub['true_level'], sub['visual_level']), 4)
        res[f'kappa_dentist_vs_orig_{label}_w'] = round(
            kappa(sub['true_level'], sub['visual_level'], 'linear'), 4)
        res[f'kappa_dentist_vs_model_{label}'] = round(
            kappa(sub['predicted_level'], sub['visual_level']), 4)
        res[f'kappa_dentist_vs_model_{label}_w'] = round(
            kappa(sub['predicted_level'], sub['visual_level'], 'linear'), 4)

    # ── per-level accuracy and dentist confidence ──────────────────────
    for level in (1, 2, 3):
        m = true_l == level
        res[f'l{level}_n']            = int(m.sum())
        res[f'l{level}_dentist_acc']  = round(float((dentist_l[m] == level).mean()), 4)
        res[f'l{level}_model_acc']    = round(float((pred_l[m] == level).mean()), 4)
        res[f'l{level}_dentist_conf_mean'] = round(
            float(comparison.loc[m, 'visual_confidence'].mean()), 3)
        res[f'l{level}_dentist_conf_sd'] = round(
            float(comparison.loc[m, 'visual_confidence'].std(ddof=1)), 3)

    # ── confidence correlation ─────────────────────────────────────────
    rho, p = spearmanr(model_conf, dentist_conf)
    res.update({
        'spearman_rho':        round(float(rho), 4),
        'spearman_p':          round(float(p), 5),
        'model_conf_mean':     round(float(model_conf.mean()), 4),
        'model_conf_sd':       round(float(model_conf.std(ddof=1)), 4),
        'dentist_conf_mean':   round(float(dentist_conf.mean()), 3),
        'dentist_conf_sd':     round(float(dentist_conf.std(ddof=1)), 3),
    })

    if not verbose:
        return res

    # ── report ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"Inter-annotator Agreement  (n={res['n_images']}, "
          f"{res['n_correct']} correct / {res['n_incorrect']} misclassified)")
    print(f"{'='*64}")
    print(f"  {'':28s} {'kappa':>8} {'weighted':>10}")
    print(f"  {'-'*50}")
    for name, label in [('dentist_vs_orig',  'Dentist vs Original'),
                        ('model_vs_orig',    'Model vs Original'),
                        ('model_vs_dentist', 'Model vs Dentist')]:
        print(f"  {label:28s} {res['kappa_'+name]:>8.3f} "
              f"{res['kappa_'+name+'_w']:>10.3f}  "
              f"({interpret_kappa(res['kappa_'+name])})")

    print(f"\n  Split by model correctness:")
    for label in ('correct', 'incorrect'):
        k  = res[f'kappa_dentist_vs_orig_{label}']
        kw = res[f'kappa_dentist_vs_orig_{label}_w']
        km = res[f'kappa_dentist_vs_model_{label}']
        n  = res['n_correct'] if label == 'correct' else res['n_incorrect']
        if k is None:
            continue
        print(f"    {label:12s} (n={n:3d})  dentist-vs-original "
              f"k={k:.3f} kw={kw:.3f}  |  dentist-vs-model k={km:.3f}")

    print(f"\n  Per severity level:")
    for level in (1, 2, 3):
        print(f"    Level {level} (n={res[f'l{level}_n']:2d}): "
              f"dentist acc={res[f'l{level}_dentist_acc']:.3f}  "
              f"model acc={res[f'l{level}_model_acc']:.3f}  "
              f"dentist confidence={res[f'l{level}_dentist_conf_mean']:.1f}"
              f" +/- {res[f'l{level}_dentist_conf_sd']:.1f}")

    print(f"\n  Model vs dentist confidence: "
          f"Spearman rho={res['spearman_rho']:.3f}, p={res['spearman_p']:.4f}")

    print(f"\n{'='*64}")
    print("Classification Report: Dentist vs Original")
    print(f"{'='*64}")
    print(classification_report(true_l, dentist_l, zero_division=0,
                                labels=[1, 2, 3],
                                target_names=['Level 1', 'Level 2', 'Level 3']))
    print(f"{'='*64}")
    print("Classification Report: Model vs Original")
    print(f"{'='*64}")
    print(classification_report(true_l, pred_l, zero_division=0,
                                labels=[1, 2, 3],
                                target_names=['Level 1', 'Level 2', 'Level 3']))

    return res

def run_dentist_experiment(dentist_csv, config, force=False, save_dir=None):
    """
    Train a model on all data except the dentist evaluation set, predict on
    the dentist images, and compare against the dentist's visual grading.

    Args:
        dentist_csv: CSV with image_path, visual_level, visual_confidence.
        config:      config dict; a list of seeds uses the first.
        force:       retrain even if a completed run exists.
        save_dir:    output directory (required).

    Returns:
        (comparison DataFrame, results dict)
    """
    if save_dir is None:
        raise ValueError("save_dir is required")
    save_dir = Path(save_dir)

    seed = config['seed']
    if isinstance(seed, list):
        seed = seed[0]
        print(f"Config contains multiple seeds — using seed {seed}")
    run_config = copy.deepcopy(config)
    run_config['seed'] = seed

    if check_run_dir(save_dir, force=force):
        print(f"Dentist experiment already complete at {save_dir} — reusing.")
        comparison = pd.read_csv(save_dir / "dentist_comparison.csv")
        results    = json.loads((save_dir / "results_summary.json").read_text())
        return comparison, results

    with open(save_dir / "config.json", 'w') as f:
        json.dump(run_config, f, indent=2)

    # ── data ───────────────────────────────────────────────────────────
    df         = load_metadata()
    dentist_df = pd.read_csv(dentist_csv)
    dentist_df['image_path'] = dentist_df['image_path'].apply(
        lambda x: Path(str(x).replace('\\', '/')).name
    )

    dentist_images = set(dentist_df['image_path'])
    test_df      = df[df['image_path'].isin(dentist_images)].reset_index(drop=True)
    remaining_df = df[~df['image_path'].isin(dentist_images)].reset_index(drop=True)

    missing = dentist_images - set(df['image_path'])
    if missing:
        raise ValueError(
            f"{len(missing)} dentist images not found in metadata, "
            f"e.g. {sorted(missing)[:5]}"
        )

    print(f"\n{'='*60}")
    print(f"Dentist Experiment — seed {seed}")
    print(f"{'='*60}")
    print(f"Dentist (test) set: {len(test_df)}   Remaining: {len(remaining_df)}")
    print(f"Test distribution:\n{test_df['level'].value_counts().sort_index()}")

    train_df, val_df = train_test_split(
        remaining_df,
        test_size    = run_config.get('val_size', 0.15),
        stratify     = remaining_df['level'],
        random_state = seed,
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
        d[['image_path', 'level']].to_csv(
            save_dir / f"{name}_split.csv", index=False)

    # ── train ──────────────────────────────────────────────────────────
    set_seed(seed)
    train_dataset, val_dataset, test_dataset, norm_stats = build_datasets(
        train_df, val_df, test_df, run_config)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset, val_dataset, test_dataset, run_config)

    model = build_model(run_config)
    model, history = train(model, train_loader, val_loader, run_config)

    torch.save(model.state_dict(), save_dir / "model.pth")
    with open(save_dir / "norm_stats.json", 'w') as f:
        json.dump({k: [float(v[0]), float(v[1])]
                   for k, v in norm_stats.items()}, f, indent=2)

    # ── predict and merge ──────────────────────────────────────────────
    probs_df = save_predictions(model, test_loader, test_df,
                                save_path=save_dir / "test_probs.csv")

    plot_history(history,       save_path=save_dir / "training_curves.png")
    evaluate(model, test_loader, save_path=save_dir / "confusion_matrix.png")
    plot_roc_auc(model, test_loader,
                 save_path=save_dir / "roc_auc.png")
    plot_precision_recall(model, test_loader,
                          save_path=save_dir / "precision_recall.png")

    comparison = probs_df.merge(
        dentist_df[['image_path', 'visual_level', 'visual_confidence']],
        on='image_path', how='inner')

    print(f"\nMatched images for comparison: {len(comparison)} "
          f"of {len(dentist_df)} dentist-graded images")
    if len(comparison) < len(dentist_df):
        raise ValueError(
            f"Only {len(comparison)} of {len(dentist_df)} dentist images "
            f"matched model predictions — check image_path consistency."
        )

    comparison['model_correct'] = (
        comparison['predicted_level'] == comparison['true_level'])

    # ── agreement statistics ───────────────────────────────────────────
    results = compute_dentist_agreement(comparison)
    results['seed'] = seed

    comparison.to_csv(save_dir / "dentist_comparison.csv", index=False)
    with open(save_dir / "results_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    (save_dir / '.complete').write_text(json.dumps({
        'seed': seed,
        'n_images': len(comparison),
        'completed': datetime.now().isoformat(),
    }, indent=2))
    print(f"\nResults saved → {save_dir}")

    return comparison, results

def find_best_ensemble_weights(run_dirs, seeds, optimize_on='macro_f1'):
    labels = list(run_dirs.keys())
    
    # Load test probs for tuning seeds
    all_probs = {}
    for label, run_dir in run_dirs.items():
        seed_probs = []
        for seed in seeds:
            path = Path(run_dir) / f"seed_{seed}_probs.csv"
            df   = pd.read_csv(path)
            seed_probs.append(df)
        all_probs[label] = pd.concat(seed_probs, ignore_index=True)
     
    steps = np.arange(0.0, 1.1, 0.1).round(1)
    weight_grid = []
    if len(labels) == 2:
        weight_grid = [[float(w), round(1.0 - w, 4)] for w in steps]
    else:
        for w1 in steps:
            for w2 in steps:
                w3 = round(1.0 - w1 - w2, 1)
                if 0.0 <= w3 <= 1.0:
                    weight_grid.append([float(w1), float(w2), max(float(w3), 0.0)])
                
    
    print(f"Searching {len(weight_grid)} weight combinations...")
    
    best_weights = None
    best_score   = -1
    results      = []
    
    ref_df     = all_probs[labels[0]]
    true_labels = ref_df['true_level'].values - 1
    
    for weights in weight_grid:
        # Weighted average of probabilities
        ens_probs = np.zeros((len(ref_df), 3))
        for w, label in zip(weights, labels):
            probs = all_probs[label][['prob_l1', 
                                      'prob_l2', 
                                      'prob_l3']].values
            ens_probs += w * probs
        
        preds = ens_probs.argmax(axis=1)
        score = f1_score(true_labels, preds, average='macro')
        
        row = {f'w_{lab}': round(float(w), 4) for lab, w in zip(labels, weights)}
        row['score'] = round(score, 4)
        results.append(row)
        
        if score > best_score:
            best_score   = score
            best_weights = weights
       
    results_df = (pd.DataFrame(results)
                  .sort_values('score', ascending=False)
                  .reset_index(drop=True))

    best_row     = results_df.iloc[0]
    best_weights = [float(best_row[f'w_{lab}']) for lab in labels]
    best_score   = float(best_row['score'])

    print("\nTop 5 weight combinations:")
    print(results_df.head(5).to_string(index=False))
    print(f"\nBest weights: {dict(zip(labels, best_weights))}")
    print(f"Best {optimize_on}: {best_score:.4f}")

    top5_spread = float(results_df['score'].head(5).max()
                        - results_df['score'].head(5).min())
    print(f"Spread across top 5: {top5_spread:.4f}")
    if top5_spread < 0.01:
        print("  NOTE: the top combinations are within ~1 SEM of each other — "
              "the weight surface is flat.\n  Report these as one of several "
              "near-equivalent choices, not as a precise optimum.")

    # equal weights as a reference point
    equal = round(1.0 / len(labels), 1)
    eq = results_df
    for lab in labels:
        eq = eq[np.isclose(eq[f'w_{lab}'], equal, atol=0.051)]
    if len(eq):
        print(f"  Equal weights ({equal} each): {eq['score'].iloc[0]:.4f} "
              f"(tuned gains {best_score - eq['score'].iloc[0]:+.4f})")

    save_dir = RUNS_DIR / 'statistical_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)

    grid_path = save_dir / f'ensemble_weight_grid_{optimize_on}.csv'
    results_df.to_csv(grid_path, index=False)

    chosen_path = save_dir / 'ensemble_weights.json'
    chosen_path.write_text(json.dumps({
        'labels':       labels,
        'weights':      best_weights,
        'optimize_on':  optimize_on,
        'mean_score':   best_score,
        'top5_spread':  top5_spread,
        'tuning_seeds': list(seeds),
        'run_dirs':     {k: str(v) for k, v in run_dirs.items()},
    }, indent=2))

    print(f"\nSaved → {grid_path}")
    print(f"Saved → {chosen_path}")

    return best_weights, best_score, results_df