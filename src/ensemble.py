"""
ensemble.py — Soft ensemble of saved models across seeds.
Loads saved model weights per seed and averages softmax probabilities.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RUNS_DIR
from model import build_model
from dataset import build_datasets
from utils import set_seed, split_data
from dataset import load_metadata

def load_model_from_run(run_dir, seed, config):
    """Load a saved model from a run directory for a specific seed."""
    model_path = run_dir / f"seed_{seed}_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = build_model(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_probs(model, test_loader, device):
    """Get softmax probabilities from a single model."""
    model = model.to(device)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, meta, labels in test_loader:
            images, meta = images.to(device), meta.to(device)
            outputs = model(images, meta)
            probs   = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)

def run_ensemble(run_dirs, weights=None, seeds=None, save_dir=None):
    """
    Soft ensemble across multiple run directories.
    
    Args:
        run_dirs:  dict of {label: run_dir_path} — e.g.
                   {'multimodal': 'runs/multimodal_run',
                    'image_only': 'runs/image_only_run',
                    'meta_only':  'runs/meta_only_run'}
        weights:   optional list of weights per run_dir — must sum to 1.
                   If None, simple average is used.
        seeds:     list of seeds to ensemble — must exist in all run_dirs.
                   If None, uses seeds found in first run_dir's results.csv
        save_dir:  optional Path to save plots and results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_dirs  = {k: Path(v) for k, v in run_dirs.items()}
    labels    = list(run_dirs.keys())
    n_models  = len(labels)

    if weights is None:
        weights = [1.0 / n_models] * n_models
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1"
    assert len(weights) == n_models, "One weight per run_dir required"

    # Load config from first run_dir
    config_path = list(run_dirs.values())[0] / "config.json"
    with open(config_path) as f:
        base_config = json.load(f)

    # Get seeds from results.csv if not provided
    if seeds is None:
        results_path = list(run_dirs.values())[0] / "results.csv"
        seeds = pd.read_csv(results_path)['seed'].tolist()
    print(f"Ensembling seeds: {seeds}")
    print(f"Models: {labels}")
    print(f"Weights: {dict(zip(labels, weights))}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)



    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        set_seed(seed)

        # Rebuild the same split used during training
        seed_config        = base_config.copy()
        seed_config['seed'] = seed
        df = load_metadata()
        train_df, val_df, test_df = split_data(df, seed_config)

        # Build test loader — use multimodal dataset for full features
        seed_config['fusion_mode'] = 'multimodal'
        _, _, test_dataset, norm_stats = build_datasets(
            train_df, val_df, test_df, seed_config
        )

        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=seed_config['batch_size'],
            shuffle=False,
            num_workers=seed_config.get('num_workers', 2),
            pin_memory=True,
            persistent_workers=True
        )

        # Collect weighted probabilities from each model
        ensemble_probs = None
        all_labels     = None

        for label, run_dir, weight in zip(labels, run_dirs.values(), weights):
            # Load config specific to this run
            cfg_path = run_dir / "config.json"
            with open(cfg_path) as f:
                run_config = json.load(f)
            run_config['seed'] = seed

            # Build correct test loader for this fusion mode
            _, _, test_ds, _ = build_datasets(
                train_df, val_df, test_df, run_config
            )
            tl = DataLoader(
                test_ds,
                batch_size=run_config['batch_size'],
                shuffle=False,
                num_workers=run_config.get('num_workers', 2),
                pin_memory=True,
                persistent_workers=True
            )

            model = load_model_from_run(run_dir, seed, run_config)
            probs, labels_arr = get_probs(model, tl, device)

            print(f"  {label} — probs shape: {probs.shape}  "
                  f"weight: {weight:.3f}")

            if ensemble_probs is None:
                ensemble_probs = weight * probs
                all_labels     = labels_arr
            else:
                ensemble_probs += weight * probs

        # Final predictions from averaged probabilities
        preds  = ensemble_probs.argmax(axis=1)
        y_bin  = label_binarize(all_labels, classes=[0, 1, 2])
        report = classification_report(
            all_labels, preds, output_dict=True,
            target_names=["Level 1", "Level 2", "Level 3"]
        )

        print(classification_report(
            all_labels, preds,
            target_names=["Level 1", "Level 2", "Level 3"]
        ))

        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), ensemble_probs.ravel())
        all_fpr  = np.linspace(0, 1, 200)
        mean_tpr = np.mean([
            np.interp(all_fpr, *roc_curve(y_bin[:, i], ensemble_probs[:, i])[:2])
            for i in range(3)
        ], axis=0)

        row = {
            'seed':          seed,
            'fusion_mode':   'ensemble_' + '_'.join(labels),
            'weights':       str(dict(zip(labels, weights))),
            'backbone_name': base_config['backbone_name'],

            'accuracy':        report['accuracy'],
            'macro_f1':        report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall':    report['macro avg']['recall'],
            'micro_auc':       auc(fpr_micro, tpr_micro),
            'macro_auc':       auc(all_fpr, mean_tpr),
            'micro_ap':        average_precision_score(y_bin, ensemble_probs, average='micro'),
            'macro_ap':        average_precision_score(y_bin, ensemble_probs, average='macro'),

            'l1_precision': report['Level 1']['precision'],
            'l1_recall':    report['Level 1']['recall'],
            'l1_f1':        report['Level 1']['f1-score'],
            'l1_ap':        average_precision_score(y_bin[:, 0], ensemble_probs[:, 0]),

            'l2_precision': report['Level 2']['precision'],
            'l2_recall':    report['Level 2']['recall'],
            'l2_f1':        report['Level 2']['f1-score'],
            'l2_ap':        average_precision_score(y_bin[:, 1], ensemble_probs[:, 1]),

            'l3_precision': report['Level 3']['precision'],
            'l3_recall':    report['Level 3']['recall'],
            'l3_f1':        report['Level 3']['f1-score'],
            'l3_ap':        average_precision_score(y_bin[:, 2], ensemble_probs[:, 2]),
        }

        all_results.append(row)
        print(f"Seed {seed} done — acc: {row['accuracy']:.4f}  "
              f"macro_f1: {row['macro_f1']:.4f}  "
              f"macro_auc: {row['macro_auc']:.4f}")

    results_df = pd.DataFrame(all_results)

    # Summary
    metric_cols = ['accuracy', 'macro_f1', 'macro_auc', 'macro_ap',
                   'l1_f1', 'l2_f1', 'l3_f1',
                   'l1_recall', 'l2_recall', 'l3_recall']
    print(f"\n{'='*60}")
    print("Ensemble summary across seeds:")
    print(f"{'='*60}")
    print(results_df[metric_cols].agg(['mean', 'std']).round(4))

    if save_dir:
        results_df.to_csv(save_dir / "ensemble_results.csv", index=False)
        print(f"\nResults saved → {save_dir / 'ensemble_results.csv'}")

        # Confusion matrix for last seed as representative
        cm = confusion_matrix(all_labels, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Level 1", "Level 2", "Level 3"],
                    yticklabels=["Level 1", "Level 2", "Level 3"])
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Ensemble Confusion Matrix")
        plt.savefig(save_dir / "ensemble_confusion_matrix.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    return results_df


if __name__ == "__main__":
    # Simple average ensemble
    
    '''
    results = run_ensemble(
        run_dirs={
            'multimodal': 'D:/research/brar-classification/runs/multi_dens_20_20260416_174306',
            'image_only': 'D:/research/brar-classification/runs/image_dens_20_20260416_145552',
            'meta_only':  'D:/research/brar-classification/runs/meta_only_20_20260416_033922',
        },
        weights=None,   # None = equal weights [1/3, 1/3, 1/3]
        seeds=None,     # None = read from first run's results.csv
        save_dir='runs/ensemble_results'
    )
    '''
    
    # Or weighted ensemble — giving multimodal more weight
    results_weighted = run_ensemble(
        run_dirs={
            'multimodal': '../runs/multi_dens_17_20260417_222055',
            'image_only': '../runs/image_dens_20_20260418_010249',
            'meta_only':  '../runs/meta_only_20_20260418_033603',
        },
        #weights=[0.6, 0.3, 0.1],  # multimodal gets most weight
        weights=[0.6, 0.4, 0],
        seeds=None,
        #save_dir='runs/ensemble_weighted_results'
        save_dir=f'{RUNS_DIR}/ensemble_2model'
        
    )
