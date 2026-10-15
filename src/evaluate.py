from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
import pandas as pd
from pytorch_grad_cam import GradCAM
from pathlib import Path
import torch.nn as nn
from src.transforms import get_transforms
from PIL import Image


def get_predictions(model, test_loader):
    """Shared inference — call once, reuse in all eval functions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, meta, labels in test_loader:
            images, meta = images.to(device), meta.to(device)
            outputs = model(images, meta)
            probs   = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    return (np.concatenate(all_probs, axis=0),
            np.array(all_preds),
            np.array(all_labels))

def evaluate_metrics(model, test_loader):
    probs, preds, labels = get_predictions(model, test_loader)
    y_bin = label_binarize(labels, classes=[0, 1, 2])
    return {
        "accuracy":  accuracy_score(labels, preds),
        "macro_f1":  f1_score(labels, preds, average="macro"),
        "macro_auc": roc_auc_score(y_bin, probs, average="macro", multi_class="ovr"),
        "macro_ap":  average_precision_score(y_bin, probs, average="macro"),
    }

def evaluate(model, test_loader, save_path=None):
    probs, preds, labels = get_predictions(model, test_loader)
    print(classification_report(labels, preds,
                                target_names=["Level 1", "Level 2", "Level 3"]))
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Level 1", "Level 2", "Level 3"],
                yticklabels=["Level 1", "Level 2", "Level 3"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        
    #plt.show()
    plt.close('all')

def plot_history(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    boundaries = history.get("stage_boundaries", [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (train_key, val_key), ylabel in [
        (ax1, ("train_loss", "val_loss"), "Loss"),
        (ax2, ("train_acc",  "val_acc"),  "Accuracy"),
    ]:
        ax.plot(epochs, history[train_key], label="Train",      color="#2196F3")
        ax.plot(epochs, history[val_key],   label="Validation", color="#FF5722", linestyle="--")
        for i, b in enumerate(boundaries[:-1]):
            ax.axvline(x=b, color="gray", linestyle=":", alpha=0.7)
            ax.text(b + 0.3, ax.get_ylim()[1] * 0.97, f"S{i+2}", fontsize=9, color="gray")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    #plt.show()
    plt.close('all')

def plot_roc_auc(model, test_loader, save_path = None):
    probs, _, labels = get_predictions(model, test_loader)
    classes     = [0, 1, 2]
    class_names = ["Level 1", "Level 2", "Level 3"]
    y_bin       = label_binarize(labels, classes=classes)
    colors      = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One vs Rest"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="#9C27B0", lw=2, linestyle="--",
            label=f"Micro-avg (AUC = {auc_micro:.3f})")
    all_fpr  = np.linspace(0, 1, 200)
    mean_tpr = np.mean([np.interp(all_fpr, *roc_curve(y_bin[:,i], probs[:,i])[:2])
                        for i in range(3)], axis=0)
    auc_macro = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="#FF9800", lw=2, linestyle="-.",
            label=f"Macro-avg (AUC = {auc_macro:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Averaged"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #plt.show()
    plt.close('all')
    
    print(f"\nMicro-avg AUC: {auc_micro:.4f}")
    print(f"Macro-avg AUC: {auc_macro:.4f}")
        
def plot_precision_recall(model, test_loader, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    probs, _, labels = get_predictions(model, test_loader)  # reuse shared inference
    all_probs  = probs
    all_labels = labels

    classes     = [0, 1, 2]
    class_names = ["Level 1", "Level 2", "Level 3"]
    y_bin       = label_binarize(all_labels, classes=classes)
    colors      = ["#2196F3", "#FF5722", "#4CAF50"]

    # Class support for baseline reference lines
    support = y_bin.sum(axis=0) / len(all_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: per-class PR curves ───────────────────────────────────────────
    ax = axes[0]
    for i, (name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AP = {ap:.3f})")
        # Baseline: random classifier = class prevalence
        ax.axhline(y=support[i], color=color, lw=1, linestyle=":", alpha=0.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — One vs Rest")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    # ── Right: macro & micro average PR ────────────────────────────────────
    ax = axes[1]

    # Micro average
    ap_micro = average_precision_score(y_bin, all_probs, average="micro")
    p_micro, r_micro, _ = precision_recall_curve(y_bin.ravel(), all_probs.ravel())
    ax.plot(r_micro, p_micro, color="#9C27B0", lw=2, linestyle="--",
            label=f"Micro-avg (AP = {ap_micro:.3f})")

    # Macro average — interpolate onto common recall grid
    all_recall = np.linspace(0, 1, 200)
    mean_prec  = np.zeros_like(all_recall)
    for i in range(len(classes)):
        p_i, r_i, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
        # precision_recall_curve returns decreasing recall — flip for interp
        mean_prec += np.interp(all_recall, r_i[::-1], p_i[::-1])
    mean_prec /= len(classes)
    ap_macro = average_precision_score(y_bin, all_probs, average="macro")
    ax.plot(all_recall, mean_prec, color="#FF9800", lw=2, linestyle="-.",
            label=f"Macro-avg (AP = {ap_macro:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Averaged")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.suptitle("Precision-Recall Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    #plt.show()
    plt.close('all')
    
    print(f"\nPer-class Average Precision:")
    for i, name in enumerate(class_names):
        ap = average_precision_score(y_bin[:, i], all_probs[:, i])
        print(f"  {name}: AP = {ap:.4f}  (baseline = {support[i]:.3f})")
    print(f"\nMicro-avg AP: {ap_micro:.4f}")
    print(f"Macro-avg AP: {ap_macro:.4f}")

def get_misclassified(probs_df, sort_by='true_level', save_path=None):
    """
    Filter and report misclassified samples from probs_df.

    Args:
        probs_df:  DataFrame returned by save_predictions
        sort_by:   'true_level'  — group errors by severity
                   'confidence'  — most confidently wrong first
                   'margin'      — most ambiguous first
        save_path: optional Path to save CSV

    Returns:
        DataFrame of misclassified samples
    """
    df_errors = probs_df[probs_df['correct'] == False].copy()

    if df_errors.empty:
        print("No misclassified samples found.")
        return df_errors

    # Add confidence and margin columns
    prob_cols = ['prob_l1', 'prob_l2', 'prob_l3']

    df_errors['confidence'] = df_errors.apply(
        lambda row: row[f"prob_l{int(row['predicted_level'])}"], axis=1
    )
    df_errors['true_class_prob'] = df_errors.apply(
        lambda row: row[f"prob_l{int(row['true_level'])}"], axis=1
    )
    df_errors['margin'] = (df_errors['confidence'] -
                           df_errors['true_class_prob']).round(4)

    # Sort
    if sort_by == 'true_level':
        df_errors = df_errors.sort_values('true_level', ascending=True)
    elif sort_by == 'confidence':
        df_errors = df_errors.sort_values('confidence', ascending=False)
    elif sort_by == 'margin':
        df_errors = df_errors.sort_values('margin', ascending=True)
    else:
        raise ValueError(f"sort_by must be 'true_level', 'confidence', or 'margin'. "
                         f"Got: {sort_by}")

    df_errors = df_errors.reset_index(drop=True)

    # Print summary
    total = len(probs_df)
    n_errors = len(df_errors)
    print(f"\n{'='*70}")
    print(f"Misclassified: {n_errors} / {total} ({100*n_errors/total:.1f}%)")
    print(f"Sorted by: {sort_by}")
    print(f"{'='*70}")

    print("\nError breakdown:")
    breakdown = df_errors.groupby(['true_level', 'predicted_level']).size()
    for (true, pred), count in breakdown.items():
        print(f"  True L{true} → Predicted L{pred}: {count} cases")

    print(f"\n{'='*70}")
    display_cols = ['image_path', 'true_level', 'predicted_level',
                    'confidence', 'true_class_prob', 'margin']
    #print(df_errors[display_cols].to_string(index=True))

    if save_path:
        df_errors.to_csv(save_path, index=False)
        print(f"\nMisclassified saved → {save_path}")

    return df_errors

class _GradCAMWrapper(nn.Module):
    """Wraps multimodal model to accept single image input for GradCAM."""
    def __init__(self, model, meta):
        super().__init__()
        self.model = model
        self.meta  = meta

    def forward(self, image):
        return self.model(image, self.meta)

def _get_target_layer(model):
    """Returns the correct GradCAM target layer based on backbone."""
    backbone = model.backbone_name
    if backbone == 'densenet121':
        return model.image_branch.features.denseblock4.denselayer16.conv2
    elif backbone.startswith('efficientnet'):
        return model.image_branch.features[-2][-1]
    else:
        raise ValueError(f"No target layer defined for backbone: {backbone}")

def select_gradcam_images(probs_df, n_correct=1, n_misclassified=1,
                          image_ids=None):
    """
    Select images for GradCAM visualization.

    Args:
        probs_df:        DataFrame from save_predictions
        n_correct:       number of correct predictions to select per level
        n_misclassified: number of misclassified cases to select per boundary
        image_ids:       optional list of specific image_path values to use
                         — overrides auto-selection if provided

    Returns:
        DataFrame of selected images with a 'selection_reason' column
    """
    if image_ids is not None:
        selected = probs_df[probs_df['image_path'].isin(image_ids)].copy()
        selected['selection_reason'] = 'manual'
        return selected.reset_index(drop=True)


    # chamge above block to this to print block to append the reason to 
    # the filename 
    
    #if image_ids is not None:
    #    selected = probs_df[probs_df['image_path'].isin(image_ids)].copy()
    #    selected['selection_reason'] = selected.apply(
    #        lambda r: (f"correct_L{int(r['true_level'])}"
    #                   if r['true_level'] == r['predicted_level']
    #                   else f"misclassified_L{int(r['true_level'])}"
    #                        f"_as_L{int(r['predicted_level'])}"),
    #        axis=1)
    #    return selected.reset_index(drop=True)


    selected_rows = []

    # Correct predictions — highest confidence per level
    for level in [1, 2, 3]:
        candidates = probs_df[
            (probs_df['true_level'] == level) &
            (probs_df['correct'] == True)
        ].sort_values(f'prob_l{level}', ascending=False)

        for _, row in candidates.head(n_correct).iterrows():
            r = row.to_dict()
            r['selection_reason'] = f'correct_L{level}'
            selected_rows.append(r)

    # Misclassified — clinically important boundaries only
    boundaries = [
        (3, 2, 'misclassified_L3_as_L2'),   # severe → moderate (most critical)
        (2, 3, 'misclassified_L2_as_L3'),   # moderate → severe
        (1, 2, 'misclassified_L1_as_L2'),   # mild → moderate
        (2, 1, 'misclassified_L2_as_L1'),   # moderate → mild
        (1, 3, 'misclassified_L1_as_L3'),   # 
        (3, 1, 'misclassified_L3_as_L1'),   # 
    ]

    for true_level, pred_level, reason in boundaries:
        candidates = probs_df[
            (probs_df['true_level']      == true_level) &
            (probs_df['predicted_level'] == pred_level)
        ].sort_values(f'prob_l{pred_level}', ascending=False)

        if candidates.empty:
            print(f"No cases found for {reason} — skipping")
            continue

        for _, row in candidates.head(n_misclassified).iterrows():
            r = row.to_dict()
            r['selection_reason'] = reason
            selected_rows.append(r)

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    print(f"\nSelected {len(selected_df)} images for GradCAM:")
    
    for _, row in selected_df.iterrows():
        print(f"  {row['image_path']:30s} "
              f"True: L{int(row['true_level'])}  "
              f"Pred: L{int(row['predicted_level'])}  "
              f"Reason: {row['selection_reason']}")

    return selected_df

def plot_gradcam1(model, selected_df, config, norm_stats, save_dir=None):
    """
    Generate GradCAM visualizations for selected images.
    GradCAM is overlaid on the original full-resolution image.

    Args:
        model:       trained model
        selected_df: DataFrame from select_gradcam_images
        config:      config dict — for transforms and DATA_DIR
        norm_stats:  normalization stats from training — {col: (min, max)}
        save_dir:    optional Path to save figures
    """
    from src.config import DATA_DIR

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = model.to(device).eval()
    transform    = get_transforms(config, train=False)

    meta_cols      = ["age", "gender", "number_of_missing_teeth",
                      "implant", "residual_root", "functional_tooth_logarithm"]
    normalize_cols = ["age", "number_of_missing_teeth", "implant",
                      "residual_root", "functional_tooth_logarithm"]

    for _, row in selected_df.iterrows():
        image_path = (DATA_DIR / 'manual_cropped_dataset' /
                      f"level_{int(row['true_level'])}" /
                      row['image_path'])

        if not image_path.exists():
            print(f"Image not found: {image_path} — skipping")
            continue

        # Load original full-resolution image
        pil_original = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_original.size

        # Transform for model input (224x224)
        img_tensor = transform(pil_original).unsqueeze(0).to(device)

        # Build normalized meta tensor
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

        # Wrap model for GradCAM
        wrapper = _GradCAMWrapper(model, meta_tensor)
        for param in wrapper.model.image_branch.parameters():
            param.requires_grad = True

        target_layer = _get_target_layer(wrapper.model)

        if not any(module is target_layer
                   for _, module in wrapper.named_modules()):
            raise RuntimeError(
                f"Target layer not found in wrapper — check backbone: "
                f"{model.backbone_name}"
            )

        # Target class — predicted level (0-indexed)
        pred_class = int(row['predicted_level']) - 1
        targets    = [ClassifierOutputTarget(pred_class)]

        # Generate GradCAM
        with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        # Handle shape
        grayscale_cam = np.array(grayscale_cam)
        if grayscale_cam.ndim == 3:
            grayscale_cam = grayscale_cam[0]
        assert grayscale_cam.ndim == 2, \
            f"Expected 2D grayscale_cam, got shape {grayscale_cam.shape}"

        # Resize GradCAM from 224x224 to original resolution using PIL
        grayscale_cam_pil  = Image.fromarray(
            (grayscale_cam * 255).astype(np.uint8)
        )
        grayscale_cam_orig = np.array(
            grayscale_cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0

        # Original image as float32 [0,1] — brightened for display
        img_original_np = np.array(pil_original).astype(np.float32) / 255.0
        img_display     = np.clip(img_original_np * 1.5, 0, 1)

        # Overlay GradCAM on original resolution image
        #cam_overlay = show_cam_on_image(img_display, grayscale_cam_orig,
        #                                use_rgb=True)

        import cv2
        cam_overlay = show_cam_on_image(img_display, grayscale_cam_orig,
                                        use_rgb=True, colormap=cv2.COLORMAP_INFERNO)

        # Plot — proportional figsize to preserve aspect ratio
        aspect = orig_w / orig_h
        fig_h  = 5
        fig_w  = fig_h * aspect * 2  # two panels

        reason = row.get('selection_reason', '')
        conf   = row.get(f"prob_l{int(row['predicted_level'])}", 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

        axes[0].imshow(img_display)
        axes[0].set_title(f"Original\nTrue: L{int(row['true_level'])}")
        axes[0].axis('off')

        axes[1].imshow(cam_overlay)
        axes[1].set_title(
            f"GradCAM Overlay\nPred: L{int(row['predicted_level'])}"
        )
        axes[1].axis('off')

        plt.suptitle(
            f"{Path(row['image_path']).stem} | {reason} | "
            f"Confidence: {float(conf):.3f}",
            fontsize=11
        )
        plt.tight_layout()

        if save_dir:
            basefile = Path(row['image_path']).stem
            fname    = f"{basefile}_{reason}.png"
            plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
            print(f"Saved → {save_dir / fname}")

        plt.close('all')
        
def plot_gradcam(model, selected_df, config, norm_stats, save_dir=None,
                 overlay_only=True, show_titles=True):
    """
    Generate GradCAM visualizations for selected images.
    GradCAM is overlaid on the original full-resolution image.

    Args:
        model:        trained model
        selected_df:  DataFrame from select_gradcam_images
        config:       config dict — for transforms and DATA_DIR
        norm_stats:   normalization stats from training — {col: (min, max)}
        save_dir:     optional Path to save figures
        overlay_only: if True, save only the GradCAM overlay panel (no
                      side-by-side original). Useful when assembling the
                      panels into a grid figure.
        show_titles:  if False, omit all titles and padding — produces a bare
                      image suitable for tiling. Ignored when overlay_only
                      is False.
    """
    from src.config import DATA_DIR

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = model.to(device).eval()
    transform    = get_transforms(config, train=False)

    meta_cols      = config.get('metadata_cols', [
        "age", "gender", "number_of_missing_teeth",
        "implant", "residual_root", "functional_tooth_logarithm"])
    normalize_cols = config.get('normalize_cols', [
        "age", "number_of_missing_teeth", "implant",
        "residual_root", "functional_tooth_logarithm"])

    for _, row in selected_df.iterrows():
        image_path = (DATA_DIR / 'manual_cropped_dataset' /
                      f"level_{int(row['true_level'])}" /
                      row['image_path'])

        if not image_path.exists():
            print(f"Image not found: {image_path} — skipping")
            continue

        # Load original full-resolution image
        pil_original = Image.open(image_path).convert('RGB')
        orig_w, orig_h = pil_original.size

        # Transform for model input (224x224)
        img_tensor = transform(pil_original).unsqueeze(0).to(device)

        # Build normalized meta tensor
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

        # Wrap model for GradCAM
        wrapper = _GradCAMWrapper(model, meta_tensor)
        for param in wrapper.model.image_branch.parameters():
            param.requires_grad = True

        target_layer = _get_target_layer(wrapper.model)

        if not any(module is target_layer
                   for _, module in wrapper.named_modules()):
            raise RuntimeError(
                f"Target layer not found in wrapper — check backbone: "
                f"{model.backbone_name}"
            )

        # Target class — predicted level (0-indexed)
        pred_class = int(row['predicted_level']) - 1
        targets    = [ClassifierOutputTarget(pred_class)]

        # Generate GradCAM
        with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

        # Handle shape
        grayscale_cam = np.array(grayscale_cam)
        if grayscale_cam.ndim == 3:
            grayscale_cam = grayscale_cam[0]
        assert grayscale_cam.ndim == 2, \
            f"Expected 2D grayscale_cam, got shape {grayscale_cam.shape}"

        # Resize GradCAM from 224x224 to original resolution using PIL
        grayscale_cam_pil  = Image.fromarray(
            (grayscale_cam * 255).astype(np.uint8)
        )
        grayscale_cam_orig = np.array(
            grayscale_cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
        ) / 255.0

        # Original image as float32 [0,1] — brightened for display
        img_original_np = np.array(pil_original).astype(np.float32) / 255.0
        img_display     = np.clip(img_original_np * 1.5, 0, 1)

        # Overlay GradCAM on original resolution image
        #cam_overlay = show_cam_on_image(img_display, grayscale_cam_orig,
        #                                use_rgb=True)

        cam_overlay = show_cam_on_image(img_display, grayscale_cam_orig,
                                        use_rgb=True, 
                                        image_weight=0.70)
                
        reason = row.get('selection_reason', '')
        conf   = row.get(f"prob_l{int(row['predicted_level'])}", 0.0)
        aspect = orig_w / orig_h
        fig_h  = 5  # was 5

        if overlay_only:
            fig, ax = plt.subplots(figsize=(fig_h * aspect, fig_h))
            ax.imshow(cam_overlay)
            ax.axis('off')

            if show_titles:
                ax.set_title(
                    f"True: L{int(row['true_level'])}  |  "
                    f"Pred: L{int(row['predicted_level'])}  |  "
                    f"Confidence: {float(conf):.3f}",
                    fontsize=11
                )
                plt.tight_layout()
                savefig_kwargs = dict(dpi=300, bbox_inches='tight')
            else:
                # bare image — no margins, no padding, for tiling
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                savefig_kwargs = dict(dpi=300, bbox_inches='tight',
                                      pad_inches=0)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(fig_h * aspect * 2, fig_h))

            axes[0].imshow(img_display)
            axes[0].set_title(f"Original | Reference: L{int(row['true_level'])}",
                              fontsize=22)
            axes[0].axis('off')

            axes[1].imshow(cam_overlay)
            
            d = abs(int(row['predicted_level']) - int(row['true_level']))
            category = ['Correct', 'One-grade error', 'Two-grade error'][d]
            
            '''
            axes[1].set_title(
                f"GradCAM | Predicted: L{int(row['predicted_level'])} | "
                f"Confidence: {float(conf):.2f}", fontsize=22)
            '''
            
            axes[1].set_title(
                f"Predicted: L{int(row['predicted_level'])} | {category} | "
                f"Confidence: {float(conf):.2f}", fontsize=22)
            
            axes[1].axis('off')

            plt.tight_layout()
            savefig_kwargs = dict(dpi=300, bbox_inches='tight')
            '''
            fig, axes = plt.subplots(1, 2, figsize=(fig_h * aspect * 2, fig_h))

            axes[0].imshow(img_display)
            axes[0].set_title(f"Original | Reference: L{int(row['true_level'])}",
                              fontsize=8)
            axes[0].axis('off')

            axes[1].imshow(cam_overlay)
            axes[1].set_title(
                f"GradCAM | Pred: L{int(row['predicted_level'])} | "
                f"Confidence: {float(conf):.2f}", fontsize=8)
            axes[1].axis('off')

            fig.subplots_adjust(left=0.0, right=1.0, top=0.88,
                                bottom=0.0, wspace=0.01)
            savefig_kwargs = dict(dpi=300, bbox_inches='tight')
            '''
            
            '''
            fig, axes = plt.subplots(1, 2, figsize=(fig_h * aspect * 2, fig_h))

            axes[0].imshow(img_display)
            axes[0].set_title(f"Original\nTrue: L{int(row['true_level'])}")
            axes[0].axis('off')

            axes[1].imshow(cam_overlay)
            axes[1].set_title(
                f"GradCAM Overlay\nPred: L{int(row['predicted_level'])}"
            )
            axes[1].axis('off')

            plt.suptitle(
                f"{Path(row['image_path']).stem} | {reason} | "
                f"Confidence: {float(conf):.3f}",
                fontsize=11
            )
            plt.tight_layout()
            savefig_kwargs = dict(dpi=300, bbox_inches='tight')
            '''
            
        if save_dir:
            basefile = Path(row['image_path']).stem
            suffix   = '_overlay' if overlay_only else ''
            fname    = f"{basefile}_{reason}{suffix}.png"
            plt.savefig(Path(save_dir) / fname, **savefig_kwargs)
            print(f"Saved → {Path(save_dir) / fname}")

        plt.close('all')
             
def print_results_table(results_path, title=None):
    """
    Print a formatted performance table from results CSV.
    Shows mean ± std for all metrics across seeds.

    Args:
        results_path: path to results.csv
        title:        optional title for the table
    """
    df = pd.read_csv(results_path)

    # Metric groups for organized display
    metric_groups = {
        'Overall': [
            'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
            'macro_auc', 'micro_auc', 'macro_ap', 'micro_ap'
        ],
        'Level 1 (Mild)': [
            'l1_precision', 'l1_recall', 'l1_f1', 'l1_ap'
        ],
        'Level 2 (Moderate)': [
            'l2_precision', 'l2_recall', 'l2_f1', 'l2_ap'
        ],
        'Level 3 (Severe)': [
            'l3_precision', 'l3_recall', 'l3_f1', 'l3_ap'
        ],
    }

    # Pretty metric names
    metric_names = {
        'accuracy':        'Accuracy',
        'macro_f1':        'Macro F1',
        'macro_precision': 'Macro Precision',
        'macro_recall':    'Macro Recall',
        'macro_auc':       'Macro AUC-ROC',
        'micro_auc':       'Micro AUC-ROC',
        'macro_ap':        'Macro AP',
        'micro_ap':        'Micro AP',
        'l1_precision':    'Precision',
        'l1_recall':       'Recall',
        'l1_f1':           'F1-Score',
        'l1_ap':           'AP',
        'l2_precision':    'Precision',
        'l2_recall':       'Recall',
        'l2_f1':           'F1-Score',
        'l2_ap':           'AP',
        'l3_precision':    'Precision',
        'l3_recall':       'Recall',
        'l3_f1':           'F1-Score',
        'l3_ap':           'AP',
    }

    # Header
    n_seeds = len(df)
    backbone = df['backbone_name'].iloc[0] if 'backbone_name' in df.columns else ''
    fusion   = df['fusion_mode'].iloc[0]   if 'fusion_mode'   in df.columns else ''

    print(f"\n{'='*65}")
    if title:
        print(f"  {title}")
    else:
        print(f"  Results: {fusion} | {backbone} | {n_seeds} seeds")
    print(f"{'='*65}")
    print(f"  {'Metric':<22} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for group, metrics in metric_groups.items():
        print(f"\n  [{group}]")
        for metric in metrics:
            if metric not in df.columns:
                continue
            mean = df[metric].mean()
            std  = df[metric].std()
            mn   = df[metric].min()
            mx   = df[metric].max()
            name = metric_names.get(metric, metric)
            print(f"  {name:<22} {mean:>10.4f} {std:>10.4f} "
                  f"{mn:>10.4f} {mx:>10.4f}")

    print(f"\n{'='*65}")
    print(f"  Seeds: {sorted(df['seed'].tolist())}")
    print(f"{'='*65}\n")

def print_results_tables(results_paths, labels, title=None, save_path=None):
    """
    Print a formatted comparison table from multiple results CSVs.
    Each row = metric (mean ± std), each column = model.

    Args:
        results_paths: list of paths to results.csv files
        labels:        list of column labels — one per results file
        title:         optional table title
    """
    assert len(results_paths) == len(labels), \
        "results_paths and labels must have the same length"

    # Load all dataframes
    dfs = {label: pd.read_csv(path)
           for label, path in zip(labels, results_paths)}

    # Metric groups
    metric_groups = {
        'Overall': [
            'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
            'macro_auc', 'micro_auc', 'macro_ap', 'micro_ap'
        ],
        'Level 1 (Mild)': [
            'l1_precision', 'l1_recall', 'l1_f1', 'l1_ap'
        ],
        'Level 2 (Moderate)': [
            'l2_precision', 'l2_recall', 'l2_f1', 'l2_ap'
        ],
        'Level 3 (Severe)': [
            'l3_precision', 'l3_recall', 'l3_f1', 'l3_ap'
        ],
    }

    # Pretty metric names
    metric_names = {
        'accuracy':        'Accuracy',
        'macro_f1':        'Macro F1',
        'macro_precision': 'Macro Precision',
        'macro_recall':    'Macro Recall',
        'macro_auc':       'Macro AUC-ROC',
        'micro_auc':       'Micro AUC-ROC',
        'macro_ap':        'Macro AP',
        'micro_ap':        'Micro AP',
        'l1_precision':    'L1 Precision',
        'l1_recall':       'L1 Recall',
        'l1_f1':           'L1 F1-Score',
        'l1_ap':           'L1 AP',
        'l2_precision':    'L2 Precision',
        'l2_recall':       'L2 Recall',
        'l2_f1':           'L2 F1-Score',
        'l2_ap':           'L2 AP',
        'l3_precision':    'L3 Precision',
        'l3_recall':       'L3 Recall',
        'l3_f1':           'L3 F1-Score',
        'l3_ap':           'L3 AP',
    }

    # Column widths
    metric_col_w = 20
    data_col_w   = 18  # wide enough for "0.6671 ± 0.0318"

    # Total width
    total_w = metric_col_w + data_col_w * len(labels) + 2

    # Header
    print(f"\n{'='*total_w}")
    if title:
        print(f"  {title}")
    print(f"{'='*total_w}")

    # Column headers
    header = f"  {'Metric':<{metric_col_w}}"
    for label in labels:
        header += f"{label:^{data_col_w}}"
    print(header)
    print(f"  {'-'*metric_col_w}" + f"{'-'*data_col_w}" * len(labels))

    # Seeds info
    n_seeds = {label: len(df) for label, df in dfs.items()}

    for group, metrics in metric_groups.items():
        print(f"\n  [{group}]")
        for metric in metrics:
            # Skip if metric missing in all dataframes
            if not any(metric in df.columns for df in dfs.values()):
                continue

            name = metric_names.get(metric, metric)
            row  = f"  {name:<{metric_col_w}}"

            best_mean  = -1.0
            best_label = None

            # Find best mean across models
            for label, df in dfs.items():
                if metric in df.columns:
                    mean = df[metric].mean()
                    if mean > best_mean:
                        best_mean  = mean
                        best_label = label

            # Build row
            for label, df in dfs.items():
                if metric not in df.columns:
                    row += f"{'—':^{data_col_w}}"
                    continue
                mean = df[metric].mean()
                std  = df[metric].std()
                cell = f"{mean:.4f} ± {std:.4f}"
                # Bold best with asterisk
                if label == best_label:
                    cell = f"*{cell}"
                row += f"{cell:^{data_col_w}}"

            print(row)

    print(f"\n{'='*total_w}")
    for label, n in n_seeds.items():
        print(f"  {label}: {n} seeds")
    print(f"{'='*total_w}\n")

    # Build summary DataFrame
    rows = []
    for group, metrics in metric_groups.items():
        for metric in metrics:
            row = {'group': group, 'metric': metric_names.get(metric, metric)}
            for label, df in dfs.items():
                if metric in df.columns:
                    mean = df[metric].mean()
                    std  = df[metric].std()
                    row[f'{label}_mean'] = round(mean, 4)
                    row[f'{label}_std']  = round(std, 4)
                    row[f'{label}_str']  = f"{mean:.2f} ± {std:.2f}"
            rows.append(row)

    summary_df = pd.DataFrame(rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        print(f"\nTable saved → {save_path}")

    return summary_df

def plot_interannotator_confusion(comparison_df, save_path=None):
    """
    Three confusion matrices side by side:
    Left:   Dentist vs Original
    Middle: Model vs Original  
    Right:  Model vs Dentist
    """
    from sklearn.metrics import confusion_matrix, cohen_kappa_score
    import seaborn as sns

    true_labels    = comparison_df['true_level'].values
    pred_labels    = comparison_df['predicted_level'].values
    dentist_labels = comparison_df['visual_level'].values

    pairs = [
        (true_labels, dentist_labels, 
         'Dentist vs Original',
         'Original', 'Dentist'),
        (true_labels, pred_labels,    
         'Model vs Original',
         'Original', 'Model'),
        (dentist_labels, pred_labels, 
         'Model vs Dentist',
         'Dentist', 'Model'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (y_true, y_pred, title, ylabel, xlabel) in zip(axes, pairs):
        cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3])
        
        kappa = cohen_kappa_score(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['L1', 'L2', 'L3'],
                    yticklabels=['L1', 'L2', 'L3'],
                    cbar=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nCohen's κ = {kappa:.3f}")

    # Add note about model vs original bias
    fig.text(0.38, -0.02,
             '* Model vs Original reflects purposefully selected subset '
             '— not representative of overall model performance.',
             ha='center', fontsize=8, style='italic', color='gray')

    #plt.suptitle('Inter-annotator Agreement Analysis', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.close('all')
    
def plot_dentist_agreement_by_level(comparison_df, save_path=None):
    """
    Bar chart showing dentist accuracy and mean confidence per severity level.
    Highlights the L2 paradox — high confidence, low accuracy.
    """
    levels      = [1, 2, 3]
    level_names = ['Level 1\n(Mild)', 'Level 2\n(Moderate)', 'Level 3\n(Severe)']
    colors      = ['#2196F3', '#FF5722', '#4CAF50']

    dentist_acc  = []
    model_acc    = []
    mean_conf    = []
    std_conf     = []

    for level in levels:
        mask = comparison_df['true_level'] == level
        dentist_acc.append(
            (comparison_df.loc[mask, 'visual_level'] == level).mean()
        )
        model_acc.append(
            (comparison_df.loc[mask, 'predicted_level'] == level).mean()
        )
        mean_conf.append(comparison_df.loc[mask, 'visual_confidence'].mean())
        std_conf.append(comparison_df.loc[mask, 'visual_confidence'].std())

    x      = np.arange(len(levels))
    width  = 0.25
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Bars — dentist and model accuracy
    bars1 = ax1.bar(x - width/2, dentist_acc, width,
                    label='Dentist Accuracy', color='#2196F3', alpha=0.85)
    bars2 = ax1.bar(x + width/2, model_acc,   width,
                    label='Model Accuracy',   color='#FF9800', alpha=0.85)

    ax1.set_xlabel('Severity Level')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(level_names)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom',
                 fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{bar.get_height():.2f}', ha='center', va='bottom',
                 fontsize=9)

    # Second axis — dentist confidence
    ax2 = ax1.twinx()
    ax2.errorbar(x, mean_conf, yerr=std_conf,
                 fmt='o-', color='#9C27B0', linewidth=2,
                 markersize=8, capsize=5,
                 label='Dentist Confidence (mean ± std)')
    ax2.set_ylabel('Dentist Confidence (1–10)')
    ax2.set_ylim(0, 10)
    ax2.legend(loc='upper right')

    plt.title('Dentist and Model Accuracy vs Dentist Confidence by Severity Level',
              fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved → {save_path}")
    plt.close('all')
    















