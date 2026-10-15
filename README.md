# BRAR Multimodal

[![DOI](https://zenodo.org/badge/1420020573.svg)](https://doi.org/10.5281/zenodo.23405144)

Deep learning classification of periodontal bone resorption severity from panoramic
radiographs, combining image features with routinely available clinical metadata.

This repository contains the full pipeline used to produce the results reported in:

> Salman A, Bishtawi IR. **Multimodal deep learning for automated assessment of
> periodontal bone resorption severity using panoramic radiographs and structured
> metadata.** *BMC Medical Imaging*. 2026.
> [doi:10.1186/s12880-026-02928-6](https://doi.org/10.1186/s12880-026-02928-6)

Every table and figure in the paper is generated from the code and configuration in
this repository: `main.py` produces the results, and the three top-level scripts
produce the figures.

---

## What the model does

Each radiograph is assigned a **bone resorption to age ratio (BRAR)** severity grade:

```
BRAR = (bone_resorption × 100) / age
```

Grades are defined by thresholds on BRAR:

| Grade | Description | BRAR range |
|-------|-------------|------------|
| Level 1 | Mild | < 0.25 |
| Level 2 | Moderate | 0.25 – 1.0 |
| Level 3 | Severe | ≥ 1.0 |

Three model variants are trained and compared:

- **image_only** — DenseNet-121 backbone on the radiograph alone
- **meta_only** — clinical metadata alone (no image)
- **multimodal** — joint fusion of both, via concatenated projections

---

## Dataset

988 panoramic radiographs, manually cropped to the dental-arch region of interest.
Class distribution: Level 1 = 170, Level 2 = 560, Level 3 = 258.

Each seed produces a stratified 70/15/15 partition: 691 training, 148 validation and
149 test images (26 / 84 / 39 per level in the test partition).

**Data availability:** the radiographs are derived from the BRAR-anchored
multimodal dataset (Xia Y, Li Z, Lin Z, Wang S, Wang Y, Xie Y. *BRAR-anchored
multimodal dataset*. figshare, 2025.
[doi:10.6084/m9.figshare.30155974](https://doi.org/10.6084/m9.figshare.30155974)),
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The images in this repository are **modified**: manually cropped to the
dental-arch region of interest as described in the Methods. All 988 radiographs
available in the deposit are included, although the deposit's description refers
to 1,104 patients. `meta_data.csv` is derived from the same deposit. Original
data © the dataset authors, reused and adapted under CC BY 4.0.

Expected layout:

```
data/
├── meta_data.csv
└── manual_cropped_dataset/
    ├── level_1/
    ├── level_2/
    └── level_3/
```

`meta_data.csv` must contain an `image_path` column (filenames, with or without a
directory prefix), a `level` column with values 1–3, and the six metadata columns
listed under Configuration below.

---

## Installation

```bash
conda create -n ml_env python=3.10
conda activate ml_env
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training. The analysis, comparison and figure steps
run on CPU from saved prediction files — see *Reproducing the tables without a GPU*
below.

---

## Running the pipeline

```bash
python main.py
```

This runs all stages in order. Completed stages are detected by a `.complete`
sentinel file in each run directory and are reused rather than recomputed, so the
script can be safely re-run after an interruption.

| Stage | What it does |
|-------|--------------|
| Training | Trains image_only, meta_only and multimodal over the 20 evaluation seeds |
| Tuning runs | Trains the same three models on the held-out tuning seeds |
| Ensemble weights | Grid-searches probability-averaging weights on the tuning seeds |
| Ensemble | Builds the tuned ensemble on the evaluation seeds |
| Late fusion | Weighted probability averaging of the separately trained unimodal models |
| Metadata ablation | Image + age only, and image + non-age metadata only |
| Comparisons | Paired Wilcoxon tests between models, with Holm–Bonferroni correction |
| Confidence intervals | Mean and 95% CI for each model and metric |
| MAGE | Mean absolute grade error |
| Grad-CAM | Activation-map visualisations for selected cases |
| Expert analysis | Agreement against the independent dentist grading |

---

## Figures

The three scripts at the top level are standalone: they read the saved results under
`runs/` and write the paper figures. Run them after `main.py` has completed.

```bash
python paper_figures.py            # Figures 2, 4, 5 and 6
python paired_slope_figure.py      # Figure 3
python check_lowest_auc_class.py   # verification check, no figure
```

| Script | What it produces |
|--------|------------------|
| `paper_figures.py` | Paired metric distributions across the 20 splits (box plots); the effect-size forest plot of median paired differences with bootstrap 95% CIs; the ordinal confusion matrix and representative-split panel; the expert error-analysis figure |
| `paired_slope_figure.py` | Within-split paired slope plot: one line per split, coloured by the direction of the difference, for multimodal vs image-only and multimodal vs metadata-only |
| `check_lowest_auc_class.py` | Prints, for each of the 20 splits, which severity level had the lowest per-class AUC. Supports the statement in the paper that Level 2 was lowest in all 20 splits |

All figures use the Okabe–Ito colour-blind-safe palette, with a fixed colour per
model and per severity level, and are written as vector PDFs.

---

## Configuration

All hyperparameters live in `src/config.py`. The values in this repository are
those used for the published results; `config.json` is also written into every run
directory, so each result set carries the configuration that produced it.

Key settings:

```python
'seed':            [20 evaluation seeds]     # each seed = one split + one init
'val_size':        0.15
'test_size':       0.15
'backbone_name':   'densenet121'             # or efficientnet_b0…b3
'fusion_mode':     'multimodal'              # or image_only / meta_only
'metadata_cols':   ['age', 'gender', 'number_of_missing_teeth',
                    'implant', 'residual_root', 'functional_tooth_logarithm']
```

**Seeds control the data split as well as model initialisation.** Each seed
produces a different stratified train/validation/test partition, so variation
across seeds reflects generalisation variance, not just training variance. All
reported confidence intervals should be read on that basis.

The metadata ablations are produced by changing `metadata_cols` and the matching
`normalize_cols`: keep only `'age'` for the image-plus-age model, drop `'age'` for
the image-plus-non-age model. The model reads the feature count from the column
list, so no other change is needed.

---

## Method summary

**Splits.** Stratified by severity level. Test is held out first, then validation
is taken from the remainder with the ratio adjusted so that both are 15% of the
full dataset. Per-seed split membership is saved as
`seed_<n>_{train,val,test}_split.csv` in each run directory.

**Preprocessing.** Images are resized to 224×224 and normalised with ImageNet
statistics. Metadata features are min–max normalised using statistics computed on
the **training split only**, then applied to validation and test.

**Class weighting.** Fourth-root inverse-frequency weights, which counter the class
imbalance without over-weighting the smallest class.

**Training.** Three stages with progressively lower learning rates and higher
dropout:

1. Frozen backbone; train projection and classifier heads only
2. Unfreeze `denseblock3`, `denseblock4`, `transition3`, `norm5`
3. Unfreeze the full network

Each stage builds a fresh optimiser over the currently trainable parameters, uses
`ReduceLROnPlateau`, and early-stops on validation loss. The best checkpoint is
carried across stages, so a stage that fails to improve on the running best is
discarded.

**Evaluation.** Three primary metrics are prespecified: macro F1, macro AUC-ROC
(one-vs-rest), and Level 3 recall. Comparisons between models use the Wilcoxon
signed-rank test paired by seed (`zero_method='zsplit'`), with Holm–Bonferroni
correction applied across the three primaries. All other metrics are reported as
exploratory and uncorrected. Effect sizes are given as the median paired difference
with a bootstrap percentile 95% confidence interval; absolute performance is
reported as the mean with a t-distribution 95% confidence interval.

**Ensemble weights** are selected by grid search on 10 tuning seeds that are
disjoint from the 20 evaluation seeds, and are recorded in
`runs/statistical_analysis/ensemble_weights.json`.

---

## Outputs

```
runs/
├── provenance.json                  # git commit and environment for this runs/ tree
├── provenance_log.jsonl             # every invocation, appended
├── <run_id>/
│   ├── .complete                    # sentinel; absent means the run is partial
│   ├── config.json                  # exact configuration used
│   ├── results.csv                  # one row of metrics per seed
│   ├── seed_<n>_probs.csv           # per-sample softmax probabilities
│   ├── seed_<n>_{train,val,test}_split.csv
│   ├── seed_<n>_model.pth
│   ├── seed_<n>_norm_stats.json
│   └── seed_<n>_*.png               # training curves, confusion matrix, ROC, PR
└── statistical_analysis/
    ├── mean_ci.csv / mean_ci_wide.csv
    ├── ensemble_weights.json
    └── *_wilcoxon.csv
```

Every metric in `results.csv` is derivable from the corresponding
`seed_<n>_probs.csv`, so metric definitions can be changed and tables rebuilt
without retraining.

---

## Reproducing the tables without a GPU

If you have the `runs/` directory but no GPU, the analysis stages run directly on
the saved probability files:

```python
from src.experiment import compare_experiments, mean_ci_metrics

mean_ci_metrics()
compare_experiments('runs/multi_dens/results.csv',
                    'runs/image_only_dens/results.csv',
                    save_path='runs/statistical_analysis/multi_vs_image.csv')
```

The three figure scripts also run on CPU from the same files.

---

## Repository layout

```
src/
├── config.py       # all hyperparameters; PROJECT_ROOT, DATA_DIR, RUNS_DIR
├── dataset.py      # metadata loading, Dataset classes, dataloaders
├── transforms.py   # image augmentation and normalisation
├── model.py        # backbones and the three fusion architectures
├── train.py        # three-stage training loop, class weighting, early stopping
├── evaluate.py     # metrics, plots, Grad-CAM
├── ensemble.py     # probability-level combination and weight grid search
├── experiment.py   # run orchestration, ensembling, statistical comparison
└── utils.py        # seeding and data splitting
main.py                    # pipeline entry point
paper_figures.py           # Figures 2, 4, 5, 6
paired_slope_figure.py     # Figure 3
check_lowest_auc_class.py  # per-class AUC verification check
requirements.txt
```

---

## Citation

Please cite the article:

```bibtex
@article{salman2026brar,
  title   = {Multimodal deep learning for automated assessment of periodontal
             bone resorption severity using panoramic radiographs and
             structured metadata},
  author  = {Salman, Adnan and Bishtawi, Ibtehal Raghib},
  journal = {BMC Medical Imaging},
  year    = {2026},
  doi     = {10.1186/s12880-026-02928-6}
}
```

To cite this software archive specifically:

```bibtex
@software{salman2026brar_code,
  title     = {BRAR Multimodal: code and data for multimodal periodontal bone
               resorption severity classification},
  author    = {Salman, Adnan and Bishtawi, Ibtehal Raghib},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23405144}
}
```

The DOI above resolves to the latest release; each version also has its own DOI.

## License

**Code** — MIT License, see [LICENSE](LICENSE).

**Data** (`data/`) — the cropped radiographs and `meta_data.csv` are adapted from
the BRAR-anchored multimodal dataset, © Xia et al., used and redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See *Data
availability* above and `data/README.md`.

**Article** — published open access under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

Adnan Salman — aalshaikh@najah.edu
