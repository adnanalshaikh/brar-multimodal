# Data

## Source and licence

The panoramic radiographs in this directory are derived from:

> Xia Y, Li Z, Lin Z, Wang S, Wang Y, Xie Y. **BRAR-anchored multimodal dataset.**
> figshare. 2025.
> [doi:10.6084/m9.figshare.30155974](https://doi.org/10.6084/m9.figshare.30155974)

Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Modifications

The material here has been **modified** from the original deposit:

- Images were manually cropped to the dental-arch region of interest, as
  described in the Methods section of the accompanying article.
- The set is restricted to the 988 cases with complete metadata; the source
  deposit covers 1,104 patients.
- `meta_data.csv` was constructed from the deposit's annotations, retaining the
  severity level and the six metadata fields used by the models.

Original data © the dataset authors, reused and adapted under CC BY 4.0. These
modifications do not imply endorsement by the original authors.

## Layout

```
data/
├── meta_data.csv
└── manual_cropped_dataset/
    ├── level_1/
    ├── level_2/
    └── level_3/
```

Severity levels follow the BRAR thresholds: Level 1 < 0.25, Level 2 0.25–1.0,
Level 3 ≥ 1.0, where BRAR = (bone_resorption × 100) / age.
