# Cryptojacking Detection Validation

**Replication Package for the Empirical Evaluation in the Comprehensive Examination**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and reproducible pipeline for the empirical evaluation (Section 5) of:

> **AI-Based Cryptojacking Detection in Cloud Environments: A Systematic Literature Review**  
> *Amitabh Chakravorty*  
> School of Information Technology, University of Cincinnati, USA

The examination studies this issue from two perspectives. The first is a systematic literature review of 41 peer-reviewed research papers on AI-based cryptojacking detection in cloud and cloud-adjacent environments, organized around three research questions:

- **RQ1:** What attack strategies and target environments are reported?
- **RQ2:** What AI-based approaches are used to detect cryptojacking?
- **RQ3:** What data, evaluation and operational challenges make reliable deployment difficult?

The second perspective is an empirical evaluation. Six classical ML models from the reviewed literature were tested on DS2OS and NSL-KDD as behavioral proxy datasets. Performance on DS2OS deteriorated after leakage-prone information was removed. Models tested on NSL-KDD that reached near-perfect accuracy with a random split achieved only 76.76 to 80.82 percent when tested with the official holdout set that contains attacks not present in the training set.

## Empirical Results

### Table 7: Overall Accuracy Across Both Proxy Datasets

| Model | DS2OS | NSL-KDD |
|-------|-------|---------|
| Random Forest | 96.26% | 77.17% |
| XGBoost | 96.26% | 80.82% |
| LightGBM | 96.26% | 80.35% |
| Gradient Boosting | 96.23% | 78.25% |
| Decision Tree | 96.26% | 77.66% |
| KNN | 99.21% | 76.76% |

*DS2OS uses the leakage-aware stratified 70/30 split. NSL-KDD uses the official KDDTrain+/KDDTest+ partition containing attack types absent from training.*

### Table 8: Performance Changes Under Stricter Evaluation Conditions

| Evaluation | Initial Condition | Stricter Condition | Observed Change |
|------------|-------------------|-------------------|-----------------|
| DS2OS leakage | 12 raw features, XGBoost | Five-feature leakage-aware pipeline | Accuracy: ~99.995% → 96.26%; attack-class F1: 0.999 → ~0.59 |
| NSL-KDD generalization | Random stratified split | Official unseen-attack holdout | Accuracy: 99.00-99.60% → 76.76-80.82% |

### Why the Gap?

- **DS2OS (~3% accuracy drop):** Original studies kept identifier columns (timestamp, sourceID, sourceAddress) that leak the target variable. The timestamp was acting as an almost deterministic proxy for the label because attacks had been recorded in sequential time blocks. After removing those identifiers, only five behavioral features remained and the tree-based models fell to an attack-class F1-score near 0.59.
- **NSL-KDD (~22% accuracy drop):** Original studies tested on random splits of training data, so models only encountered attack types they had already been trained on. The official KDDTest+ holdout contains novel attacks (mscan, saint, apache2, processtable) absent from training. Several attack families that were absent from training received zero recall, while familiar families remained close to perfectly detected.

> **Note:** Both datasets are behavioral proxies for cloud cryptojacking. The review found zero publicly available datasets that capture actual cloud VM, container or Kubernetes telemetry with labeled cryptomining activity (Section 4.4.1). That gap is one of the key findings from the review.

## Repository Structure

```
cryptojacking-validation/
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
│
├── src/                             # Run scripts in order
│   ├── 1_setup_and_download.py     # Directory setup + Kaggle download
│   ├── 2_explore.py                # Dataset exploration + distribution charts
│   ├── 3_preprocess.py             # Cleaning, SMOTE, scaling
│   └── 4_train_and_evaluate.py     # Training, metrics, visualizations
│
├── data/
│   ├── raw/                         # Downloaded datasets (gitignored)
│   └── processed/                   # Preprocessed .npy arrays (gitignored)
│
├── models/                          # Saved .pkl model files (gitignored)
│
├── results/
│   ├── figures/                     # Confusion matrices, accuracy/F1 bars, heatmap
│   └── metrics/                     # CSV performance tables
│
├── scripts/
│   └── utils.py                     # Shared helper functions
│
└── docs/
    ├── METHODOLOGY.md               # Detailed methodology write-up
    └── GITHUB_SETUP_GUIDE.md
```

## Quick Start

```bash
git clone https://github.com/AmitabhCh822/cryptojacking-research-empirical-evaluation.git
cd cryptojacking-research-empirical-evaluation

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run scripts in order
python src/1_setup_and_download.py --base_path .
python src/2_explore.py --base_path .
python src/3_preprocess.py --base_path .
python src/4_train_and_evaluate.py --base_path .
```

A [Kaggle account](https://www.kaggle.com/) and API key (`~/.kaggle/kaggle.json`) are needed for the dataset download step.

## Datasets

### DS2OS (Distributed Smart Space Orchestration System)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/libamariyam/ds2os-dataset) [5]
- **Samples:** 357,952
- **Original features:** 12 (IoT device telemetry)
- **After leakage removal:** 5 features (sourceType, sourceLocation, destinationServiceAddress, destinationServiceType, destinationLocation)
- **Removed for leakage:** sourceID, sourceAddress, timestamp, value, accessedNodeAddress, accessedNodeType, operation
- **Class split:** 97.2% normal, 2.8% attack
- **Resampling:** SMOTE applied to training set only (severe class imbalance)

### NSL-KDD
- **Source:** [UNB CIC](https://www.unb.ca/cic/datasets/nsl.html) [43]
- **Train:** 125,973 samples (KDDTrain+)
- **Test:** 22,544 samples (KDDTest+ with novel attack types absent from training)
- **Features:** 41 (network traffic patterns)
- **Class split:** ~53% normal, ~47% attack
- **Resampling:** Not needed; classes are approximately balanced

### Why Proxy Datasets?

The systematic review (Section 4.4.1) found no public datasets with real cloud cryptojacking telemetry. The closest options (CREMEv2, VIKRANT honeypot, AWS simulation repo) capture only host-level sequences or network flows. None includes hypervisor metrics, Kubernetes pod stats or container runtime telemetry.

## Models

Six model families selected based on frequency of appearance in the 41 reviewed studies. Classical ML covers 57% of the studies.

| Model | Configuration |
|-------|--------------|
| Random Forest | 100 estimators, max_depth=20, min_samples_split=5 |
| XGBoost | 100 estimators, max_depth=10, learning_rate=0.1 |
| LightGBM | 100 estimators, max_depth=10 |
| Gradient Boosting | 100 estimators, max_depth=5, learning_rate=0.1 |
| Decision Tree | max_depth=15, min_samples_split=5 |
| KNN | 5 neighbors |

Configurations match commonly reported settings in the literature. No automated tuning. The purpose is reproducibility, not chasing the highest number.

## Preprocessing Pipeline (Section 5.1)

```
Raw Data
    │
    ├── Remove identifier columns correlated with the target (leakage check)
    │
    ├── Label-encode categorical features
    │
    ├── Stratified 70/30 train/test split (random_state=42)
    │
    ├── SMOTE on training set only (DS2OS; severe ~97/3 imbalance)
    │   └── NSL-KDD already ~53/47, no resampling needed
    │
    └── StandardScaler (zero mean, unit variance)
```

KDDTest+ is used as-is. Resampling it would defeat the purpose of testing generalization to novel attacks.

## Key Findings (Section 5)

1. **Data leakage matters (Section 5.4).** Removing identifier columns from DS2OS drops accuracy by ~3%. The near-perfect result did not survive once the shortcut through timestamp and sourceID was removed.
2. **Generalization is the real test (Section 5.5).** The 22% accuracy drop on NSL-KDD shows that models tested only on familiar attacks massively overstate production readiness. Several attack families absent from training received zero recall.
3. **Operational metrics beyond accuracy (Section 5.3).** Weighted F1-scores initially look strong (~0.97 on DS2OS), but attack-class precision falls to about 0.43. MCC reaches 0.844 for KNN on DS2OS while the remaining models stay around 0.61 to 0.66.
4. **Classical models converge (Section 5.2).** After limiting the available signal, many of the classical models converged significantly. All tree-based models land at ~96.26% on DS2OS because only 5 low-cardinality features survive.
5. **Class imbalance needs handling.** Without SMOTE on DS2OS, models achieve 97% accuracy by predicting everything as normal with zero attack recall.

## Reproducing Results

Results are based on single stratified train-test splits to match how the primary studies ran their experiments. Treat the numbers as point estimates, not guarantees.

**Environment:** Python 3.10, scikit-learn 1.3.0, XGBoost 2.0.0, LightGBM 4.0.0.

## Citation

### Software / Replication Package

```bibtex
@software{chakravorty2026cryptojacking_code,
  title   = {Cryptojacking Validation: Replication Package},
  author  = {Chakravorty, Amitabh},
  year    = {2026},
  url     = {https://github.com/AmitabhCh822/cryptojacking-research-empirical-evaluation}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- University of Cincinnati CECH Impact Accelerator Grant
- Canadian Institute for Cybersecurity (NSL-KDD dataset)
- DS2OS dataset contributors

## Contact

**Amitabh Chakravorty** - [chakraa4@mail.uc.edu](mailto:chakraa4@mail.uc.edu)

---

This repository is the replication package for the empirical evaluation in the comprehensive examination. The main takeaway: reported performance was strongly affected by the information available to the model and the conditions under which it was evaluated. The field needs public cloud-specific cryptojacking datasets before detection approaches can be taken seriously for operational deployment.
