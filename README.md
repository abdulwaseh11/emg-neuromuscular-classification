# EMG-Based Neuromuscular Signal Classification

A machine learning pipeline for classifying electromyography (EMG) signals, applied both to hand movement recognition (Ninapro DB1) and to neuromuscular disease classification (PhysioNet EMG Database: healthy vs. myopathy vs. neuropathy).

## Overview

This project explores signal processing and machine learning methods for EMG analysis across two complementary tasks:

1. **Movement classification** (Ninapro DB1), 17-class hand gesture recognition from 10-channel surface EMG across 27 subjects, used as a large-scale benchmark for developing and validating the feature extraction and modeling pipeline.
2. **Disease classification** (PhysioNet EMG Database), distinguishing healthy, myopathic, and neuropathic EMG signals from real clinical needle-EMG recordings, directly reflecting neuromuscular disease biomarker extraction.

Both a raw-signal 1D-CNN and classical ML models trained on hand-engineered time/frequency-domain EMG features are implemented and compared.

## Datasets

- **[Ninapro DB1](http://ninapro.hevs.ch/)**, 27 subjects, 10-channel surface EMG, 17 hand movement classes plus rest.
- **[PhysioNet EMG Database](https://physionet.org/content/emgdb/)**, single-channel needle EMG recordings for healthy, myopathic, and neuropathic subjects, collected at Beth Israel Deaconess Medical Center / Harvard Medical School.

## Methods

### Signal Processing / Feature Engineering
Six features extracted per channel from sliding windows of raw EMG:
- **Time domain**: RMS, mean absolute value (MAV), zero-crossing rate, waveform length, variance
- **Frequency domain**: mean frequency (via FFT)

### Models
- 1D Convolutional Neural Network (raw signal input, PyTorch)
- Random Forest, SVM (RBF kernel), MLP, trained on engineered features, with hyperparameter tuning via `GridSearchCV` and `StratifiedKFold` cross-validation
- MLP trained directly on flattened raw signal, for comparison

### Validation Methodology
- **Ninapro**: repetition-based train/val/test split (no subject's repetitions mixed across splits) to avoid data leakage
- **PhysioNet**: time-based split (train on the first portion of each recording, test on a later, non-overlapping portion) with a purge window to prevent overlapping windows from appearing in both train and test, this was adopted after an initial random split raised concerns about leakage between adjacent overlapping windows

## Results

### Ninapro DB1 (17-class movement classification)

| Model | Test Macro-F1 |
|---|---|
| CNN (raw signal) | 0.821 |
| Random Forest (tuned) | 0.803 |
| Random Forest (default) | 0.797 |
| SVM (tuned) | 0.749 |
| MLP (raw signal) | 0.745 |
| MLP (engineered features) | 0.702 |
| SVM (default) | 0.616 |

### PhysioNet (healthy / myopathy / neuropathy classification)

| Split | Test Macro-F1 |
|---|---|
| Random split | 0.954 |
| **Time-based split (leakage-resistant)** | **0.958** |

Per-class performance (time-based split): healthy 0.91 F1, myopathy 0.99 F1, neuropathy 0.98 F1.

Additional models benchmarked on the time-based split: Logistic Regression (0.906), SVM-RBF (0.922), confirming the result is robust across model types, not specific to Random Forest.

## Limitations

- The PhysioNet dataset contains only **one recording per diagnostic class**. Reported Macro-F1 reflects generalization across time-segments *within* each recording, not generalization to unseen patients or recordings. This should be read as a proof-of-concept validation of the feature-extraction and classification pipeline on real clinical EMG data, not a claim of clinical-grade diagnostic accuracy across patients.
- A larger, multi-patient dataset (e.g., the [Mendeley 241-subject EMG dataset](https://data.mendeley.com/datasets/543xpjycj9)) would be needed to properly assess generalization to new patients.

## Engineering Practices

- Reproducibility via fixed random seeds across NumPy, PyTorch, and CUDA
- Experiment tracking with MLflow (parameters, metrics logged per run)
- Model versioning with timestamped artifacts and metadata

## Tech Stack

Python, PyTorch, scikit-learn, NumPy, pandas, matplotlib/seaborn, MLflow, WFDB (PhysioNet data access)

## Author

AbdulWaseh, Master's in Data Science (Biomedical Data Science), Universiti Teknologi Malaysia
