# Wearable Stress Detection with Deep Learning

Stress detection from physiological signals using the WESAD dataset. This project builds on prior work that found traditional ML (Random Forest, F1: 0.67) outperforming deep learning models — several of which completely failed with 0.00 F1. We identify why those models failed and fix it with a proper CNN-Transformer pipeline, class-imbalance handling, and leave-one-subject-out cross-validation.

Built on top of [kjspring/stress-detection-wearable-devices](https://github.com/kjspring/stress-detection-wearable-devices), extended with clean Python scripts, two deep learning architectures, and a rigorous evaluation protocol.

## What was broken in the original

The original project tried CNN+LSTM and Transformer+LSTM but got 0.00 F1 on stress detection for several models. Two reasons: the models were fed hand-extracted statistical features instead of raw signals, which strips out the temporal structure CNNs and Transformers actually need, and class imbalance was never handled — models just predicted everything as non-stress. The evaluation also used a single random split rather than leave-one-subject-out, which inflates performance by leaking subject-specific patterns into the test set.

This project fixes all three.

## What we changed

- Raw signal windowing (10s at 100 Hz) instead of pre-computed statistical features
- WeightedRandomSampler and weighted cross-entropy loss to handle stress/non-stress imbalance
- Leave-one-subject-out (LOSO) cross-validation for proper subject-independent evaluation
- CNN-Transformer architecture combining local waveform extraction with temporal modeling
- LightweightCNN as a simpler deep learning comparison
- Random Forest baseline re-run under the same LOSO protocol for fair comparison

## Dataset

[WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) — 15 subjects, chest-worn RespiBAN device at 700 Hz. Four chest modalities used: ECG, EDA, RESP, TEMP. Signals downsampled to 100 Hz and segmented into 10-second windows with 50% overlap.

Labels: 1 (baseline) mapped to 0 (non-stress), 2 (stress) mapped to 1. Other conditions (amusement, meditation) excluded. S1 and S12 excluded due to sensor malfunction per the original dataset documentation.

Download WESAD and place it in the `WESAD/` folder in the project root.

## Model Architectures

### StressTransformer

Two CNN blocks extract local patterns from raw physiological signals (R-peaks, EDA bursts, respiration cycles), followed by a Transformer encoder that models how those patterns evolve across the 10-second window. Uses dual avg+max pooling before classification, same design principle as the ECG ResNet baseline.

```
Input (batch, 4, 1000)
    -> ConvBlock(4->32, k=7, stride=2)
    -> ConvBlock(32->64, k=5, stride=2)
    -> TransformerEncoder(d=64, heads=4, layers=2)
    -> AdaptiveAvgPool + AdaptiveMaxPool
    -> Linear(128->2)
```

### LightweightCNN

Three-block CNN with global average pooling. Simpler than StressTransformer — useful to verify whether the Transformer component adds meaningful value on this dataset size.

### Random Forest

Re-run under the same LOSO protocol using per-channel mean, std, and IQR as features. This is the original project's best model — our baseline to beat.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Training:

```bash
python train.py --data-dir WESAD --model stress_transformer --use-gpu --epochs 30
python train.py --data-dir WESAD --model lightweight_cnn --use-gpu --epochs 30
```

Evaluation:

```bash
python evaluate.py --data-dir WESAD --model stress_transformer --use-gpu
python evaluate.py --data-dir WESAD --model lightweight_cnn --use-gpu
python evaluate.py --data-dir WESAD --model random_forest
```

## Results

LOSO cross-validation across 15 subjects (avg across all folds):

| Model | Avg F1 | Avg Recall | Avg Precision | Avg Accuracy |
|---|---|---|---|---|
| Random Forest (original baseline, re-run) | 0.6643 | 0.7161 | 0.7578 | 0.7862 |
| LightweightCNN | 0.7946 | 0.8719 | 0.8373 | 0.8319 |
| StressTransformer | 0.7661 | 0.7993 | 0.8614 | 0.8230 |

Both deep learning models outperform the Random Forest under the same rigorous LOSO protocol. LightweightCNN leads on F1 and recall, while StressTransformer achieves higher precision — both represent meaningful improvements over the baseline.

The original project reported RF F1 of 0.67 on a single random split. Under proper LOSO evaluation, that baseline is 0.6643, and our models reach 0.7946 and 0.7661 respectively.

## Why LOSO Matters

The original project split data randomly, which leaks subject-specific patterns into training. LOSO holds out one complete subject per fold, testing whether models generalize to people they have never seen. This is the standard evaluation protocol for wearable stress detection and explains why some subjects (S2, S4) are harder than others — they have less distinctive physiological responses.

## Signals Used

| Signal | Why |
|---|---|
| ECG | Heart rate variability is one of the strongest indicators of stress |
| EDA | Electrodermal activity reflects sympathetic nervous system arousal |
| RESP | Respiration rate and depth change under stress |
| TEMP | Peripheral temperature drops during acute stress response |

## File Structure

```
├── dataset.py          — WESAD loader, windowing, LOSO splits
├── models.py           — StressTransformer and LightweightCNN
├── train.py            — LOSO training with weighted sampler
├── evaluate.py         — evaluation and Random Forest baseline
├── utils.py            — metrics
├── notebooks/          — original baseline notebook (reference)
├── WESAD/              — dataset (not included, download separately)
└── results/            — CSV results per model
```

## Acknowledgements

Original notebook baseline: [kjspring/stress-detection-wearable-devices](https://github.com/kjspring/stress-detection-wearable-devices).
Dataset: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI 2018.
