# Wearable Stress Detection with Deep Learning

Stress detection from physiological signals using the WESAD dataset. This project builds on prior work that found traditional ML (Random Forest, F1: 0.67) outperforming deep learning models — several of which completely failed (F1: 0.00). We identify why those models failed and fix it with a proper CNN-Transformer pipeline and class-imbalance handling.

Built on top of [kjspring/stress-detection-wearable-devices](https://github.com/kjspring/stress-detection-wearable-devices) (original notebook baseline), extended with clean Python scripts, a CNN-Transformer architecture, and leave-one-subject-out cross-validation.

## What was broken in the original

The original project tried CNN+LSTM and Transformer+LSTM models but got 0.00 F1 on stress detection. Two reasons: (1) the models were fed hand-extracted statistical features instead of raw signals, which removes the temporal structure CNNs and Transformers actually need, and (2) class imbalance was never handled — models just predicted everything as non-stress.

This project fixes both.

## What we changed

- Raw signal windowing instead of pre-computed statistical features
- Weighted cross-entropy loss to handle stress/non-stress imbalance
- Leave-one-subject-out (LOSO) cross-validation for proper generalization testing
- CNN-Transformer model that combines local waveform feature extraction with long-range temporal modeling
- Lightweight CNN as an additional comparison point
- Random Forest baseline re-run under the same LOSO protocol for fair comparison

## Dataset

[WESAD (Wearable Stress and Affect Detection)](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) — 15 subjects, chest-worn RespiBAN device at 700 Hz. We use four chest modalities: ECG, EDA, RESP, and TEMP. Signals are downsampled to 100 Hz and segmented into 10-second windows with 50% overlap.

Labels: 1 (baseline) mapped to 0 (non-stress), 2 (stress) mapped to 1 (stress). Other conditions (amusement, meditation) are excluded.

Download WESAD and extract to the `WESAD/` folder in the project root.

## Model Architecture

### StressTransformer (ours)

Two CNN blocks extract local patterns from the raw physiological signals (R-peaks, EDA bursts, respiration cycles), followed by a Transformer encoder that captures how those patterns evolve across the 10-second window.

```
Input (batch, 4, 1000)
    -> ConvBlock(4->32, k=7, stride=2)
    -> ConvBlock(32->64, k=5, stride=2)
    -> TransformerEncoder(d=64, heads=4, layers=2)
    -> AdaptiveAvgPool + AdaptiveMaxPool
    -> Linear(128->2)
```

### LightweightCNN (comparison)

Smaller 3-block CNN with global average pooling. Useful to check whether the Transformer adds meaningful value over a simpler architecture.

### Random Forest (baseline)

Re-run under LOSO protocol using per-channel mean, std, and IQR features. This is the original project's best model — the one we're trying to beat.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Training with LOSO cross-validation:

```bash
# StressTransformer
python train.py --data-dir WESAD --model stress_transformer --use-gpu --epochs 30

# LightweightCNN
python train.py --data-dir WESAD --model lightweight_cnn --use-gpu --epochs 30
```

Evaluation:

```bash
python evaluate.py --data-dir WESAD --model stress_transformer --use-gpu
python evaluate.py --data-dir WESAD --model lightweight_cnn --use-gpu
python evaluate.py --data-dir WESAD --model random_forest
```

## Results

LOSO cross-validation across 15 subjects (avg ± std):

| Model | F1 | Recall | Precision | Accuracy | AUC |
|---|---|---|---|---|---|
| Random Forest (baseline) | - | - | - | - | - |
| LightweightCNN | - | - | - | - | - |
| StressTransformer (ours) | - | - | - | - | - |

*Results will be filled in after experiments complete.*

The original project's Random Forest achieved F1: 0.67 on a single train/test split. Our target is to beat this under the more rigorous LOSO protocol.

## Why LOSO

The original project split data randomly, which means training and test windows from the same subject end up in both sets. That inflates performance — a model can learn subject-specific patterns rather than generalizing across people. LOSO tests on a completely held-out subject every fold, which is the standard evaluation protocol for wearable stress detection.

## Signals Used

| Signal | Device | Why |
|---|---|---|
| ECG | Chest (RespiBAN) | Heart rate variability is strongly linked to stress |
| EDA | Chest (RespiBAN) | Electrodermal activity reflects sympathetic nervous system arousal |
| RESP | Chest (RespiBAN) | Respiration rate and depth change under stress |
| TEMP | Chest (RespiBAN) | Peripheral temperature drops during acute stress |

## Acknowledgements

Original notebook baseline: [kjspring/stress-detection-wearable-devices](https://github.com/kjspring/stress-detection-wearable-devices).
Dataset: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection," ICMI 2018.
