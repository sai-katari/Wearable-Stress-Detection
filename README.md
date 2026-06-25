# Wearable Stress Detection with Deep Learning

Stress detection from physiological signals using the WESAD multimodal dataset. This project reimplements and significantly improves on a prior baseline that found Random Forest (F1: 0.67) outperforming every deep learning model it tried, including CNN+LSTM and Transformer variants that scored exactly 0.00 F1 on stress detection.

The core issue was not the architectures themselves. It was how the data was prepared and how class imbalance was handled. This project fixes both and shows that deep learning substantially outperforms the RF baseline when implemented correctly.

---

## Background

The original project fed hand-extracted statistical features (mean, std over 1 and 5 minute intervals) into CNN and Transformer models. These models need raw temporal signals, not pre-summarized statistics. The temporal structure gets thrown away in the feature extraction step, which is exactly what makes CNNs and Transformers useful in the first place.

On top of that, the dataset has roughly 36% stress windows and 64% non-stress windows. Without handling that imbalance, models find it easier to just predict non-stress every time. That explains the 0.00 F1 pattern.

The original evaluation also used a single random train/test split, which lets models memorize subject-specific physiological patterns rather than learning generalizable features.

All three of these issues are addressed here.

---

## Dataset

WESAD (Wearable Stress and Affect Detection) from Schmidt et al., ICMI 2018. 15 subjects wearing a chest-mounted RespiBAN device during a controlled stress protocol. Two subjects (S1 and S12) excluded due to sensor malfunction per the original documentation.

We use four chest signals sampled at 700 Hz: ECG, EDA (electrodermal activity), respiration, and skin temperature. These are downsampled to 100 Hz and segmented into 10-second windows with 50% overlap. Only baseline (label 1, remapped to 0) and stress (label 2, remapped to 1) conditions are kept. Amusement and meditation segments are excluded.

This gives roughly 350-380 windows per subject, with about 36% labeled stress.

Download the dataset from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) and place it in the `WESAD/` folder.

---

## Models

### StressTransformer

A CNN frontend extracts local waveform patterns from the raw physiological signals, then a Transformer encoder models how those patterns change over the 10-second window. The intuition is that the CNN handles things like R-peak morphology and EDA burst shape, while the Transformer handles the temporal evolution of arousal across the window.

```
Input: (batch, 4 channels, 1000 time steps)
ConvBlock(4->32, kernel=7, stride=2)
ConvBlock(32->64, kernel=5, stride=2)
TransformerEncoder(d_model=64, heads=4, layers=2)
AdaptiveAvgPool1d + AdaptiveMaxPool1d
Linear(128 -> 2)
```

### LightweightCNN

Three convolutional blocks with global pooling. No Transformer component. Included to check whether the added complexity of the Transformer actually helps on a dataset of this size.

```
Input: (batch, 4 channels, 1000 time steps)
ConvBlock(4->32, kernel=7, stride=2)
ConvBlock(32->64, kernel=5, stride=2)
ConvBlock(64->64, kernel=3, stride=2)
AdaptiveAvgPool1d
Linear(64 -> 2)
```

### Random Forest

Re-run under the same LOSO protocol using per-channel mean, standard deviation, and IQR as features. This reproduces the original project's best result under a fair evaluation setup.

---

## Training Details

Class imbalance is handled two ways simultaneously. A `WeightedRandomSampler` oversamples stress windows during training so each batch is roughly balanced. The loss function also uses class weights computed from training label counts. Using both together was necessary to get the models off the all-non-stress prediction collapse.

Other training decisions: Adam optimizer with weight decay 1e-4, learning rate 0.0001, cosine annealing schedule, gradient clipping at 1.0, batch size 32, 30 epochs per fold. These were determined by observing that the original attempts at LR=0.001 caused gradient instability on this dataset size.

---

## Evaluation: Leave-One-Subject-Out

Each fold trains on 14 subjects and tests on the 15th. This tests whether the model can generalize to a person it has never seen, which is the actual deployment scenario for a wearable device. A random split does not test this because training and test windows from the same person end up on both sides of the split.

---

## Results

Average across all 15 subjects:

| Model | Avg F1 | Avg Recall | Avg Precision | Avg Accuracy |
|---|---|---|---|---|
| Random Forest (re-run, LOSO) | 0.6643 | 0.7161 | 0.7578 | 0.7862 |
| StressTransformer | 0.7661 | 0.7993 | 0.8614 | 0.8230 |
| LightweightCNN | 0.7946 | 0.8719 | 0.8373 | 0.8319 |

Both deep learning models outperform the Random Forest baseline. LightweightCNN has a slight edge on F1 and recall while StressTransformer has better precision, suggesting StressTransformer is more conservative about predicting stress. The standard deviations are high (around 0.24-0.26 F1) because some subjects have much more distinctive physiological stress responses than others. S2 and S4 were consistently hard across all models, while S8, S11, S14, S15, S16, and S17 were well above average.

Per-subject results are saved to `results/` as CSV files after running evaluation.

---

## Setup and Usage

```bash
pip install -r requirements.txt
```

Training (runs full LOSO, saves one model checkpoint per subject fold):

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

---

## File Structure

```
dataset.py        signal loading, downsampling, windowing, LOSO splits
models.py         StressTransformer and LightweightCNN architectures
train.py          LOSO training loop with weighted sampler
evaluate.py       evaluation script and Random Forest baseline
utils.py          metric computation
notebooks/        original baseline notebook kept for reference
results/          per-model CSV result files (generated after running)
WESAD/            dataset folder (not included, download separately)
```

---

## Reference

Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., and Van Laerhoven, K. Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection. ICMI 2018.

Original baseline notebook this project builds on: [kjspring/stress-detection-wearable-devices](https://github.com/kjspring/stress-detection-wearable-devices)
