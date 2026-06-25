import os
import pickle
import numpy as np
from scipy.signal import resample
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# chest signals we use — ECG, EDA, RESP, TEMP
# ACC and EMG dropped: less relevant for stress, adds noise
SIGNAL_KEYS = ['ECG', 'EDA', 'Resp', 'Temp']
CHEST_SIGNAL_KEYS = ['ACC', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']

ORIG_FS = 700    # RespiBAN sampling rate
TARGET_FS = 100  # downsample to 100 Hz — manageable for CNN
WINDOW_SEC = 10  # 10 second windows
STRIDE_SEC = 5   # 50% overlap

WINDOW_SIZE = TARGET_FS * WINDOW_SEC   # 1000 samples
STRIDE = TARGET_FS * STRIDE_SEC        # 500 samples

# subjects in dataset — S1 and S12 missing due to sensor malfunction
ALL_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]


def load_subject(data_dir, subject_id):
    pkl_path = os.path.join(data_dir, f'S{subject_id}', f'S{subject_id}.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def extract_signals(data):
    # pull ECG, EDA, RESP, TEMP from chest RespiBAN
    chest = data['signal']['chest']
    signals = []
    for key in SIGNAL_KEYS:
        sig = chest[key].squeeze()
        signals.append(sig)
    return np.stack(signals, axis=0)  # (4, n_samples)


def downsample_signals(signals):
    # signals: (4, n_samples) at 700 Hz -> 100 Hz
    n_channels, n_samples = signals.shape
    n_new = int(n_samples * TARGET_FS / ORIG_FS)
    downsampled = resample(signals, n_new, axis=1)
    return downsampled


def extract_labels(data):
    labels = data['label'].squeeze()  # at 700 Hz
    n_new = int(len(labels) * TARGET_FS / ORIG_FS)
    # downsample labels by taking majority in each block
    block_size = len(labels) // n_new
    labels_down = []
    for i in range(n_new):
        block = labels[i * block_size: (i + 1) * block_size]
        labels_down.append(np.bincount(block.astype(int)).argmax())
    return np.array(labels_down)


def create_windows(signals, labels):
    # signals: (4, n_samples), labels: (n_samples,)
    # keep only label 1 (baseline) and label 2 (stress)
    windows, targets = [], []
    n = signals.shape[1]
    for start in range(0, n - WINDOW_SIZE, STRIDE):
        end = start + WINDOW_SIZE
        window_labels = labels[start:end]
        # majority label in window
        counts = np.bincount(window_labels.astype(int), minlength=8)
        majority = np.argmax(counts)
        # skip windows with mixed or irrelevant labels
        if majority not in [1, 2]:
            continue
        # check label is consistent enough (>70% same label)
        if counts[majority] / len(window_labels) < 0.7:
            continue
        windows.append(signals[:, start:end])
        # remap: 1 (baseline) -> 0, 2 (stress) -> 1
        targets.append(0 if majority == 1 else 1)
    return np.array(windows), np.array(targets)


def load_all_subjects(data_dir):
    all_windows, all_labels, all_subject_ids = [], [], []
    for sid in ALL_SUBJECTS:
        try:
            data = load_subject(data_dir, sid)
            signals = extract_signals(data)
            signals = downsample_signals(signals)
            labels = extract_labels(data)
            windows, targets = create_windows(signals, labels)
            if len(windows) == 0:
                print(f'S{sid}: no valid windows, skipping')
                continue
            all_windows.append(windows)
            all_labels.append(targets)
            all_subject_ids.extend([sid] * len(windows))
            print(f'S{sid}: {len(windows)} windows, stress={targets.sum()}, non-stress={(targets==0).sum()}')
        except Exception as e:
            print(f'S{sid}: failed — {e}')
    return (np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels, axis=0),
            np.array(all_subject_ids))


def normalize_windows(X_train, X_test):
    # normalize per channel across training windows
    n_channels = X_train.shape[1]
    X_train_norm = X_train.copy().astype(np.float32)
    X_test_norm = X_test.copy().astype(np.float32)
    for c in range(n_channels):
        scaler = StandardScaler()
        # fit on training channel data flattened
        scaler.fit(X_train[:, c, :].reshape(-1, 1))
        X_train_norm[:, c, :] = scaler.transform(
            X_train[:, c, :].reshape(-1, 1)).reshape(-1, WINDOW_SIZE)
        X_test_norm[:, c, :] = scaler.transform(
            X_test[:, c, :].reshape(-1, 1)).reshape(-1, WINDOW_SIZE)
    return X_train_norm, X_test_norm


def loso_split(windows, labels, subject_ids, test_subject):
    # leave one subject out split
    test_mask = subject_ids == test_subject
    train_mask = ~test_mask
    X_train = windows[train_mask]
    y_train = labels[train_mask]
    X_test = windows[test_mask]
    y_test = labels[test_mask]
    X_train, X_test = normalize_windows(X_train, X_test)
    return X_train, y_train, X_test, y_test


class WESADDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = torch.from_numpy(windows).float()
        self.labels = torch.from_numpy(labels).long()

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)