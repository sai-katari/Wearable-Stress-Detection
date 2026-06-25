import argparse
import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from dataset import load_all_subjects, loso_split, WESADDataset, ALL_SUBJECTS
from models import stress_transformer, lightweight_cnn
from utils import compute_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='WESAD')
    parser.add_argument('--model', type=str, default='stress_transformer',
                        choices=['stress_transformer', 'lightweight_cnn', 'random_forest'])
    parser.add_argument('--use-gpu', default=False, action='store_true')
    parser.add_argument('--results-dir', type=str, default='results')
    return parser.parse_args()


def eval_random_forest(windows, labels, subject_ids):
    # flatten windows for RF: (n_windows, 4*1000) -> use statistical features
    print('Running Random Forest baseline (statistical features)...')
    subjects = np.unique(subject_ids)
    all_metrics = []
    for test_subject in subjects:
        mask = subject_ids == test_subject
        X_train = windows[~mask].reshape(len(windows[~mask]), -1)
        y_train = labels[~mask]
        X_test = windows[mask].reshape(len(windows[mask]), -1)
        y_test = labels[mask]

        # use mean + std per channel as features (same as original paper approach)
        def extract_features(X):
            n = X.shape[0]
            X_reshaped = X.reshape(n, 4, -1)
            feats = np.concatenate([
                X_reshaped.mean(axis=2),
                X_reshaped.std(axis=2),
                np.percentile(X_reshaped, 25, axis=2),
                np.percentile(X_reshaped, 75, axis=2),
            ], axis=1)
            return feats

        X_train_feat = extract_features(X_train.reshape(-1, 4, 1000))
        X_test_feat = extract_features(X_test.reshape(-1, 4, 1000))

        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train_feat, y_train)
        y_pred = rf.predict(X_test_feat)
        metrics = compute_metrics(y_test, y_pred)
        all_metrics.append(metrics)
        print(f'S{test_subject}:', end=' ')
        print_metrics(metrics)

    print('\nRandom Forest LOSO Average:')
    for key in ['f1', 'recall', 'precision', 'accuracy']:
        vals = [m[key] for m in all_metrics]
        print(f'Avg {key.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    return all_metrics


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device('cuda:0') if args.use_gpu and torch.cuda.is_available() else torch.device('cpu')

    print('Loading subjects...')
    windows, labels, subject_ids = load_all_subjects(args.data_dir)

    if args.model == 'random_forest':
        eval_random_forest(windows, labels, subject_ids)
    else:
        subjects = np.unique(subject_ids)
        all_metrics = []
        for test_subject in subjects:
            X_train, y_train, X_test, y_test = loso_split(
                windows, labels, subject_ids, test_subject)
            test_dataset = WESADDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            model_path = f'models/{args.model}_S{test_subject}.pth'
            if not os.path.exists(model_path):
                print(f'S{test_subject}: no saved model found, skipping')
                continue

            if args.model == 'stress_transformer':
                model = stress_transformer().to(device)
            else:
                model = lightweight_cnn().to(device)

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            all_preds, all_true, all_probs = [], [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = model(x)
                    probs = torch.softmax(out, dim=1)[:, 1]
                    preds = out.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(y.numpy())
                    all_probs.extend(probs.cpu().numpy())

            y_true = np.array(all_true)
            y_pred = np.array(all_preds)
            y_prob = np.array(all_probs)
            metrics = compute_metrics(y_true, y_pred, y_prob)
            all_metrics.append(metrics)
            print(f'S{test_subject}:', end=' ')
            print_metrics(metrics)

            # save confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=['Non-stress', 'Stress'])
            disp.plot()
            plt.title(f'{args.model} — Subject S{test_subject}')
            plt.savefig(os.path.join(args.results_dir, f'cm_{args.model}_S{test_subject}.png'))
            plt.close()

        print(f'\n{args.model} LOSO Average:')
        for key in ['f1', 'recall', 'precision', 'accuracy', 'auc']:
            vals = [m[key] for m in all_metrics if key in m]
            print(f'Avg {key.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

        df = pd.DataFrame(all_metrics, index=[f'S{s}' for s in subjects[:len(all_metrics)]])
        df.to_csv(os.path.join(args.results_dir, f'{args.model}_eval_results.csv'))