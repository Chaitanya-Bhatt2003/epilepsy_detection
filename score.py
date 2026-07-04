"""
Grade the seizure detector on a labelled test set (blind demo).

Feeds an unlabelled EEG file to the model, then compares each prediction
against a separate answer key and reports accuracy.

Usage:
    python score.py                      # uses mini_test_eeg.csv + answer_key.csv
    python score.py my_eeg.csv my_key.csv --threshold 0.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("artifacts", "cnn_lstm_seizure.keras")
SCALER_PATH = os.path.join("artifacts", "scaler.joblib")
N_SAMPLES = 178


def main():
    ap = argparse.ArgumentParser(description="Score the model against an answer key.")
    ap.add_argument("eeg", nargs="?", default="mini_test_eeg.csv",
                    help="CSV of EEG segments (rows of 178 samples, no labels)")
    ap.add_argument("key", nargs="?", default="answer_key.csv",
                    help="CSV answer key with columns: segment, true_label")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="seizure probability cutoff (default 0.5)")
    args = ap.parse_args()

    for p in (MODEL_PATH, SCALER_PATH, args.eeg, args.key):
        if not os.path.exists(p):
            sys.exit(f"Missing required file: {p}")

    import tensorflow as tf
    import joblib

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    eeg = pd.read_csv(args.eeg).select_dtypes("number").iloc[:, :N_SAMPLES]
    key = pd.read_csv(args.key)

    if len(eeg) != len(key):
        sys.exit(f"Row mismatch: {len(eeg)} EEG segments vs {len(key)} answers.")

    X = scaler.transform(eeg.values.astype("float32"))[..., np.newaxis]
    proba = model.predict(X, verbose=0).ravel()
    pred = np.where(proba >= args.threshold, "seizure", "non-seizure")

    truth = key["true_label"].str.strip().values
    correct = pred == truth

    report = pd.DataFrame({
        "segment": key["segment"].values,
        "true_label": truth,
        "predicted": pred,
        "probability": proba.round(3),
        "result": np.where(correct, "OK", "WRONG"),
    })

    acc = correct.mean()
    print(f"\nBlind test - {len(report)} held-out EEG segments "
          f"(threshold = {args.threshold})\n")
    print(report.to_string(index=False))
    print(f"\nScore: {correct.sum()}/{len(correct)} correct  =  {acc:.0%} accuracy")

    # per-class breakdown
    for cls in ("seizure", "non-seizure"):
        m = truth == cls
        if m.any():
            print(f"  {cls:>11}: {correct[m].sum()}/{m.sum()} correct")


if __name__ == "__main__":
    main()
