"""
Epileptic Seizure Detection - inference on new EEG data.

Usage:
    python predict.py sample_new_eeg.csv
    python predict.py path/to/your_data.csv --threshold 0.5 --out predictions.csv

Input CSV: each ROW is one 1-second EEG segment of 178 samples.
- An optional leading id column (non-numeric) is ignored.
- An optional trailing label column named 'y' or 'label' is ignored for prediction.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("artifacts", "cnn_lstm_seizure.keras")
SCALER_PATH = os.path.join("artifacts", "scaler.joblib")
N_SAMPLES = 178


def load_segments(csv_path):
    """Read a CSV and return an (n, 178) float array of EEG segments."""
    df = pd.read_csv(csv_path)

    # drop a leading non-numeric id column if present (e.g. "X21.V1.791")
    if df.shape[1] and df.iloc[:, 0].dtype == object:
        df = df.iloc[:, 1:]
    # drop a label column if the file happens to include one
    for lbl in ("y", "label", "Y", "Label"):
        if lbl in df.columns:
            df = df.drop(columns=lbl)

    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] < N_SAMPLES:
        sys.exit(f"Error: found {df.shape[1]} numeric columns, need {N_SAMPLES} per segment.")
    if df.shape[1] > N_SAMPLES:
        df = df.iloc[:, :N_SAMPLES]  # keep the first 178 signal columns
    return df.values.astype("float32")


def main():
    ap = argparse.ArgumentParser(description="Detect seizures in new EEG segments.")
    ap.add_argument("csv", help="CSV file; each row = 178 EEG samples")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability cutoff for the seizure class (default 0.5)")
    ap.add_argument("--out", default="predictions.csv", help="where to save results")
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found at {MODEL_PATH}. Run the notebook to train and save it first.")

    import tensorflow as tf  # imported here so --help stays fast
    import joblib

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = load_segments(args.csv)
    Xs = scaler.transform(X)[..., np.newaxis]
    proba = model.predict(Xs, verbose=0).ravel()
    pred = (proba >= args.threshold).astype(int)

    result = pd.DataFrame({
        "segment": np.arange(len(proba)),
        "seizure_probability": proba.round(4),
        "prediction": np.where(pred == 1, "SEIZURE", "non-seizure"),
    })

    pd.set_option("display.width", 100)
    print(f"\nLoaded {len(X)} segment(s) from {args.csv}  (threshold = {args.threshold})\n")
    print(result.to_string(index=False))
    n_seiz = int(pred.sum())
    print(f"\nDetected {n_seiz} seizure segment(s) out of {len(pred)}.")

    result.to_csv(args.out, index=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
