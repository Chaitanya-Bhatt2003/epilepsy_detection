# Epileptic Seizure Detection from EEG (CNN-LSTM)

Detects epileptic seizures from single-channel EEG signals using a CNN-LSTM deep
learning model. Trained on the UCI *Epileptic Seizure Recognition* dataset
(11,500 one-second EEG segments, 178 samples each).

**Results on held-out test data:** 98.4% accuracy · ROC-AUC 0.9988 · 343 of 345 seizures caught.

## Contents

| File | Description |
|------|-------------|
| `main_1.ipynb` | Full project notebook — EDA, preprocessing, baseline, CNN-LSTM, evaluation, inference |
| `index.html` | Presentation web page (open in any browser) |
| `app.py` | Live interactive demo — run it, open http://localhost:8000 |
| `predict.py` | Detect seizures in a new CSV: `python predict.py your_data.csv` |
| `score.py` | Grade the model on a labelled set: `python score.py` |
| `DEMO_GUIDE.md` | Presentation walkthrough / talking points |
| `cheatsheet_print.html` | One-page printable cheat sheet |
| `artifacts/` | Saved trained model (`cnn_lstm_seizure.keras`) + scaler |
| `sample_new_eeg.csv`, `mini_test_eeg.csv`, `answer_key.csv` | Demo / test data |

## Quick start

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib

# 1. Reproduce the whole project
jupyter notebook main_1.ipynb

# 2. Detect seizures in new EEG data (each row = 178 samples)
python predict.py sample_new_eeg.csv

# 3. Launch the live interactive demo
python app.py        # then open http://localhost:8000
```

## Data format

Each input row is one 1-second EEG segment = **178 numbers** (sampled at 178 Hz),
matching the training data. The dataset CSV is not included in this repo.

## Model

`Conv1D → BatchNorm → MaxPool` ×2 → `Bidirectional LSTM` ×2 → `Dense → Sigmoid`,
trained with class weighting for the ~20/80 seizure imbalance, early stopping, and
a clinically-aware threshold that favours seizure recall.
