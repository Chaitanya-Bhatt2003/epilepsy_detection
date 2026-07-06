# Epileptic Seizure Detection from EEG (CNN-LSTM)

Detects epileptic seizures from single-channel EEG signals using a CNN-LSTM deep
learning model. Trained on the UCI *Epileptic Seizure Recognition* dataset
(11,500 one-second EEG segments, 178 samples each).

**Results on held-out test data:** 98.4% accuracy · ROC-AUC 0.9988 · 343 of 345 seizures caught.

### ▶ [Try it live »](https://epilepsy-detection-chaitanya-bhatt-s-projects.vercel.app)

A fully interactive site (hosted on Vercel) with the project overview **and** a working
**seizure detector that runs entirely in your browser** — the CNN-LSTM model is served as
TensorFlow.js, so predictions happen client-side with no backend and nothing to install.
Click **Launch detector**, then try a random unseen recording, upload a CSV, or paste 178
numbers. (A GitHub Pages mirror of the overview is also at
`chaitanya-bhatt2003.github.io/epilepsy_detection`.)

Running `python app.py` still gives the same detector locally against the full held-out set.

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
| `web/` | Static site deployed to Vercel — in-browser detector (TensorFlow.js model, scaler + samples as JSON) |
| `scripts/convert_to_web.py` | Rebuilds the `web/` assets: converts the Keras model to TensorFlow.js and exports the scaler + samples |

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

## Deployment

The live site is hosted on **Vercel** from the `web/` folder and is connected to this
GitHub repo — every push to `main` automatically deploys to production (root directory
`web`, no build step). To rebuild the browser assets after retraining the model:

```bash
python scripts/convert_to_web.py   # regenerates web/model, web/scaler.json, web/samples.json
git commit -am "Update model" && git push   # auto-deploys
```
