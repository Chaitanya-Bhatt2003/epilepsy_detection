"""
One-off build script: convert the trained Keras 3 CNN-LSTM model into a
browser-ready TensorFlow.js model, and export the scaler + sample recordings
as JSON, so the whole detector can run client-side on Vercel (no backend).

Run from the repo root:  python scripts/convert_to_web.py
"""
import io
import json
import os
import sys
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# tensorflowjs's package __init__ eagerly imports the TFDF, TF-Hub and JAX
# converter backends. TFDF has no Windows wheel and JAX isn't needed for the
# Keras conversion path we use, so stub the ones that aren't installable/needed.
def _stub(name):
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
    return mod

_stub("tensorflow_decision_forests")
_jax = _stub("jax")
_jax_exp = _stub("jax.experimental")
_jax2tf = _stub("jax.experimental.jax2tf")
_jax_mon = _stub("jax.monitoring")   # keras 3 probes jax.monitoring.record_scalar
_jax.__version__ = "0.0.0-stub"
_jax.experimental = _jax_exp
_jax.monitoring = _jax_mon
_jax_exp.jax2tf = _jax2tf

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import tf_keras                      # Keras 2 -> guaranteed tfjs-loadable
from tensorflowjs.converters import save_keras_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
MODEL_OUT = os.path.join(WEB, "model")
os.makedirs(MODEL_OUT, exist_ok=True)

MODEL_PATH = os.path.join(ROOT, "artifacts", "cnn_lstm_seizure.keras")
SCALER_PATH = os.path.join(ROOT, "artifacts", "scaler.joblib")
N = 178

print("Loading trained Keras 3 model + scaler ...")
src = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def build_keras2():
    """Rebuild the exact same architecture using tf-keras (Keras 2)."""
    L = tf_keras.layers
    m = tf_keras.Sequential([
        L.Input(shape=(N, 1)),
        L.Conv1D(64, 5, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling1D(2),
        L.Conv1D(128, 3, padding="same", activation="relu"),
        L.BatchNormalization(),
        L.MaxPooling1D(2),
        L.Bidirectional(L.LSTM(64, return_sequences=True)),
        L.Dropout(0.3),
        L.Bidirectional(L.LSTM(32)),
        L.Dropout(0.3),
        L.Dense(64, activation="relu"),
        L.Dropout(0.4),
        L.Dense(1, activation="sigmoid"),
    ])
    return m


dst = build_keras2()
# Copy weights layer-for-layer (identical architecture + gate ordering).
dst.set_weights(src.get_weights())
print("Rebuilt model in tf-keras and copied weights.")

# ---- parity check on the bundled sample recordings -------------------------
mini = pd.read_csv(os.path.join(ROOT, "mini_test_eeg.csv"))
Xraw = mini.iloc[:, :N].values.astype("float32")
Xs = scaler.transform(Xraw)[..., np.newaxis]
p_src = src.predict(Xs, verbose=0).ravel()
p_dst = dst.predict(Xs, verbose=0).ravel()
max_diff = float(np.max(np.abs(p_src - p_dst)))
print(f"Max |Keras3 - Keras2| probability diff on samples: {max_diff:.3e}")
if max_diff > 1e-4:
    sys.exit("ERROR: rebuilt model diverges from original; aborting.")

# ---- convert the tf-keras model to TensorFlow.js ---------------------------
save_keras_model(dst, MODEL_OUT)
print(f"Saved TensorFlow.js model -> {MODEL_OUT}")

# ---- export the scaler (StandardScaler: (x - mean) / scale) ----------------
with open(os.path.join(WEB, "scaler.json"), "w") as f:
    json.dump({"mean": scaler.mean_.tolist(),
               "scale": scaler.scale_.tolist()}, f)
print("Wrote web/scaler.json")

# ---- bundle the 10 labelled recordings for the random-sample button --------
ans = pd.read_csv(os.path.join(ROOT, "answer_key.csv"))
label_map = {"seizure": "SEIZURE", "non-seizure": "non-seizure"}
samples = []
for i in range(len(mini)):
    truth = label_map[str(ans.iloc[i]["true_label"]).strip().lower()]
    samples.append({"signal": [int(v) for v in Xraw[i]],
                    "truth": truth,
                    "ref_proba": round(float(p_dst[i]), 6)})
with open(os.path.join(WEB, "samples.json"), "w") as f:
    json.dump(samples, f)
print(f"Wrote web/samples.json ({len(samples)} labelled recordings)")

print("\nReference probabilities (browser must reproduce these):")
for i, s in enumerate(samples):
    print(f"  #{i}  proba={s['ref_proba']:.4f}  truth={s['truth']}")
print("\nDone.")
