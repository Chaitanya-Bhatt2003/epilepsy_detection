# Demo Cheat-Sheet — Epileptic Seizure Detection

A step-by-step script for presenting this project. Keep this open on your phone or a
second screen. Times are rough — the whole demo fits in **6–8 minutes**.

---

## 0. Before the examiner arrives (2 minutes of setup)

Open a terminal **inside the project folder**
`C:\Users\acer\Desktop\epilepsy_detection` and start the live app:

```
python app.py
```

Wait ~15 seconds until it prints **"Seizure Detector is running!"**, then open a browser
tab at **http://localhost:8000**. Leave it ready.

Also have these open in tabs, in this order:
1. `index.html` — the project website (double-click it, or it may already be open)
2. `main_1.ipynb` — the notebook (in VS Code or Jupyter)
3. The browser tab at **http://localhost:8000** — the live demo

> If `python app.py` fails, no problem — the website and notebook alone are enough.
> The live app is the "wow", not a crutch.

---

## 1. Open with the website  (~1.5 min)

Show `index.html` full-screen. Scroll slowly, top to bottom.

**Say:**
> "My project detects epileptic seizures from EEG brain-wave signals using a deep
> learning model. This is the title slide. As I scroll, you can see a live animation of
> an EEG trace — and when a seizure pattern passes through, the monitor flags it in red,
> which is exactly what my model does."

Pause on the **results section**.

**Say:**
> "On unseen test data the model reaches **98.4% accuracy** and an AUC of **0.9988**.
> Most importantly, out of 345 real seizures it caught 343 — it almost never misses one,
> which is what matters clinically."

---

## 2. Explain the method briefly  (~1 min)

Point at the **pipeline** and **architecture** sections on the page.

**Say:**
> "The data is the UCI Epileptic Seizure dataset — 11,500 one-second EEG segments.
> I framed it as seizure vs. non-seizure. The model is a **CNN-LSTM**: the convolution
> layers learn the local shape of the signal — the spikes — and the LSTM models how those
> patterns evolve across the one second. I also trained a Random Forest as a baseline to
> prove the deep model was worth it."

---

## 3. The live demo — the highlight  (~2.5 min)

Switch to the browser tab **http://localhost:8000**.

### a) Random unseen recording
Click **"🎲 Try a random unseen recording"** two or three times.

**Say:**
> "This pulls a real EEG recording the model has **never seen during training**, predicts,
> and then reveals the true answer. Watch — it predicts SEIZURE with 99% confidence, and
> the checkmark confirms it was correct. Let me try a few more."

Click **"5 random recordings"** once.

**Say:**
> "Here are five at once — you can see the model's confidence bar and the actual brain-wave
> shape for each, and how many it got right."

### b) The threshold slider (shows you understand the trade-off)
Drag the **decision-threshold slider** left, then right.

**Say:**
> "This slider is the clinical trade-off. Slide it low and the model catches more seizures
> but raises more false alarms; slide it high and false alarms drop. Because a missed
> seizure is far more dangerous than a false alarm, in practice I'd set it to favour
> catching every seizure."

### c) (Optional) Let the examiner drive
**Say:**
> "You're welcome to upload your own file or click the buttons yourself."

---

## 4. Show it's real code, not smoke and mirrors  (~1 min)

Switch to `main_1.ipynb`. Scroll to **Section 8 – Detecting seizures in new EEG data**.

**Say:**
> "Everything is reproducible in the notebook. This section loads the saved model and runs
> it on new data. I can also run a blind test from the terminal."

*(Optional live terminal flourish — in a second terminal:)*
```
python score.py
```
**Say:**
> "This grades the model on 10 held-out recordings against a hidden answer key — 10 out
> of 10 correct."

---

## 5. Close  (~30 sec)

**Say:**
> "To summarise: I built a CNN-LSTM that detects seizures from EEG at 98% accuracy, wrapped
> it in a live tool that works on new data, and validated it honestly on data the model
> never saw. The natural next step is patient-independent testing to prove it generalises
> across people."

---

## Likely examiner questions — and honest answers

**Q: Does this work on any EEG machine's data?**
> Not directly. The model expects the same format as its training data — single-channel EEG
> at 178 samples per second. Data from a different device would need resampling and rescaling
> first. That's why my live demo draws from real recordings in this exact format.

**Q: How do you know it isn't just memorising / overfitting?**
> I used a stratified train / validation / test split with a fixed seed, and the scaler was
> fit on training data only, so no test information leaks in. All the numbers I report are on
> the held-out **test** set the model never trained on. I also used early stopping.

**Q: The classes are imbalanced (20% seizure) — did you handle that?**
> Yes. I used class weighting so the rare seizure class counts more during training, and I
> report recall and precision for the seizure class, not just accuracy, because accuracy alone
> can hide poor performance on the minority class.

**Q: Why CNN **and** LSTM, not just one?**
> The CNN captures the local shape — the spikes and sharp waves. The LSTM captures the temporal
> pattern over the full second. My Random Forest baseline (AUC 0.9952) shows the deep model
> (AUC 0.9988) genuinely adds value.

**Q: What's the false-alarm rate?**
> On the test set, 26 false alarms out of 1,380 normal segments — about 2% — while missing only
> 2 of 345 seizures. The threshold slider lets me trade these off.

**Q: What would you improve with more time?**
> Patient-independent splits (grouping by subject) to prove it generalises to new people,
> k-fold cross-validation for tighter confidence, spectrogram features, the full 5-class
> version, and exporting to TensorFlow Lite for a real wearable device.

**Q: What are the real-world applications?**
> Wearable seizure-alert devices, automated screening to assist neurologists, and long-term
> monitoring where reviewing hours of EEG by hand is impractical.

---

## Quick troubleshooting

| Problem | Fix |
|---|---|
| `python app.py` won't start / "port in use" | Change `PORT = 8000` near the top of `app.py` to `8080`, rerun, open `http://localhost:8080`. |
| Browser page is blank | Wait for the terminal to say "running", then refresh. |
| "Model not found" | Run the notebook once to train and save the model into `artifacts/`. |
| App is slow on first click | First prediction warms up TensorFlow; it's instant afterwards. |
| No internet in the exam room | Fine — everything runs locally and offline. |

---

## The one-sentence pitch (memorise this)

> **"I built an AI that reads one second of brain-wave data and detects epileptic seizures
> with 98% accuracy, and I can feed it brand-new recordings live and watch it flag the
> seizures correctly."**
