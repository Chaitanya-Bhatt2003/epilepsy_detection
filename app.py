"""
Interactive Seizure Detector - a local web app for live demos.

Run it:
    python app.py
Then open http://localhost:8000 in a browser.

The examiner can:
  * click "Try a random unseen recording" -> model predicts, then reveals the true answer
  * upload their own CSV (each row = 178 EEG samples)
  * paste 178 numbers
  * slide the decision threshold and watch predictions change

Uses only the standard library + the already-installed model stack.
"""
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join("artifacts", "cnn_lstm_seizure.keras")
SCALER_PATH = os.path.join("artifacts", "scaler.joblib")
DATA_PATH = next((p for p in [
    "Epileptic Seizure TUH dataset.csv",
    "../Epileptic Seizure TUH dataset.csv",
] if os.path.exists(p)), None)
N = 178
PORT = 8000

print("Loading model ...")
import tensorflow as tf          # noqa: E402
import joblib                    # noqa: E402
MODEL = tf.keras.models.load_model(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
_lock = threading.Lock()         # Keras predict is not thread-safe

# ---- held-out test pool for the "random recording" button (honest: unseen data)
TEST_POOL = None
if DATA_PATH:
    from sklearn.model_selection import train_test_split
    _df = pd.read_csv(DATA_PATH)
    if _df.columns[0].lower().startswith("unnamed"):
        _df = _df.drop(columns=_df.columns[0])
    _df["label"] = (_df["y"] == 1).astype(int)
    _feat = [c for c in _df.columns if c.startswith("X")]
    _idx = np.arange(len(_df))
    _tr, _tmp = train_test_split(_idx, test_size=.30, stratify=_df["label"], random_state=42)
    _va, _te = train_test_split(_tmp, test_size=.50, stratify=_df["label"].iloc[_tmp], random_state=42)
    TEST_POOL = _df.iloc[_te].reset_index(drop=True)
    _POOL_FEAT = _feat
    print(f"Loaded {len(TEST_POOL)} held-out recordings for the random-sample demo.")
else:
    print("Dataset not found -> 'random recording' button disabled (upload/paste still work).")


def _predict(mat, threshold):
    """mat: (n,178) float array -> list of {probability, prediction}."""
    Xs = SCALER.transform(mat.astype("float32"))[..., np.newaxis]
    with _lock:
        proba = MODEL.predict(Xs, verbose=0).ravel()
    out = []
    for p in proba:
        out.append({
            "probability": round(float(p), 4),
            "prediction": "SEIZURE" if p >= threshold else "non-seizure",
        })
    return out


def _parse_csv(text):
    """Parse pasted / uploaded CSV text into an (n,178) array. Raises ValueError."""
    df = pd.read_csv(io.StringIO(text), header=None)
    # if the first row looks like a header (non-numeric), re-read with header
    if df.iloc[0].apply(lambda v: isinstance(v, str)).any():
        df = pd.read_csv(io.StringIO(text))
    if df.shape[1] and df.iloc[:, 0].dtype == object:
        df = df.iloc[:, 1:]                       # drop id column
    for lbl in ("y", "label", "Y", "Label"):
        if lbl in df.columns:
            df = df.drop(columns=lbl)             # drop label if present
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[1] < N:
        raise ValueError(f"Each row needs {N} numbers, but found only {df.shape[1]} columns.")
    df = df.iloc[:, :N]
    if df.isnull().values.any():
        raise ValueError("Some values are not numbers. Every cell must be a number.")
    if len(df) == 0:
        raise ValueError("No data rows found.")
    return df.values


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # quiet console
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, "not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or "{}")
            threshold = float(req.get("threshold", 0.5))

            truth = None
            if req.get("mode") == "random":
                if TEST_POOL is None:
                    raise ValueError("Dataset not available for random sampling.")
                k = int(req.get("n", 1))
                rows = TEST_POOL.sample(k)
                mat = rows[_POOL_FEAT].values
                truth = ["SEIZURE" if v == 1 else "non-seizure" for v in rows["label"].values]
            else:
                mat = _parse_csv(req.get("csv", ""))

            preds = _predict(mat, threshold)
            segments = []
            for i, pr in enumerate(preds):
                seg = {"index": i, **pr, "signal": [int(x) for x in mat[i]]}
                if truth is not None:
                    seg["truth"] = truth[i]
                    seg["correct"] = (truth[i] == pr["prediction"])
                segments.append(seg)
            self._send(200, json.dumps({"ok": True, "segments": segments,
                                        "has_truth": truth is not None}))
        except Exception as e:                       # noqa: BLE001
            self._send(200, json.dumps({"ok": False, "error": str(e)}))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seizure Detector - Live Demo</title>
<style>
:root{--bg:#eef1f7;--surface:#fff;--surface2:#f6f8fc;--border:#d7deec;--ink:#0f1826;
--soft:#3f4a5e;--muted:#6b7688;--accent:#0c8f86;--accsoft:#d6efec;--seiz:#cf3339;
--seizsoft:#fbe2e2;--trace:#0c8f86;--font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:ui-monospace,"Cascadia Code",Consolas,monospace;
--sh:0 1px 2px rgba(15,24,38,.05),0 8px 30px rgba(15,24,38,.06);}
@media(prefers-color-scheme:dark){:root{--bg:#090e17;--surface:#111927;--surface2:#0c131f;
--border:#1d2a3d;--ink:#e9eef7;--soft:#aab6c9;--muted:#6f7c92;--accent:#26cbbe;
--accsoft:rgba(38,203,190,.13);--seiz:#ff6a63;--seizsoft:rgba(255,106,99,.13);--trace:#26cbbe;
--sh:0 1px 2px rgba(0,0,0,.4),0 12px 40px rgba(0,0,0,.35);}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);
line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);font-weight:600}
h1{font-size:clamp(26px,4vw,40px);font-weight:800;letter-spacing:-.02em;margin:.3em 0}
.sub{color:var(--soft);max-width:60ch}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;
padding:22px;box-shadow:var(--sh);margin-top:20px}
.title{font-family:var(--mono);font-size:12px;letter-spacing:.1em;text-transform:uppercase;
color:var(--muted);font-weight:600;margin-bottom:14px}
.btn{font-family:var(--font);font-size:15px;font-weight:600;border:1px solid var(--border);
background:var(--surface2);color:var(--ink);padding:12px 18px;border-radius:10px;cursor:pointer;
transition:transform .05s,border-color .2s}
.btn:hover{border-color:var(--accent)}.btn:active{transform:translateY(1px)}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
@media(prefers-color-scheme:dark){.btn.primary{color:#04120f}}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.drop{border:2px dashed var(--border);border-radius:12px;padding:22px;text-align:center;
color:var(--muted);cursor:pointer;transition:.2s;font-size:14px}
.drop.hover{border-color:var(--accent);background:var(--accsoft)}
textarea{width:100%;min-height:70px;border:1px solid var(--border);border-radius:10px;
background:var(--surface2);color:var(--ink);font-family:var(--mono);font-size:12px;padding:10px;resize:vertical}
label.small{font-family:var(--mono);font-size:12px;color:var(--muted)}
input[type=range]{accent-color:var(--accent);vertical-align:middle}
.result{display:flex;gap:16px;align-items:center;padding:14px 0;border-top:1px solid var(--border)}
.result:first-child{border-top:none}
.pill{font-family:var(--mono);font-weight:700;font-size:13px;padding:8px 14px;border-radius:999px;
white-space:nowrap;flex:none;min-width:132px;text-align:center}
.pill.seiz{background:var(--seizsoft);color:var(--seiz);border:1px solid var(--seiz)}
.pill.non{background:var(--accsoft);color:var(--accent);border:1px solid var(--accent)}
.meta{flex:1;min-width:0}
.bar{height:8px;border-radius:6px;background:var(--surface2);overflow:hidden;margin-top:6px;border:1px solid var(--border)}
.bar>i{display:block;height:100%;background:var(--seiz);border-radius:6px}
.pct{font-family:var(--mono);font-size:12px;color:var(--muted)}
canvas{width:120px;height:48px;flex:none}
.truth{font-family:var(--mono);font-size:12px;font-weight:700;flex:none}
.truth.ok{color:var(--accent)}.truth.bad{color:var(--seiz)}
.summary{font-family:var(--mono);font-size:14px;margin-top:8px;color:var(--soft)}
.err{color:var(--seiz);font-family:var(--mono);font-size:13px;margin-top:10px}
.hint{font-family:var(--mono);font-size:11.5px;color:var(--muted);margin-top:10px;line-height:1.6}
a{color:var(--accent)}
</style></head><body><div class="wrap">
<div class="eyebrow">Live model - CNN-LSTM</div>
<h1>Seizure Detector</h1>
<p class="sub">Give the model one second of EEG (178 numbers). It returns a seizure verdict and its confidence. Use a real unseen recording, upload a file, or paste your own numbers.</p>

<div class="card">
  <div class="title">1 - Choose input</div>
  <div class="row">
    <button class="btn primary" id="rnd">&#127922; Try a random unseen recording</button>
    <button class="btn" id="rnd5">5 random recordings</button>
    <span class="hint" id="poolInfo"></span>
  </div>
  <div style="margin-top:16px">
    <div class="drop" id="drop">&#128193; <b>Upload a CSV</b> - or drag &amp; drop here<br>
      <span style="font-size:12px">each row = 178 EEG samples</span>
      <input type="file" id="file" accept=".csv,text/csv" hidden></div>
  </div>
  <details style="margin-top:14px"><summary class="label small" style="cursor:pointer">or paste numbers</summary>
    <textarea id="paste" placeholder="Paste 178 comma-separated numbers (one recording per line)"></textarea>
    <button class="btn" id="pasteBtn" style="margin-top:8px">Check pasted data</button>
  </details>
</div>

<div class="card">
  <div class="title">2 - Decision threshold</div>
  <div class="row">
    <input type="range" id="thr" min="0.05" max="0.95" step="0.01" value="0.5" style="flex:1">
    <span class="pct" id="thrVal">0.50</span>
  </div>
  <div class="hint">Lower = catch more seizures (fewer misses, more false alarms). Higher = fewer false alarms. This is the clinical trade-off from the project.</div>
</div>

<div class="card" id="out" style="display:none">
  <div class="title">Result</div>
  <div id="results"></div>
  <div class="summary" id="summary"></div>
  <div class="err" id="err"></div>
</div>

<p class="hint">Local demo running on your machine - nothing is uploaded to the internet. Input must match the training format: single-channel EEG, 178 samples per second.</p>
</div>
<script>
var thr=document.getElementById('thr'),thrVal=document.getElementById('thrVal');
var lastPayload=null;
thr.oninput=function(){thrVal.textContent=(+thr.value).toFixed(2);
  if(lastPayload){lastPayload.threshold=+thr.value;send(lastPayload);}};
document.getElementById('poolInfo').textContent="";

function drawSig(cv,sig){var c=cv.getContext('2d'),w=cv.width=120,h=cv.height=48;
  c.clearRect(0,0,w,h);var mx=1;for(var i=0;i<sig.length;i++)mx=Math.max(mx,Math.abs(sig[i]));
  var s=(h*0.42)/mx,mid=h/2,st=getComputedStyle(document.body).getPropertyValue('--trace');
  c.strokeStyle=st;c.lineWidth=1.5;c.beginPath();
  for(var j=0;j<sig.length;j++){var x=j/(sig.length-1)*w,y=mid-sig[j]*s;
    j?c.lineTo(x,y):c.moveTo(x,y);}c.stroke();}

function render(d){
  var out=document.getElementById('out'),res=document.getElementById('results'),
      err=document.getElementById('err'),sum=document.getElementById('summary');
  out.style.display='block';res.innerHTML='';err.textContent='';sum.textContent='';
  if(!d.ok){err.textContent='&#9888; '+d.error;return;}
  var seiz=0,correct=0;
  d.segments.forEach(function(s){
    if(s.prediction==='SEIZURE')seiz++;
    var div=document.createElement('div');div.className='result';
    var isS=s.prediction==='SEIZURE';
    var truthHtml='';
    if(s.truth!==undefined){if(s.correct)correct++;
      truthHtml='<span class="truth '+(s.correct?'ok':'bad')+'">'+
        (s.correct?'✔ correct':'✗ actual: '+s.truth)+'</span>';}
    div.innerHTML=
      '<span class="pill '+(isS?'seiz':'non')+'">'+(isS?'⚠ SEIZURE':'non-seizure')+'</span>'+
      '<div class="meta"><span class="pct">recording #'+s.index+' &middot; confidence '+
        (s.probability*100).toFixed(1)+'%</span>'+
        '<div class="bar"><i style="width:'+(s.probability*100).toFixed(1)+'%"></i></div></div>'+
      '<canvas></canvas>'+truthHtml;
    res.appendChild(div);drawSig(div.querySelector('canvas'),s.signal);
  });
  var msg=d.segments.length+' recording(s) &middot; '+seiz+' flagged as seizure';
  if(d.has_truth)msg+=' &middot; '+correct+'/'+d.segments.length+' matched the true answer';
  sum.innerHTML=msg;
}

function send(payload){lastPayload=payload;
  fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)}).then(r=>r.json()).then(render)
    .catch(e=>render({ok:false,error:e.message}));}

document.getElementById('rnd').onclick=function(){send({mode:'random',n:1,threshold:+thr.value});};
document.getElementById('rnd5').onclick=function(){send({mode:'random',n:5,threshold:+thr.value});};
document.getElementById('pasteBtn').onclick=function(){
  send({csv:document.getElementById('paste').value,threshold:+thr.value});};

var drop=document.getElementById('drop'),file=document.getElementById('file');
drop.onclick=function(){file.click();};
file.onchange=function(){var f=file.files[0];if(!f)return;var r=new FileReader();
  r.onload=function(){send({csv:r.result,threshold:+thr.value});};r.readAsText(f);};
['dragover','dragenter'].forEach(function(ev){drop.addEventListener(ev,function(e){
  e.preventDefault();drop.classList.add('hover');});});
['dragleave','drop'].forEach(function(ev){drop.addEventListener(ev,function(e){
  e.preventDefault();drop.classList.remove('hover');});});
drop.addEventListener('drop',function(e){var f=e.dataTransfer.files[0];if(!f)return;
  var r=new FileReader();r.onload=function(){send({csv:r.result,threshold:+thr.value});};
  r.readAsText(f);});
</script></body></html>"""


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("\n" + "=" * 52)
    print(f"  Seizure Detector is running!")
    print(f"  Open this in your browser:  http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop.")
    print("=" * 52 + "\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
