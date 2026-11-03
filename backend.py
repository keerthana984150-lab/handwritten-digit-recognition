"""
Backend for the Handwritten Digit Recognition Dashboard.

INTEGRATION POINT: `load_dataset()` and `CLASSIFIERS` / `REGRESSORS` are the only
places that touch your ML code. Point `load_dataset()` at your own data loader and
replace the factories with your own estimators (anything with fit / predict, plus
predict_proba for classifiers). Everything else (metrics, history, persistence)
is computed from real evaluation results and stored in SQLite.
"""
import base64, io, json, re, sqlite3, time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            f1_score, mean_absolute_error, mean_squared_error,
                            precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

BASE = Path(__file__).parent
DB_PATH = BASE / "data" / "dashboard.db"
MODEL_DIR = BASE / "models"
DB_PATH.parent.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42

# --------------------------------------------------------------------------- data
def load_dataset():
    """Return X (n, 64) pixel values scaled to 0-1, y (n,), raw images (n, 8, 8).
    Replace with your own loader if your project uses a different dataset."""
    d = load_digits()
    return d.data / 16.0, d.target, d.images

def get_split():
    X, y, imgs = load_dataset()
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    return X[tr], X[te], y[tr], y[te], tr, te

# ------------------------------------------------------------------------ models
class KerasCNN:
    """Small CNN (needs TensorFlow). Exposes a scikit-learn style API."""
    classes_ = np.arange(10)
    impl = "TensorFlow/Keras CNN"
    def __init__(self, epochs=30): self.epochs = epochs; self.model = None
    def _build(self):
        from tensorflow import keras
        L = keras.layers
        m = keras.Sequential([L.Input((8, 8, 1)), L.Conv2D(32, 3, padding="same", activation="relu"),
                              L.Conv2D(64, 3, padding="same", activation="relu"), L.MaxPooling2D(),
                              L.Flatten(), L.Dense(128, activation="relu"), L.Dropout(0.3),
                              L.Dense(10, activation="softmax")])
        m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return m
    def fit(self, X, y):
        self.model = self._build()
        self.model.fit(X.reshape(-1, 8, 8, 1), y, epochs=self.epochs, batch_size=32,
                       validation_split=0.1, verbose=0)
        return self
    def predict_proba(self, X): return self.model.predict(np.asarray(X).reshape(-1, 8, 8, 1), verbose=0)
    def predict(self, X): return self.predict_proba(X).argmax(1)
    def __getstate__(self): return {"epochs": self.epochs, "w": self.model.get_weights()}
    def __setstate__(self, s):
        self.epochs = s["epochs"]; self.model = self._build(); self.model.set_weights(s["w"])

def make_cnn():
    try:
        import tensorflow  # noqa
        return KerasCNN()
    except Exception:
        m = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=SEED)
        m.impl = "MLP stand-in (TensorFlow not installed)"
        return m

CLASSIFIERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
    "Decision Tree": lambda: DecisionTreeClassifier(random_state=SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "CNN": make_cnn,
}
REGRESSORS = {
    "Linear Regression": lambda: LinearRegression(),
    "Decision Tree Regressor": lambda: DecisionTreeRegressor(random_state=SEED),
    "Random Forest Regressor": lambda: RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
}

# ---------------------------------------------------------------------- database
def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    return c

def init_db():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS training_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT, task TEXT, model TEXT, version INTEGER,
            trained_at TEXT, duration REAL, dataset_size INTEGER, train_score REAL,
            test_score REAL, status TEXT, impl TEXT, metrics_json TEXT);
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, image TEXT, actual INTEGER,
            predicted INTEGER, confidence REAL, model TEXT, status TEXT, source TEXT);
        """)

def q(sql, params=()):
    with conn() as c:
        return pd.read_sql_query(sql, c, params=params)

def _slug(n): return re.sub(r"\W+", "_", n.lower())

def has_runs(): return int(q("SELECT COUNT(*) n FROM training_runs").n[0]) > 0

# ---------------------------------------------------------------------- training
def _train_one(task, name, factory, data):
    Xtr, Xte, ytr, yte = data
    model = factory()
    t0 = time.time(); status = "Completed"
    try:
        model.fit(Xtr, ytr)
    except Exception as e:  # keep history even if a model fails
        status = f"Failed: {e}"[:80]
    dur = time.time() - t0
    if status != "Completed":
        _log(task, name, dur, len(ytr) + len(yte), None, None, status, "", {}); return
    p_tr, p_te = model.predict(Xtr), model.predict(Xte)
    if task == "classification":
        m = dict(accuracy=accuracy_score(yte, p_te),
                 precision=precision_score(yte, p_te, average="weighted", zero_division=0),
                 recall=recall_score(yte, p_te, average="weighted", zero_division=0),
                 f1=f1_score(yte, p_te, average="weighted", zero_division=0),
                 cm=confusion_matrix(yte, p_te, labels=range(10)).tolist(),
                 report=classification_report(yte, p_te, output_dict=True, zero_division=0))
        tr_s, te_s = accuracy_score(ytr, p_tr), m["accuracy"]
    else:
        mse = mean_squared_error(yte, p_te)
        m = dict(mae=mean_absolute_error(yte, p_te), mse=mse, rmse=float(np.sqrt(mse)),
                 r2=r2_score(yte, p_te), y_true=[int(v) for v in yte],
                 y_pred=[round(float(v), 3) for v in p_te])
        tr_s, te_s = r2_score(ytr, p_tr), m["r2"]
    joblib.dump(model, MODEL_DIR / f"{task}_{_slug(name)}.joblib")
    _log(task, name, dur, len(ytr) + len(yte), tr_s, te_s, status, getattr(model, "impl", type(model).__name__), m)

def _log(task, name, dur, n, tr, te, status, impl, metrics):
    with conn() as c:
        v = c.execute("SELECT COALESCE(MAX(version),0)+1 FROM training_runs WHERE task=? AND model=?",
                      (task, name)).fetchone()[0]
        c.execute("""INSERT INTO training_runs(task,model,version,trained_at,duration,dataset_size,
                     train_score,test_score,status,impl,metrics_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                  (task, name, v, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), dur, n, tr, te,
                   status, impl, json.dumps(metrics)))

def train_all(progress=None):
    Xtr, Xte, ytr, yte, _, _ = get_split()
    jobs = [("classification", n, f) for n, f in CLASSIFIERS.items()] + \
           [("regression", n, f) for n, f in REGRESSORS.items()]
    for i, (task, name, f) in enumerate(jobs):
        if progress: progress(i / len(jobs), f"Training {name}…")
        _train_one(task, name, f, (Xtr, Xte, ytr, yte))
    if progress: progress(1.0, "Done")

def latest_runs(task=None):
    df = q("""SELECT * FROM training_runs WHERE id IN
              (SELECT MAX(id) FROM training_runs WHERE status='Completed' GROUP BY task, model)""")
    if task: df = df[df.task == task]
    if df.empty: return df
    m = df.metrics_json.apply(json.loads).apply(pd.Series)
    return pd.concat([df.drop(columns="metrics_json").reset_index(drop=True), m.reset_index(drop=True)], axis=1)

# -------------------------------------------------------------------- prediction
def load_model(name, task="classification"):
    return joblib.load(MODEL_DIR / f"{task}_{_slug(name)}.joblib")

def to_png_b64(arr01, size=64):
    im = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype("uint8").reshape(8, 8)).resize((size, size), Image.NEAREST)
    b = io.BytesIO(); im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()

def preprocess(img: Image.Image):
    """Any PIL image -> (64,) vector in 0-1, matching the training data format
    (white ink on black, digit centred with margin, 8x8)."""
    g = np.asarray(img.convert("L"), dtype=float)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if border.mean() > 127: g = 255 - g           # dark ink on light paper -> invert
    g = g - g.min()
    if g.max() == 0: return None
    g = g / g.max()
    ys, xs = np.where(g > 0.25)
    if len(ys) == 0: return None
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = int(max(g.shape) * 1.4) + 2
    sq = np.zeros((s, s)); oy, ox = (s - g.shape[0]) // 2, (s - g.shape[1]) // 2
    sq[oy:oy + g.shape[0], ox:ox + g.shape[1]] = g
    small = Image.fromarray((sq * 255).astype("uint8")).resize((8, 8), Image.LANCZOS)
    v = np.asarray(small, dtype=float) / 255.0
    return (v / v.max()).ravel() if v.max() > 0 else None

def predict_and_log(model_name, x, actual=None, source="Upload"):
    model = load_model(model_name)
    proba = model.predict_proba(x.reshape(1, -1))[0]
    pred = int(model.classes_[int(np.argmax(proba))]); conf = float(proba.max())
    status = "Unknown" if actual is None else ("Correct" if pred == actual else "Incorrect")
    with conn() as c:
        c.execute("""INSERT INTO predictions(ts,image,actual,predicted,confidence,model,status,source)
                     VALUES(?,?,?,?,?,?,?,?)""",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), to_png_b64(x), actual, pred, conf,
                   model_name, status, source))
    return pred, conf, proba, status

def predictions():
    df = q("SELECT * FROM predictions ORDER BY id DESC")
    if not df.empty: df["ts"] = pd.to_datetime(df.ts)
    return df

def clear_predictions():
    with conn() as c: c.execute("DELETE FROM predictions")

# ----------------------------------------------------------------- dataset stats
def dataset_stats():
    X, y, imgs = load_dataset()
    _, _, ytr, yte, _, _ = get_split()
    dist = pd.DataFrame({"Digit": range(10),
                         "Train": np.bincount(ytr, minlength=10),
                         "Test": np.bincount(yte, minlength=10)})
    dist["Total"] = dist.Train + dist.Test
    checks = pd.DataFrame([
        ("Missing values", "None found" if not np.isnan(X).any() else "Present", not np.isnan(X).any()),
        ("Pixel normalisation", f"Scaled to [{X.min():.1f}, {X.max():.1f}]", X.max() <= 1.0),
        ("Image shape", f"{imgs.shape[1]}×{imgs.shape[2]} grayscale", True),
        ("Duplicate images", f"{len(X) - len(np.unique(X, axis=0))} duplicates", True),
        ("Stratified split", "80% train / 20% test", True),
    ], columns=["Step", "Result", "ok"])
    return dict(total=len(y), train=len(ytr), test=len(yte), dist=dist, checks=checks, X=X, y=y, imgs=imgs)

def init():
    init_db()
