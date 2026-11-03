import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

import backend as B

st.set_page_config(page_title="Digit Recognition Dashboard", page_icon="🧠", layout="wide")
B.init()

# ------------------------------------------------------------------------- theme
if "dark" not in st.session_state: st.session_state.dark = True
DARK = st.session_state.dark
PAL = dict(
    bg="linear-gradient(135deg,#0b1030 0%,#1a1147 55%,#2d1b69 100%)" if DARK else "linear-gradient(135deg,#f4f6ff 0%,#ece6ff 100%)",
    side="linear-gradient(180deg,#0a0e27 0%,#1b1250 100%)" if DARK else "linear-gradient(180deg,#ffffff 0%,#efe9ff 100%)",
    card="rgba(255,255,255,0.06)" if DARK else "rgba(255,255,255,0.9)",
    border="rgba(139,92,246,0.35)" if DARK else "rgba(109,40,217,0.2)",
    text="#e8eaff" if DARK else "#1e1b4b", sub="#a5a9d6" if DARK else "#5b5f8f")
st.markdown(f"""<style>
.stApp {{ background:{PAL['bg']}; color:{PAL['text']}; }}
[data-testid="stSidebar"] {{ background:{PAL['side']}; border-right:1px solid {PAL['border']}; }}
[data-testid="stHeader"] {{ background:transparent; }}
.stApp, .stApp p, .stApp label, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stSidebar"] * {{ color:{PAL['text']}; }}
.block-container {{ padding-top:1.5rem; max-width:1500px; }}
.hero h1 {{ font-size:2rem; margin:0; background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.hero p {{ color:{PAL['sub']} !important; margin:.2rem 0 1.2rem; }}
.kpi {{ background:{PAL['card']}; border:1px solid {PAL['border']}; border-radius:16px; padding:16px 18px;
  backdrop-filter:blur(8px); box-shadow:0 8px 24px rgba(0,0,0,.18); margin-bottom:12px; position:relative; overflow:hidden; }}
.kpi:before {{ content:""; position:absolute; left:0; top:0; height:100%; width:4px;
  background:linear-gradient(180deg,#60a5fa,#a855f7); }}
.kpi .l {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:{PAL['sub']} !important; }}
.kpi .v {{ font-size:1.8rem; font-weight:700; line-height:1.2; }}
.kpi .s {{ font-size:.78rem; color:{PAL['sub']} !important; }}
.best {{ background:linear-gradient(90deg,rgba(96,165,250,.25),rgba(168,85,247,.25)); border:1px solid {PAL['border']};
  border-radius:12px; padding:10px 16px; margin:6px 0 14px; }}
div[role="radiogroup"] label {{ background:transparent; border-radius:10px; padding:8px 12px; margin:2px 0; width:100%; }}
div[role="radiogroup"] label:hover {{ background:rgba(139,92,246,.18); }}
div[role="radiogroup"] label > div:first-child {{ display:none; }}
.stButton>button, .stDownloadButton>button {{ background:linear-gradient(90deg,#3b82f6,#8b5cf6); color:#fff !important;
  border:none; border-radius:10px; font-weight:600; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ filter:brightness(1.12); }}
[data-testid="stSidebar"] .stButton>button p, .stDownloadButton>button p, .stButton>button p {{ color:#fff !important; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; }}
</style>""", unsafe_allow_html=True)

def style(fig, h=380):
    fig.update_layout(template="plotly_dark" if DARK else "plotly_white", height=h,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45, b=10), font=dict(color=PAL["text"]),
                      colorway=["#60a5fa", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#fb7185"])
    return fig

def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub or "&nbsp;"}</div></div>', unsafe_allow_html=True)

def header(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)

def csv_btn(df, name, label="⬇️ Download CSV", key=None):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv",
                       key=key or name, use_container_width=False)

# ---------------------------------------------------------------- sidebar / state
NAV = ["📊 Dashboard Overview", "🎯 Classification Performance", "📉 Regression Performance",
       "🕘 Prediction History", "🧪 Model Training History", "🗂️ Dataset Analytics",
       "⚡ Live Prediction Analytics", "🏆 Model Comparison", "✍️ Make a Prediction"]

with st.sidebar:
    st.markdown("### 🧠 DigitVision ML")
    page = st.radio("Navigation", NAV, label_visibility="collapsed")
    st.divider()
    st.markdown("**Date range** (history pages)")
    dr = st.date_input("Range", value=(date.today() - timedelta(days=90), date.today()),
                       label_visibility="collapsed", key="dr")
    st.toggle("🌙 Dark mode", key="dark")
    st.divider()
    if st.button("🔄 Retrain all models", use_container_width=True):
        bar = st.progress(0.0)
        B.train_all(lambda p, m: bar.progress(p, text=m))
        st.success("Training complete. Metrics updated."); st.rerun()

if not B.has_runs():
    bar = st.progress(0.0, text="First launch: training models and evaluating on the held-out test set…")
    B.train_all(lambda p, m: bar.progress(p, text=m)); st.rerun()

def in_range(df, col):
    if df.empty: return df
    d0, d1 = (dr[0], dr[-1]) if isinstance(dr, (tuple, list)) and len(dr) else (dr, dr)
    t = pd.to_datetime(df[col])
    return df[(t.dt.date >= d0) & (t.dt.date <= d1)]

CLS, REG = B.latest_runs("classification"), B.latest_runs("regression")
DS = B.dataset_stats()

# ------------------------------------------------------------------------- pages
def overview():
    header("Dashboard Overview", "Real evaluation results from the latest trained models")
    runs = B.q("SELECT COUNT(*) n FROM training_runs WHERE status='Completed'").n[0]
    bc, br = CLS.loc[CLS.accuracy.idxmax()], REG.loc[REG.r2.idxmax()]
    r1 = st.columns(4); r2 = st.columns(3)
    kpi(r1[0], "Total dataset images", f"{DS['total']:,}", "8×8 grayscale digits")
    kpi(r1[1], "Training samples", f"{DS['train']:,}", "80% stratified")
    kpi(r1[2], "Testing samples", f"{DS['test']:,}", "20% held-out")
    kpi(r1[3], "Digit classes", "10", "Digits 0–9")
    kpi(r2[0], "Best classification accuracy", f"{bc.accuracy*100:.2f}%", bc.model)
    kpi(r2[1], "Best regression R²", f"{br.r2:.4f}", br.model)
    kpi(r2[2], "Total models trained", f"{runs}", f"{len(CLS)+len(REG)} distinct models · all runs logged")
    c1, c2 = st.columns(2)
    f = px.bar(CLS.sort_values("accuracy"), x="accuracy", y="model", orientation="h", text=CLS.sort_values("accuracy").accuracy.map("{:.2%}".format),
               title="Classification accuracy", color="accuracy", color_continuous_scale="Purples")
    c1.plotly_chart(style(f.update_coloraxes(showscale=False)), use_container_width=True)
    f = px.bar(REG.sort_values("r2"), x="r2", y="model", orientation="h", title="Regression R²",
               text=REG.sort_values("r2").r2.map("{:.3f}".format), color="r2", color_continuous_scale="Blues")
    c2.plotly_chart(style(f.update_coloraxes(showscale=False)), use_container_width=True)
    preds = B.predictions()
    st.markdown(f'<div class="best">🔮 <b>{len(preds)}</b> predictions logged · latest training: '
                f'<b>{B.q("SELECT MAX(trained_at) t FROM training_runs").t[0]}</b></div>', unsafe_allow_html=True)

def classification():
    header("Classification Performance", "Metrics are weighted averages on the held-out test set")
    t = CLS.assign(Accuracy=(CLS.accuracy * 100).round(2))
    tbl = t[["model", "Accuracy", "precision", "recall", "f1", "duration", "impl"]].rename(
        columns={"model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1-score",
                 "duration": "Train time (s)", "impl": "Implementation"})
    cols = st.columns(len(t))
    for c, (_, r) in zip(cols, t.iterrows()):
        kpi(c, r.model, f"{r.accuracy*100:.2f}%", f"F1 {r.f1:.3f}")
    st.dataframe(tbl.round(4), use_container_width=True, hide_index=True)
    csv_btn(tbl, "classification_scores.csv")
    m = CLS.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1"], var_name="Metric")
    st.plotly_chart(style(px.bar(m, x="model", y="value", color="Metric", barmode="group",
                                 title="Model comparison – accuracy, precision, recall, F1"
                                 ).update_yaxes(range=[max(0, m.value.min() - .05), 1])), use_container_width=True)
    st.markdown("#### Confusion matrix & classification report")
    sel = st.selectbox("Model", CLS.model.tolist(), index=int(CLS.accuracy.values.argmax()))
    r = CLS[CLS.model == sel].iloc[0]
    c1, c2 = st.columns([1.1, 1])
    cm = np.array(r.cm)
    norm = st.toggle("Normalise rows (%)", False)
    z = cm / cm.sum(1, keepdims=True) * 100 if norm else cm
    f = px.imshow(z, text_auto=".1f" if norm else True, x=list(range(10)), y=list(range(10)),
                  color_continuous_scale="Purples" if not DARK else "Viridis", aspect="auto",
                  labels=dict(x="Predicted", y="Actual", color="%" if norm else "Count"), title=f"{sel} – confusion matrix")
    c1.plotly_chart(style(f, 450), use_container_width=True)
    rep = pd.DataFrame(r.report).T.rename(index={"macro avg": "macro avg", "weighted avg": "weighted avg"})
    rep["support"] = rep["support"].astype(int)
    rep = rep.drop(index="accuracy", errors="ignore").round(3).reset_index().rename(columns={"index": "class"})
    c2.markdown(f"**Classification report – {sel}**")
    c2.dataframe(rep, use_container_width=True, hide_index=True, height=420)
    with c2: csv_btn(rep, f"report_{sel}.csv", key="rep")

def regression():
    header("Regression Performance", "Models regress the digit value (0–9) from pixel intensities")
    cols = st.columns(len(REG))
    for c, (_, r) in zip(cols, REG.iterrows()): kpi(c, r.model, f"R² {r.r2:.4f}", f"RMSE {r.rmse:.3f}")
    tbl = REG[["model", "mae", "mse", "rmse", "r2", "duration"]].rename(
        columns={"model": "Model", "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R² Score", "duration": "Train time (s)"})
    st.dataframe(tbl.round(4), use_container_width=True, hide_index=True)
    csv_btn(tbl, "regression_scores.csv")
    c1, c2 = st.columns(2)
    m = REG.melt(id_vars="model", value_vars=["mae", "mse", "rmse"], var_name="Error metric")
    c1.plotly_chart(style(px.bar(m, x="model", y="value", color="Error metric", barmode="group",
                                 title="Error metrics (lower is better)")), use_container_width=True)
    c2.plotly_chart(style(px.bar(REG, x="model", y="r2", color="model", text=REG.r2.round(3),
                                 title="R² score (higher is better)").update_layout(showlegend=False)),
                    use_container_width=True)
    sel = st.selectbox("Actual vs predicted – model", REG.model.tolist())
    r = REG[REG.model == sel].iloc[0]
    f = px.scatter(x=r.y_true, y=r.y_pred, labels=dict(x="Actual digit", y="Predicted value"),
                   title=f"{sel}: actual vs predicted", opacity=.6)
    f.add_trace(go.Scatter(x=[0, 9], y=[0, 9], mode="lines", name="Perfect", line=dict(dash="dash", color="#f472b6")))
    st.plotly_chart(style(f, 420), use_container_width=True)

def history():
    header("Prediction History", "Every prediction is stored in SQLite and survives restarts")
    df = in_range(B.predictions(), "ts")
    if df.empty:
        st.info("No predictions in this date range yet. Use **✍️ Make a Prediction** to create some."); return
    c = st.columns([2, 1.5, 1.5, 1.2, 1])
    s = c[0].text_input("🔍 Search (ID, digit, model)")
    mf = c[1].multiselect("Model", sorted(df.model.unique()))
    sf = c[2].multiselect("Status", ["Correct", "Incorrect", "Unknown"])
    sort = c[3].selectbox("Sort by", ["Date", "Confidence", "Predicted", "ID"])
    asc = c[4].toggle("Ascending", False)
    if s:
        s = s.lower()
        df = df[df.apply(lambda r: s in f"{r.id} {r.actual} {r.predicted} {r.model} {r.status} {r.source}".lower(), axis=1)]
    if mf: df = df[df.model.isin(mf)]
    if sf: df = df[df.status.isin(sf)]
    col = {"Date": "ts", "Confidence": "confidence", "Predicted": "predicted", "ID": "id"}[sort]
    df = df.sort_values(col, ascending=asc)
    out = pd.DataFrame({"Prediction ID": df.id, "Image": df.image,
                        "Actual": df.actual.apply(lambda v: "—" if pd.isna(v) else str(int(v))),
                        "Predicted": df.predicted, "Confidence": (df.confidence * 100).round(2),
                        "Model": df.model, "Date & time": df.ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Status": df.status, "Source": df.source})
    st.caption(f"{len(out)} record(s)")
    st.dataframe(out, use_container_width=True, hide_index=True, height=520, column_config={
        "Image": st.column_config.ImageColumn("Digit image", width="small"),
        "Confidence": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100, format="%.1f")})
    c1, c2 = st.columns([1, 5])
    with c1: csv_btn(out.drop(columns="Image"), "prediction_history.csv")
    with c2:
        if st.button("🗑️ Clear history"): B.clear_predictions(); st.rerun()

def training_history():
    header("Model Training History", "Every training run is logged with its version and scores")
    df = in_range(B.q("SELECT * FROM training_runs ORDER BY id DESC"), "trained_at")
    if df.empty: st.info("No training runs in this date range."); return
    task = st.radio("Task", ["All", "classification", "regression"], horizontal=True)
    if task != "All": df = df[df.task == task]
    out = pd.DataFrame({"Model": df.model, "Task": df.task, "Trained at": df.trained_at,
                        "Duration (s)": df.duration.round(2), "Dataset size": df.dataset_size,
                        "Training score": df.train_score.round(4), "Testing score": df.test_score.round(4),
                        "Version": "v" + df.version.astype(str), "Status": df.status})
    st.caption("Scores are accuracy for classifiers and R² for regressors.")
    st.dataframe(out, use_container_width=True, hide_index=True, height=360)
    csv_btn(out, "training_history.csv")
    d = df[df.status == "Completed"].sort_values("id").assign(Run=lambda x: "v" + x.version.astype(str))
    if d.empty: return
    c1, c2 = st.columns(2)
    f = px.line(d, x="trained_at", y="test_score", color="model", markers=True, title="Testing score trend")
    c1.plotly_chart(style(f), use_container_width=True)
    f = px.line(d, x="trained_at", y="duration", color="model", markers=True, title="Training duration (s)")
    c2.plotly_chart(style(f), use_container_width=True)
    if (d.version.max() < 2): st.info("Trends become lines once you retrain (sidebar → Retrain all models).")

def dataset():
    header("Dataset Analytics", "Digit distribution, split, preprocessing and samples")
    c = st.columns(4)
    kpi(c[0], "Total images", f"{DS['total']:,}"); kpi(c[1], "Training", f"{DS['train']:,}")
    kpi(c[2], "Testing", f"{DS['test']:,}"); kpi(c[3], "Features / image", "64", "8×8 pixels")
    d = DS["dist"]
    c1, c2, c3 = st.columns([1.4, 1, 1])
    c1.plotly_chart(style(px.bar(d.melt(id_vars="Digit", value_vars=["Train", "Test"]), x="Digit", y="value",
                                 color="variable", barmode="stack", title="Digit-wise distribution",
                                 labels=dict(value="Images", variable="Split")).update_xaxes(dtick=1)), use_container_width=True)
    c2.plotly_chart(style(px.pie(values=[DS["train"], DS["test"]], names=["Training", "Testing"], hole=.55,
                                 title="Train / test split")), use_container_width=True)
    c3.plotly_chart(style(px.pie(d, values="Total", names="Digit", hole=.4, title="Class share")), use_container_width=True)
    st.markdown("#### Preprocessing status")
    ck = DS["checks"].assign(Status=lambda x: x.ok.map({True: "✅ OK", False: "⚠️ Check"})).drop(columns="ok")
    st.dataframe(ck, use_container_width=True, hide_index=True)
    mean_img = np.array([DS["imgs"][DS["y"] == k].mean(0) for k in range(10)])
    f = px.imshow(mean_img, facet_col=0, facet_col_wrap=5, color_continuous_scale="Purples" if not DARK else "Magma",
                  title="Average image per digit")
    f.for_each_annotation(lambda a: a.update(text=f"Digit {a.text.split('=')[-1]}"))
    f.update_xaxes(showticklabels=False); f.update_yaxes(showticklabels=False); f.update_coloraxes(showscale=False)
    st.plotly_chart(style(f, 380), use_container_width=True)
    st.markdown("#### Sample gallery")
    g1, g2 = st.columns([1, 3])
    digit = g1.selectbox("Digit", ["All"] + list(range(10)))
    n = g2.slider("Images", 10, 60, 20, step=10)
    idx = np.arange(len(DS["y"])) if digit == "All" else np.where(DS["y"] == digit)[0]
    idx = np.random.RandomState(0).permutation(idx)[:n]
    cols = st.columns(10)
    for i, k in enumerate(idx):
        cols[i % 10].image(B.to_png_b64(DS["X"][k], 96), caption=f"{DS['y'][k]}", use_container_width=True)

def live():
    header("Live Prediction Analytics", "Computed from the stored prediction log")
    df = in_range(B.predictions(), "ts")
    if df.empty: st.info("No predictions yet – make one from **✍️ Make a Prediction**."); return
    known = df[df.status != "Unknown"]
    c = st.columns(4)
    kpi(c[0], "Total predictions", len(df))
    kpi(c[1], "Most predicted digit", int(df.predicted.mode()[0]), f"{(df.predicted == df.predicted.mode()[0]).sum()} times")
    kpi(c[2], "Average confidence", f"{df.confidence.mean()*100:.1f}%")
    kpi(c[3], "Accuracy (labelled)", f"{(known.status == 'Correct').mean()*100:.1f}%" if len(known) else "—", f"{len(known)} labelled")
    c1, c2 = st.columns(2)
    cnt = df.predicted.value_counts().reindex(range(10), fill_value=0).reset_index()
    cnt.columns = ["Digit", "Count"]
    c1.plotly_chart(style(px.bar(cnt, x="Digit", y="Count", title="Prediction distribution",
                                 color="Count", color_continuous_scale="Purples").update_xaxes(dtick=1)
                          .update_coloraxes(showscale=False)), use_container_width=True)
    per = df.set_index("ts").resample("D" if (df.ts.max() - df.ts.min()).days >= 2 else "60min").size().reset_index(name="Predictions")
    c2.plotly_chart(style(px.area(per, x="ts", y="Predictions", title="Prediction activity")), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(style(px.histogram(df, x="confidence", nbins=20, title="Confidence distribution")), use_container_width=True)
    c4.plotly_chart(style(px.pie(df, names="model", hole=.5, title="Predictions by model")), use_container_width=True)
    st.markdown("#### Recent activity")
    rec = df.head(10)
    st.dataframe(pd.DataFrame({"ID": rec.id, "Image": rec.image, "Predicted": rec.predicted,
                               "Confidence %": (rec.confidence * 100).round(1), "Model": rec.model,
                               "Status": rec.status, "Time": rec.ts.dt.strftime("%Y-%m-%d %H:%M:%S")}),
                 use_container_width=True, hide_index=True, column_config={"Image": st.column_config.ImageColumn("Digit")})

def comparison():
    header("Model Comparison", "Best models are highlighted using the appropriate metric for each task")
    t1, t2 = st.tabs(["Classification", "Regression"])
    with t1:
        by = st.selectbox("Rank by", ["f1", "accuracy", "precision", "recall"], key="cby")
        d = CLS.sort_values(by, ascending=False).reset_index(drop=True)
        st.markdown(f'<div class="best">🏆 Best classifier: <b>{d.model[0]}</b> – {by} = <b>{d[by][0]:.4f}</b></div>', unsafe_allow_html=True)
        tb = d[["model", "accuracy", "precision", "recall", "f1", "duration"]].rename(columns={"duration": "train_s"})
        st.dataframe(tb.style.highlight_max(subset=["accuracy", "precision", "recall", "f1"], color="#6d28d9")
                     .highlight_min(subset=["train_s"], color="#0e7490").format({c: "{:.4f}" for c in tb.columns[1:]}),
                     use_container_width=True, hide_index=True)
        csv_btn(tb, "classification_comparison.csv", key="cc")
        c1, c2 = st.columns(2)
        f = go.Figure()
        for _, r in CLS.iterrows():
            f.add_trace(go.Scatterpolar(r=[r.accuracy, r.precision, r.recall, r.f1], theta=["Accuracy", "Precision", "Recall", "F1"],
                                        fill="toself", name=r.model, opacity=.55))
        f.update_layout(polar=dict(radialaxis=dict(range=[max(0, CLS.accuracy.min() - .05), 1])), title="Radar comparison")
        c1.plotly_chart(style(f, 420), use_container_width=True)
        f = px.scatter(CLS, x="duration", y="accuracy", size=[20] * len(CLS), color="model", title="Accuracy vs training time (s)")
        c2.plotly_chart(style(f, 420), use_container_width=True)
    with t2:
        by = st.selectbox("Rank by", ["rmse", "mae", "mse", "r2"], key="rby")
        asc = by != "r2"
        d = REG.sort_values(by, ascending=asc).reset_index(drop=True)
        st.markdown(f'<div class="best">🏆 Best regressor: <b>{d.model[0]}</b> – {by.upper()} = <b>{d[by][0]:.4f}</b> '
                    f'({"lower" if asc else "higher"} is better)</div>', unsafe_allow_html=True)
        tb = d[["model", "mae", "mse", "rmse", "r2"]]
        st.dataframe(tb.style.highlight_min(subset=["mae", "mse", "rmse"], color="#6d28d9").highlight_max(subset=["r2"], color="#6d28d9")
                     .format({c: "{:.4f}" for c in tb.columns[1:]}), use_container_width=True, hide_index=True)
        csv_btn(tb, "regression_comparison.csv", key="rc")
        m = REG.melt(id_vars="model", value_vars=["mae", "mse", "rmse", "r2"])
        st.plotly_chart(style(px.bar(m, x="model", y="value", color="model", facet_col="variable", title="All regression metrics")
                              .update_yaxes(matches=None).update_layout(showlegend=False), 400), use_container_width=True)

def predict_page():
    header("Make a Prediction", "Upload, draw, or pick a test image – results are saved automatically")
    models = CLS.sort_values("accuracy", ascending=False).model.tolist()
    sel = st.multiselect("Model(s)", models, default=models[:1])
    mode = st.radio("Input", ["Upload image", "Draw digit", "Dataset test sample"], horizontal=True)
    x, actual, src = None, None, mode
    c1, c2 = st.columns([1, 1])
    with c1:
        if mode == "Upload image":
            up = st.file_uploader("PNG / JPG of a single digit", type=["png", "jpg", "jpeg"])
            if up: x = B.preprocess(Image.open(up))
        elif mode == "Draw digit":
            try:
                from streamlit_drawable_canvas import st_canvas
                cv = st_canvas(stroke_width=18, stroke_color="#fff", background_color="#000", height=280, width=280,
                               drawing_mode="freedraw", key="canvas")
                if cv.image_data is not None and cv.image_data[..., :3].max() > 0:
                    x = B.preprocess(Image.fromarray(cv.image_data.astype("uint8")).convert("RGB"))
            except ImportError:
                st.warning("Install `streamlit-drawable-canvas` to enable drawing (pip install streamlit-drawable-canvas).")
        else:
            _, Xte, _, yte, _, _ = B.get_split()
            i = st.slider("Test sample #", 0, len(yte) - 1, 0)
            x, actual = Xte[i], int(yte[i])
        if mode != "Dataset test sample":
            a = st.selectbox("Actual digit (optional)", ["Unknown"] + list(range(10)))
            actual = None if a == "Unknown" else int(a)
        else:
            st.caption(f"Actual digit: **{actual}**")
        if x is not None: st.image(B.to_png_b64(x, 160), caption="What the model sees (8×8)")
    with c2:
        if st.button("🔮 Predict", disabled=x is None or not sel):
            for mname in sel:
                p, conf, proba, status = B.predict_and_log(mname, x, actual, src)
                icon = {"Correct": "✅", "Incorrect": "❌", "Unknown": "❔"}[status]
                st.markdown(f'<div class="kpi"><div class="l">{mname}</div><div class="v">{p} <span style="font-size:1rem">'
                            f'{conf*100:.1f}% {icon}</span></div></div>', unsafe_allow_html=True)
                st.plotly_chart(style(px.bar(x=list(range(10)), y=proba, labels=dict(x="Digit", y="Probability")
                                             ).update_xaxes(dtick=1), 200), use_container_width=True, key=f"pp{mname}")
            st.success("Saved to prediction history.")
        elif x is None:
            st.info("Provide an input to enable prediction.")

{"📊 Dashboard Overview": overview, "🎯 Classification Performance": classification,
 "📉 Regression Performance": regression, "🕘 Prediction History": history,
 "🧪 Model Training History": training_history, "🗂️ Dataset Analytics": dataset,
 "⚡ Live Prediction Analytics": live, "🏆 Model Comparison": comparison,
 "✍️ Make a Prediction": predict_page}[page]()
