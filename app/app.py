# Sentiment Analysis — Fashion E-commerce

import re
import pickle
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ── CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Insights · Fashion",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── STYLE ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #ffffff;
    color: #111;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 980px;
    padding-top: 0px;
    padding-bottom: 100px;
}

.hero-container {
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    background-color: #000;
    padding: 120px 0;
}

.hero-content {
    max-width: 980px;
    margin: 0 auto;
    padding: 0 24px;
}

.hero-title {
    font-size: 4.5rem;
    font-weight: 900;
    color: #fff;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    line-height: 1;
    margin-bottom: 16px;
}

.hero-sub {
    font-size: 0.9rem;
    font-weight: 400;
    color: #ccc;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    max-width: 420px;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    background: #eee;
    border: 1px solid #eee;
    margin-bottom: 40px;
}
.metric-card {
    background: #fff;
    padding: 28px 24px;
}
.metric-number {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #999;
}
.positive { color: #1a5c2a; }
.neutral  { color: #7a5c00; }
.negative { color: #8b1a1a; }
.total    { color: #111; }

/* Result box */
.result-box {
    border: 1px solid #eee;
    padding: 28px 32px;
    margin-top: 20px;
}
.result-tag {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 8px;
}
.result-value {
    font-size: 2rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.result-excerpt {
    font-size: 0.82rem;
    color: #aaa;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #f0f0f0;
    line-height: 1.6;
    font-weight: 300;
}
.alert-box {
    border-left: 3px solid #111;
    padding: 12px 16px;
    margin-top: 12px;
    background: #f9f9f9;
    font-size: 0.82rem;
}

/* Section title */
.section-title {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 24px;
    padding-bottom: 12px;
    border-bottom: 1px solid #eee;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 32px;
    justify-content: center;
    margin-top: 30px;
    margin-bottom: 40px;
    border-bottom: 1px solid #eee;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #bbb;
    padding-bottom: 16px;
}
.stTabs [aria-selected="true"] {
    color: #111 !important;
    border-bottom: 2px solid #111 !important;
}

/* Inputs */
.stTextArea label { display: none !important; }
textarea {
    border: 1px solid #eee !important;
    border-radius: 0 !important;
    padding: 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
textarea:focus {
    border-color: #111 !important;
    box-shadow: none !important;
}

/* Button */
.stButton > button {
    background: #111 !important;
    color: #fff !important;
    border-radius: 0 !important;
    border: none !important;
    padding: 12px 32px !important;
    letter-spacing: 0.12em !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { background: #333 !important; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: #111 !important;
    border: 1px solid #eee !important;
    border-radius: 0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}

hr { border: none; border-top: 1px solid #eee; margin: 48px 0; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ─────────────────────────────────────────────────────────
LABELS = ["Positive", "Neutral", "Negative"]


def normalize(label: str) -> str:
    """Normalize any casing → 'Positive' / 'Neutral' / 'Negative'."""
    return str(label).strip().capitalize()


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.strip()


@st.cache_resource
def load_model_from_file():
    with open("svm_pipeline.pkl", "rb") as f:
        return pickle.load(f)


def make_chart(counts: dict):
    labels = [l for l in LABELS if counts.get(l, 0) > 0]
    values = [counts[l] for l in labels]
    colors = {"Positive": "#d4edda", "Neutral": "#fff3cd", "Negative": "#f8d7da"}
    edges  = {"Positive": "#1a5c2a", "Neutral": "#7a5c00", "Negative": "#8b1a1a"}

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#fff")
    ax.set_facecolor("#fff")
    bars = ax.bar(
        labels, values,
        color=[colors[l] for l in labels],
        edgecolor=[edges[l] for l in labels],
        linewidth=0.8, width=0.45, zorder=3,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:,}", ha="center", va="bottom",
            fontsize=11, color="#555", fontweight="600",
        )
    ax.set_ylim(0, max(values) * 1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.tick_params(colors="#aaa", labelsize=10, bottom=False)
    ax.grid(False)
    fig.tight_layout(pad=0.5)
    return fig


# ── LOAD MODEL ──────────────────────────────────────────────────────
# Model loads automatically from svm_pipeline.pkl in the repo root.
# No upload needed.
model = load_model_from_file()


# ── HERO ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-content">
        <div class="hero-title">Customer Insights</div>
        <div class="hero-sub">
            Analyze customer reviews to understand product performance
            and improve decision making
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:48px;'></div>", unsafe_allow_html=True)


# ── TABS ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Single review", "Batch analysis", "Model performance"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Single review
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([5, 4], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Write a review</div>', unsafe_allow_html=True)
        review = st.text_area(
            "",
            placeholder="e.g. «The fabric is beautiful and the fit is exactly as described...»",
            height=160,
            label_visibility="collapsed",
        )
        analyze_btn = st.button("Analyze")

    with col_right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

        if analyze_btn:
            if not review.strip():
                st.warning("Write a review to analyze.")
            else:
                pred    = normalize(model.predict([preprocess(review)])[0])
                css_cls = pred.lower()
                messages = {
                    "Positive": "Strong product performance detected.",
                    "Neutral":  "Mixed signals — customer has reservations.",
                    "Negative": "Potential product issue detected.",
                }
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-tag">Detected sentiment</div>
                    <div class="result-value {css_cls}">{pred}</div>
                    <div class="result-excerpt">"{review[:200]}{'...' if len(review) > 200 else ''}"</div>
                </div>
                <div class="alert-box">{messages[pred]}</div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box" style="border-style:dashed;">
                <div class="result-tag">Awaiting input</div>
                <div style="font-size:0.85rem;color:#ccc;font-weight:300;margin-top:6px;">
                    The prediction will appear here.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — Batch analysis
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Upload a CSV with reviews</div>', unsafe_allow_html=True)

    csv_file = st.file_uploader("CSV file", type=["csv"], key="csv")

    if csv_file:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        st.caption(f"{len(df):,} rows · {len(df.columns)} columns")

        col_a, col_b, _ = st.columns([2, 2, 3])
        with col_a:
            text_col = st.selectbox("Text column", df.columns.tolist())

        if st.button("Run analysis"):
            with st.spinner("Running predictions..."):
                raw_preds       = model.predict(df[text_col].fillna("").apply(preprocess).tolist())
                df["Predicted"] = [normalize(p) for p in raw_preds]

            # Count using normalized labels — this is what fixed the 0s bug
            counts   = {l: int((df["Predicted"] == l).sum()) for l in LABELS}
            total    = len(df)
            neg_rate = counts["Negative"] / total * 100 if total else 0
            pos_rate = counts["Positive"] / total * 100 if total else 0

            # Metric cards
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-number positive">{counts['Positive']:,}</div>
                    <div class="metric-label">Positive</div>
                </div>
                <div class="metric-card">
                    <div class="metric-number neutral">{counts['Neutral']:,}</div>
                    <div class="metric-label">Neutral</div>
                </div>
                <div class="metric-card">
                    <div class="metric-number negative">{counts['Negative']:,}</div>
                    <div class="metric-label">Negative</div>
                </div>
                <div class="metric-card">
                    <div class="metric-number total">{total:,}</div>
                    <div class="metric-label">Total reviews</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, _ = st.columns([2, 2, 3])
            with col1:
                st.metric("Negative rate", f"{neg_rate:.1f}%")
            with col2:
                st.metric("Positive rate", f"{pos_rate:.1f}%")

            col_chart, _ = st.columns([3, 2])
            with col_chart:
                st.pyplot(make_chart(counts), use_container_width=True)

            st.markdown('<div class="section-title" style="margin-top:32px;">Preview</div>',
                        unsafe_allow_html=True)
            st.dataframe(df[[text_col, "Predicted"]].head(50),
                         use_container_width=True, height=260)
            st.download_button(
                "Download predictions",
                df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════
# TAB 3 — Model performance
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Test set results · 3,397 reviews</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-number total">81.7%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-number total">0.61</div>
            <div class="metric-label">F1 macro</div>
        </div>
        <div class="metric-card">
            <div class="metric-number total">LinearSVC</div>
            <div class="metric-label">Algorithm</div>
        </div>
        <div class="metric-card">
            <div class="metric-number total">5,000</div>
            <div class="metric-label">TF-IDF features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Per-class breakdown</div>', unsafe_allow_html=True)
    perf_df = pd.DataFrame({
        "Class":     ["Positive", "Neutral", "Negative"],
        "Precision": [0.91, 0.39, 0.53],
        "Recall":    [0.93, 0.33, 0.57],
        "F1":        [0.92, 0.36, 0.55],
        "Support":   [2618, 424, 355],
    }).set_index("Class")
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("""
    <hr>
    <div style="font-size:0.72rem;color:#bbb;letter-spacing:0.06em;line-height:1.8;">
        The Neutral class is the hardest to classify — the language in 3-star reviews
        often overlaps with both positive and negative vocabulary.
        The Positive class performs strongly due to its majority representation in the training data.
    </div>
    """, unsafe_allow_html=True)

    if hasattr(model, "steps"):
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Pipeline parameters</div>', unsafe_allow_html=True)
        rows = []
        for step_name, obj in model.steps:
            for k, v in obj.get_params().items():
                rows.append({"Step": step_name, "Parameter": k, "Value": str(v)})
        st.dataframe(pd.DataFrame(rows).set_index("Step"), use_container_width=True)


# ── FOOTER ──────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="display:flex;justify-content:center;align-items:center;padding-bottom:24px;">
    <div style="font-size:0.68rem;color:#ccc;letter-spacing:0.08em;text-transform:uppercase;">
        Women's Clothing E-Commerce
    </div>
</div>
""", unsafe_allow_html=True)
