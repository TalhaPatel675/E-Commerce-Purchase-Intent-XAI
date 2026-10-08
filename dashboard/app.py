"""Streamlit dashboard with six sections (PRD §4.8)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.main import predict as predict_local
from src.api.schemas import PredictRequest
from src.utils.logger import configure_logging, get_logger  # noqa: E402

configure_logging(level="INFO")
log = get_logger(__name__)

REPO = ROOT
DATA_CSV = REPO / "dataset" / "ecommerce_sessions.csv"
DATA_DICT_CSV = REPO / "dataset" / "data_dictionary.csv"
REPORTS_DIR = REPO / "reports"
FIG_DIR = REPORTS_DIR / "figures"
SUMMARY_JSON = REPORTS_DIR / "explainability_summary.json"
MODEL_CARD = REPORTS_DIR / "MODEL_CARD.md"
ABLATION_JSON = REPORTS_DIR / "pagevalues_ablation.json"

st.set_page_config(page_title="Purchase Intent Dashboard", page_icon="🛒", layout="wide")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_CSV)


@st.cache_data(show_spinner=False)
def load_data_dict() -> pd.DataFrame:
    return pd.read_csv(DATA_DICT_CSV)


@st.cache_data(show_spinner=False)
def load_summary() -> dict | None:
    if SUMMARY_JSON.exists():
        return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    return None


def section_overview() -> None:
    st.header("🛒 Project overview")
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", f"{len(df):,}")
    c2.metric("Conversion rate", f"{df['Converted'].mean():.1%}")
    c3.metric("Buyers (1)", f"{int(df['Converted'].sum()):,}")
    c4.metric("Non-buyers (0)", f"{int((df['Converted'] == 0).sum()):,}")
    fig = px.histogram(
        df, x="Month", color="Converted", barmode="group", title="Conversion by month"
    )
    st.plotly_chart(fig, use_container_width=True)


def section_live_scoring() -> None:
    st.header("🎯 Live session scoring")
    configured_api_url = os.getenv("DASHBOARD_API_URL", "").strip()
    if configured_api_url:
        api_url = st.text_input("API URL", value=configured_api_url)
        use_remote_api = True
    else:
        st.caption("Using the local model directly (Streamlit Cloud mode).")
        api_url = ""
        use_remote_api = False
    with st.form("predict-form"):
        c1, c2, c3 = st.columns(3)
        Administrative = c1.number_input("Administrative", 0, 100, 3)
        Administrative_Duration = c1.number_input("Administrative_Duration (s)", 0.0, 5000.0, 80.0)
        Informational = c1.number_input("Informational", 0, 100, 0)
        Informational_Duration = c1.number_input("Informational_Duration (s)", 0.0, 5000.0, 0.0)
        ProductRelated = c1.number_input("ProductRelated", 0, 500, 25)
        ProductRelated_Duration = c1.number_input(
            "ProductRelated_Duration (s)", 0.0, 20000.0, 600.0
        )
        BounceRates = c2.number_input("BounceRates", 0.0, 1.0, 0.02)
        ExitRates = c2.number_input("ExitRates", 0.0, 1.0, 0.03)
        PageValues = c2.number_input("PageValues", 0.0, 500.0, 5.0)
        SpecialDay = c2.number_input("SpecialDay", 0.0, 1.0, 0.0)
        Month = c3.selectbox(
            "Month",
            ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        OperatingSystems = c3.number_input("OperatingSystems (id)", 1, 8, 2)
        Browser = c3.number_input("Browser (id)", 1, 13, 2)
        Region = c3.number_input("Region (id)", 1, 9, 1)
        TrafficType = c3.number_input("TrafficType (id)", 1, 20, 2)
        VisitorType = c3.selectbox("VisitorType", ["Returning_Visitor", "New_Visitor", "Other"])
        Weekend = c3.checkbox("Weekend", value=False)
        submitted = st.form_submit_button("Score session")

    if submitted:
        payload = {
            "session": {
                "Administrative": int(Administrative),
                "Administrative_Duration": float(Administrative_Duration),
                "Informational": int(Informational),
                "Informational_Duration": float(Informational_Duration),
                "ProductRelated": int(ProductRelated),
                "ProductRelated_Duration": float(ProductRelated_Duration),
                "BounceRates": float(BounceRates),
                "ExitRates": float(ExitRates),
                "PageValues": float(PageValues),
                "SpecialDay": float(SpecialDay),
                "Month": Month,
                "OperatingSystems": int(OperatingSystems),
                "Browser": int(Browser),
                "Region": int(Region),
                "TrafficType": int(TrafficType),
                "VisitorType": VisitorType,
                "Weekend": bool(Weekend),
            }
        }
        try:
            if use_remote_api:
                r = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
                r.raise_for_status()
                out = r.json()
            else:
                local_response = predict_local(
                    PredictRequest(session=payload["session"])
                )
                out = local_response.model_dump()
            st.success(
                f"Prediction: **{out['prediction']}** — probability **{out['conversion_probability']:.1%}** (confidence {out['confidence']}, model {out.get('model_name','?')})"
            )
            df_c = pd.DataFrame(
                [{**c, "contribution_abs": abs(c["contribution"])} for c in out["top_contributors"]]
            )
            fig = px.bar(
                df_c.sort_values("contribution_abs"),
                x="contribution",
                y="feature",
                orientation="h",
                color="direction",
                title="Top contributors from local SHAP",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(
                f"API call failed: {e}. Is the FastAPI service running on {api_url}? (scripts\\run_api.bat)"
            )


def section_funnel() -> None:
    st.header("📈 Funnel & cohort analytics")
    df = load_data()
    c1, c2 = st.columns(2)
    with c1:
        df["bucket"] = pd.cut(
            df["ProductRelated"],
            bins=[-1, 5, 15, 30, 60, 1000],
            labels=["0-5", "6-15", "16-30", "31-60", "60+"],
        )
        g = df.groupby("bucket", observed=True)["Converted"].mean().reset_index()
        fig = px.bar(g, x="bucket", y="Converted", title="Conversion by ProductRelated bucket")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(
            df,
            x="VisitorType",
            y="PageValues",
            color="Converted",
            title="PageValues by VisitorType & conversion",
        )
        st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(
        px.imshow(
            df.drop(columns=["Converted"]).corr(numeric_only=True), title="Numeric correlations"
        ),
        use_container_width=True,
    )


def section_model_comparison() -> None:
    st.header("🧪 Model comparison")
    s = load_summary()
    if not s:
        st.info("Training has not been run yet. Run scripts\\train.bat first.")
        return
    rows = []
    for name, m in s.get("all_results", {}).items():
        rows.append(
            {
                "Model": name,
                "PR-AUC": round(m.get("pr_auc", 0), 3),
                "ROC-AUC": round(m.get("roc_auc", 0), 3),
                "Recall": round(m.get("recall", 0), 3),
                "Precision": round(m.get("precision", 0), 3),
                "F1": round(m.get("f1", 0), 3),
                "CV PR-AUC mean": round(m.get("cv_pr_auc_mean", 0), 3),
            }
        )
    dfm = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
    st.dataframe(dfm, use_container_width=True)
    fig = px.bar(
        dfm.melt(id_vars="Model", value_vars=["PR-AUC", "ROC-AUC", "F1"]),
        x="Model",
        y="value",
        color="variable",
        barmode="group",
        title="Metrics per model",
    )
    st.plotly_chart(fig, use_container_width=True)


def section_explainability() -> None:
    st.header("🔍 Explainability (XAI) — two lenses")
    s = load_summary()
    if not s:
        st.info("Run scripts\\train.bat first.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lens 1 — Global SHAP")
        if (FIG_DIR / "shap_global_summary.png").exists():
            st.image(str(FIG_DIR / "shap_global_summary.png"), use_column_width=True)
    with c2:
        st.subheader("Lens 2 — Permutation importance")
        if (FIG_DIR / "permutation_importance.png").exists():
            st.image(str(FIG_DIR / "permutation_importance.png"), use_column_width=True)

    if (REPORTS_DIR / "lime_local_session0.html").exists():
        st.subheader("Local LIME (sample session)")
        with open(REPORTS_DIR / "lime_local_session0.html", encoding="utf-8") as fh:
            st.components.v1.html(fh.read(), height=600, scrolling=True)

    for key, label in [("global_shap", "SHAP"), ("permutation_importance", "Permutation")]:
        fi = s.get(key, {}).get("feature_importance") or []
        if fi:
            st.subheader(f"Top features by {label}")
            if label == "SHAP":
                df_imp = pd.DataFrame(
                    [(n, m) for n, m in fi][:10], columns=["Feature", "Importance"]
                )
            else:
                df_imp = pd.DataFrame(
                    [(n, m, sd) for n, m, sd in fi][:10], columns=["Feature", "Importance", "Std"]
                )
            st.dataframe(df_imp, use_container_width=True)
            st.plotly_chart(
                px.bar(df_imp, x="Importance", y="Feature", orientation="h"),
                use_container_width=True,
            )


def section_performance() -> None:
    st.header("📊 Performance metrics")
    s = load_summary()
    if not s:
        st.info("Run scripts\\train.bat first.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected model", s.get("selected_model", "n/a"))
    c2.metric("Decision threshold", f"{s.get('threshold', 0):.3f}")
    c3.metric("Recall @ threshold", f"{s.get('recall_at_threshold', 0):.3f}")
    c4.metric("Precision @ threshold", f"{s.get('precision_at_threshold', 0):.3f}")
    if MODEL_CARD.exists():
        with open(MODEL_CARD, encoding="utf-8") as fh:
            st.markdown(fh.read())
    if ABLATION_JSON.exists():
        st.subheader("PageValues leakage audit")
        ab = json.loads(ABLATION_JSON.read_text(encoding="utf-8"))
        m = ab.get("metrics", {})
        c1, c2 = st.columns(2)
        c1.metric(
            "PR-AUC (with PageValues)",
            f"{s.get('all_results',{}).get(ab.get('selected_model','?'),{}).get('pr_auc',0):.3f}",
        )
        c2.metric("PR-AUC (without PageValues)", f"{m.get('pr_auc',0):.3f}")
        st.caption(
            "Honest take: this is the deployment-leakage audit the PRD asks for. See docs/pagevalues_ablation.md."
        )


PAGES = {
    "Overview": section_overview,
    "Live scoring": section_live_scoring,
    "Funnel analytics": section_funnel,
    "Model comparison": section_model_comparison,
    "Explainability": section_explainability,
    "Performance metrics": section_performance,
}


def main() -> None:
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
