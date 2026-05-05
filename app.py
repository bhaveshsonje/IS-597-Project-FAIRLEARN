"""
app.py — FairLearn: Student Outcome Predictor
IS 597: Human-Centered Data Science — Final Project
UIUC · Angad Chaudhry & Bhavesh Sonje

Run with:
    streamlit run app.py
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLearn — Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────────────────
MC_CLASS_LABELS = {0: "Pass", 1: "Distinction", 2: "Fail", 3: "Withdrawn"}
MC_CLASS_COLORS = {
    "Pass":        "#28a745",
    "Distinction": "#007bff",
    "Fail":        "#fd7e14",
    "Withdrawn":   "#dc3545",
}

NUMERIC_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "total_clicks",
    "num_activities", "active_days", "avg_early_score", "num_assessments_taken",
]
CATEGORICAL_FEAT = [
    "gender", "age_band", "highest_education", "disability",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
    "mlp":                 "MLP",
}

# Plain-English feature labels for SHAP chart
FEATURE_LABELS = {
    "total_clicks":          "Total VLE Clicks",
    "active_days":           "Active Days",
    "num_activities":        "Resources Accessed",
    "avg_early_score":       "Avg Early Score",
    "num_assessments_taken": "Assessments Submitted",
    "num_of_prev_attempts":  "Previous Attempts",
    "studied_credits":       "Credits Studying",
    "gender":                "Gender",
    "age_band":              "Age Band",
    "highest_education":     "Highest Education",
    "disability":            "Disability",
}

# Real OULAD students with verified outcomes — pulled from the dataset.
# The "_meta" key is dashboard-only metadata; remove before model input.
DEMO_PROFILES = {
    "🎓  High Performer": {
        "_meta": {"id": 148993, "true_outcome": "Pass", "module": "AAA"},
        "num_of_prev_attempts": 1, "studied_credits": 60,
        "total_clicks": 1117,      "num_activities": 59,   "active_days": 28,
        "avg_early_score": 77,     "num_assessments_taken": 2,
        "gender": "F", "age_band": "35-55", "disability": "Y",
        "highest_education": "A Level or Equivalent",
    },
    "⚠️  Likely Withdraw": {
        "_meta": {"id": 292923, "true_outcome": "Withdrawn", "module": "AAA"},
        "num_of_prev_attempts": 0, "studied_credits": 180,
        "total_clicks": 0,         "num_activities": 0,   "active_days": 0,
        "avg_early_score": 0,      "num_assessments_taken": 0,
        "gender": "F", "age_band": "35-55", "disability": "N",
        "highest_education": "A Level or Equivalent",
    },
    "🔍  Borderline": {
        "_meta": {"id": 334259, "true_outcome": "Fail", "module": "AAA"},
        "num_of_prev_attempts": 0, "studied_credits": 150,
        "total_clicks": 243,       "num_activities": 24,  "active_days": 11,
        "avg_early_score": 57,     "num_assessments_taken": 1,
        "gender": "F", "age_band": "0-35", "disability": "N",
        "highest_education": "A Level or Equivalent",
    },
}


# ─── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        n: joblib.load(f"models/{n}.pkl")
        for n in ["logistic_regression", "random_forest", "xgboost", "mlp"]
        if os.path.exists(f"models/{n}.pkl")
    }

@st.cache_resource
def load_models_mc():
    return {
        n: joblib.load(f"models/{n}_mc.pkl")
        for n in ["logistic_regression", "random_forest", "xgboost", "mlp"]
        if os.path.exists(f"models/{n}_mc.pkl")
    }

@st.cache_resource
def load_encoders():
    p = "models/encoders.pkl"
    return joblib.load(p) if os.path.exists(p) else {}

@st.cache_data
def load_feature_names():
    p = "models/test_data.pkl"
    return joblib.load(p)["feature_names"] if os.path.exists(p) else []

@st.cache_data
def load_fairness():
    p = "results/advanced_binary/fairness_metrics.json"
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data
def load_thresholds():
    p = "results/advanced_binary/mitigation_thresholding.json"
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


# ─── Prediction card HTML ─────────────────────────────────────────────────────
def prediction_card(model_name: str, prob_success: float,
                    threshold: float = 0.5, flipped: bool = False) -> str:
    is_ok   = prob_success >= threshold
    label   = "✅ Success" if is_ok else "🔴 At-Risk"
    bg      = "#d4edda" if is_ok else "#f8d7da"
    border  = "#28a745" if is_ok else "#dc3545"
    bar_clr = "#28a745" if is_ok else "#dc3545"
    pct     = prob_success * 100
    thresh_note = f"threshold: {threshold:.2f}" if threshold != 0.5 else "threshold: 0.50"
    flip_badge  = " &nbsp;<span style='font-size:0.7em;background:#ffc107;color:#333;padding:1px 5px;border-radius:4px;'>flipped</span>" if flipped else ""
    return f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:12px;
                padding:16px;margin:4px 0;height:118px;">
        <div style="font-size:0.78em;font-weight:600;color:#555;margin-bottom:2px;">
            {MODEL_LABELS.get(model_name, model_name)}
        </div>
        <div style="font-size:1.15em;font-weight:bold;">{label}{flip_badge}</div>
        <div style="background:#e0e0e0;border-radius:4px;height:7px;margin:8px 0 4px;">
            <div style="background:{bar_clr};width:{pct:.0f}%;height:7px;
                        border-radius:4px;"></div>
        </div>
        <div style="font-size:0.75em;color:#555;">
            P(Success) = <b>{pct:.0f}%</b>
            &nbsp;·&nbsp;<span style="color:#888;">{thresh_note}</span>
        </div>
    </div>"""


def multiclass_card(model_name: str, probs) -> str:
    """Prediction card showing 4 class probabilities for multiclass mode."""
    class_order = [0, 1, 2, 3]
    pred_idx    = int(np.argmax(probs))
    pred_label  = MC_CLASS_LABELS[pred_idx]
    pred_color  = MC_CLASS_COLORS[pred_label]

    # Build bars as compact single-line strings — leading whitespace triggers
    # Markdown code-block rendering inside st.markdown(unsafe_allow_html=True)
    bars = []
    for i in class_order:
        cname = MC_CLASS_LABELS[i]
        color = MC_CLASS_COLORS[cname]
        pct   = float(probs[i]) * 100
        bold  = "font-weight:700;" if i == pred_idx else "opacity:0.75;"
        bars.append(
            f'<div style="display:flex;align-items:center;margin:2px 0;{bold}">'
            f'<div style="width:76px;font-size:0.7em;color:#333;">{cname}</div>'
            f'<div style="flex:1;background:#e0e0e0;border-radius:3px;height:6px;margin:0 6px;">'
            f'<div style="background:{color};width:{pct:.0f}%;height:6px;border-radius:3px;"></div>'
            f'</div>'
            f'<div style="font-size:0.7em;color:#333;width:30px;text-align:right;">{pct:.0f}%</div>'
            f'</div>'
        )
    bars_html = "".join(bars)

    name_label = MODEL_LABELS.get(model_name, model_name)
    return (
        f'<div style="background:#f8f9fa;border:1.5px solid {pred_color};border-radius:12px;padding:12px 14px;margin:4px 0;">'
        f'<div style="font-size:0.78em;font-weight:600;color:#555;margin-bottom:3px;">{name_label}</div>'
        f'<div style="font-size:1.1em;font-weight:bold;color:{pred_color};margin-bottom:5px;">→ {pred_label}</div>'
        f'{bars_html}'
        f'</div>'
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 FairLearn")
    st.markdown("**Fair & Explainable Student Performance Prediction**")
    st.markdown("IS 597 · Human-Centered Data Science · UIUC")
    st.divider()

    st.markdown("#### How to use")
    st.markdown(
        "1. **Pick a preset** to load a sample student, or\n"
        "2. **Adjust the sliders & dropdowns** to create your own profile\n"
        "3. Click **🔍 Predict** to see what all four models predict\n"
        "4. The **SHAP chart** explains *why* the model made that decision"
    )
    st.divider()

    st.markdown("#### Prediction Mode")
    pred_mode = st.radio(
        "Prediction mode",
        ["Binary (Success / At-Risk)", "Multiclass (4 classes)"],
        label_visibility="collapsed",
        help=(
            "**Binary:** Pass & Distinction → Success, Fail & Withdrawn → At-Risk\n\n"
            "**Multiclass:** Predicts one of Pass / Distinction / Fail / Withdrawn"
        ),
    )
    multiclass_mode = pred_mode.startswith("Multiclass")

    if not multiclass_mode:
        st.divider()
        st.markdown("#### ⚖️ Fairness Mitigation")
        apply_mitigation = st.checkbox(
            "Apply per-group thresholding",
            value=False,
            help=(
                "Adjusts the decision threshold per demographic group to equalise "
                "True Positive Rate (TPR) across groups. "
                "Thresholds were computed on the held-out test set."
            ),
        )
        if apply_mitigation:
            mitigate_attr = st.selectbox(
                "Sensitive attribute",
                ["disability", "gender", "age_band"],
                format_func=lambda x: {
                    "disability": "Disability",
                    "gender":     "Gender",
                    "age_band":   "Age Band",
                }[x],
            )
        else:
            mitigate_attr = None
    else:
        apply_mitigation = False
        mitigate_attr    = None

    st.divider()
    st.caption("📚 Dataset: OULAD — Open University Learning Analytics")
    st.caption("🗓️ Features: First 4 weeks of student activity only")
    if multiclass_mode:
        st.caption("🎯 Target: Pass · Distinction · Fail · Withdrawn")
    else:
        st.caption("🎯 Target: Pass / Distinction → Success · Fail / Withdrawn → At-Risk")


# ─── Main header ─────────────────────────────────────────────────────────────
st.markdown("## 🔍 Student Outcome Predictor")
st.markdown(
    "Enter a student's **first 4-week activity profile** to see what four ML models predict — "
    "and *why* via SHAP explainability."
)

# ─── Preset buttons ───────────────────────────────────────────────────────────
preset_cols = st.columns(len(DEMO_PROFILES))
selected_preset = None
for col, (label, profile) in zip(preset_cols, DEMO_PROFILES.items()):
    if col.button(label, use_container_width=True):
        selected_preset = profile
        # Save meta for the "true outcome" badge
        st.session_state["preset_meta"] = profile.get("_meta")
        for k, v in profile.items():
            if k == "_meta":
                continue
            st.session_state[f"p_{k}"] = v

st.divider()

# ─── Load assets ─────────────────────────────────────────────────────────────
trained_models    = load_models()
trained_models_mc = load_models_mc()
encoders          = load_encoders()
feature_names     = load_feature_names()

if not trained_models or not encoders or not feature_names:
    st.error(
        "Saved models not found. Run `python src/save_models.py` first.",
        icon="🚨",
    )
    st.stop()

if multiclass_mode and not trained_models_mc:
    st.warning(
        "Multiclass models not found. Run `python src/save_models.py` to generate them.",
        icon="⚠️",
    )
    multiclass_mode = False

active_models = trained_models_mc if multiclass_mode else trained_models

# helper to read session state with fallback
def ss(key, default):
    return st.session_state.get(f"p_{key}", default)


# ─── Two-column layout ────────────────────────────────────────────────────────
left, right = st.columns([1, 1.2], gap="large")

# ════ LEFT — Input form ═══════════════════════════════════════════════════════
with left:
    st.markdown("### Student Profile")

    # VLE activity
    st.markdown("##### 📊 VLE Activity — First 4 Weeks")
    total_clicks = st.slider(
        "Total VLE clicks",        0, 3000, ss("total_clicks", 200), 10,
        key="p_total_clicks",
        help="Total number of clicks across all learning resources",
    )
    active_days = st.slider(
        "Active days",             0, 35,   ss("active_days", 10),   1,
        key="p_active_days",
        help="Distinct days the student logged into the platform",
    )
    num_activities = st.slider(
        "Different resources accessed", 0, 130, ss("num_activities", 15), 1,
        key="p_num_activities",
    )

    st.markdown("##### 📝 Early Assessments")
    avg_early_score = st.slider(
        "Average assessment score", 0, 100, ss("avg_early_score", 65), 1,
        key="p_avg_early_score",
    )
    num_assessments = st.slider(
        "Assessments submitted",    0, 10,  ss("num_assessments_taken", 1), 1,
        key="p_num_assessments_taken",
    )

    st.markdown("##### 👤 Student Background")
    c1, c2 = st.columns(2)
    with c1:
        num_prev = st.slider("Previous attempts", 0, 5, ss("num_of_prev_attempts", 0),
                             key="p_num_of_prev_attempts")
    with c2:
        studied_credits = st.slider(
            "Credits studying", 30, 300, ss("studied_credits", 60), 30,
            key="p_studied_credits",
            help="Total credits the student is studying",
        )

    c3, c4 = st.columns(2)
    with c3:
        gender = st.selectbox(
            "Gender", ["F", "M"],
            index=["F", "M"].index(ss("gender", "F")),
            format_func=lambda x: "Female" if x == "F" else "Male",
            key="p_gender",
        )
    with c4:
        disability = st.selectbox(
            "Disability", ["N", "Y"],
            index=["N", "Y"].index(ss("disability", "N")),
            format_func=lambda x: "No" if x == "N" else "Yes",
            key="p_disability",
        )

    age_band = st.selectbox(
        "Age band", ["0-35", "35-55", "55<="],
        index=["0-35", "35-55", "55<="].index(ss("age_band", "0-35")),
        format_func=lambda x: x.replace("55<=", "55+"),
        key="p_age_band",
    )

    ed_opts = list(encoders["highest_education"].classes_)
    highest_ed = st.selectbox(
        "Highest education", ed_opts,
        index=ed_opts.index(ss("highest_education", "A Level or Equivalent")),
        key="p_highest_education",
    )

    st.markdown("")
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")
    if predict_btn:
        # Keep the preset badge only if current slider/dropdown values still match a preset
        current_vals = {
            "num_of_prev_attempts": st.session_state.get("p_num_of_prev_attempts"),
            "studied_credits":      st.session_state.get("p_studied_credits"),
            "total_clicks":         st.session_state.get("p_total_clicks"),
            "num_activities":       st.session_state.get("p_num_activities"),
            "active_days":          st.session_state.get("p_active_days"),
            "avg_early_score":      st.session_state.get("p_avg_early_score"),
            "num_assessments_taken":st.session_state.get("p_num_assessments_taken"),
            "gender":               st.session_state.get("p_gender"),
            "age_band":             st.session_state.get("p_age_band"),
            "disability":           st.session_state.get("p_disability"),
            "highest_education":    st.session_state.get("p_highest_education"),
        }
        matched = False
        for profile in DEMO_PROFILES.values():
            if all(profile.get(k) == current_vals.get(k) for k in current_vals):
                matched = True
                break
        if not matched:
            st.session_state["preset_meta"] = None


# ════ RIGHT — Results ══════════════════════════════════════════════════════════
with right:
    if predict_btn or selected_preset:

        # ── Build & encode input row ─────────────────────────────────────────
        raw = {
            "num_of_prev_attempts":  num_prev,
            "studied_credits":       studied_credits,
            "total_clicks":          total_clicks,
            "num_activities":        num_activities,
            "active_days":           active_days,
            "avg_early_score":       float(avg_early_score),
            "num_assessments_taken": num_assessments,
            "gender":                gender,
            "age_band":              age_band,
            "highest_education":     highest_ed,
            "disability":            disability,
        }
        encoded = dict(raw)
        for col in CATEGORICAL_FEAT:
            le  = encoders[col]
            val = str(raw[col])
            encoded[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
        X = pd.DataFrame([encoded])[feature_names]

        # ── Student profile summary ──────────────────────────────────────────
        st.markdown("### Results")

        # If a real OULAD preset is loaded, show the actual outcome
        meta = st.session_state.get("preset_meta")
        if meta:
            actual = meta["true_outcome"]
            if multiclass_mode:
                actual_color = MC_CLASS_COLORS.get(actual, "#555")
                outcome_str  = actual
            else:
                actual_class = "Success" if actual in ("Pass", "Distinction") else "At-Risk"
                actual_color = "#28a745" if actual_class == "Success" else "#dc3545"
                outcome_str  = f"{actual} ({actual_class})"
            st.markdown(
                f"""<div style='background:#fff3cd;border-left:4px solid #ffc107;
                padding:10px 14px;border-radius:6px;margin-bottom:10px;'>
                📌 <b>Real OULAD student #{meta['id']}</b> from module {meta['module']}<br/>
                Actual outcome: <span style='color:{actual_color};font-weight:bold;'>
                {outcome_str}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        with st.expander("📋 Student profile summary", expanded=False):
            summary = pd.DataFrame([{
                "Clicks": total_clicks, "Active Days": active_days,
                "Resources": num_activities, "Avg Score": avg_early_score,
                "Assessments": num_assessments, "Prev Attempts": num_prev,
                "Credits": studied_credits, "Gender": "Female" if gender=="F" else "Male",
                "Age": age_band.replace("55<=","55+"),
                "Disability": "Yes" if disability=="Y" else "No",
            }]).T.rename(columns={0: "Value"})
            st.dataframe(summary, use_container_width=True)

        # ── Model predictions ─────────────────────────────────────────────────
        st.markdown("#### Model Predictions")

        if multiclass_mode:
            # Multiclass: show 4-class probability bars per model
            all_mc_probs = {}
            for name, model in active_models.items():
                all_mc_probs[name] = model.predict_proba(X)[0]  # shape (4,)

            c1, c2 = st.columns(2)
            pred_classes = {}
            for i, (name, mc_probs) in enumerate(all_mc_probs.items()):
                col = c1 if i % 2 == 0 else c2
                col.markdown(multiclass_card(name, mc_probs), unsafe_allow_html=True)
                pred_classes[name] = int(np.argmax(mc_probs))

            # Agreement badge for multiclass
            votes = Counter(pred_classes.values())
            top_class_idx, top_count = votes.most_common(1)[0]
            top_class_name  = MC_CLASS_LABELS[top_class_idx]
            top_class_color = MC_CLASS_COLORS[top_class_name]
            n_total = len(pred_classes)
            if top_count == n_total:
                agree_txt = f"**All {n_total} models agree: → {top_class_name}**"
            else:
                agree_txt = f"**{top_count}/{n_total} models predict → {top_class_name}**"
            st.markdown(
                f"<div style='text-align:center;margin:8px 0;font-size:1em;"
                f"color:{top_class_color};'>{agree_txt}</div>",
                unsafe_allow_html=True,
            )

        else:
            # Binary: success/at-risk cards
            probs = {}
            for name, model in active_models.items():
                probs[name] = float(model.predict_proba(X)[0, 1])

            # ── Resolve per-group thresholds if mitigation is on ─────────────
            thresholds_data = load_thresholds()
            group_val_map = {"disability": disability, "gender": gender, "age_band": age_band}

            if apply_mitigation and mitigate_attr:
                model_thresholds = {}
                for name in probs:
                    key  = f"{name}_{mitigate_attr}"
                    info = thresholds_data.get(key, {})
                    gval = str(group_val_map.get(mitigate_attr, ""))
                    model_thresholds[name] = float(info.get("thresholds", {}).get(gval, 0.5))
            else:
                model_thresholds = {name: 0.5 for name in probs}

            # ── Standard prediction label (always threshold=0.5) ─────────────
            attr_label = {"disability": "Disability", "gender": "Gender",
                          "age_band": "Age Band"}

            if apply_mitigation and mitigate_attr:
                std_col, mit_col = st.columns(2)
                std_col.markdown(
                    "<div style='text-align:center;font-size:0.85em;font-weight:600;"
                    "color:#555;margin-bottom:4px;'>Standard (threshold: 0.50)</div>",
                    unsafe_allow_html=True,
                )
                mit_col.markdown(
                    f"<div style='text-align:center;font-size:0.85em;font-weight:600;"
                    f"color:#6f42c1;margin-bottom:4px;'>Mitigated "
                    f"({attr_label.get(mitigate_attr, mitigate_attr)} thresholding)</div>",
                    unsafe_allow_html=True,
                )
                for name, prob in probs.items():
                    thresh = model_thresholds[name]
                    standard_pred = prob >= 0.5
                    mitigated_pred = prob >= thresh
                    flipped = standard_pred != mitigated_pred
                    std_col.markdown(prediction_card(name, prob, 0.5), unsafe_allow_html=True)
                    mit_col.markdown(
                        prediction_card(name, prob, thresh, flipped=flipped),
                        unsafe_allow_html=True,
                    )

                # Agreement badges — standard and mitigated
                n_std = sum(1 for p in probs.values() if p >= 0.5)
                n_mit = sum(1 for name, p in probs.items() if p >= model_thresholds[name])
                n_total = len(probs)

                def agree_badge(n_ok, total, color):
                    if n_ok == total / 2:
                        v = "⚖️ Tied"
                    elif n_ok > total / 2:
                        v = "✅ Success"
                    else:
                        v = "🔴 At-Risk"
                    t = (f"**{n_ok}/{total} models → {v}**"
                         if 0 < n_ok < total else f"**All {total} models → {v}**")
                    return (f"<div style='text-align:center;margin:6px 0;font-size:0.9em;"
                            f"color:{color};'>{t}</div>")

                std_col.markdown(agree_badge(n_std, n_total, "#333"), unsafe_allow_html=True)
                mit_col.markdown(agree_badge(n_mit, n_total, "#6f42c1"), unsafe_allow_html=True)

                # Mitigation summary banner
                gval_display = str(group_val_map.get(mitigate_attr, ""))
                key_xgb   = f"xgboost_{mitigate_attr}"
                mit_info  = thresholds_data.get(key_xgb, {})
                gap_before = mit_info.get("tpr_gap_before", None)
                gap_after  = mit_info.get("tpr_gap_after",  None)
                gap_line   = ""
                if gap_before is not None and gap_after is not None:
                    gap_line = (f"<br/>XGBoost TPR gap: "
                                f"<b>{gap_before:.1%}</b> → <b>{gap_after:.1%}</b> "
                                f"after mitigation")

                st.markdown(
                    f"<div style='background:#f3e9ff;border-left:4px solid #6f42c1;"
                    f"padding:10px 14px;border-radius:6px;margin:10px 0;'>"
                    f"⚖️ <b>Mitigation active</b> — "
                    f"{attr_label.get(mitigate_attr, mitigate_attr)} thresholding · "
                    f"group value: <b>{gval_display}</b>"
                    f"{gap_line}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            else:
                # No mitigation — standard cards
                c1, c2 = st.columns(2)
                for i, (name, prob) in enumerate(probs.items()):
                    col = c1 if i % 2 == 0 else c2
                    col.markdown(prediction_card(name, prob), unsafe_allow_html=True)

                n_success = sum(1 for p in probs.values() if p >= 0.5)
                n_total   = len(probs)
                if n_success == n_total / 2:
                    verdict = "⚖️ Tied"
                elif n_success > n_total / 2:
                    verdict = "✅ Success"
                else:
                    verdict = "🔴 At-Risk"
                agree_txt = (
                    f"**{n_success}/{n_total} models predict {verdict}**"
                    if n_success != n_total and n_success != 0
                    else f"**All {n_total} models agree: {verdict}**"
                )
                st.markdown(f"<div style='text-align:center;margin:8px 0;font-size:1em;'>"
                            f"{agree_txt}</div>", unsafe_allow_html=True)

        st.divider()

        # ── SHAP explanation ──────────────────────────────────────────────────
        xgb_model = active_models.get("xgboost")
        if xgb_model:
            if multiclass_mode:
                # Predicted class drives which SHAP slice to show
                mc_probs_xgb = all_mc_probs.get("xgboost", [0, 0, 0, 0])
                explain_class = int(np.argmax(mc_probs_xgb))
                explain_label = MC_CLASS_LABELS[explain_class]
                explain_color = MC_CLASS_COLORS[explain_label]
                st.markdown(
                    f"#### Why? — XGBoost SHAP  "
                    f"<span style='color:{explain_color};font-size:0.9em;'>"
                    f"(predicted: {explain_label})</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Bars show what pushed the model toward **{explain_label}** · "
                    "Longer bar = stronger influence"
                )
            else:
                st.markdown("#### Why? — XGBoost Explanation (SHAP)")
                st.caption(
                    "🟩 Green = pushes toward **Success** · "
                    "🟥 Red = pushes toward **At-Risk** · "
                    "Longer bar = stronger influence"
                )

            with st.expander("ℹ️ What is SHAP?", expanded=False):
                if multiclass_mode:
                    st.markdown(
                        """
                        **SHAP** (SHapley Additive exPlanations) measures how much each
                        feature contributed to *this specific* prediction.

                        - Each bar shows one feature's pull on the prediction
                        - **Positive (green):** pushes toward the **predicted class**
                        - **Negative (red):** pushes away from the predicted class
                        - **Larger absolute value = stronger influence**

                        Think of it as: *"Why did the model predict this class for this student?"*
                        """
                    )
                else:
                    st.markdown(
                        """
                        **SHAP** (SHapley Additive exPlanations) measures how much each
                        feature contributed to *this specific* prediction.

                        - Each bar shows one feature's pull on the prediction
                        - **Positive (green):** pushes toward **Success**
                        - **Negative (red):** pushes toward **At-Risk**
                        - **Larger absolute value = stronger influence**

                        *Example:* a value of `+0.5` for **Avg Early Score** means good
                        early scores increased the predicted probability of success by 0.5
                        log-odds units.

                        Think of it as: *"Why did the model make this decision for this student?"*
                        """
                    )

            with st.spinner("Computing explanation…"):
                explainer = shap.TreeExplainer(xgb_model)
                shap_out  = explainer.shap_values(X.values)
                if multiclass_mode:
                    # shap_out shape: (1, n_features, n_classes) or list of (1, n_features)
                    if isinstance(shap_out, list):
                        sv = shap_out[explain_class][0]
                    elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
                        sv = shap_out[0, :, explain_class]
                    else:
                        sv = shap_out[0]
                else:
                    if isinstance(shap_out, list):
                        sv = shap_out[1][0]
                    elif isinstance(shap_out, np.ndarray) and shap_out.ndim == 3:
                        sv = shap_out[0, :, 1]
                    else:
                        sv = shap_out[0]
                sv = sv.flatten()

            order      = np.argsort(np.abs(sv))[::-1][:10][::-1]
            top_labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
            top_vals   = [float(sv[i]) for i in order]
            colors     = ["#2ca02c" if v > 0 else "#d62728" for v in top_vals]

            if multiclass_mode:
                xaxis_title = f"SHAP value  ( + → {explain_label},  − → other classes )"
            else:
                xaxis_title = "SHAP value  ( + → Success,  − → At-Risk )"

            fig = go.Figure(go.Bar(
                x=top_vals, y=top_labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in top_vals],
                textposition="outside",
                textfont=dict(size=11),
            ))
            fig.add_vline(x=0, line_color="black", line_width=1.5)
            fig.update_layout(
                height=370,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis_title=xaxis_title,
                plot_bgcolor="white",
            )
            fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── Fairness note ─────────────────────────────────────────────────────
        fairness = load_fairness()
        xgb_fair = fairness.get("xgboost", {})
        notes = []

        dis_data = xgb_fair.get("disability", {})
        if "N" in dis_data and "Y" in dis_data:
            gap = dis_data["N"]["tpr"] - dis_data["Y"]["tpr"]
            if abs(gap) > 0.03:
                if gap > 0:
                    # Disabled students are disadvantaged
                    if disability == "Y":
                        dis_msg = (
                            f"**Disability:** This student has a disability. The model has a "
                            f"**{abs(gap):.1%} lower True Positive Rate** for disabled students — "
                            f"it is less likely to correctly flag at-risk students in this group. "
                            f"Treat this prediction with extra caution."
                        )
                    else:
                        dis_msg = (
                            f"**Disability:** Students *with* disabilities have a "
                            f"**{abs(gap):.1%} lower True Positive Rate** in this model — "
                            f"the model is less reliable for at-risk disabled students. "
                            f"This student is not in the disadvantaged group."
                        )
                else:
                    # Non-disabled students are disadvantaged (unexpected)
                    if disability == "N":
                        dis_msg = (
                            f"**Disability:** This student has no disability. The model has a "
                            f"**{abs(gap):.1%} lower True Positive Rate** for non-disabled students — "
                            f"an unexpected disparity. Treat this prediction with extra caution."
                        )
                    else:
                        dis_msg = (
                            f"**Disability:** Students *without* disabilities have a "
                            f"**{abs(gap):.1%} lower True Positive Rate** in this model — "
                            f"an unexpected disparity. This student is not in the disadvantaged group."
                        )
                notes.append(dis_msg)

        gen_data = xgb_fair.get("gender", {})
        if "M" in gen_data and "F" in gen_data:
            gen_gap = abs(gen_data["F"]["tpr"] - gen_data["M"]["tpr"])
            if gen_gap > 0.03:
                disadvantaged_gender = "M" if gen_data["M"]["tpr"] < gen_data["F"]["tpr"] else "F"
                dis_label = "Male" if disadvantaged_gender == "M" else "Female"
                adv_label = "Female" if disadvantaged_gender == "M" else "Male"
                student_in_dis = gender == disadvantaged_gender
                student_group = "in the disadvantaged group" if student_in_dis else "in the advantaged group"
                student_label = "Male" if gender == "M" else "Female"
                notes.append(
                    f"**Gender:** {dis_label} students have a **{gen_gap:.1%} lower True Positive Rate** "
                    f"than {adv_label.lower()} students — at-risk {dis_label.lower()} students are less likely "
                    f"to be flagged. This student is {student_label} ({student_group})."
                )

        if notes:
            st.divider()
            st.markdown("#### ⚖️ Fairness Context")
            st.caption("Known fairness gaps in the model for this student's demographic group:")
            for note in notes:
                st.warning(note, icon="⚠️")

    else:
        # Placeholder before first prediction
        st.markdown("### Results")
        st.info(
            "👈  Fill in the student profile on the left and click **🔍 Predict**,\n\n"
            "or try one of the **quick presets** above to load a sample student instantly.",
            icon="ℹ️",
        )
        st.markdown("")
        st.markdown(
            "The predictor will show:\n"
            "- Predictions from **4 ML models** (Logistic Regression, Random Forest, "
            "XGBoost, MLP)\n"
            "- A **SHAP chart** explaining which features drove the prediction\n"
            "- A **fairness note** if the student's demographic group has a known "
            "model bias"
        )
