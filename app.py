"""SmartPure AI - Streamlit Dashboard (v2)"""
import os
import json
import datetime
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartPure AI", page_icon="💧", layout="wide")

# Train automatically the first time (needed on Streamlit Cloud)
if not (os.path.exists("quality_model.pkl") and os.path.exists("filter_model.pkl")):
    import train_models  # noqa: F401


@st.cache_resource
def load_models():
    return joblib.load("quality_model.pkl"), joblib.load("filter_model.pkl")


quality_model, filter_model = load_models()

# ---------------- Helper functions ----------------
QUALITY_NAMES = ["Safe", "Moderate", "Unsafe"]

PRESETS = {
    "Clean water": dict(ph=7.2, tds=150, turbidity=0.8, usage=60, users=3, age=40),
    "Average water": dict(ph=7.8, tds=600, turbidity=3.0, usage=80, users=4, age=60),
    "Poor water": dict(ph=8.9, tds=1100, turbidity=9.0, usage=120, users=6, age=30),
}


def load_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def check(value, safe_limit, warn_limit):
    """Traffic-light status for TDS and turbidity."""
    if value <= safe_limit:
        return "✅ Good"
    if value <= warn_limit:
        return "⚠️ Moderate"
    return "❌ High"


def check_ph(value):
    if 6.5 <= value <= 8.5:
        return "✅ Good"
    if 5.5 <= value <= 9.5:
        return "⚠️ Moderate"
    return "❌ Out of range"


# ---------------- Sidebar inputs ----------------
for key, value in PRESETS["Clean water"].items():
    st.session_state.setdefault(key, value)

st.sidebar.header("💧 Water & Usage Details")
st.sidebar.caption("Quick examples:")
c1, c2, c3 = st.sidebar.columns(3)
c1.button("Clean", on_click=load_preset, args=("Clean water",))
c2.button("Average", on_click=load_preset, args=("Average water",))
c3.button("Poor", on_click=load_preset, args=("Poor water",))

ph = st.sidebar.slider("pH", 4.0, 10.0, step=0.1, key="ph")
tds = st.sidebar.slider("TDS (mg/L)", 20, 1500, key="tds")
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 15.0, step=0.1, key="turbidity")
usage = st.sidebar.number_input("Daily usage (litres)", 10, 500, key="usage")
users = st.sidebar.number_input("Number of users", 1, 15, key="users")
age = st.sidebar.number_input("Filter age (days)", 0, 600, key="age")

# ---------------- Predictions (run on every change) ----------------
q_input = pd.DataFrame([[ph, tds, turbidity]], columns=["ph", "tds", "turbidity"])
q = int(quality_model.predict(q_input)[0])
probs = quality_model.predict_proba(q_input)[0]

f_input = pd.DataFrame(
    [[tds, turbidity, usage, users, age]],
    columns=["tds", "turbidity", "daily_usage", "users", "filter_age"],
)
days = max(0, int(filter_model.predict(f_input)[0]))
replace_on = datetime.date.today() + datetime.timedelta(days=days)

# ---------------- Page ----------------
st.title("💧 SmartPure AI")
st.caption("ML-Based Water Quality & Filter Life Prediction System")

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Info", "ℹ️ How it works"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("1. Water Quality")
        if q == 0:
            st.success("✅ Safe to drink")
        elif q == 1:
            st.warning("⚠️ Moderate - needs attention")
        else:
            st.error("❌ Unsafe - do not drink")

        st.write("Model confidence for each class:")
        st.bar_chart(pd.Series(probs * 100, index=QUALITY_NAMES))

        st.write("Parameter check:")
        st.table(pd.DataFrame({
            "Parameter": ["pH", "TDS (mg/L)", "Turbidity (NTU)"],
            "Your value": [ph, tds, turbidity],
            "Ideal range": ["6.5 - 8.5", "below 500", "below 5"],
            "Status": [check_ph(ph), check(tds, 500, 1000), check(turbidity, 5, 10)],
        }))

    with right:
        st.subheader("2. Filter Life")
        m1, m2 = st.columns(2)
        m1.metric("Remaining life", f"{days} days")
        m2.metric("Replace around", replace_on.strftime("%d %b %Y"))
        st.progress(min(days, 365) / 365)

        if days <= 15:
            st.error("🔴 Replace the filter very soon!")
        elif days <= 45:
            st.warning("🟡 Plan a filter replacement soon.")
        else:
            st.success("🟢 Filter is in good condition.")

        st.subheader("3. Recommendations")
        tips = []
        if tds > 500:
            tips.append("High TDS: check the RO membrane and consider more frequent servicing.")
        if turbidity > 5:
            tips.append("High turbidity: add or clean the sediment pre-filter.")
        if not 6.5 <= ph <= 8.5:
            tips.append("pH is outside 6.5-8.5: a pH-balancing or mineral cartridge may help.")
        if days <= 45:
            tips.append("Book a filter change before the date shown above.")
        if not tips:
            tips.append("Everything looks fine. Keep up the regular maintenance.")
        for tip in tips:
            st.write("• " + tip)

    report = (
        "SmartPure AI - Report\n"
        f"Date: {datetime.date.today()}\n\n"
        f"pH: {ph}\nTDS: {tds} mg/L\nTurbidity: {turbidity} NTU\n"
        f"Daily usage: {usage} L\nUsers: {users}\nFilter age: {age} days\n\n"
        f"Water quality: {QUALITY_NAMES[q]}\n"
        f"Filter life remaining: {days} days (replace around {replace_on})\n"
        "Recommendations:\n" + "\n".join("- " + t for t in tips)
    )
    st.download_button("📄 Download report", report, file_name="smartpure_report.txt")

with tab2:
    st.subheader("Model performance")
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            m = json.load(f)
        a, b, c = st.columns(3)
        a.metric("Quality accuracy", f"{m['accuracy']}%")
        b.metric("Filter life error (MAE)", f"{m['mae']} days")
        c.metric("Filter life R²", m["r2"])
        st.caption(f"Trained on {m['samples']} synthetic samples (80% training, 20% testing).")

    st.subheader("Which inputs matter most?")
    g1, g2 = st.columns(2)
    with g1:
        st.write("Water quality model")
        st.bar_chart(pd.Series(quality_model.feature_importances_, index=q_input.columns))
    with g2:
        st.write("Filter life model")
        st.bar_chart(pd.Series(filter_model.feature_importances_, index=f_input.columns))

with tab3:
    st.markdown(
        """
**Model 1 - Water Quality (Random Forest Classifier)**
Takes pH, TDS and turbidity and predicts Safe, Moderate or Unsafe.

**Model 2 - Filter Life (Random Forest Regressor)**
Takes TDS, turbidity, daily usage, number of users and filter age and predicts the remaining days.

**Random Forest** = 100 decision trees. For the classifier they vote, for the regressor
their answers are averaged.

**Data:** synthetic, created from standard water quality limits. With real sensor data
from a purifier, the same code can be retrained.
        """
    )
