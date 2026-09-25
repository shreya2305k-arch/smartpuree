"""SmartPure AI - Streamlit Dashboard (v3, restyled)"""
import os
import json
import datetime
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SmartPure AI", page_icon="\U0001F4A7", layout="wide")

# Train automatically the first time (needed on Streamlit Cloud)
if not (os.path.exists("quality_model.pkl") and os.path.exists("filter_model.pkl")):
    import train_models  # noqa: F401


@st.cache_resource
def load_models():
    return joblib.load("quality_model.pkl"), joblib.load("filter_model.pkl")


quality_model, filter_model = load_models()

# ---------------- Design tokens ----------------
INK = "#12262E"
PAPER = "#F4F6F4"
CARD = "#FFFFFF"
LINE = "#DCE3E0"
TEAL = "#127C77"
TEAL_SOFT = "#E1F0EE"
AMBER = "#C08A22"
AMBER_SOFT = "#FBF0DA"
RED = "#B03A2E"
RED_SOFT = "#F8E3E0"

STATUS = {
    0: dict(word="Safe to drink", verdict="looks good", color=TEAL, soft=TEAL_SOFT, mood="Everything checked out."),
    1: dict(word="Needs a look", verdict="is moderate", color=AMBER, soft=AMBER_SOFT, mood="A couple of readings are outside the ideal range."),
    2: dict(word="Do not drink yet", verdict="is unsafe", color=RED, soft=RED_SOFT, mood="More than one reading is well outside the safe range."),
}
QUALITY_NAMES = ["Safe", "Moderate", "Unsafe"]
QUALITY_COLORS = [TEAL, AMBER, RED]

PRESETS = {
    "Clean water": dict(ph=7.2, tds=150, turbidity=0.8, usage=60, users=3, age=40),
    "Average water": dict(ph=7.8, tds=600, turbidity=3.0, usage=80, users=4, age=60),
    "Poor water": dict(ph=8.9, tds=1100, turbidity=9.0, usage=120, users=6, age=30),
}


def load_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def check(value, safe_limit, warn_limit):
    if value <= safe_limit:
        return "Good", TEAL
    if value <= warn_limit:
        return "Moderate", AMBER
    return "High", RED


def check_ph(value):
    if 6.5 <= value <= 8.5:
        return "Good", TEAL
    if 5.5 <= value <= 9.5:
        return "Moderate", AMBER
    return "Out of range", RED


# ---------------- Global styling ----------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {INK};
}}
.stApp {{ background-color: {PAPER}; }}
section[data-testid="stSidebar"] {{ background-color: {CARD}; border-right: 1px solid {LINE}; }}
#MainMenu, footer {{ visibility: hidden; }}
div.block-container {{ padding-top: 2rem; max-width: 1180px; }}

.sp-eyebrow {{ font-size: 0.85rem; color: #5C6B67; margin-bottom: 0.2rem; }}
.sp-hero {{
    background: {STATUS[0]['soft']};
    border-radius: 6px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}}
.sp-hero h1 {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.1rem;
    margin: 0 0 0.3rem 0;
    line-height: 1.15;
}}
.sp-hero p {{ margin: 0; color: {INK}; opacity: 0.85; font-size: 0.98rem; }}
.sp-confidence {{ text-align: right; }}
.sp-confidence .num {{ font-family: 'Fraunces', serif; font-size: 2.6rem; font-weight: 600; line-height: 1; }}
.sp-confidence .lbl {{ font-size: 0.85rem; opacity: 0.75; }}

.sp-section-title {{ font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 600; margin: 0.2rem 0 0.9rem 0; }}

.sp-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 0; border-bottom: 1px solid {LINE};
}}
.sp-row:last-child {{ border-bottom: none; }}
.sp-row .name {{ font-weight: 500; }}
.sp-row .range {{ color: #5C6B67; font-size: 0.85rem; }}
.sp-dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 0.5rem; }}
.sp-tag {{ font-size: 0.82rem; font-weight: 500; padding: 0.15rem 0.55rem; border-radius: 20px; }}

.sp-tip {{
    padding: 0.55rem 0.9rem; margin-bottom: 0.5rem; border-radius: 4px;
    border-left: 3px solid {TEAL}; background: {CARD}; font-size: 0.95rem;
}}

.sp-stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; padding: 1rem 1.2rem; }}
.sp-stat .num {{ font-family: 'Fraunces', serif; font-size: 1.8rem; font-weight: 600; }}
.sp-stat .lbl {{ font-size: 0.85rem; color: #5C6B67; }}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar inputs ----------------
for key, value in PRESETS["Clean water"].items():
    st.session_state.setdefault(key, value)

st.sidebar.markdown("### \U0001F4A7 Water & usage details")
st.sidebar.caption("Quick examples")
c1, c2, c3 = st.sidebar.columns(3)
c1.button("Clean", on_click=load_preset, args=("Clean water",), use_container_width=True)
c2.button("Average", on_click=load_preset, args=("Average water",), use_container_width=True)
c3.button("Poor", on_click=load_preset, args=("Poor water",), use_container_width=True)
st.sidebar.markdown("---")

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
s = STATUS[q]

# ---------------- Page ----------------
st.markdown('<div class="sp-eyebrow">SmartPure AI</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="sp-hero" style="background:{s['soft']}">
  <div>
    <h1 style="color:{s['color']}">{s['word']}</h1>
    <p>pH {ph}, TDS {tds} mg/L, turbidity {turbidity} NTU. {s['mood']}</p>
  </div>
  <div class="sp-confidence">
    <div class="num" style="color:{s['color']}">{probs[q]*100:.0f}%</div>
    <div class="lbl">model confidence</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "Model info", "How it works"])

with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="sp-section-title">Water readings</div>', unsafe_allow_html=True)
        rows = [
            ("pH", ph, "6.5 - 8.5", *check_ph(ph)),
            ("TDS", f"{tds} mg/L", "below 500", *check(tds, 500, 1000)),
            ("Turbidity", f"{turbidity} NTU", "below 5", *check(turbidity, 5, 10)),
        ]
        rows_html = "".join(
            f'<div class="sp-row"><span><span class="sp-dot" style="background:{color}"></span>'
            f'<span class="name">{name}</span></span>'
            f'<span class="range">{val} &middot; ideal {ideal}</span>'
            f'<span class="sp-tag" style="background:{color}22;color:{color}">{label}</span></div>'
            for name, val, ideal, label, color in rows
        )
        st.markdown(f'<div class="sp-stat">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sp-section-title" style="margin-top:1.5rem">Model confidence</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=probs * 100, y=QUALITY_NAMES, orientation="h",
            marker_color=QUALITY_COLORS,
            text=[f"{p*100:.0f}%" for p in probs], textposition="outside",
        ))
        fig.update_layout(
            height=200, margin=dict(l=0, r=20, t=10, b=10),
            xaxis=dict(range=[0, 105], showgrid=False, visible=False),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=INK, size=13),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown('<div class="sp-section-title">Filter life</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(days, 365),
            number={"suffix": " days", "font": {"family": "Fraunces", "size": 34, "color": INK}},
            gauge={
                "axis": {"range": [0, 365], "tickcolor": LINE},
                "bar": {"color": s["color"], "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 15], "color": RED_SOFT},
                    {"range": [15, 45], "color": AMBER_SOFT},
                    {"range": [45, 365], "color": TEAL_SOFT},
                ],
            },
        ))
        gauge.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0),
                             paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=INK))
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<p style="text-align:center;margin-top:-1rem;color:#5C6B67">'
            f'Replace around <strong style="color:{INK}">{replace_on.strftime("%d %b %Y")}</strong></p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sp-section-title" style="margin-top:0.5rem">Recommendations</div>', unsafe_allow_html=True)
        tips = []
        if tds > 500:
            tips.append(("High TDS. Check the RO membrane and consider more frequent servicing.", RED))
        if turbidity > 5:
            tips.append(("High turbidity. Add or clean the sediment pre-filter.", RED))
        if not 6.5 <= ph <= 8.5:
            tips.append(("pH is outside 6.5-8.5. A pH-balancing or mineral cartridge may help.", AMBER))
        if days <= 45:
            tips.append(("Book a filter change before the date shown above.", AMBER))
        if not tips:
            tips.append(("Everything looks fine. Keep up the regular maintenance.", TEAL))
        for text, color in tips:
            st.markdown(f'<div class="sp-tip" style="border-left-color:{color}">{text}</div>', unsafe_allow_html=True)

    report = (
        "SmartPure AI - Report\n"
        f"Date: {datetime.date.today()}\n\n"
        f"pH: {ph}\nTDS: {tds} mg/L\nTurbidity: {turbidity} NTU\n"
        f"Daily usage: {usage} L\nUsers: {users}\nFilter age: {age} days\n\n"
        f"Water quality: {QUALITY_NAMES[q]}\n"
        f"Filter life remaining: {days} days (replace around {replace_on})\n"
        "Recommendations:\n" + "\n".join("- " + t for t, _ in tips)
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("Download report", report, file_name="smartpure_report.txt")

with tab2:
    st.markdown('<div class="sp-section-title">Model performance</div>', unsafe_allow_html=True)
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            m = json.load(f)
        a, b, c = st.columns(3)
        for col, num, lbl in [
            (a, f"{m['accuracy']}%", "Quality accuracy"),
            (b, f"{m['mae']} days", "Filter life error (MAE)"),
            (c, f"{m['r2']}", "Filter life R2"),
        ]:
            col.markdown(f'<div class="sp-stat"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
                          unsafe_allow_html=True)
        st.caption(f"Trained on {m['samples']} synthetic samples (80% training, 20% testing).")

    st.markdown('<div class="sp-section-title" style="margin-top:1.5rem">Which inputs matter most?</div>', unsafe_allow_html=True)

    def importance_fig(model, columns):
        imp = model.feature_importances_
        order = imp.argsort()
        fig = go.Figure(go.Bar(
            x=imp[order], y=[columns[i] for i in order], orientation="h",
            marker_color=TEAL,
        ))
        fig.update_layout(
            height=220, margin=dict(l=0, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, visible=False),
            font=dict(family="Inter", color=INK, size=13),
        )
        return fig

    g1, g2 = st.columns(2)
    with g1:
        st.caption("Water quality model")
        st.plotly_chart(importance_fig(quality_model, list(q_input.columns)),
                         use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.caption("Filter life model")
        st.plotly_chart(importance_fig(filter_model, list(f_input.columns)),
                         use_container_width=True, config={"displayModeBar": False})

with tab3:
    st.markdown('<div class="sp-section-title">How SmartPure AI works</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="sp-stat" style="margin-bottom:0.8rem">
<strong>Model 1, water quality (Random Forest classifier)</strong><br>
Takes pH, TDS and turbidity and predicts Safe, Moderate or Unsafe.
</div>
<div class="sp-stat" style="margin-bottom:0.8rem">
<strong>Model 2, filter life (Random Forest regressor)</strong><br>
Takes TDS, turbidity, daily usage, number of users and filter age and predicts the remaining days.
</div>
<div class="sp-stat" style="margin-bottom:0.8rem">
<strong>Random Forest</strong><br>
100 decision trees. For the classifier they vote, for the regressor their answers are averaged.
</div>
<div class="sp-stat">
<strong>Data</strong><br>
Synthetic, created from standard water quality limits. With real sensor data from a purifier, the same code can be retrained.
</div>
""",
        unsafe_allow_html=True,
    )
