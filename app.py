import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Assets
model = joblib.load('assets/champion_model.pkl')
le = joblib.load('assets/label_encoder.pkl')

st.set_page_config(page_title="PrediFix AI", layout="wide")

# ===== FIXED CSS (DARK + READABLE) =====
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    border-left: 6px solid #4e73df;
}

/* Metric text */
div[data-testid="stMetric"] * {
    color: black !important;
}

/* Expander */
div[data-testid="stExpander"] {
    background-color: #ffffff;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("Industrial Fault Detection Using Machine Learning")
st.write("Real-time predictive maintenance powered by Gradient Boosting Machines.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("🎮 Sensor Input Simulation")
    m_type = st.selectbox("Product Grade", ["L", "M", "H"])
    air_t = st.slider("Air Temp [K]", 295, 305, 300)
    proc_t = st.slider("Process Temp [K]", 305, 315, 310)
    rpm = st.slider("Rotational Speed [RPM]", 1100, 2900, 1500)
    torque = st.slider("Torque [Nm]", 3, 80, 40)
    wear = st.slider("Tool Wear [min]", 0, 260, 100)

# --- DATA PROCESSING ---
type_enc = le.transform([m_type])[0]
temp_diff = proc_t - air_t
pwr = torque * rpm
features = np.array([[type_enc, air_t, proc_t, rpm, torque, wear, temp_diff, pwr]])
prob = model.predict_proba(features)[0][1]

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
health_score = int((1 - prob) * 100)

with col1:
    st.metric("Failure Risk", f"{prob:.1%}")
with col2:
    st.metric("Health Score", f"{health_score}/100")
with col3:
    status = "CRITICAL" if prob > 0.5 else "WARNING" if prob > 0.1 else "OPTIMAL"
    st.metric("System Status", status)
with col4:
    st.metric("Energy Draw", f"{pwr/1000:.1f} kW")

# --- HEALTH BAR ---
st.write("### 🩺 Machine Vitality")
bar_color = "#e74c3c" if prob > 0.5 else "#f39c12" if prob > 0.1 else "#2ecc71"

st.markdown(f"""
<div style="
    width:100%;
    background:#2c2c2c;
    border-radius:25px;
    overflow:hidden;
    height:45px;
">
  <div style="
      width:{health_score}%;
      background:{bar_color};
      height:100%;
      display:flex;
      align-items:center;
      justify-content:center;
      color:white;
      font-weight:700;
      white-space:nowrap;
      min-width:80px;
  ">
    {health_score}% Healthy
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- FEATURE IMPORTANCE ---
imp_df = pd.DataFrame({
    'Feature': ['Type', 'Air T', 'Proc T', 'RPM', 'Torque', 'Wear', 'Δ Temp', 'Power'],
    'Importance': model.feature_importances_
}).sort_values('Importance')

fig, ax = plt.subplots()
ax.barh(imp_df['Feature'], imp_df['Importance'])
ax.set_title("Global Feature Importance")
st.pyplot(fig)

st.caption("Developed with ☕")
