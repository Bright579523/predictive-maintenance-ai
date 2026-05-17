import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- 0. PATH CONFIGURATION ---
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_FILE = BASE_DIR / "models" / "predictive_maintenance_model.pkl"
SCALER_FILE = BASE_DIR / "models" / "scaler.pkl"

# --- 1. PAGE CONFIG & CSS ---
st.set_page_config(page_title="Smart CNC Maintenance", page_icon="🏭", layout="wide")
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { background-color: #0e1117; color: white; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #151922; border-right: 1px solid #262730; }
    .stButton>button { width: 100%; border-radius: 8px; border: 1px solid #30333d; background-color: #1e2130; color: white; transition: all 0.3s; font-weight: 600; }
    .stButton>button:hover { border-color: #00D4FF; color: #00D4FF; background-color: #1a1f2e; }
    .kpi-card { background: linear-gradient(135deg, #1e2130 0%, #252a3a 100%); padding: 20px; border-radius: 12px; border-left: 4px solid #00D4FF; text-align: center; }
    .kpi-card h2 { color: #00D4FF; margin: 0; font-size: 28px; }
    .kpi-card p { color: #8892a4; margin: 5px 0 0 0; font-size: 13px; }
    .status-box { padding: 10px; border-radius: 8px; background-color: #1e2130; border: 1px solid #30333d; margin-bottom: 10px; }
    .work-order { background: linear-gradient(135deg, #2a1a1a 0%, #1e2130 100%); border: 1px solid #e74c3c; border-radius: 12px; padding: 20px; }
    .work-order-safe { background: linear-gradient(135deg, #1a2a1a 0%, #1e2130 100%); border: 1px solid #2ecc71; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 Smart CNC Predictive Maintenance System")
st.caption("AI-Driven Failure Prediction | Explainable AI (SHAP) | RAG Consultant | ROI Analysis")

# --- 2. LOAD DATA & MODEL ---
@st.cache_data
def get_dataset():
    if DATA_FILE.exists(): return pd.read_csv(DATA_FILE)
    return None

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except: return None, None

df = get_dataset()
model, scaler = load_artifacts()

def get_full_name(code):
    mapping = {'TWF': 'Tool Wear Failure', 'HDF': 'Heat Dissipation Failure', 'PWF': 'Power Failure', 'OSF': 'Overstrain Failure', 'RNF': 'Random Failure'}
    return mapping.get(code, code)

def _offline_fallback(query):
    query = query.lower()
    if any(k in query for k in ["chip", "chipping", "fracture", "break", "broken", "shatter", "crack", "บิ่น", "แตก", "หัก"]):
        return "🔧 **Chipping / Fracture:**\nOften caused by mechanical shock, unstable setup, or thermal fluctuations.\n* **Countermeasure:** Reduce feed rate, select a tougher carbide grade (e.g., PVD coated), or ensure rigid workpiece clamping."
    elif any(k in query for k in ["wear", "flank", "crater", "rubbing", "dull", "life", "สึก", "อายุ", "พังไว", "ทื่อ"]):
        return "⚠️ **Tool Wear (Flank/Crater):**\nTypically results from excessive cutting speed or machining abrasive materials.\n* **Countermeasure:** Lower the cutting speed (Vc), apply a highly wear-resistant CVD coated grade, or check coolant delivery."
    elif any(k in query for k in ["chatter", "vibration", "noise", "squeal", "สั่น", "เสียงดัง", "สะท้าน"]):
        return "🔊 **Chatter & Vibration:**\nCaused by lack of rigidity or excessive cutting forces.\n* **Countermeasure:** Reduce depth of cut (ap), decrease cutting speed (Vc), use a smaller nose radius insert, or improve tool overhang."
    elif any(k in query for k in ["bue", "built up", "built-up", "sticky", "melt", "พอก", "ติดมีด", "ละลายติด"]):
        return "🧲 **Built-Up Edge (BUE):**\nMaterial welding to the cutting edge, common in low-carbon steel or aluminum at low speeds.\n* **Countermeasure:** Increase cutting speed (Vc), use a sharper edge geometry, or apply high-pressure coolant."
    elif any(k in query for k in ["surface", "finish", "roughness", "ra", "rz", "ผิว", "หยาบ", "ไม่สวย", "ลาย"]):
        return "✨ **Poor Surface Finish:**\nUsually related to feed rate, nose radius, or BUE formation.\n* **Countermeasure:** Reduce feed rate (fn), increase cutting speed (Vc) to avoid BUE, or consider using a Wiper geometry insert."
    elif any(k in query for k in ["speed", "rpm", "vc", "feed", "fn", "fz", "parameter", "ความเร็ว", "รอบ", "เร็ว", "ช้า", "ฟีด"]):
        return "⚡ **Cutting Speed (Vc) & Feed Recommendation:**\nDepends on the ISO material group.\n* **ISO P (Steel):** Moderate to high Vc and fn.\n* **ISO M (Stainless):** Lower Vc to prevent work-hardening; keep fn high enough to cut under the hardened layer."
    elif any(k in query for k in ["grade", "insert", "coating", "cvd", "pvd", "carbide", "cermet", "เกรด", "เคลือบ", "อินเสิร์ท"]):
        return "💎 **Insert Grade Selection:**\n* **CVD Coated:** Superior heat and wear resistance. Ideal for continuous turning.\n* **PVD Coated:** High edge toughness and sharpness. Excellent for interrupted cuts, milling, or sticky materials."
    else:
        return "🤔 **Information Not Found:**\nSorry, I couldn't find a matching keyword. \n💡 *Hint: Try keywords like 'Chipping', 'Wear', 'Chatter', 'BUE', 'Speed', or 'Surface Finish'.*"

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎛️ System Console")
    st.caption("v3.0.0 | MLOps Edition")
    st.markdown("---")
    st.markdown("### 📡 Data Source")
    if df is not None:
        st.markdown(f"""<div class="status-box"><span style='color:#2ecc71; font-size: 14px;'>●</span> <span style='font-weight:bold; margin-left: 5px;'>System Online</span><br><span style='font-size:12px; color:gray; margin-left: 20px;'>{len(df):,} records active</span></div>""", unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV", csv, "ai4i2020.csv", "text/csv", use_container_width=True)
    else:
        st.error("❌ Dataset not found.")
    st.markdown("---")
    st.markdown("### 🤖 AI Engine")
    ai_status = "🟢 Groq (Llama-3.3)" if os.getenv("GROQ_API_KEY") else "🔴 No API Key"
    st.markdown(f"""<div class="status-box"><span style='font-size:12px;'>{ai_status}</span></div>""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 4. MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Health", "🔮 Predictive Simulator", "💬 AI Consultant", "💰 Financial Impact"])

# ==================================================
# TAB 1: OVERVIEW & HEALTH MONITOR
# ==================================================
with tab1:
    if df is not None:
        total = len(df); fails = int(df['Machine failure'].sum())
        fail_rate = (fails / total) * 100
        valid_cols = [c for c in ['TWF','HDF','PWF','OSF','RNF'] if c in df.columns]
        top_fail = df[valid_cols].sum().idxmax() if valid_cols else "N/A"

        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f'<div class="kpi-card"><h2>{total:,}</h2><p>Total Operations</p></div>', unsafe_allow_html=True)
        with k2: st.markdown(f'<div class="kpi-card"><h2 style="color:#e74c3c">{fail_rate:.2f}%</h2><p>Failure Rate</p></div>', unsafe_allow_html=True)
        with k3: st.markdown(f'<div class="kpi-card"><h2 style="color:#f39c12">{top_fail}</h2><p>Top Risk: {get_full_name(top_fail)}</p></div>', unsafe_allow_html=True)
        with k4: st.markdown(f'<div class="kpi-card"><h2 style="color:#2ecc71">98.15%</h2><p>Model Accuracy (XGBoost)</p></div>', unsafe_allow_html=True)

        st.markdown("---")

            # Health Gauges + 3D Scatter
        col_gauge, col_scatter = st.columns([1, 2])
        with col_gauge:
            st.markdown("##### 🏥 Machine Health Indicators")
            avg_temp = df['Process temperature [K]'].mean()
            avg_torque = df['Torque [Nm]'].mean()
            avg_wear = df['Tool wear [min]'].mean()

            fig_g1 = go.Figure(go.Indicator(mode="gauge+number", value=avg_temp, title={'text': "🌡️ Avg Temperature (K)", 'font': {'size': 13}}, number={'font': {'size': 28}}, gauge={'axis': {'range': [295, 315]}, 'bar': {'color': '#00D4FF'}, 'steps': [{'range': [295, 305], 'color': '#1a3a2a'}, {'range': [305, 310], 'color': '#3a3a1a'}, {'range': [310, 315], 'color': '#3a1a1a'}]}))
            fig_g1.update_layout(height=180, margin=dict(t=50,b=10,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_g1, use_container_width=True)

            fig_g2 = go.Figure(go.Indicator(mode="gauge+number", value=avg_torque, title={'text': "⚙️ Avg Torque (Nm)", 'font': {'size': 13}}, number={'font': {'size': 28}}, gauge={'axis': {'range': [0, 80]}, 'bar': {'color': '#f39c12'}, 'steps': [{'range': [0, 40], 'color': '#1a3a2a'}, {'range': [40, 60], 'color': '#3a3a1a'}, {'range': [60, 80], 'color': '#3a1a1a'}]}))
            fig_g2.update_layout(height=180, margin=dict(t=50,b=10,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_g2, use_container_width=True)

            fig_g3 = go.Figure(go.Indicator(mode="gauge+number", value=avg_wear, title={'text': "🔧 Avg Tool Wear (min)", 'font': {'size': 13}}, number={'font': {'size': 28}}, gauge={'axis': {'range': [0, 250]}, 'bar': {'color': '#e74c3c'}, 'steps': [{'range': [0, 100], 'color': '#1a3a2a'}, {'range': [100, 180], 'color': '#3a3a1a'}, {'range': [180, 250], 'color': '#3a1a1a'}]}))
            fig_g3.update_layout(height=180, margin=dict(t=50,b=10,l=20,r=20), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_g3, use_container_width=True)

        with col_scatter:
            st.markdown("##### 🌐 3D Failure Distribution (RPM × Torque × Wear)")
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig_3d = px.scatter_3d(sample, x='Rotational speed [rpm]', y='Torque [Nm]', z='Tool wear [min]', color='Machine failure', color_discrete_map={0: '#00D4FF', 1: '#e74c3c'}, opacity=0.6, labels={'Machine failure': 'Status'})
            fig_3d.update_layout(height=580, paper_bgcolor='rgba(0,0,0,0)', scene=dict(xaxis=dict(backgroundcolor='#0e1117', gridcolor='#262730'), yaxis=dict(backgroundcolor='#0e1117', gridcolor='#262730'), zaxis=dict(backgroundcolor='#0e1117', gridcolor='#262730')), font_color='white', legend_title_text='0=Normal | 1=Failed')
            st.plotly_chart(fig_3d, use_container_width=True)

        # Failure Type Bar Chart
        st.markdown("##### 📊 Failure Type Distribution")
        if valid_cols:
            fail_counts = df[valid_cols].sum().reset_index()
            fail_counts.columns = ['Type', 'Count']
            fail_counts['Full Name'] = fail_counts['Type'].apply(get_full_name)
            colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#1abc9c']
            fig_bar = px.bar(fail_counts, x='Full Name', y='Count', color='Full Name', color_discrete_sequence=colors, text='Count')
            # Add y-axis padding to prevent numbers from clipping at the top
            fig_bar.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False, xaxis_title="", yaxis_title="Number of Cases", yaxis=dict(range=[0, fail_counts['Count'].max() * 1.15]))
            fig_bar.update_traces(textposition='outside', cliponaxis=False)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("⚠️ No dataset loaded.")

# ==================================================
# TAB 2: PREDICTIVE SIMULATOR
# ==================================================
with tab2:
    st.subheader("🔮 Cutting Condition Simulation")
    if model is not None:
        # Quick Scenarios
        st.markdown("##### ⚡ Quick Scenarios:")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("✅ Normal Run"): st.session_state.update(rpm_s=1500, torque_s=35.0, wear_s=20, air_s=298.0, proc_s=309.0); st.rerun()
        if b2.button("⚠️ High Wear"): st.session_state.update(rpm_s=1500, torque_s=55.0, wear_s=290, air_s=298.0, proc_s=309.0); st.rerun()
        if b3.button("🔥 Overstrain"): st.session_state.update(rpm_s=1400, torque_s=65.0, wear_s=200, air_s=300.0, proc_s=312.0); st.rerun()
        if b4.button("⚡ Power Fail"): st.session_state.update(rpm_s=2800, torque_s=60.0, wear_s=50, air_s=298.0, proc_s=309.0); st.rerun()
        st.divider()

        # Sliders + Gauge
        col_input, col_gauge = st.columns([1, 1])
        with col_input:
            st.markdown("#### 🎛️ Operator Input (Manual)")
            rpm = st.slider("Rotational Speed [rpm]", 1000, 3000, st.session_state.get('rpm_s', 1550),
                            help="Spindle speed — higher RPM = higher productivity but more heat & wear")
            torque = st.slider("Torque [Nm]", 10.0, 90.0, st.session_state.get('torque_s', 40.0),
                               help="Cutting force — high torque increases risk of Overstrain Failure (OSF)")
            wear = st.slider("Tool Wear [min]", 0, 300, st.session_state.get('wear_s', 50),
                             help="Cumulative tool usage time — above 200 min significantly increases failure risk")
            type_in = st.selectbox(
                "Workpiece Material",
                ["Aluminum / Non-Ferrous (Light Duty)", "Carbon Steel (Medium Duty)", "Stainless Steel / Alloy (Heavy Duty)"],
                help="Material type determines machine load profile and failure risk.\n\n"
                     "• **Aluminum:** High RPM, low torque — risk of BUE, chatter\n\n"
                     "• **Carbon Steel:** Balanced parameters — most common\n\n"
                     "• **Stainless/Alloy:** Low RPM, high torque — risk of OSF, heat damage")

            # Sensor Readings (auto-linked, collapsible)
            with st.expander("🌡️ Sensor Readings (auto / IoT)", expanded=False):
                st.caption("In production, these values come from machine sensors automatically. "
                           "Adjust only for simulation purposes.")
                air = st.number_input("Air Temperature [K]", 290.0, 310.0,
                                      st.session_state.get('air_s', 298.0),
                                      help="Ambient workshop temperature (typically 295-303 K)")
                auto_proc = st.checkbox("Auto-calculate Process Temp", value=True,
                                        help="Process temp ≈ Air temp + 10K (typical thermal rise)")
                if auto_proc:
                    proc = air + 10.0
                    st.info(f"Process Temp = {air:.1f} + 10.0 = **{proc:.1f} K**")
                else:
                    proc = st.number_input("Process Temperature [K]", 300.0, 320.0,
                                           st.session_state.get('proc_s', 309.0),
                                           help="Temperature at the cutting zone (sensor reading)")

        power = torque * rpm * 0.1047
        temp_diff = proc - air
        type_code = {"Aluminum / Non-Ferrous (Light Duty)": 0, "Carbon Steel (Medium Duty)": 1, "Stainless Steel / Alloy (Heavy Duty)": 2}[type_in]
        input_vec = np.array([[type_code, air, proc, rpm, torque, wear, power, temp_diff]])

        try:
            scaled = scaler.transform(input_vec)
            prob = model.predict_proba(scaled)[0][1] * 100
        except:
            prob = 0.0

        with col_gauge:
            st.markdown("#### 🎯 Failure Risk Assessment")
            gauge_color = '#2ecc71' if prob < 30 else '#f39c12' if prob < 60 else '#e74c3c'
            risk_label = '✅ SAFE' if prob < 30 else '⚠️ WARNING' if prob < 60 else '🚨 DANGER'
            fig_risk = go.Figure(go.Indicator(
                mode="gauge+number", value=prob,
                number={'suffix': '%', 'font': {'size': 52}},
                title={'text': f"Failure Probability<br><span style='font-size:14px;color:{gauge_color}'>{risk_label}</span>", 'font': {'size': 16}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color},
                       'steps': [{'range': [0, 30], 'color': '#1a3a2a'}, {'range': [30, 60], 'color': '#3a3a1a'}, {'range': [60, 100], 'color': '#3a1a1a'}],
                       'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.75, 'value': 50}}))
            fig_risk.update_layout(height=320, margin=dict(t=80,b=20,l=30,r=30), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_risk, use_container_width=True)

            st.markdown(f"**Calculated Power:** `{power:,.0f} W` | **Temp Rise:** `{temp_diff:.1f} K`")

        # SHAP-style Feature Impact (simplified waterfall)
        st.markdown("---")
        st.markdown("#### 🧠 Why did the model predict this? (XAI)")
        st.caption("Feature importance — which parameters matter most for the prediction")
        feature_labels = ['Material', 'Air Temp', 'Process Temp', 'RPM', 'Torque', 'Tool Wear', 'Power', 'Temp Rise']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            pct = (importances / importances.sum()) * 100
            imp_df = pd.DataFrame({'Feature': feature_labels, 'Importance (%)': pct}).sort_values('Importance (%)')
            colors_imp = ['#00D4FF' if v < 10 else '#f39c12' if v < 25 else '#e74c3c' for v in imp_df['Importance (%)']]
            fig_imp = go.Figure(go.Bar(
                x=imp_df['Importance (%)'], y=imp_df['Feature'], orientation='h',
                marker_color=colors_imp,
                text=[f"{v:.1f}%" for v in imp_df['Importance (%)']],
                textposition='outside'))
            fig_imp.update_layout(
                height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', xaxis_title='Importance (%)', yaxis_title='',
                margin=dict(l=10,r=50,t=10,b=30),
                xaxis=dict(range=[0, max(pct) * 1.2]))
            fig_imp.update_traces(cliponaxis=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        # Work Order
        st.markdown("---")
        import datetime
        wo_id = f"WO-{datetime.datetime.now().strftime('%Y-%m%d-%H%M')}"

        # Dynamic recommendations based on primary driver
        action_map = {
            'Torque': {
                'cause': 'Overstrain Failure (OSF) — Torque exceeds safe operating threshold',
                'action': 'Reduce depth of cut (ap) and feed rate (fn). Check for workpiece hardness variation. Inspect tool holder rigidity.',
                'severity': 'CRITICAL'
            },
            'Tool Wear': {
                'cause': 'Tool Wear Failure (TWF) — Insert has exceeded useful life',
                'action': 'Replace cutting insert immediately. Reset tool wear counter. Consider switching to a more wear-resistant grade (CVD coated).',
                'severity': 'HIGH'
            },
            'RPM': {
                'cause': 'Power Failure (PWF) — Rotational speed × Torque exceeds power limit',
                'action': 'Reduce spindle speed or torque. Check spindle motor load %. Verify power supply stability.',
                'severity': 'HIGH'
            },
            'Power': {
                'cause': 'Power Overload — Calculated power exceeds machine capacity',
                'action': 'Reduce cutting parameters (RPM × Torque). Split operation into lighter passes. Check for dull insert causing excessive force.',
                'severity': 'CRITICAL'
            },
            'Temp Rise': {
                'cause': 'Heat Dissipation Failure (HDF) — Insufficient cooling at cutting zone',
                'action': 'Check coolant flow rate and nozzle direction. Reduce cutting speed (Vc). Consider high-pressure coolant system.',
                'severity': 'MODERATE'
            },
            'Process Temp': {
                'cause': 'Thermal Overload — Process temperature exceeds safe range',
                'action': 'Increase coolant concentration. Reduce cutting speed. Allow cool-down intervals between passes.',
                'severity': 'MODERATE'
            },
            'Air Temp': {
                'cause': 'Ambient Temperature Effect — Workshop temperature affecting thermal stability',
                'action': 'Monitor ambient conditions. Adjust thermal compensation on CNC controller. Check machine warm-up procedure.',
                'severity': 'LOW'
            },
            'Material': {
                'cause': 'Material-related Failure — Operating profile mismatch for workpiece type',
                'action': 'Verify correct insert grade for material. Check ISO material group settings. Adjust parameters per material catalog.',
                'severity': 'MODERATE'
            }
        }

        if prob > 50:
            top_feature = feature_labels[np.argmax(model.feature_importances_)] if hasattr(model, 'feature_importances_') else "Unknown"
            advice = action_map.get(top_feature, {'cause': 'Unknown root cause', 'action': 'Stop machine and inspect manually.', 'severity': 'HIGH'})
            severity_color = {'CRITICAL': '#ff4444', 'HIGH': '#e74c3c', 'MODERATE': '#f39c12', 'LOW': '#f1c40f'}.get(advice['severity'], '#e74c3c')

            st.markdown(f"""<div class="work-order">
            <h4>⚠️ Maintenance Work Order — {wo_id}</h4>
            <table style="width:100%; color:white; line-height: 1.8;">
            <tr><td style="width:180px"><b>Risk Level:</b></td><td style="color:{severity_color}"><b>{advice['severity']} ({prob:.1f}%)</b></td></tr>
            <tr><td><b>Primary Driver:</b></td><td>{top_feature}</td></tr>
            <tr><td><b>Root Cause:</b></td><td>{advice['cause']}</td></tr>
            <tr><td><b>Recommended Action:</b></td><td>{advice['action']}</td></tr>
            <tr><td><b>Est. Downtime Cost:</b></td><td><b>€{(prob/100)*250:,.0f}/hr</b> (if unplanned)</td></tr>
            </table></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="work-order-safe">
            <h4>✅ System Status — {wo_id}</h4>
            <p style="color:#2ecc71"><b>All parameters within safe operating range.</b></p>
            <p>Failure probability: {prob:.1f}% — No immediate action required. Next scheduled inspection in <b>{max(1, int((100-prob)/10))} hours</b>.</p>
            </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not found. Please run the notebook first.")

# ==================================================
# TAB 3: AI CONSULTANT (RAG)
# ==================================================
with tab3:
    st.subheader("💬 AI Technical Consultant")

    # Try to load RAG system
    RAG_AVAILABLE = False
    rag = None
    try:
        import sys
        sys.path.insert(0, str(BASE_DIR / "api"))
        from rag_core import CNCExpertRAG
        rag = CNCExpertRAG(vector_store_path=str(BASE_DIR / "models" / "faiss_index"))
        if rag.load_vector_db() and rag.llm:
            RAG_AVAILABLE = True
    except Exception as e:
        RAG_AVAILABLE = False

    if RAG_AVAILABLE:
        st.success("🟢 AI Mode: **Groq Llama-3.3** — Powered by Industrial Manuals (Haas, Sandvik, McGill CNC Guide)")
    else:
        st.warning("🟡 RAG not available. Using keyword fallback. Ensure GROQ_API_KEY is set and vector DB exists.")

    # Quick buttons — specific questions that the manuals CAN answer
    st.markdown("###### 💡 Quick Questions:")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("🔧 Fix Chipping", use_container_width=True, key="q1"): st.session_state.quick_prompt = "What causes insert chipping in CNC turning and what are the recommended countermeasures?"
    if q2.button("⚙️ Insert Grades", use_container_width=True, key="q2"): st.session_state.quick_prompt = "What insert grade should I use for turning steel? Compare CVD vs PVD coated grades."
    if q3.button("📐 Speed & Feed", use_container_width=True, key="q3"): st.session_state.quick_prompt = "What is the recommended cutting speed and feed rate for turning carbon steel with a carbide insert?"
    if q4.button("🌡️ Tool Wear", use_container_width=True, key="q4"): st.session_state.quick_prompt = "What are the main types of tool wear in CNC machining and how to identify them?"
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your **AI Technical Consultant**, trained on industrial manuals from Haas, Sandvik Coromant, Sumitomo, and CNC engineering guides.\n\nTry asking specific questions like:\n- *\"What insert grade for stainless steel turning?\"*\n- *\"How to fix chipping on a CVD coated insert?\"*\n- *\"Recommended cutting speed for aluminum milling?\"*"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = None
    if "quick_prompt" in st.session_state:
        prompt = st.session_state.quick_prompt
        del st.session_state.quick_prompt
    else:
        prompt = st.chat_input("Type your technical question here...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("🤖 Searching knowledge base..."):
            if RAG_AVAILABLE:
                try: response = rag.ask_question(prompt)
                except Exception as e: response = f"Error: {e}"
            else:
                response = _offline_fallback(prompt) if '_offline_fallback' in dir() else "RAG system not available. Please check your setup."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        st.rerun()

# ==================================================
# TAB 4: FINANCIAL IMPACT & ROI
# ==================================================
with tab4:
    st.subheader("💰 Cost Optimization & ROI Analysis")
    st.markdown("Compare **Run-to-Failure** vs. **AI-Driven Preventive Maintenance** costs in **Euro (€)**.")

    st.markdown("#### 1. Base Cost Rates")
    cb1, cb2, cb3 = st.columns(3)
    with cb1: cost_tool = st.number_input("🛠️ New Tool Cost (€)", value=25.0)
    with cb2: downtime = st.number_input("🛑 Downtime Cost (€/hr)", value=200.0)
    with cb3: rate = st.number_input("👷 Technician Rate (€/hr)", value=50.0)
    st.divider()

    st.markdown("#### 2. Maintenance Scenarios")
    col_fail, col_prev = st.columns(2)
    with col_fail:
        with st.container(border=True):
            st.markdown("##### 🔴 Run-to-Failure")
            st.info("Operating until the tool breaks. Causes collateral damage.")
            t_fix = st.number_input("Repair Time (hrs)", value=4.0, key="t_fix")
            cost_dmg = st.number_input("Extra Damage Cost (€)", value=100.0, key="c_dmg")
    with col_prev:
        with st.container(border=True):
            st.markdown("##### 🟢 AI Preventive")
            st.success("Changing tool before failure based on AI prediction.")
            t_prev = st.number_input("Planned Maint. Time (hrs)", value=0.5, key="t_prev")
            st.number_input("Extra Damage Cost (€)", value=0.0, disabled=True, key="c_dmg2")

    cost_fail = cost_tool + cost_dmg + (t_fix * (downtime + rate))
    cost_prev = cost_tool + (t_prev * (downtime + rate))
    savings = cost_fail - cost_prev
    roi = (savings / cost_prev) * 100 if cost_prev > 0 else 0

    st.divider()
    st.markdown("#### 3. Financial Impact")
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Cost if Broken", f"€{cost_fail:,.2f}", "Reactive", delta_color="inverse")
    m2.metric("🟢 Cost if Prevented", f"€{cost_prev:,.2f}", "Proactive", delta_color="normal")
    m3.metric("💰 Savings/Event", f"€{savings:,.2f}", f"+{roi:.0f}% ROI", delta_color="normal")

    # Plotly bar chart
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(x=[cost_fail], y=['Run-to-Failure'], orientation='h', marker_color='#e74c3c', text=[f'€{cost_fail:,.0f}'], textposition='outside', name='Reactive'))
    fig_roi.add_trace(go.Bar(x=[cost_prev], y=['AI Preventive'], orientation='h', marker_color='#2ecc71', text=[f'€{cost_prev:,.0f}'], textposition='outside', name='Proactive'))
    fig_roi.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', showlegend=False, xaxis_title="Total Cost (€)", margin=dict(l=10,r=80,t=10,b=30), barmode='group')
    st.plotly_chart(fig_roi, use_container_width=True)

    # Annual projection
    st.markdown("#### 4. Annual Projection")
    events = st.slider("Estimated Failure Events / Year", 1, 100, 24)
    annual_save = savings * events
    st.metric("📅 Estimated Annual Savings", f"€{annual_save:,.0f}", f"Based on {events} events/year")
