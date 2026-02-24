import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 0. PATH CONFIGURATION ---
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "ai4i2020.csv"
MODEL_FILE = BASE_DIR / "predictive_maintenance_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI-Driven Maintenance", page_icon="🏭", layout="wide")

# CSS Styling (ปรับ Sidebar ให้ดู Premium)
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #151922; border-right: 1px solid #262730; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e6e6e6; font-weight: 600; font-family: 'Inter', sans-serif; }
    .stButton>button { width: 100%; border-radius: 6px; border: 1px solid #30333d; background-color: #262730; color: white; transition: all 0.3s; }
    .stButton>button:hover { border-color: #0068c9; color: #0068c9; }
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 8px; border-left: 4px solid #0068c9; }
    .status-box { padding: 10px; border-radius: 5px; background-color: #1e2130; border: 1px solid #30333d; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🏭 AI-Driven Maintenance & Decision Support System")
st.markdown("**AI-Driven Failure Prediction, Solutions & Cost Analysis**")

# --- 2. FUNCTIONS  ---
def download_uci_dataset():
    if DATA_FILE.exists():
        return True, "✅ Local dataset 'ai4i2020.csv' found and loaded!"
    else:
        return False, "❌ 'ai4i2020.csv' missing. Please ensure it's in the same folder as app.py"

@st.cache_data
def get_dataset():
    if DATA_FILE.exists(): return pd.read_csv(DATA_FILE)
    return None

df = get_dataset()

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_artifacts()

def get_full_name(code):
    mapping = {
        'TWF': 'Tool Wear Failure (TWF)', 
        'HDF': 'Heat Dissipation Failure (HDF)', 
        'PWF': 'Power Failure (PWF)', 
        'OSF': 'Overstrain Failure (OSF)', 
        'RNF': 'Random Failure (RNF)'
    }
    return mapping.get(code, code)

# ==========================================
# 🧠 SMART OFFLINE SEARCH (60+ Keywords)
# ==========================================
def smart_offline_search(query):
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

def dark_plot(fig, ax):
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.tick_params(colors='#e0e0e0', which='both')
    ax.xaxis.label.set_color('#e0e0e0')
    ax.yaxis.label.set_color('#e0e0e0')
    ax.title.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#404040')
    return fig

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🎛️ System Console")
    st.caption("v2.5.0 | Enterprise Edition")
    st.markdown("---")
    
    st.markdown("### 📡 Data Source")
    if df is not None:
        st.markdown(f"""
        <div class="status-box">
            <span style='color:#2ecc71; font-size: 14px;'>●</span> 
            <span style='font-weight:bold; margin-left: 5px;'>System Online</span><br>
            <span style='font-size:12px; color:gray; margin-left: 20px;'>{len(df):,} records active</span>
        </div>
        """, unsafe_allow_html=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export CSV Data", csv, "ai4i2020.csv", "text/csv", use_container_width=True)
    else:
        st.markdown(f"""
        <div class="status-box" style="border-color: #e74c3c;">
            <span style='color:#e74c3c; font-size: 14px;'>●</span> 
            <span style='font-weight:bold; margin-left: 5px;'>No Data</span><br>
            <span style='font-size:12px; color:gray; margin-left: 20px;'>Connection Required</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("☁️ Connect & Download", use_container_width=True): 
            success, msg = download_uci_dataset()
            if success: st.rerun()
            else: st.error(msg)

    st.markdown("---")
    st.markdown("### 📚 Documentation")
    st.caption("Technical references and manuals.")
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1: st.markdown("📄")
    with col_d2: 
        st.link_button("Technical Guidance (PDF)", "https://www.sumitool.com/en/downloads/cutting-tools/general-catalog/assets/pdf/n2.pdf", use_container_width=True)

    st.markdown("### ⚙️ System Actions")
    if st.button("🔄 Reset Dashboard", use_container_width=True): 
        st.session_state.clear()
        st.rerun()

# --- 4. MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Consultant", "📊 Failure Pattern Analysis", "🔮 Process Simulation", "💰 Financial Impact & ROI"])

# ==================================================
# TAB 1: CHATBOT
# ==================================================
with tab1:
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<h3 style='text-align: center;'>💬 AI Technical Consultant</h3>", unsafe_allow_html=True)
        st.markdown("###### 💡 Quick Troubleshooting:")
        q1, q2, q3 = st.columns(3)
        if q1.button("Fix Chipping?", use_container_width=True): st.session_state.quick_prompt = "Fix Chipping?"
        if q2.button("Explain Grades", use_container_width=True): st.session_state.quick_prompt = "Explain Grades"
        if q3.button("Rec. Speed?", use_container_width=True): st.session_state.quick_prompt = "Rec. Speed?"
        st.markdown("---")

    if "messages" not in st.session_state: 
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Technical Consultant. Ask me about **Chipping, Wear, Chatter, Speed, or Insert Grades**."}]
    
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
        
        response = smart_offline_search(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        st.rerun()

# ==================================================
# TAB 2: DATA DIAGNOSIS (THE MISSING BOXPLOT!)
# ==================================================
with tab2:
    if df is not None:
        st.subheader("📊 Dataset Diagnosis")
        c1, c2, c3 = st.columns(3)
        total = len(df); fails = df['Machine failure'].sum()
        c1.metric("Total Operations", f"{total:,}", border=True)
        c2.metric("Failures Detected", f"{fails:,}", delta=f"{(fails/total)*100:.1f}% Rate", delta_color="inverse", border=True)
        
        valid_cols = [c for c in ['TWF', 'HDF', 'PWF', 'OSF', 'RNF'] if c in df.columns]
        if valid_cols:
            top_code = df[valid_cols].sum().idxmax()
            c3.metric("Top Failure Mode", get_full_name(top_code), f"{df[top_code].sum()} Cases", delta_color="inverse", border=True)
            
        st.divider()
        c_left, c_right = st.columns([1, 2])
        with c_left:
            sel_fail_code = st.selectbox("Select Failure Type:", valid_cols, format_func=get_full_name)
            st.info(f"Comparing **{get_full_name(sel_fail_code)}** vs Healthy Machines.")
        with c_right:
            target_map = {'TWF': 'Tool wear [min]', 'HDF': 'Rotational speed [rpm]', 'PWF': 'Torque [Nm]', 'OSF': 'Torque [Nm]'}
            target = target_map.get(sel_fail_code, 'Process temperature [K]')
            
            plot_data = pd.DataFrame({
                'Status': ['Normal'] * len(df[df['Machine failure']==0]) + ['Failed'] * len(df[df[sel_fail_code]==1]),
                target: pd.concat([df[df['Machine failure']==0][target], df[df[sel_fail_code]==1][target]])
            })
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.boxplot(x='Status', y=target, data=plot_data, palette={'Normal': '#2ecc71', 'Failed': '#e74c3c'}, ax=ax)
            ax.set_title(f"Parameter Analysis: {target}")
            dark_plot(fig, ax); st.pyplot(fig)
    else:
        st.warning("⚠️ No dataset loaded. Please go to the Sidebar.")

# ==================================================
# TAB 3: PREDICTION (SIMULATION)
# ==================================================
with tab3:
    st.subheader("🔮 Cutting Condition Simulation")
    
    if model is not None:
        if 'rpm_slider' not in st.session_state: st.session_state.rpm_slider = 1550
        if 'torque_slider' not in st.session_state: st.session_state.torque_slider = 40.0
        if 'wear_slider' not in st.session_state: st.session_state.wear_slider = 50
        if 'air_slider' not in st.session_state: st.session_state.air_slider = 298.0
        if 'proc_slider' not in st.session_state: st.session_state.proc_slider = 309.0

        st.markdown("##### ⚡ Quick Scenarios (Click to Load):")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("✅ Normal Run"):
            st.session_state.rpm_slider = 1500; st.session_state.torque_slider = 35.0; st.session_state.wear_slider = 20
            st.rerun()
        if b2.button("⚠️ High Wear"):
            st.session_state.rpm_slider = 1500; st.session_state.torque_slider = 55.0; st.session_state.wear_slider = 290     
            st.rerun()
        if b3.button("🔥 Overstrain"):
            st.session_state.rpm_slider = 1400; st.session_state.torque_slider = 65.0; st.session_state.wear_slider = 200
            st.rerun()
        if b4.button("⚡ Power Fail"):
            st.session_state.rpm_slider = 2800; st.session_state.torque_slider = 60.0; st.session_state.wear_slider = 50
            st.rerun()
        st.divider()

        with st.container(border=True):
            st.markdown("#### 🎛️ Operation Parameters")
            cp1, cp2, cp3 = st.columns(3)
            with cp1: rpm = st.slider("Speed [rpm]", 1000, 3000, key='rpm_slider')
            with cp2: torque = st.slider("Torque [Nm]", 10.0, 90.0, key='torque_slider')
            with cp3: wear = st.slider("Tool Wear [min]", 0, 300, key='wear_slider')
            
            ce1, ce2, ce3 = st.columns(3)
            with ce1: type_in = st.selectbox("Material Grade", ["L (Low)", "M (Medium)", "H (High)"])
            with ce2: air = st.number_input("Air Temp [K]", 290.0, 310.0, key='air_slider')
            with ce3: proc = st.number_input("Process Temp [K]", 300.0, 320.0, key='proc_slider')

        power = torque * rpm * 0.1047
        temp_diff = proc - air
        
        st.info(f"""
        - **Calculated Power:** {power:,.2f} W (Limit: 3500-9000 W)
        - **Temp Rise:** {temp_diff:.2f} K
        """)
        
        type_code = {"L (Low)": 0, "M (Medium)": 1, "H (High)": 2}[type_in]
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            input_vec = np.array([[type_code, air, proc, rpm, torque, wear, power, temp_diff]])
            try:
                scaled = scaler.transform(input_vec)
                prob = model.predict_proba(scaled)[0][1] * 100
                st.divider()
                if prob > 50: st.error(f"🚨 **FAILURE PREDICTED ({prob:.2f}%)**")
                else: st.success(f"✅ **NORMAL ({prob:.2f}%)**")
            except Exception as e: st.error(f"⚠️ Model needs to be loaded or Error: {e}")
    else:
        st.warning("⚠️ Machine Learning Model (predictive_maintenance_model.pkl) not found. Please upload it to the directory.")

# ==================================================
# TAB 4: ROI Analysis (Redesigned for Clarity)
# ==================================================
with tab4:
    st.subheader("💰 Cost Optimization & ROI Analysis")
    st.markdown("Evaluate the financial impact by comparing **Run-to-Failure** vs. **AI-Driven Preventive Maintenance**.")
    
    # 1. Base Cost Rates (ต้นทุนพื้นฐานที่ต้องใช้เหมือนกัน)
    st.markdown("#### 1. Base Cost Rates")
    col_base1, col_base2, col_base3 = st.columns(3)
    with col_base1:
        cost_tool = st.number_input("🛠️ New Tool/Part Cost (€)", value=25.0, help="Cost of a new replacement tool or insert")
    with col_base2:
        downtime = st.number_input("🛑 Downtime Cost (€/hr)", value=200.0, help="Estimated lost revenue per hour of machine downtime")
    with col_base3:
        rate = st.number_input("👷 Technician Rate (€/hr)", value=50.0, help="Hourly labor cost for the maintenance technician")

    st.divider()
    
    # 2. Scenarios (แยก 2 กรณีชัดเจน ซ้าย-ขวา)
    st.markdown("#### 2. Maintenance Scenarios")
    col_fail, col_prev = st.columns(2)
    
    # กล่องฝั่งซ้าย: กรณีปล่อยให้พังคาเครื่อง
    with col_fail:
        with st.container(border=True):
            st.markdown("##### 🔴 Case 1: Run-to-Failure")
            st.info("Operating until the tool breaks. Often causes collateral damage and requires longer repair time.")
            t_fix = st.number_input("⏱️ Breakdown Repair Time (hrs)", value=4.0, help="Time taken to fix the machine after a sudden crash")
            cost_dmg = st.number_input("💥 Extra Damage Cost (€)", value=100.0, help="Collateral damage (e.g., scrapped part, damaged holder)")

    # กล่องฝั่งขวา: กรณีให้ AI เตือนแล้วเปลี่ยนก่อน
    with col_prev:
        with st.container(border=True):
            st.markdown("##### 🟢 Case 2: Preventive (AI Alert)")
            st.success("Changing the tool just before failure based on AI prediction. Fast and safe.")
            t_prev = st.number_input("⏱️ Planned Maint. Time (hrs)", value=0.5, help="Time taken for a scheduled tool change")
            # ล็อคช่อง Extra Damage ไว้ที่ 0 เพราะเปลี่ยนก่อนพัง งานเลยไม่เสีย
            st.number_input("💥 Extra Damage Cost (€)", value=0.0, disabled=True, help="No collateral damage since the tool is changed before breaking")

    # 3. คำนวณต้นทุน (Calculate Costs)
    cost_fail = cost_tool + cost_dmg + (t_fix * (downtime + rate))
    cost_prev = cost_tool + (t_prev * (downtime + rate))
    savings = cost_fail - cost_prev
    roi = (savings / cost_prev) * 100 if cost_prev > 0 else 0
    
    st.divider()
    
    # 4. แสดงผลลัพธ์ (Show Results)
    st.markdown("#### 3. Financial Impact Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Cost if Broken", f"€{cost_fail:,.2f}", "Reactive Strategy", delta_color="inverse")
    m2.metric("🟢 Cost if Prevented", f"€{cost_prev:,.2f}", "Proactive Strategy", delta_color="normal")
    m3.metric("💰 Savings per Event", f"€{savings:,.2f}", f"+{roi:.0f}% ROI", delta_color="normal")
    
    # พล็อตกราฟแท่ง (Plot Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    bars = ax.barh(['Run-to-Failure', 'AI Preventive'], [cost_fail, cost_prev], color=['#e74c3c', '#2ecc71'])
    ax.set_xlabel('Total Estimated Cost (€)')
    # ใส่ตัวเลขกำกับไว้ท้ายกราฟแท่ง
    for bar in bars:
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, f'€{bar.get_width():,.0f}', va='center', fontweight='bold', color='white')
    dark_plot(fig, ax) 

    st.pyplot(fig, use_container_width=False)