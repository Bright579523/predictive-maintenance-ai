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

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="AI-Driven Maintenance", page_icon="🏭", layout="wide")

# CSS Styling (ปรับ Sidebar ให้ดู Premium)
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0e1117; color: white; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { 
        background-color: #151922; 
        border-right: 1px solid #262730;
    }
    
    /* Headers in Sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #e6e6e6;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button { 
        width: 100%; 
        border-radius: 6px; 
        border: 1px solid #30333d;
        background-color: #262730;
        color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #0068c9;
        color: #0068c9;
    }
    
    /* Metrics & Cards */
    .metric-card { background-color: #1e2130; padding: 15px; border-radius: 8px; border-left: 4px solid #0068c9; }
    
    /* Status Indicator Style */
    .status-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #1e2130;
        border: 1px solid #30333d;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 AI-Driven Maintenance & Decision Support System")
st.markdown("**AI-Driven Failure Prediction, Solutions & Cost Analysis**")

# --- 2. FUNCTIONS  ---

def download_uci_dataset():
    url = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall()
            return True, "✅ Download Complete!"
        else: return False, f"❌ Connection Error: {r.status_code}"
    except Exception as e: return False, f"❌ Error: {e}"

@st.cache_data
def get_dataset():
    if os.path.exists('ai4i2020.csv'): return pd.read_csv('ai4i2020.csv')
    return None

df = get_dataset()

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('predictive_maintenance_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except: return None, None

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

def smart_offline_search(query):
    query = query.lower()
    kb = {
        "chipping": {"keywords": ["chipping", "chip", "break"], "response": "💡 **Chipping:** Occurs due to vibration or unstable clamping.\n**Solution:** Use a Tougher Grade insert or reduce Feed Rate."},
        "wear": {"keywords": ["wear", "flank", "life"], "response": "💡 **Flank Wear:** Occurs due to high cutting speed.\n**Solution:** Reduce Cutting Speed (Vc) or increase Coolant."},
        "speed": {"keywords": ["speed", "vc", "rpm"], "response": "⚙️ **Speed (Vc):** Affects cutting temperature.\n- Steel: 150-250 m/min\n- Aluminum: 300+ m/min"},
        "grade": {"keywords": ["grade", "material"], "response": "📘 **Grade Selection:**\n- CVD Coated: Heat resistant (Turning)\n- PVD Coated: Toughness (Milling)"}
    }
    for topic, data in kb.items():
        if any(k in query for k in data["keywords"]): return data["response"]
    return "❓ Topic not found. Try keywords: **Chipping, Wear, Speed, Grade**"

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
    # 3.1 Header
    st.markdown("## 🎛️ System Console")
    st.caption("v2.4.0 | Enterprise Edition")
    st.markdown("---")
    
    # 3.2 Data Status 
    st.markdown("### 📡 Data Source")
    if df is not None:
        # Custom Status Box
        st.markdown(f"""
        <div class="status-box">
            <span style='color:#2ecc71; font-size: 14px;'>●</span> 
            <span style='font-weight:bold; margin-left: 5px;'>System Online</span><br>
            <span style='font-size:12px; color:gray; margin-left: 20px;'>{len(df):,} records active</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Download Button
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

    # 3.3 Documentation
    st.markdown("### 📚 Documentation")
    st.caption("Technical references and manuals.")
    col_d1, col_d2 = st.columns([1, 4])
    with col_d1: st.markdown("📄")
    with col_d2: 
        st.link_button("Technical Guidance (PDF)", "https://www.sumitool.com/en/downloads/cutting-tools/general-catalog/assets/pdf/n2.pdf", use_container_width=True)

    st.markdown("---")

    # 3.4 System Actions
    st.markdown("### ⚙️ System Actions")
    if st.button("🔄 Reset Dashboard", use_container_width=True): 
        st.cache_data.clear()
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
        st.markdown("###### 💡 Quick Ask:")
        q1, q2, q3 = st.columns(3)
        if q1.button("Fix Chipping?", use_container_width=True): st.session_state.prompt_trigger = "Fix Chipping?"
        if q2.button("Explain Grades", use_container_width=True): st.session_state.prompt_trigger = "Explain Grades"
        if q3.button("Rec. Speed?", use_container_width=True): st.session_state.prompt_trigger = "Rec. Speed?"
        st.markdown("---")

    if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about **Chipping, Wear, Speed, or Grades**."}]
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
    
    prompt = st.chat_input("Type your question here...")
    if "prompt_trigger" in st.session_state and st.session_state.prompt_trigger:
        prompt = st.session_state.prompt_trigger; del st.session_state.prompt_trigger

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response = smart_offline_search(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# ==================================================
# TAB 2: DATA DIAGNOSIS
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
            st.session_state.rpm_slider = 1500; st.session_state.torque_slider = 45.0; st.session_state.wear_slider = 230
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
            with cp1: rpm = st.slider("Speed [rpm]", 1000, 3000, key='rpm_slider', help="Rotation Speed (RPM)")
            with cp2: torque = st.slider("Torque [Nm]", 10.0, 90.0, key='torque_slider', help="Torque (Nm)")
            with cp3: wear = st.slider("Tool Wear [min]", 0, 300, key='wear_slider', help="Accumulated Tool Wear (min)")
            
            ce1, ce2, ce3 = st.columns(3)
            with ce1: type_in = st.selectbox("Material Grade", ["L (Low)", "M (Medium)", "H (High)"], help="L=Light, M=Medium, H=Heavy Duty")
            with ce2: air = st.number_input("Air Temp [K]", 290.0, 310.0, key='air_slider', help="Ambient Temperature (K)")
            with ce3: proc = st.number_input("Process Temp [K]", 300.0, 320.0, key='proc_slider', help="Process Temperature (K)")

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
            except Exception as e: st.error(f"Error: {e}")

# ==================================================
# TAB 4: ROI Analysis
# ==================================================
with tab4:
    st.subheader("💰 Cost Optimization & Savings")
    st.markdown("Compare the cost of 'Run-to-Failure' (Reactive) vs. 'Preventive Maintenance' (Proactive).")
    
    st.markdown("#### 1. Cost Parameters")
    c_inp1, c_inp2, c_inp3 = st.columns(3)
    
    with c_inp1:
        st.markdown("**🛠️ Tool & Parts**")
        cost_tool = st.number_input("New Tool Cost (€)", value=25, help="Cost of a new replacement tool")
        cost_dmg = st.number_input("Extra Damage Risk (€)", value=100, help="Potential cost of damage to the workpiece/machine if it breaks")
        
    with c_inp2:
        st.markdown("**⏱️ Operational Time**")
        downtime = st.number_input("Downtime Cost (€/hr)", value=200, help="Cost per hour when the machine is not running")
        t_fix = st.number_input("Repair Time: Breakdown (hrs)", value=4.0, help="Time required to fix the machine after failure")
        t_prev = st.number_input("Maint. Time: Preventive (hrs)", value=0.5, help="Time required for scheduled maintenance")
        
    with c_inp3:
        st.markdown("**👷 Technician**")
        rate = st.number_input("Technician Rate (€/hr)", value=50, help="Hourly labor cost for the technician")

    st.divider()
    
    cost_fail = cost_tool + cost_dmg + (t_fix * (downtime + rate))
    cost_prev = cost_tool + (t_prev * (downtime + rate))
    savings = cost_fail - cost_prev
    roi = (savings / cost_prev) * 100
    
    st.markdown("#### 2. Cost Comparison Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Cost if Broken (Run-to-Fail)", f"€{cost_fail:,.2f}", "Worst Case")
    m2.metric("🟢 Cost if Prevented", f"€{cost_prev:,.2f}", "Best Case")
    m3.metric("💰 Savings per Event", f"€{savings:,.2f}", f"+{roi:.0f}% ROI", delta_color="normal")
    
    fig, ax = plt.subplots(figsize=(5, 1.5))
    bars = ax.barh(['Run-to-Failure', 'Preventive'], [cost_fail, cost_prev], color=['#e74c3c', '#2ecc71'])
    ax.set_xlabel('Total Cost (€)')
    for bar in bars:
        ax.text(bar.get_width()+10, bar.get_y()+bar.get_height()/2, f'€{bar.get_width():,.0f}', va='center', fontweight='bold', color='white')
    dark_plot(fig, ax) 
    st.pyplot(fig, use_container_width=False)