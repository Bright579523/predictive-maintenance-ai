# 🏭 Smart CNC Predictive Maintenance System

> **AI-Driven Failure Prediction | Explainable AI (SHAP) | RAG Consultant | ROI Analysis**

An enterprise-grade predictive maintenance platform for CNC machining operations. Built with XGBoost, LangChain RAG, and Streamlit — designed to reduce unplanned downtime by up to 35% and provide real-time, explainable failure risk assessment.

---

## 🎯 Project Overview

This project tackles the **#1 cost driver in manufacturing** — unplanned machine downtime (averaging **€250/hr** per CNC machine). Instead of reactive "fix when broken" maintenance, this system uses machine learning to **predict failures before they happen** and provides AI-powered troubleshooting via a Retrieval-Augmented Generation (RAG) chatbot trained on industrial manuals.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| 🔮 **Failure Prediction** | XGBoost model (98.15% accuracy) predicts machine failure from sensor data |
| 🧠 **Explainable AI (XAI)** | Feature importance analysis shows *why* a failure is predicted |
| 💬 **RAG AI Consultant** | LLM chatbot (Llama-3.3-70B) answers CNC questions using indexed technical manuals |
| 💰 **Financial Impact** | ROI calculator comparing reactive vs. proactive maintenance costs in € |
| 📊 **Interactive Dashboard** | 4-tab Plotly + Streamlit dashboard with real-time simulation |

---

## 📸 Dashboard Preview

### Tab 1 — Overview & Health Monitor
- 4 KPI cards (Total Ops, Failure Rate, Top Risk, Model Accuracy)
- 3 health gauges (Temperature, Torque, Tool Wear)
- 3D scatter plot (RPM × Torque × Wear) color-coded by failure status
- Failure type distribution bar chart

### Tab 2 — Predictive Simulator
- Real-time failure probability gauge with risk level indicator (SAFE / WARNING / DANGER)
- Workpiece material selector (Aluminum, Carbon Steel, Stainless Steel)
- Quick scenario buttons (Normal Run, High Wear, Overstrain, Power Fail)
- XAI feature importance chart (% contribution per parameter)
- Dynamic maintenance Work Order with root cause analysis

### Tab 3 — AI Technical Consultant (RAG)
- Groq Llama-3.3-70B powered chatbot
- Trained on 1,000+ document chunks from Sandvik, Sumitomo, and CNC engineering manuals
- Quick question buttons for common troubleshooting topics
- FAISS vector database for semantic search

### Tab 4 — Financial Impact Analysis
- Reactive vs. Proactive cost comparison
- Annual savings projection slider
- ROI calculation with cost breakdown

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | XGBoost (scikit-learn pipeline) |
| **Feature Engineering** | Power (W) = Torque × RPM × 0.1047, Temp_Diff (K) |
| **Explainability** | Feature Importances (built-in XGBoost) |
| **RAG Pipeline** | LangChain + FAISS + HuggingFace Embeddings |
| **LLM** | Groq API → Llama-3.3-70B-Versatile |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Dashboard** | Streamlit + Plotly |
| **Dataset** | UCI AI4I 2020 Predictive Maintenance (10,000 records) |

---

## 📁 Project Structure

```
Smart_Dashboard/
├── app.py                  # Main Streamlit dashboard (4 tabs)
├── ingest_all.py           # Script to build FAISS vector DB from docs/
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── api/
│   └── rag_core.py         # RAG pipeline (LangChain + FAISS + Groq)
│
├── data/
│   └── ai4i2020.csv        # UCI AI4I 2020 dataset (10,000 records)
│
├── docs/
│   ├── information.txt     # Machine specification reference
│   ├── Sandvik.pdf         # Sandvik Coromant turning guide (not in repo)
│   ├── sumitomo_manual.pdf # Sumitomo insert catalog (not in repo)
│   └── Fundamentals_of_CNC_Machining.pdf  # McGill CNC guide (not in repo)
│
├── models/
│   ├── predictive_maintenance_model.pkl  # Trained XGBoost model
│   ├── scaler.pkl                        # StandardScaler for features
│   └── faiss_index/                      # FAISS vector DB (regenerate via ingest_all.py)
│
├── notebooks/
│   └── setup_notebook.py   # ML training & evaluation pipeline
│
└── Report/
    └── Note_project.txt    # Development log & project notes
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Bright579523/predictive-maintenance-ai.git
cd predictive-maintenance-ai
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com)

### 3. (Optional) Build RAG Knowledge Base

Place your CNC manual PDFs in the `docs/` folder, then run:

```bash
python ingest_all.py
```

This will create the FAISS vector database in `models/faiss_index/`.

### 4. Launch Dashboard

```bash
streamlit run app.py
```

Open [Streamlit]((https://cnc-predictivewithai.streamlit.app/)) in your browser.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.15% |
| **Precision** | 97.8% |
| **Recall** | 96.2% |
| **F1-Score** | 97.0% |
| **AUC-ROC** | 0.994 |

### Feature Importance (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Power (W) | ~44% |
| 2 | RPM | ~15% |
| 3 | Torque (Nm) | ~12% |
| 4 | Tool Wear (min) | ~10% |
| 5 | Temp Diff (K) | ~8% |

---

## 🔄 What Changed (v2.0 Redesign)

This project was originally a simple Matplotlib + keyword-chatbot dashboard. It has been **completely redesigned** with the following improvements:

### Dashboard
- ❌ Static Matplotlib charts → ✅ **Interactive Plotly** (3D scatter, gauges, bar charts)
- ❌ Basic layout → ✅ **4-tab enterprise UI** with KPI cards, dark theme, gradient CSS
- ❌ Generic "Material Grade L/M/H" → ✅ **Real material names** (Aluminum, Carbon Steel, Stainless Steel)
- ❌ Static Work Orders → ✅ **Dynamic root cause analysis** per failure type

### AI & ML
- ❌ Keyword-based chatbot → ✅ **RAG pipeline** (LangChain + FAISS + Llama-3.3-70B)
- ❌ No explainability → ✅ **XAI feature importance** (% contribution chart)
- ❌ No knowledge base → ✅ **1,033 embedded chunks** from Sandvik, Sumitomo, and CNC manuals
- ❌ Old Llama-3 70B model → ✅ **Llama-3.3-70B-Versatile** (latest, most capable)

### Infrastructure
- ❌ Flat file structure → ✅ **Organized directories** (api/, data/, docs/, models/, notebooks/)
- ❌ No environment management → ✅ **`.env` + `.gitignore`** for secure API key handling
- ❌ Manual model loading → ✅ **Automated ingestion script** (`ingest_all.py`)

---

## 📚 Knowledge Sources (RAG)

The AI Consultant is trained on these industrial references:

| Source | Type | Content |
|--------|------|---------|
| **Sandvik Coromant** | Turning Guide | Insert grades, cutting conditions, troubleshooting |
| **Sumitomo Electric** | Insert Catalog | Grade recommendations, wear countermeasures |
| **Fundamentals of CNC Machining** | Textbook (McGill) | Speed/feed calculations, machine operation, safety |
| **AI4I 2020 Dataset Info** | Reference | Failure type definitions, sensor specifications |

---

## ⚠️ Requirements

- Python 3.10+
- Groq API key (free tier available)
- ~2GB RAM for FAISS embeddings

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 👤 Author

**Watcharapol (Bright)**  
Data Science & AI Engineering  
[GitHub](https://github.com/Bright579523)
