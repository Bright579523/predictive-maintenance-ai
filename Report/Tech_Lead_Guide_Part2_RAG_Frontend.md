# 🏭 Tech Lead Deep-Dive Guide — Part 2: RAG Pipeline & Frontend
> อ้างอิงจากโค้ดจริงใน `api/rag_core.py`, `ingest_all.py`, และ `app.py`

---

## Chapter 6: RAG Architecture — ภาพรวมก่อนเจาะโค้ด

```
User Question
     ↓
[Embedding Model] → แปลงคำถามเป็น Vector (384 มิติ)
     ↓
[FAISS Index] → ค้นหา 5 Chunks ที่ "ใกล้เคียง" ที่สุด (Cosine Similarity)
     ↓
[Prompt Builder] → รวม Context 5 ชิ้น + คำถาม + System Instruction
     ↓
[Groq API / Llama-3.3-70B] → สร้างคำตอบจาก Context
     ↓
User Answer
```

---

## Chapter 7: Data Ingestion Pipeline

### `ingest_all.py` — Script หลักในการสร้าง Knowledge Base

```python
# ingest_all.py บรรทัด 1-46
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "api"))  # เพิ่ม api/ ใน Python path
from rag_core import CNCExpertRAG

rag = CNCExpertRAG(vector_store_path="models/faiss_index")

docs_to_ingest = [
    "docs/information.txt",
    "docs/sumitomo_manual.pdf",
    "docs/Sandvik.pdf",
    "docs/Fundamentals_of_CNC_Machining.pdf"
]

# ลบ index เก่าก่อนเสมอ เพื่อ rebuild ใหม่จากศูนย์
if os.path.exists("models/faiss_index"):
    shutil.rmtree("models/faiss_index")

for doc in docs_to_ingest:
    rag.ingest_document(str(doc_path))
```

---

## Chapter 8: `rag_core.py` — เจาะลึกทุกบรรทัด

### 8.1 Initialization — เลือก Tools ที่ถูกต้อง

```python
# api/rag_core.py บรรทัด 17-44
class CNCExpertRAG:
    def __init__(self, vector_store_path="models/faiss_index"):
        # Embedding: ฟรี, รันในเครื่อง, ไม่ต้องจ่ายเงิน API
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # LLM: อ่าน Key แบบ Priority (เครื่อง → Cloud)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            try:                                    # ← Fallback สำหรับ Streamlit Cloud
                import streamlit as st
                if "GROQ_API_KEY" in st.secrets:
                    groq_api_key = st.secrets["GROQ_API_KEY"]
            except:
                pass
        
        self.llm = ChatGroq(
            temperature=0.1,                         # ← ต่ำ = ตอบตรงๆ ไม่สร้างสรรค์
            model_name="llama-3.3-70b-versatile",
            api_key=groq_api_key
        )
```

**ทำไม `temperature=0.1` ไม่ใช่ 0.7 หรือ 1.0?**
- `0.0` = ตอบเหมือนกันทุกครั้ง (Deterministic) — ดีสำหรับคำถามที่มีคำตอบเดียว
- `0.1` = ตอบตรงประเด็น แต่มีความยืดหยุ่นเล็กน้อยในการสรุปภาษา
- `0.7-1.0` = Creative มาก เหมาะกับงานเขียน แต่อาจ "มั่ว" ข้อมูลเทคนิคได้

### 8.2 Document Ingestion — จาก PDF สู่ Vector

```python
# rag_core.py บรรทัด 46-76
def ingest_document(self, file_path):
    # Step 1: Load
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)      # แปลง PDF → list of Document objects
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    
    # Step 2: Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # ← ตัดทุก 1000 ตัวอักษร
        chunk_overlap=200   # ← ซ้อนทับ 200 ตัวอักษร ป้องกันประโยคขาดตอน
    )
    docs = text_splitter.split_documents(documents)
    
    # Step 3: Embed + Store (Append ไม่ใช่ Overwrite!)
    if self.vector_store is not None:
        self.vector_store.add_documents(docs)
    else:
        if not self.load_vector_db():                        # ลอง Load ก่อน
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vector_store.add_documents(docs)            # Append ต่อท้าย
    
    self.vector_store.save_local(self.vector_store_path)
```

**ทำไม `chunk_overlap=200`?**

ลองนึกภาพว่าข้อความว่า:
> "...การใช้ Grade GC4325 เหมาะสำหรับการกลึง Carbon Steel ในสภาวะ Continuous Cut ส่วน Grade GC4315 เหมาะกับงานที่ต้องการทนความร้อนสูง..."

ถ้าตัดตรง 1000 ตัวอักษรพอดี ประโยคเรื่อง GC4315 อาจหายไปอยู่ใน Chunk ต่อไป การ Overlap 200 ตัวอักษรทำให้ทั้ง 2 Chunk มีข้อมูลส่วนนี้ร่วมกัน

**`RecursiveCharacterTextSplitter` ต่างจาก `CharacterTextSplitter` อย่างไร?**
- `CharacterTextSplitter`: ตัดตาม delimiter เดียว (เช่น `\n`)
- `RecursiveCharacterTextSplitter`: ลองตัดตาม `["\n\n", "\n", " ", ""]` ตามลำดับ → ตัดที่ย่อหน้าก่อน ถ้าไม่พอค่อยตัดที่บรรทัด ถ้าไม่พอค่อยตัดที่คำ → ได้ Chunk ที่สมบูรณ์ทางความหมายมากกว่า

### 8.3 Prompt Engineering — หัวใจของ RAG

```python
# rag_core.py บรรทัด 135-155
prompt_template = """You are an expert CNC Machining and Cutting Tool Consultant 
with 15+ years of experience.
You have access to context extracted from official technical manuals 
(Sandvik, Sumitomo, Haas, and CNC engineering guides).

Instructions:
1. Use the provided context as your primary source to answer the question.
2. If the context contains related information, synthesize a helpful answer 
   from it — even if it doesn't match the question exactly.
3. Use bullet points and bold text for clarity.
4. Include specific values (speeds, feeds, grades) when available in the context.
5. Only say you cannot answer if the context is completely unrelated to the question.

Context from manuals:
{context}

User's Question: {question}

Expert Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
```

**Prompt Engineering Techniques ที่ใช้:**

| Technique | โค้ดที่ใช้ | เหตุผล |
|-----------|-----------|--------|
| **Persona Injection** | "15+ years of experience" | ทำให้ LLM ตอบในโทนผู้เชี่ยวชาญ |
| **Source Grounding** | "Use provided context as primary source" | ป้องกัน Hallucination |
| **Format Control** | "Use bullet points and bold text" | บังคับให้อ่านง่าย |
| **Graceful Fallback** | "Only say cannot answer if completely unrelated" | ทำให้ตอบได้กว้างขึ้น |
| **Specificity Boost** | "Include specific values (speeds, feeds)" | ดึงตัวเลขจริงจากคู่มือ |

### 8.4 RAG Chain — เชื่อมทุกอย่างเข้าด้วยกัน

```python
# rag_core.py บรรทัด 157-167
qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",                                    # ← วิธี combine context
    retriever=self.vector_store.as_retriever(
        search_kwargs={"k": 5}                             # ← ดึง 5 chunks
    ),
    chain_type_kwargs={"prompt": PROMPT}
)

response = qa_chain.invoke(query)
return response['result']
```

**`chain_type="stuff"` คืออะไร?**

LangChain มีหลาย chain_type สำหรับการ combine documents:

| Chain Type | วิธีทำงาน | เหมาะกับ |
|-----------|----------|---------|
| **`stuff`** | ยัด Context ทั้งหมดเข้า Prompt ครั้งเดียว | Context ไม่มาก (k≤5) |
| `map_reduce` | แยกถาม LLM ทีละ Chunk แล้วรวมคำตอบ | Context เยอะมาก |
| `refine` | ถาม LLM ทีละ Chunk และปรับปรุงคำตอบทีละครั้ง | ต้องการคำตอบละเอียดมาก |

เราใช้ `"stuff"` เพราะดึงแค่ 5 Chunks (~5,000 ตัวอักษร) ซึ่งพอดีกับ Context Window ของ Llama-3.3

---

## Chapter 9: Streamlit Frontend — สถาปัตยกรรม UI

### 9.1 Path Configuration (app.py บรรทัด 13-17)

```python
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_FILE = BASE_DIR / "models" / "predictive_maintenance_model.pkl"
SCALER_FILE = BASE_DIR / "models" / "scaler.pkl"
```

**ทำไมต้องใช้ `Path(__file__).parent` ไม่ใช่แค่ `"./data/..."`?**
- `./` หมายถึง Current Working Directory ของ terminal ที่รัน
- `__file__` หมายถึง path ของ `app.py` เอง
- ถ้า Deploy บน Cloud หรือรันจาก directory อื่น `./` จะหาไฟล์ไม่เจอ แต่ `Path(__file__).parent` จะถูกเสมอ

### 9.2 Session State — หน่วยความจำของ Streamlit

```python
# app.py บรรทัด 97-99, 174-177
if st.button("🔄 Reset Dashboard"):
    st.session_state.clear()   # ← ล้างทุกอย่าง
    st.rerun()

# Quick Scenario buttons
if b2.button("⚠️ High Wear"):
    st.session_state.update(rpm_s=1500, torque_s=55.0, wear_s=290, ...)
    st.rerun()   # ← บังคับ rerun เพื่อให้ slider อัปเดต
```

**`st.session_state` ทำงานอย่างไร?**

Streamlit rerun script ทั้งหมดทุกครั้งที่มี interaction ดังนั้น ตัวแปรปกติจะถูก Reset ทุกครั้ง
`session_state` คือ dict พิเศษที่รอดการ rerun ได้ โดยทำงานเหมือน Cookie ใน Browser

```
ผู้ใช้กด "High Wear"
    → st.session_state['wear_s'] = 290
    → st.rerun()
    → Streamlit รัน app.py ใหม่ทั้งหมด
    → Slider อ่านค่าจาก st.session_state.get('wear_s', 50) = 290 ✅
```

### 9.3 Dynamic Work Order — Business Logic (app.py บรรทัด 247-334)

```python
# คำนวณ Feature Importance เพื่อหา Primary Driver
feature_labels = ['Material','Air Temp','Process Temp','RPM',
                  'Torque','Tool Wear','Power','Temp Rise']
importances = model.feature_importances_   # array ขนาด 8 ตัว จาก XGBoost
top_feature = feature_labels[np.argmax(importances)]  # ← หา index ที่มากที่สุด

# Map Primary Driver → Specific Action
action_map = {
    'Torque':     {'cause': 'Overstrain Failure (OSF)', 'severity': 'CRITICAL',
                   'action': 'Reduce depth of cut (ap) and feed rate (fn)...'},
    'Tool Wear':  {'cause': 'Tool Wear Failure (TWF)', 'severity': 'HIGH',
                   'action': 'Replace cutting insert immediately...'},
    'RPM':        {'cause': 'Power Failure (PWF)', 'severity': 'HIGH',
                   'action': 'Reduce spindle speed or torque...'},
    'Power':      {'cause': 'Power Overload', 'severity': 'CRITICAL',
                   'action': 'Reduce cutting parameters...'},
    ...
}

advice = action_map.get(top_feature, {...})
severity_color = {
    'CRITICAL': '#ff4444',
    'HIGH': '#e74c3c',
    'MODERATE': '#f39c12',
    'LOW': '#f1c40f'
}.get(advice['severity'], '#e74c3c')
```

**`np.argmax(importances)`:**
- คืน index ของค่าที่มากที่สุดใน array
- ตัวอย่าง: `importances = [0.02, 0.05, 0.06, 0.12, 0.10, 0.09, 0.44, 0.12]`
- `np.argmax(...)` = 6 → `feature_labels[6]` = `'Power'`

---

## Chapter 10: Offline Fallback — ระบบสำรองเมื่อ RAG ล้มเหลว

```python
# app.py บรรทัด 61-78
def _offline_fallback(query):
    query = query.lower()
    if any(k in query for k in ["chip", "chipping", "fracture", "break", "บิ่น"]):
        return "🔧 **Chipping / Fracture:**\n..."
    elif any(k in query for k in ["wear", "flank", "crater", "สึก", "ทื่อ"]):
        return "⚠️ **Tool Wear:**\n..."
    ...
    else:
        return "🤔 **Information Not Found:**\n..."
```

**Design Pattern: Graceful Degradation**

ถ้า RAG ไม่สามารถเชื่อมต่อได้ (ไม่มี API Key, ไม่มี FAISS index) ระบบจะไม่ Crash แต่จะ fall back ไปที่ Keyword Matching แบบง่ายๆ

- `any(k in query for k in [...])` = เช็คว่าคำในคำถามมีคำ keyword ใดบ้างไหม
- รองรับทั้ง English และ ภาษาไทย เพื่อรองรับผู้ใช้ในโรงงานจริง

---

## Chapter 11: Deployment Architecture

### การโหลด API Key แบบ Dual-Mode (rag_core.py บรรทัด 24-42)

```python
# Priority 1: ไฟล์ .env (สำหรับรันในเครื่อง)
groq_api_key = os.environ.get("GROQ_API_KEY")

# Priority 2: Streamlit Secrets (สำหรับ Cloud)
if not groq_api_key:
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            groq_api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass  # ไม่ crash ถ้า import st ไม่ได้ (เช่นตอนรัน test)
```

**Deployment Flow:**

```
GitHub Repository (main branch)
          ↓  Auto-detect push
Streamlit Cloud
    - pip install -r requirements.txt
    - streamlit run app.py
    - อ่าน GROQ_API_KEY จาก st.secrets (ตั้งค่าผ่าน Dashboard)
          ↓
Public URL: https://xxx.streamlit.app
```

**ไฟล์สำคัญที่ต้องมีเมื่อ Deploy:**

| ไฟล์ | สถานะใน Git | เหตุผล |
|------|------------|--------|
| `app.py` | ✅ Committed | Main app |
| `api/rag_core.py` | ✅ Committed | RAG logic |
| `models/faiss_index/` | ✅ Committed (2.5MB) | Pre-built KB |
| `models/predictive_maintenance_model.pkl` | ✅ Committed | Trained XGBoost |
| `data/ai4i2020.csv` | ✅ Committed | Dataset สำหรับ Tab 1 |
| `docs/*.pdf` | ❌ .gitignored | ใหญ่เกิน (8-114MB) |
| `.env` | ❌ .gitignored | มี API Key (ห้าม Public!) |

---

## Chapter 12: คำถามสัมภาษณ์ที่น่าจะโดนถาม + แนวตอบ

**Q: "ทำไมใช้ FAISS แทน Pinecone หรือ Weaviate?"**
> A: "สำหรับ Prototype และ Portfolio ขนาดนี้ที่มีข้อมูล ~1,000 chunks FAISS ทำงานได้ดีมาก รันในเครื่องได้เลย ไม่มีค่าใช้จ่าย และ Latency ต่ำกว่าการเรียก Cloud API ทุกครั้ง แต่ถ้า Production Scale ที่ต้องรองรับหลาย Tenant และ Millions of Vectors จะ migrate ไป Managed Vector DB เช่น Pinecone"

**Q: "SMOTE กับ class_weight='balanced' ต่างกันอย่างไร?"**
> A: "SMOTE สร้างข้อมูลสังเคราะห์ใหม่ในพื้นที่ Feature Space เพิ่ม Diversity ให้ Minority Class แต่เพิ่มเวลาเทรน ส่วน class_weight='balanced' แค่ปรับน้ำหนัก Loss Function ของแต่ละ sample ไม่สร้างข้อมูลใหม่ เราใช้ SMOTE เพราะ XGBoost ไม่มี native class_weight ในแบบเดียวกับ sklearn estimators ทั่วไป นอกจากนี้เรายังใช้ scale_pos_weight ใน Optuna เป็น backup อีกชั้น"

**Q: "RAG มีปัญหาอะไรบ้าง?"**
> A: "ปัญหาหลักคือ 1) Chunk Quality — ถ้าตัดข้อความผิดที่ความหมายจะหายไป แก้ด้วย Overlap 200 ตัว 2) Retrieval Precision — ถ้า k ใหญ่เกินไป Context จะมี Noise มาก ถ้าเล็กเกินอาจพลาดข้อมูลสำคัญ เราตั้งไว้ที่ k=5 เป็นจุดสมดุล 3) Hallucination ยังเป็นไปได้ถ้า LLM ตีความ Context ผิด แก้ด้วย Prompt ที่บอกให้ Stick กับ Context ที่ให้มา"
