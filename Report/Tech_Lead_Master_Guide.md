# 🚀 Tech Lead Master Guide: Smart CNC Predictive Maintenance & AI Consultant

ในฐานะ Tech Lead & AI Agent Partner ของคุณ นี่คือเอกสารสรุปสถาปัตยกรรม (Architecture), เฟรมเวิร์ก (Frameworks), และเจาะลึกโค้ด (Code Deep-Dive) ของโปรเจคที่เราเพิ่งสร้างเสร็จ เพื่อให้คุณเข้าใจทะลุปรุโปร่ง สามารถนำไปพรีเซนต์ หรือตอบคำถามสัมภาษณ์ได้อย่างมั่นใจระดับ Senior/Lead Engineer ครับ

---

## 1. 🌟 The Big Picture (ภาพรวมสถาปัตยกรรมระบบ)

ระบบนี้ไม่ได้เป็นแค่ ML Model ธรรมดา แต่เป็น **Intelligent Application** ที่ผสาน Machine Learning เข้ากับ Generative AI โดยมี 3 แกนหลักที่ทำงานสอดประสานกัน:

1. **Predictive Layer (ML):** ใช้ XGBoost วิเคราะห์ Sensor Data แบบ Real-time เพื่อหาโอกาสพัง (Failure Probability)
2. **Explainability Layer (XAI):** แกะกล่องดำของ ML ด้วย Feature Importance เพื่อบอกว่า "อะไรคือต้นเหตุ" (Primary Driver)
3. **Cognitive Layer (RAG):** เมื่อรู้ปัญหาแล้ว AI Consultant (LLM) จะทำหน้าที่แนะนำวิธีแก้ปัญหาโดยดึงความรู้มาจากคู่มืออุตสาหกรรมเฉพาะทาง

---

## 2. 🛠️ Frameworks ที่สำคัญ (Tech Stack)

คุณต้องอธิบายได้ว่าทำไมเราถึงเลือกใช้เครื่องมือเหล่านี้:

*   **XGBoost:** ทำไมไม่ใช้ Deep Learning? เพราะสำหรับข้อมูลแบบตาราง (Tabular Data) ที่มีฟีเจอร์ไม่เยอะ XGBoost ให้ความแม่นยำสูงที่สุด ฝึกสอนเร็ว และที่สำคัญคือ *สามารถถอดรหัส (Explainable)* หา Feature Importance ได้ง่าย
*   **LangChain:** เป็น Framework ตัวกลาง (Orchestrator) ที่ช่วยเชื่อม LLM เข้ากับข้อมูลของเรา (RAG) ทำให้เราไม่ต้องเขียนโค้ดต่อ API ดิบๆ เอง ลดความยุ่งยากไปได้ 80%
*   **FAISS (Facebook AI Similarity Search):** เป็น Vector Database แบบ Local (รันในเครื่อง/ไฟล์) ที่ทำงานไวมาก เหมาะกับโปรเจคขนาดกลางที่ไม่ต้องการตั้ง Server Database แยก (เช่น Pinecone, Qdrant)
*   **Sentence-Transformers (HuggingFace):** โมเดลแปลงข้อความ (Text) เป็นตัวเลข (Vector/Embedding) เราใช้รุ่น `all-MiniLM-L6-v2` เพราะมัน *ฟรี เบา และแม่นยำพอสำหรับการจับคู่ความหมาย* โดยไม่ต้องเสียเงินเรียก OpenAI API
*   **Groq API (Llama-3.3-70B):** Groq มีจุดเด่นคือ LPU (Language Processing Unit) ทำให้การ Gen text เร็วที่สุดในโลก (เร็วกว่า GPU ปกติ) เราเลือก Llama 3.3 70B เพราะฉลาดเทียบเท่า GPT-4 แต่รันผ่าน Groq ได้ฟรี
*   **Streamlit + Plotly:** Streamlit ทำให้ขึ้นโครงเว็บได้ด้วย Python ล้วน ส่วน Plotly ทำให้กราฟ (Gauge, 3D Scatter) สามารถโต้ตอบได้ (Interactive) ซึ่ง Matplotlib ทำไม่ได้

---

## 3. ⚙️ เจาะลึกการทำงานแต่ละส่วน (End-to-End Flow)

### Phase 1: Machine Learning Pipeline (ไฟล์ `setup_notebook.py`)

**กระบวนการ:**
1.  **Feature Engineering:** เราไม่ได้ใช้แค่ค่าดิบ แต่เราสร้างฟีเจอร์ใหม่ที่สะท้อน "หลักฟิสิกส์" ของเครื่องจักร คือ `Power` (RPM × Torque) และ `Temp_Diff` (Process Temp - Air Temp)
2.  **Training:** โยนเข้า XGBoostClassifier 
3.  **Serialization:** บันทึกโมเดล (`predictive_maintenance_model.pkl`) และ Scaler (`scaler.pkl`) ลงไฟล์ด้วย `joblib` เพื่อให้ Streamlit เอาไปโหลดใช้งานแบบไม่ต้องเทรนใหม่

### Phase 2: RAG Pipeline (โฟลเดอร์ `api/rag_core.py` และ `ingest_all.py`)

RAG (Retrieval-Augmented Generation) คือการให้ AI เปิดหนังสือดูก่อนตอบ แทนที่จะตอบจากความจำ (ซึ่งอาจจะมั่วหรือ Hallucinate)

**1. Data Ingestion (`ingest_all.py`) - ทำครั้งเดียว:**
*   ใช้ `PyPDFLoader` หรือ `TextLoader` โหลดไฟล์
*   ใช้ `RecursiveCharacterTextSplitter` สับไฟล์เป็นชิ้นเล็กๆ (Chunk) ขนาด 1000 ตัวอักษร ซ้อนทับกัน (Overlap) 200 ตัวอักษร เพื่อไม่ให้ประโยคขาดตอน
*   แปลงด้วย Embeddings และบันทึกเป็นไฟล์ `models/faiss_index`

**2. Query & Generation (`api/rag_core.py`) - ทำทุกครั้งที่ถาม:**
*   โหลด `faiss_index` ขึ้นมาบน Memory
*   เมื่อผู้ใช้ถาม: `retriever` จะเอาคำถามไปแปลงเป็น Vector ค้นหา Chunk ที่เหมือนที่สุด 5 ชิ้น (k=5)
*   **Prompt Engineering:** สำคัญมาก เราเขียน Prompt บังคับไว้ว่า: *"ใช้เฉพาะ Context ที่ให้มาเท่านั้น ถ้าไม่มีห้ามมั่วเด็ดขาด"*
*   ส่ง Context 5 ชิ้น + คำถาม เข้าไปที่ `ChatGroq` เพื่อให้สรุปออกมาเป็นคำตอบ

### Phase 3: Frontend & UI (ไฟล์ `app.py`)

**การเชื่อมต่อระบบ (Integration):**
*   **State Management:** ใช้ `st.session_state` ในการเก็บประวัติแชท (messages) เพื่อให้ตอนกดปุ่มหรือสลับแท็บ แชทจะไม่หายไป
*   **Dynamic Work Order:** เป็นจุดที่โชว์ความเป็น Tech Lead เราไม่ได้โชว์แค่ความน่าจะเป็น (Probability) แต่ดึงค่า `model.feature_importances_` มาดูว่า ตัวแปรไหน (เช่น RPM, Torque) ที่ทำให้เกิดความเสี่ยงสูงที่สุดในจังหวะนั้น แล้วเปลี่ยนข้อความ Action Plan ให้ตรงกับปัญหานั้น (เช่น Torque สูง -> แนะนำลด Depth of cut)
*   **Secrets Fallback:** การโหลดคีย์ GROQ_API_KEY ออกแบบให้รองรับทั้งรันในเครื่อง (`.env` ผ่าน `os.environ`) และบน Cloud (`st.secrets`) ทำให้โค้ดชุดเดียวรันได้ทุกที่

---

## 4. 🌐 ขั้นตอนการเชื่อมต่อทั้งระบบ (Data Flow)

1. **User Input:** ผู้ใช้ปรับ Slider 5 ตัวใน Tab 2 (หรือกดปุ่ม Quick Scenario)
2. **Preprocessing:** โค้ดคำนวณ `Power` และ `Temp Diff` สดๆ แล้วเอาเข้า `scaler.transform()`
3. **Prediction:** โยนให้ `model.predict_proba()` ได้เปอร์เซ็นต์ความเสี่ยง
4. **Explainability:** ถ้าความเสี่ยง > 50% ระบบจะเช็ค Feature Importance และออกใบ Work Order ทันที
5. **AI Consult:** ถ้าผู้ใช้ไม่รู้จะแก้ปัญหาตาม Work Order ยังไง จะข้ามไป Tab 3 เพื่อพิมพ์ถาม แชทบอทจะไปค้น FAISS และตอบกลับมา

---

## 5. 💡 Tech Lead's Advice (ถ้ามีคนถามว่า "โปรเจคนี้จะสเกลไปใช้จริงในโรงงานยังไง?")

หากคุณโดนสัมภาษณ์ ให้ตอบแนวทางต่อยอดดังนี้:

1. **Data Streaming (IoT Integration):**
   * *ปัจจุบัน:* รับค่าจาก UI Slider
   * *สเกลลิ่ง:* ต้องเชื่อมต่อกับโปรโตคอล **OPC UA** หรือ **MQTT** ของเครื่อง CNC เพื่อสตรีมข้อมูลเซ็นเซอร์มาที่ Kafka/RabbitMQ แล้วให้โมเดลรันทำนายทุกๆ วินาทีแบบ Real-time
2. **Vector DB Scaling:**
   * *ปัจจุบัน:* ใช้ FAISS แบบ Local File
   * *สเกลลิ่ง:* หากต้องดึงข้อมูลคู่มือ 100,000 หน้า จากเครื่องจักร 50 ยี่ห้อ ควรเปลี่ยนไปใช้ Managed Vector DB เช่น **Pinecone**, **Qdrant**, หรือ **Weaviate** เพื่อความเร็วและสามารถแบ่ง Tenant ของข้อมูลได้
3. **MLOps & Model Drift:**
   * *ปัจจุบัน:* โมเดลแบบ Static (รันบน Notebook แล้วจบ)
   * *สเกลลิ่ง:* ต้องติดตั้ง MLflow เต็มรูปแบบบน Server เมื่อพบว่า Machine เริ่มเสื่อมสภาพตามกาลเวลา ทำให้ Data Drift โมเดลจะต้องถูก Retrain อัตโนมัติด้วยข้อมูลใหม่ๆ

---

เอกสารฉบับนี้คือสรุปแก่นสารของสิ่งที่เราพัฒนาร่วมกันครับ คุณสามารถใช้เป็น Script อ้างอิงตอนอัดคลิปพรีเซนต์ หรือเขียนลง Portfolio ได้เลย! 🚀
