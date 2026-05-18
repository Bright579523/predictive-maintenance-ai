# 🏭 Tech Lead Deep-Dive Guide — Part 1: ML Pipeline
> อ้างอิงจากโค้ดจริงใน `notebooks/setup_notebook.py` และ `app.py`

---

## Chapter 1: Feature Engineering — ทำไมถึงสร้างฟีเจอร์ใหม่?

### โค้ดจริง (setup_notebook.py บรรทัด 21-34)
```python
df['Power_W'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * 0.1047
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
type_mapping = {'L': 0, 'M': 1, 'H': 2}
df['Type_encoded'] = df['Type'].map(type_mapping)

features = ['Type_encoded', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Power_W', 'Temp_Diff']
```

### อธิบายทีละบรรทัด:

**`Power_W = Torque × RPM × 0.1047`**
- นี่คือสูตรฟิสิกส์จริง: **P (Watt) = τ (Nm) × ω (rad/s)**
- `ω = RPM × 2π/60 ≈ RPM × 0.1047`
- ทำไมต้องสร้าง? เพราะ Torque กับ RPM แยกกันอาจไม่ทำนายได้ดี แต่ **ผลคูณ (Power)** คือค่าที่แท้จริงที่ทำให้เครื่องพัง — มอเตอร์ของ CNC มีขีดจำกัด Watt ไม่ใช่ขีดจำกัด Torque หรือ RPM อย่างใดอย่างหนึ่ง

**`Temp_Diff = Process Temp - Air Temp`**
- คือ **ความต่างอุณหภูมิ** ระหว่างบริเวณตัดกับอากาศภายนอก
- ถ้า Temp_Diff สูง = ระบบระบายความร้อน (coolant) ทำงานผิดปกติ → เสี่ยง Heat Dissipation Failure (HDF)
- ถ้าใส่แค่ค่าดิบ โมเดลต้องเรียนรู้ความสัมพันธ์นี้เอง แต่ถ้าเราคำนวณให้ = โมเดลฉลาดขึ้นทันที

**`type_mapping = {'L': 0, 'M': 1, 'H': 2}`**
- Label Encoding แปลง text → number เพราะ XGBoost ทำงานกับตัวเลขเท่านั้น
- **ทำไมไม่ใช้ One-Hot Encoding?** เพราะ L/M/H มีลำดับความรุนแรง (Ordinal) อยู่ในตัว (H > M > L) Label Encoding รักษาความสัมพันธ์นี้ไว้ได้ ขณะที่ One-Hot จะทำให้สูญเสียข้อมูลส่วนนี้ไป

---

## Chapter 2: Train-Test Split และกับดัก Data Leakage

### โค้ดจริง (setup_notebook.py บรรทัด 45-55)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ← fit AND transform
X_test_scaled = scaler.transform(X_test)         # ← transform ONLY!

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
```

### กับดักสำคัญที่คนทั่วไปมักทำผิด:

**❌ ผิด:** `scaler.fit_transform(X)` แล้วค่อย split
**✅ ถูก:** split ก่อน แล้วค่อย `fit_transform` บน Train เท่านั้น

> **เหตุผล (Data Leakage):** ถ้า fit scaler บนข้อมูลทั้งหมด หมายความว่า Mean และ Std ของ Test Set ถูกแอบนำมาคำนวณด้วย โมเดลจึงรู้ข้อมูล "อนาคต" ล่วงหน้า → ตัวเลขผลลัพธ์จะสูงเกินจริง แต่พอ Deploy จริงจะแย่ลงทันที

**`stratify=y`**
- บังคับให้สัดส่วน 0/1 ใน Train และ Test เท่ากัน
- สำคัญมากเพราะชุดข้อมูลนี้ **Imbalanced** (Failure มีแค่ ~3.39%) ถ้าไม่ Stratify อาจได้ Test Set ที่ไม่มี Failure เลย

**SMOTE (Synthetic Minority Over-sampling Technique)**
- ปัญหา: เครื่องพัง 339 ครั้ง / ปกติ 9,661 ครั้ง = สัดส่วน 1:28
- ถ้าไม่แก้: โมเดลจะทายว่า "ไม่พัง" ตลอด แล้วก็ได้ Accuracy 96%+ ทั้งๆ ที่ไม่มีประโยชน์อะไรเลย
- SMOTE แก้โดย **สร้างตัวอย่าง Failure ปลอมๆ (Synthetic)** ด้วยการ interpolate ระหว่างจุด Failure ที่ใกล้กัน ทำให้สมดุลขึ้น
- **ทำบน Train เท่านั้น!** ไม่แตะ Test เพราะ Test ต้องเป็นข้อมูลจริงเท่านั้น

---

## Chapter 3: Baseline Model — ทำไมต้องมี Logistic Regression ก่อน?

### โค้ดจริง (setup_notebook.py บรรทัด 67-82)
```python
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_train_smote, y_train_smote)

y_pred_base = baseline_model.predict(X_test_scaled)
y_prob_base = baseline_model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_base):.4f}")
print(f"Recall:   {recall_score(y_test, y_pred_base):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_prob_base):.4f}")
```

### ทำไม Recall สำคัญกว่า Accuracy ในงานนี้?

| Metric | ความหมาย | ตัวอย่าง |
|--------|----------|---------|
| **Accuracy** | ทายถูกทั้งหมด / ทั้งหมด | ถ้าทายว่า "ไม่พัง" ตลอด = 96.6% Accuracy → ใช้ไม่ได้! |
| **Precision** | จากที่ทาย "พัง" ทั้งหมด ทายถูกกี่ % | สำคัญถ้า False Alarm มีต้นทุนสูง |
| **Recall** | จากที่พังจริงทั้งหมด โมเดลจับได้กี่ % | **สำคัญที่สุดในงานนี้** — พลาด 1 ครั้ง = โรงงานหยุดยาว |
| **F1** | ค่าเฉลี่ยของ Precision กับ Recall | ใช้วัดโดยรวมเมื่อทั้งคู่สำคัญ |
| **ROC AUC** | ความสามารถในการแยก Class 0 กับ 1 | ยิ่งใกล้ 1.0 ยิ่งดี |

**Baseline มีไว้เพื่อ:** พิสูจน์ว่า XGBoost "ดีกว่า" อย่างมีนัยสำคัญ ไม่ใช่แค่บังเอิญ

---

## Chapter 4: Optuna — Bayesian Hyperparameter Tuning

### โค้ดจริง (setup_notebook.py บรรทัด 86-113)
```python
def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 1, 20),
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test_scaled)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
```

### เจาะลึก Hyperparameters แต่ละตัว:

| Parameter | บทบาท | ค่าต่ำ → สูง |
|-----------|--------|-------------|
| `n_estimators` | จำนวนต้นไม้ใน Forest | น้อย=เร็ว/Underfit → มาก=ช้า/Overfit |
| `max_depth` | ความลึกของแต่ละต้นไม้ | ตื้น=ง่าย → ลึก=ซับซ้อน/Overfit |
| `learning_rate` | ขนาดก้าวการเรียนรู้ (log=True ค้นบน Log scale) | เล็ก=ละเอียด/ช้า → ใหญ่=หยาบ/เร็ว |
| `subsample` | % ข้อมูลที่สุ่มต่อต้นไม้ | ป้องกัน Overfit |
| `colsample_bytree` | % Feature ที่สุ่มต่อต้นไม้ | ป้องกัน Overfit + เพิ่ม Diversity |
| `scale_pos_weight` | น้ำหนักเพิ่มให้ Class Minority (Failure) | ทดแทน SMOTE ได้อีกทาง |

**Optuna vs Grid Search:**
- Grid Search ทดลองทุกคู่ผสม (O(n^k)) → ช้ามาก
- Optuna ใช้ **Bayesian Optimization (TPE algorithm)** — ทดลอง trial ก่อน แล้วเดาว่า parameter ไหนน่าจะดี → ทดลองแถวนั้น → ทำซ้ำ ใช้ 30 trials แทน Grid ที่ต้องการหลายพัน trials

---

## Chapter 5: การ Save และ Load Model (app.py)

### โค้ดจริง (app.py บรรทัด 41-55)
```python
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
```

### `@st.cache_data` vs `@st.cache_resource`:

| Decorator | ใช้กับ | ทำงานอย่างไร |
|-----------|--------|-------------|
| `@st.cache_data` | DataFrame, dict, list | Copy ค่ากลับเสมอ (Thread-safe สำหรับข้อมูล) |
| `@st.cache_resource` | Model, DB Connection, Large Object | Return **reference เดิม** (ไม่ Copy) — เหมาะกับ Object ขนาดใหญ่ที่ Copy แล้วช้า |

> **หลักการ:** Streamlit จะ rerun สคริปต์ทั้งหมดทุกครั้งที่ผู้ใช้โต้ตอบ decorator ทั้งสองนี้ทำให้โมเดลถูกโหลดครั้งเดียวเท่านั้น ประหยัด RAM และเวลาได้มาก

### การทำนายจริงใน Simulator (app.py บรรทัด 215-224)
```python
power = torque * rpm * 0.1047        # คำนวณ Feature ใหม่เหมือนตอน Train
temp_diff = proc - air
type_code = {"Aluminum...": 0, "Carbon Steel...": 1, "Stainless...": 2}[type_in]
input_vec = np.array([[type_code, air, proc, rpm, torque, wear, power, temp_diff]])

scaled = scaler.transform(input_vec)          # ← transform เท่านั้น! ไม่ fit ใหม่
prob = model.predict_proba(scaled)[0][1] * 100  # ← [0]=แถวแรก, [1]=probability ของ Class 1
```

> **จุดสำคัญ:** `predict_proba()` คืน `[[prob_0, prob_1]]` เราต้องการ `[0][1]` = ความน่าจะเป็นของ class "พัง" (=1) แล้วคูณ 100 ให้เป็น % แสดงบน Gauge
