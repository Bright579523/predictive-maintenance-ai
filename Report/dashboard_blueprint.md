# 🏭 Smart CNC Dashboard — UI Blueprint (Tab 1 & 2)

## Tab 1: 📊 Overview & Health Monitor
![Tab 1 - Overview & Health Monitor](C:\Users\Bright\.gemini\antigravity\brain\5857ee7c-7aab-4ab2-a583-1a82968f3395\tab1_overview_1778968811417.png)

### องค์ประกอบหลัก:
| Section | รายละเอียด |
|---------|------------|
| **KPI Cards (บนสุด)** | 4 การ์ดโชว์ตัวเลข: Total Machines, Failure Rate, Top Risk Factor, Model Accuracy |
| **Health Gauges (ซ้าย)** | เข็มวัดสุขภาพ 3 ตัว (อุณหภูมิ / แรงบิด / ความสึกหรอ) เปลี่ยนสี 🟢→🟡→🔴 อัตโนมัติ |
| **3D Scatter Plot (ขวา)** | กราฟ 3 มิติ RPM vs Torque vs Tool Wear (จุดแดง = เครื่องพัง) หมุนซูมได้ |
| **Failure Distribution (ล่าง)** | กราฟแท่งแยกประเภทความเสียหาย (TWF / HDF / PWF / OSF / RNF) |

---

## Tab 2: 🔮 Predictive Simulator
![Tab 2 - Predictive Simulator](C:\Users\Bright\.gemini\antigravity\brain\5857ee7c-7aab-4ab2-a583-1a82968f3395\tab2_simulator_1778968823059.png)

### องค์ประกอบหลัก:
| Section | รายละเอียด |
|---------|------------|
| **Input Sliders (ซ้าย)** | สไลเดอร์ปรับค่า: Air Temp, Process Temp, RPM, Torque, Tool Wear |
| **Risk Gauge (ขวา)** | เข็มไมล์วงกลมแสดง Failure Probability (0%–100%) เปลี่ยนสีตามระดับ |
| **SHAP Waterfall (กลาง)** | กราฟอธิบายว่าฟีเจอร์ไหนดันความเสี่ยงขึ้น/ลงเท่าไหร่ |
| **Work Order (ล่าง)** | ใบสั่งซ่อมอัตโนมัติ: บอกสาเหตุ, คำแนะนำ, ต้นทุน Downtime (€) |

---

## 🎨 Design Tokens
- **Background:** `#0E1117` (Streamlit Dark)
- **Card Background:** `rgba(30, 30, 46, 0.8)` (Glassmorphism)
- **Accent Colors:** Cyan `#00D4FF`, Green `#2ECC71`, Red `#E74C3C`, Orange `#F39C12`
- **Currency:** Euro (€)
