import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ตั้งค่าหน้าจอ Streamlit
st.set_page_config(page_title="Edge Seam AI & Optimization System", layout="wide")

# ==========================================
# 1. ฟังก์ชันโหลด Model และ Artifacts
# ==========================================
@st.cache_resource
def load_models():
    # ตรวจสอบและโหลดโมเดล/ไฟล์ตัวแปร
    model = tf.keras.models.load_model('edge_seam_ann.keras', compile=False)
    knn_imputer = joblib.load('knn_imputer.save')
    scaler_num = joblib.load('scaler_num.save')

    # กรณีที่มีการใช้ Categorical เผื่อไว้
    if os.path.exists('encoder_cat.save'):
        encoder_cat = joblib.load('encoder_cat.save')
    else:
        encoder_cat = None

    top_features = joblib.load('best_features.save')
    col_names = joblib.load('col_names.save')

    # โหลดค่า Threshold (หากไม่มีให้ใช้ 0.5 เป็นค่าเริ่มต้น)
    if os.path.exists('optimal_threshold.txt'):
        with open('optimal_threshold.txt', 'r') as f:
            threshold = float(f.read())
    else:
        threshold = 0.5

    return model, knn_imputer, scaler_num, encoder_cat, top_features, col_names, threshold

try:
    model, imputer, scaler, encoder, top_features, col_names, threshold = load_models()
    num_cols = col_names['num_cols']
    cat_cols = col_names['cat_cols']
except Exception as e:
    st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลด Model หรือ Artifacts: {e}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ดนตรีจำพวก .keras, .save, และ .txt อยู่ในโฟลเดอร์เดียวกันกับ app.py")
    st.stop()

# ==========================================
# 2. คลาสจำลองสำหรับ Prescriptive Engine (AI Suggestion)
# ==========================================
class EdgeSeamOptimizer:
    def __init__(self, model, imputer, scaler, encoder, top_features, threshold):
        self.model = model
        self.imputer = imputer
        self.scaler = scaler
        self.encoder = encoder
        self.top_features = top_features
        self.threshold = threshold

        # ตัวแปรที่อนุญาตให้ AI แนะนำการปรับเปลี่ยนได้ตามโจทย์
        self.controllable_cols = [
            'FT_HEAD', 'CT_HEAD', 'XVPTF8', 'RMEXTG',
            'PSDRFT1', 'PSDRFT2', 'PSDRFT3', 'PSDRFT4', 'PSDRFT5',
            'PSRCMS1', 'PSRCMS2', 'PSRCMS3', 'PSRCMS4', 'PSRCMS5'            
        ]
        self.adjustment_limit_pct = 0.05 # จำกัดการปรับไม่เกิน +/- 5% จากค่าปัจจุบัน

    def optimize(self, current_df):
        # 1. ตรวจสอบค่าปัจจุบันก่อน
        X_curr = self._preprocess(current_df.copy())
        current_prob = self.model.predict(X_curr, verbose=0)[0][0]

        if current_prob < self.threshold:
            return None, current_prob, "safe"

        # 2. ปรับตัวแปรหน้างานแบบจำลอง (Monte Carlo Simulation - 800 รูปแบบ)
        n_sims = 800
        sim_df = pd.concat([current_df] * n_sims, ignore_index=True)

        for col in self.controllable_cols:
            if col in sim_df.columns:
                curr_val = current_df.loc[0, col]
                # สุ่มปรับค่าอยู่ในช่วง +/- 5%
                sim_df[col] = np.random.uniform(
                    curr_val * (1 - self.adjustment_limit_pct),
                    curr_val * (1 + self.adjustment_limit_pct),
                    size=n_sims
                )

        # 3. ให้โมเดลทำนายผลลัพธ์ของข้อมูลจำลองทั้งหมด
        X_sims = self._preprocess(sim_df)
        sim_probs = self.model.predict(X_sims, verbose=0).flatten()

        # 4. เลือกชุดข้อมูลที่รอด และมีการเปลี่ยนแปลงค่าน้อยที่สุด (Minimal Distance)
        safe_idx = np.where(sim_probs < self.threshold)[0]
        if len(safe_idx) == 0:
            return None, current_prob, "unable"

        safe_sims = sim_df.iloc[safe_idx]
        safe_probs = sim_probs[safe_idx]

        # คำนวณระยะห่างทางคณิตศาสตร์จากค่าปัจจุบัน
        orig_vals = current_df[self.controllable_cols].values[0]
        distances = np.linalg.norm(safe_sims[self.controllable_cols].values - orig_vals, axis=1)

        best_idx = np.argmin(distances)
        best_suggestion = safe_sims.iloc[best_idx]
        best_prob = safe_probs[best_idx]

        # สรุปรายการคำแนะนำออกมา
        suggestions = {}
        for col in self.controllable_cols:
            orig_v = current_df.loc[0, col]
            sugg_v = best_suggestion[col]
            diff = sugg_v - orig_v
            if abs(diff) > 0.001: # แสดงผลเฉพาะที่มีการแนะนำให้ปรับจริงๆ
                suggestions[col] = {
                    "Current": orig_v,
                    "Suggested": sugg_v,
                    "Change": diff
                }
        return suggestions, best_prob, "optimized"

    def _preprocess(self, df):
        # เติมค่าว่างด้วย Imputer
        df_num = self.imputer.transform(df[num_cols])
        df_num_scaled = self.scaler.transform(df_num)

        if self.encoder and cat_cols:
            df_cat = self.encoder.transform(df[cat_cols])
            X_all = np.hstack([df_num_scaled, df_cat])
        else:
            X_all = df_num_scaled

        # สร้าง DataFrame และดึงเฉพาะ Top Features ที่ใช้เทรนโมเดล
        all_features = num_cols + (list(self.encoder.get_feature_names_out(cat_cols)) if (self.encoder and cat_cols) else [])
        X_df = pd.DataFrame(X_all, columns=all_features)
        return X_df[self.top_features]

optimizer = EdgeSeamOptimizer(model, imputer, scaler, encoder, top_features, threshold)

# ==========================================
# 3. หน้าตาแอปพลิเคชันบน Streamlit (UI/UX)
# ==========================================
st.title("🏭 Edge Seam Defect Prediction & Parameter Optimization")
st.markdown("ระบบวิเคราะห์จุดเสี่ยงการเกิด Edge Seam Defect และแนะนำการตั้งค่าพารามิเตอร์ลูกรีดอัตโนมัติ")
st.markdown("---")

# ส่วนที่ 1: ข้อมูลกายภาพ/สเปคของชิ้นงาน (ควบคุมไม่ได้หน้างาน)
st.subheader("📋 1. ข้อมูลคุณลักษณะเหล็กแผ่น (Uncontrollable Base Parameters)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    slabs_id = st.text_input("SLPRNU (SLAB Supplier)", value="ID2605")
    comqua = st.text_input("COMQUA (เกรดเหล็ก)", value="SS400")
with col2:
    slabth = st.number_input("SLABTH (ความหนา SLAB: mm)", min_value=200.0, max_value=220.0, value=210.0, step=0.5)
    slabwi = st.number_input("SLABWI (ความกว้าง SLAB: mm)", value=1250.0, step=10.0)
with col3:
    slabwe = st.number_input("SLABWE (น้ำหนัก SLAB: kg)", value=22000.0, step=0.1)
    slfuti = st.number_input("SLFUTI (เวลาในเตา)", value=3.5, step=0.1)
with col4:
    hnspdi = st.number_input("HNSPDI (ความหนา)", value=3.2, step=0.1)
    wnspdi = st.number_input("WNSPDI (ความกว้าง)", value=1219.0, step=5.0)

# ส่วนที่ 2: ค่าพารามิเตอร์กระบวนการรีด (ควบคุมได้หน้างาน)
st.subheader("⚙️ 2. พารามิเตอร์การผลิตปัจจุบัน (Controllable Setup Parameters)")
st.caption("ระบุค่าที่กำลังตั้งค่าอยู่ในปัจจุบันเพื่อตรวจสอบจุดเสี่ยง")

tab_temp, tab_draft, tab_roll = st.tabs(["🌡️ อุณหภูมิและความเร็ว", "📉 อัตราการกดลูกรีด (RM Draft)", "🔄 ความเร็วลูกรีด (RM Speed)"])

with tab_temp:
    c1, c2, c3, c4 = st.columns(4)
    with c1: ft_head = st.number_input("FT_HEAD (องศา)", value=850.0, step=10.0)
    with c2: ct_head = st.number_input("CT_HEAD (องศา)", value=580.0, step=10.0)
    with c3: xvptf8 = st.number_input("XVPTF8 (ความเร็วลูกรีด)", value=8.5, step=0.5)
    with c4: rmextg = st.number_input("RMEXTG (BarThk)", value=32.0, step=1.0)

with tab_draft:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: psdrft1 = st.number_input("PSDRFT1 (Pass1 Draft)", value=42.2, step=0.5)
    with c2: psdrft2 = st.number_input("PSDRFT2 (Pass2 Draft)", value=48.1, step=0.5)
    with c3: psdrft3 = st.number_input("PSDRFT3 (Pass3 Draft)", value=48.6, step=0.5)
    with c4: psdrft4 = st.number_input("PSDRFT4 (Pass4 Draft)", value=48.8, step=0.5)
    with c5: psdrft5 = st.number_input("PSDRFT5 (Pass5 Draft)", value=50.2, step=0.5)

with tab_roll:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: psrcms1 = st.number_input("PSRCMS1 (Pass1 Speed)", value=1.5, step=0.1)
    with c2: psrcms2 = st.number_input("PSRCMS2 (Pass2 Speed)", value=1.5, step=0.1)
    with c3: psrcms3 = st.number_input("PSRCMS3 (Pass3 Speed)", value=1.5, step=0.1)
    with c4: psrcms4 = st.number_input("PSRCMS4 (Pass4 Speed)", value=1.5, step=0.1)
    with c5: psrcms5 = st.number_input("PSRCMS5 (Pass5 Speed)", value=2.5, step=0.1)

# เติมตัวแปรที่เหลือในโมเดลด้วยค่าสถิติเริ่มต้น (Default หรือ Median) เพื่อไม่ให้โมเดลพัง
base_data = {
    'SLPRNU': slabs_id, 'COMQUA': comqua, 'SLABTH': slabth, 'SLABWI': slabwi, 'SLABWE': slabwe,
    'SLFUTI': slfuti, 'HNSPDI': hnspdi, 'WNSPDI': wnspdi, 'FT_HEAD': ft_head, 'CT_HEAD': ct_head,
    'XVPTF8': xvptf8, 'RMEXTG': rmextg, 'PSDRFT1': psdrft1, 'PSDRFT2': psdrft2, 'PSDRFT3': psdrft3,
    'PSDRFT4': psdrft4, 'PSDRFT5': psdrft5, 'PSRCMS1': psrcms1, 'PSRCMS2': psrcms2, 'PSRCMS3': psrcms3,
    'PSRCMS4': psrcms4, 'PSRCMS5': psrcms5,
    # ตัวแปรเสริมอื่นๆ (เซ็ตเป็นค่ากลางเริ่มต้น)
    'TEM_DIS': 1260.0, 'PSDRFT': 50.2, 'CORPSR_M1': 8500.0, 'CORPSR_M2': 10000.0, 'CORPSR_M3': 11000.0,
    'CORPSR_M4': 12000.0, 'CORPSR_M5': 13500.0, 'RIDAMF1': 0.30, 'RIDAMF2': 0.32, 'RIDAMF3': 0.28, 'RIDAMF4': 0.23,
    'RIDAMF5': 0.20, 'RIDAMF6': 0.18, 'RIDAMF7': 0.15, 'CBTHSP': 2.5, 'CBRUSP': 4.5, 'DESCH1_MIN': 155.0,
    'DESCH2_MIN': 155.0, 'TNVTRP1': 0.80, 'TNVTRP2': 0.90, 'TNVTRP3': 0.90, 'TNVTRP4': 0.90, 'TNVTRP5': 0.90,
    'TNVTRP6': 0.9, 'TNVTRP7': 1.0, 'FTGM': 9500.0, 'LSP_Body': 1080.5, 'Entry_Body': 1000.0, 'FT_BODY': 855.0,
    'CT_BODY': 585.0, 'SLAB_QUALITY': 'C032', 'OPCCO': '0', 'LCBXON': 'N', 'ENDUSE': 'S', 'PASSNR': 5,
    'DescaleCondition': 'OK'
}

input_df = pd.DataFrame([base_data])

# ==========================================
# 4. ปุ่มประมวลผลและการแสดงผลลัพธ์
# ==========================================
st.markdown("---")
if st.button("📊 วิเคราะห์และตรวจสอบพารามิเตอร์การผลิต", type="primary", use_container_width=True):

    with st.spinner("AI กำลังวิเคราะห์จุดเสี่ยงและคำนวณทางเลือกกระบวนการผลิต..."):
        # รันการค้นหาคำแนะนำ
        suggestions, final_prob, status = optimizer.optimize(input_df)

    # แสดงโซนผลลัพธ์การทำนาย
    st.subheader("🎯 ผลการวิเคราะห์จากระบบ AI")

    if status == "safe":
        st.success(f"🟩 **สถานะปกติ (GOOD):** พารามิเตอร์ปัจจุบันมีความปลอดภัยสูง (โอกาสเกิด Defect เพียง {final_prob * 100:.2f}%)")
        st.balloons()

    else:
        st.error(f"🟥 **จุดเสี่ยงระดับสูง (NG DETECTED):** พารามิเตอร์ปัจจุบันมีความเสี่ยงเกิดแตกขอบสูงเกินเกณฑ์กำหนด (Risk Score: {final_prob * 100:.2f}%)")

        if status == "optimized" and suggestions:
            st.warning(f"💡 **AI Setup Guidelines:** ค้นพบแนวทางการปรับปรุงพารามิเตอร์เพื่อลดความเสี่ยงลงต่ำกว่าจุดตัด (ความเสี่ยงจะลดลงเหลือ {final_prob * 100:.2f}%)")

            # แปลงผลลัพธ์แสดงเป็นตารางเปรียบเทียบ Setup Guide
            guide_data = []
            for k, v in suggestions.items():
                guide_data.append({
                    "Parameter": k,
                    "ค่าที่ตั้งไว้ปัจจุบัน (Current)": f"{v['Current']:.2f}",
                    "คำแนะนำปรับปรุงโดย AI (Guideline)": f"{v['Suggested']:.2f}",
                    "ทิศทางการปรับเปลี่ยน (Action)": f"🔺 เพิ่มขึ้น (+{v['Change']:.2f})" if v['Change'] > 0 else f"🔻 ลดลง ({v['Change']:.2f})"
                })

            guide_df = pd.DataFrame(guide_data)
            st.table(guide_df.set_index("Parameter"))
            st.info("ℹ️ **ข้อเสนอแนะสำหรับวิศวกร:** โปรดพิจารณาจูนหน้างานตามข้อมูลแนวทางข้างต้นเพื่อเสถียรภาพในการผลิตแผ่นเหล็กชุดนี้")

        elif status == "unable":
            st.warning("⚠️ **ไม่พบทางแก้ไขที่ปลอดภัยภายใต้เงื่อนไข 5%:** กรุณาพิจารณาปรับลดอุณหภูมิภาพรวม หรือประสานงานฝ่ายเทคนิคเพื่อตรวจสอบโครงสร้างสแลบเป็นกรณีพิเศษ")
