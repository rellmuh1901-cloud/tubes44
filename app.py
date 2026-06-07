
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Risiko Stroke", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px;
        text-align: center; margin-bottom: 2rem; color: white;
    }
    .header-box h1 { color: white; font-size: 2.2rem; margin: 0; }
    .header-box p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0 0; }
    .result-stroke {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: bold;
    }
    .result-aman {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white; padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; padding: 0.75rem 2rem;
        border-radius: 25px; font-size: 1.1rem; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('model_stroke.pkl')

model = load_model()

st.markdown("""
<div class="header-box">
    <h1>🧠 Prediksi Risiko Stroke</h1>
    <p>Masukkan data pasien untuk mengetahui risiko terkena stroke</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📋 Data Pasien")
col1, col2, col3 = st.columns(3)

with col1:
    gender        = st.selectbox("👤 Jenis Kelamin", ["Laki-laki", "Perempuan"])
    age           = st.number_input("🎂 Usia (tahun)", min_value=1, max_value=100, value=45)
    hypertension  = st.selectbox("🩺 Hipertensi", ["Tidak", "Ya"])
    heart_disease = st.selectbox("❤️ Penyakit Jantung", ["Tidak", "Ya"])

with col2:
    ever_married      = st.selectbox("💍 Status Pernikahan", ["Sudah Menikah", "Belum Menikah"])
    residence_type    = st.selectbox("🏠 Tempat Tinggal", ["Perkotaan", "Pedesaan"])
    avg_glucose_level = st.number_input("🩸 Kadar Glukosa", min_value=50.0, max_value=300.0, value=100.0)
    bmi               = st.number_input("⚖️ BMI", min_value=10.0, max_value=100.0, value=25.0)

with col3:
    work_type      = st.selectbox("💼 Jenis Pekerjaan", ["Swasta", "Wiraswasta", "PNS", "Pelajar", "Tidak Bekerja"])
    smoking_status = st.selectbox("🚬 Status Merokok", ["Tidak Pernah Merokok", "Mantan Perokok", "Perokok Aktif", "Tidak Diketahui"])

st.markdown("---")

def preprocess_input():
    return pd.DataFrame([{
        'gender'                         : 1 if gender == "Laki-laki" else 0,
        'age'                            : age,
        'hypertension'                   : 1 if hypertension == "Ya" else 0,
        'heart_disease'                  : 1 if heart_disease == "Ya" else 0,
        'ever_married'                   : 1 if ever_married == "Sudah Menikah" else 0,
        'Residence_type'                 : 1 if residence_type == "Perkotaan" else 0,
        'avg_glucose_level'              : avg_glucose_level,
        'bmi'                            : bmi,
        'work_type_Govt_job'             : 1 if work_type == "PNS" else 0,
        'work_type_Never_worked'         : 1 if work_type == "Tidak Bekerja" else 0,
        'work_type_Private'              : 1 if work_type == "Swasta" else 0,
        'work_type_Self-employed'        : 1 if work_type == "Wiraswasta" else 0,
        'work_type_children'             : 1 if work_type == "Pelajar" else 0,
        'smoking_status_Unknown'         : 1 if smoking_status == "Tidak Diketahui" else 0,
        'smoking_status_formerly smoked' : 1 if smoking_status == "Mantan Perokok" else 0,
        'smoking_status_never smoked'    : 1 if smoking_status == "Tidak Pernah Merokok" else 0,
        'smoking_status_smokes'          : 1 if smoking_status == "Perokok Aktif" else 0,
    }])

if st.button("🔍 Prediksi Sekarang"):
    input_df        = preprocess_input()
    prediction_prob = model.predict_proba(input_df)[0]

    prob_tidak  = prediction_prob[0] * 100
    prob_stroke = prediction_prob[1] * 100

    # Threshold 15% - lebih sensitif tapi tidak terlalu agresif
    prediction = 1 if prediction_prob[1] >= 0.15 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.markdown(f'<div class="result-stroke">⚠️ BERISIKO STROKE — Probabilitas: {prob_stroke:.1f}%</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("⚕️ Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.markdown(f'<div class="result-aman">✅ TIDAK BERISIKO STROKE — Probabilitas Aman: {prob_tidak:.1f}%</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💚 Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("✅ Tidak Stroke", f"{prob_tidak:.1f}%")
        st.progress(prob_tidak / 100)
    with col_b:
        st.metric("⚠️ Stroke", f"{prob_stroke:.1f}%")
        st.progress(prob_stroke / 100)

    st.markdown("---")
    st.subheader("📝 Ringkasan Data Input")
    ringkasan = {
        "Jenis Kelamin"    : gender,
        "Usia"             : f"{age} tahun",
        "Hipertensi"       : hypertension,
        "Penyakit Jantung" : heart_disease,
        "Status Pernikahan": ever_married,
        "Tempat Tinggal"   : residence_type,
        "Kadar Glukosa"    : f"{avg_glucose_level} mg/dL",
        "BMI"              : f"{bmi} kg/m²",
        "Pekerjaan"        : work_type,
        "Status Merokok"   : smoking_status,
    }
    st.table(pd.DataFrame(ringkasan.items(), columns=["Fitur", "Nilai"]))

st.markdown("---")
st.caption("⚕️ Disclaimer: Aplikasi ini hanya untuk keperluan akademik dan bukan pengganti diagnosis medis.")
