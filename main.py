import streamlit as st
import joblib
import os
import pandas as pd
from data import DataProvider
from classifier_agent import ClassifierAgent
from localization_agent import LocalizationAgent
from diagnosis_agent import DiagnosisAgent
from decision_maker_agent import DecisionMakerAgent

# Setup
st.set_page_config(page_title="Engine MAS Dashboard", layout="wide")
dp = DataProvider()
c_agent = ClassifierAgent()
l_agent = LocalizationAgent()
d_agent = DiagnosisAgent()
dm_agent = DecisionMakerAgent(c_agent, l_agent, d_agent)

st.sidebar.title("🛠️ MAS Navigation")
mode = st.sidebar.radio("Pilih Modul:", ["Validasi Tugas (Academic)", "Simulator (Industrial)"])

if mode == "Validasi Tugas (Academic)":
    st.header("📋 Modul Validasi Cross-Validation (Poin B.4-B.6)")
    
    # --- STEP 1: LOAD DATA ---
    if st.button("Load & Resample Data"):
        X, y, df = dp.fetch_and_process()
        st.session_state['X'], st.session_state['y'], st.session_state['df'] = X, y, df
        
        counts = df['Engine Condition'].value_counts()
        st.success(f"✅ Data Berhasil Diproses: Total {len(df)} baris.")
        
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Kelas 0 (Sehat)", f"{counts.get(0, 0)} Rows")
        col_c2.metric("Kelas 1 (Rusak)", f"{counts.get(1, 0)} Rows")
        col_c3.metric("Kelas 2 (Warning)", f"{counts.get(2, 0)} Rows")
        
        st.write("**5 Data Awal:**")
        st.dataframe(df.head(5))

    st.divider()

    # --- STEP 2: VALIDASI & DIAGNOSA ---
    if 'X' in st.session_state:
        st.subheader("⚙️ Cross-Validation Settings")
        
        # Definisikan cv_pilihan di sini agar selalu terdeteksi oleh tombol di bawahnya
        cv_pilihan = st.selectbox("Pilih Metode CV:", ["kfold", "stratified", "loocv"])
        
        if st.button("Run Manual Loop Validation"):
            # Menggunakan cv_pilihan yang diambil dari selectbox tepat di atas
            progress = st.progress(0)
            
            # Jalankan Validasi (Loop Manual)
            # Hasilnya berupa generator dari classifier_agent
            results = list(c_agent.run_cv_task(st.session_state['X'], st.session_state['y'], cv_pilihan))
            avg_p = results.pop() # Ambil nilai terakhir sebagai rata-rata
            
            st.divider()
            
            # --- FITUR DIAGNOSA DATA ACAK ---
            st.subheader("🎲 Randomized Sample Diagnosis")
            st.info("Sistem mengambil 1 data acak untuk diuji dengan model yang baru dilatih:")
            
            # Ambil sampel acak dari data yang sudah di-load
            random_sample = st.session_state['df'].sample(1)
            st.table(random_sample) 
            
            # Siapkan fitur untuk Decision Maker (urutan harus benar)
            feature_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]
            sample_values = random_sample[feature_cols].values[0].tolist()
            
            # Proses melalui Decision Maker (Multi-Agent Collaboration)
            pred, sub, advice = dm_agent.process(sample_values, feature_cols)
            
            # Tampilan Output Diagnosa
            diag_col1, diag_col2, diag_col3 = st.columns(3)
            with diag_col1:
                labels = ["✅ SEHAT", "🚨 RUSAK KRITIS", "⚠️ PERLU PERBAIKAN"]
                st.metric("Prediksi Status", labels[pred])
            with diag_col2:
                st.metric("Sub-Sistem", sub)
            with diag_col3:
                true_label = int(random_sample['Engine Condition'].values[0])
                st.metric("Label Asli Data", labels[true_label])
            
            st.warning(f"**Analisis Agent:** {advice}")
            
            st.divider()
            
            # --- OUTPUT METRIK (Poin B.5 & B.6) ---
            st.subheader("📊 Validation Metrics")
            with st.expander("Lihat Detail Presisi Per Fold (Poin B.5)"):
                for line in results: 
                    st.write(line)
                    # Update progress bar sederhana
                    progress.progress(100) 
            
            st.metric(f"Rata-rata Nilai Presisi ({cv_pilihan})", f"{avg_p:.4f}")
            st.success("Validasi Selesai!")
else:
    # --- OPSI 2: SIMULATOR (Industrial) ---
    st.header("🎮 Dashboard Simulator Diagnosis")
    if os.path.exists("model_3class.pkl"):
        c_agent.model = joblib.load("model_3class.pkl")
        # Header untuk nama kolom
        header_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]
        
        st.write("Input Parameter Sensor (Gunakan batasan rahasia untuk tes):")
        col1, col2 = st.columns(2)
        with col1:
            rpm = st.number_input("Engine RPM", 1000, 3000, 1500)
            lub_p = st.number_input("Lub Oil Pressure", 0.0, 10.0, 4.0)
            fuel_p = st.number_input("Fuel Pressure", 0.0, 10.0, 5.0)
        with col2:
            cool_p = st.number_input("Coolant Pressure", 0.0, 10.0, 3.5)
            lub_t = st.number_input("Lub Oil Temp", 50, 150, 80)
            cool_t = st.number_input("Coolant Temp", 50, 150, 85)

        if st.button("Jalankan Diagnosa"):
            input_val = [rpm, lub_p, fuel_p, cool_p, lub_t, cool_t]
            # Decision Maker Agent bekerja di balik layar
            pred, sub, advice = dm_agent.process(input_val, header_cols)
            
            st.divider()
            cols = st.columns(3)
            
            # Pewarnaan status
            status_labels = ["✅ SEHAT", "🚨 RUSAK KRITIS", "⚠️ PERLU PERBAIKAN"]
            if pred == 0: st.success(f"### HASIL: {status_labels[pred]}")
            elif pred == 1: st.error(f"### HASIL: {status_labels[pred]}")
            else: st.warning(f"### HASIL: {status_labels[pred]}")
            
            st.write(f"**Sub-Sistem Terdampak:** {sub}")
            st.info(f"**Rekomendasi Agent:** {advice}")
    else:
        st.error("Model tidak ditemukan! Silakan jalankan 'python model.py' terlebih dahulu di terminal.")