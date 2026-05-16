import sys
import os

# Path injector agar library portable terbaca
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))

import pandas as pd
from data import DataProvider
from classifier_agent import ClassifierAgent
from localization_agent import LocalizationAgent
from diagnosis_agent import DiagnosisAgent
from decision_maker_agent import DecisionMakerAgent

def run_debug_10_samples():
    print("="*60)
    print("🔍 DEBUG OPSI 1: VALIDASI 10 SAMPEL ACAK")
    print("="*60)

    # 1. Inisialisasi Data & Agents
    dp = DataProvider()
    c_agent = ClassifierAgent()
    l_agent = LocalizationAgent()
    d_agent = DiagnosisAgent()
    dm_agent = DecisionMakerAgent(c_agent, l_agent, d_agent)

    # 2. Fetch Data & Latih Model (Simulasi proses setelah CV)
    X, y, df = dp.fetch_and_process()
    print(f"[Agent 1] Data dimuat: {len(df)} baris.")
    
    print("[Agent 2] Melatih model pada seluruh dataset untuk pengujian...")
    c_agent.model.fit(X, y)
    
    # 3. Ambil 10 Sampel Acak (Randomize Test Data)
    test_samples = df.sample(10, random_state=None) # random_state=None agar selalu berbeda tiap dijalankan
    
    # 4. Proses Diagnosa untuk tiap sampel
    feature_cols = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]
    labels_map = {0: "SEHAT", 1: "RUSAK", 2: "WARNING"}
    
    correct_predictions = 0
    
    print("\n" + "-"*85)
    print(f"{'No':<3} | {'True Label':<10} | {'Prediksi':<10} | {'Status':<10} | {'Lokasi':<15}")
    print("-"*85)

    for i, (idx, row) in enumerate(test_samples.iterrows(), 1):
        # Ambil nilai fitur
        input_vals = row[feature_cols].values.tolist()
        true_label = int(row['Engine Condition'])
        
        # Proses melalui Decision Maker (Multi-Agent Collaboration)
        pred, sub, advice = dm_agent.process(input_vals, feature_cols)
        
        # Hitung Akurasi Debug
        is_correct = "PASS" if pred == true_label else "FAIL"
        if pred == true_label: correct_predictions += 1
        
        # Cetak Ringkasan Tabel
        print(f"{i:<3} | {labels_map[true_label]:<10} | {labels_map[pred]:<10} | {is_correct:<10} | {sub:<15}")
    
    print("-"*85)
    
    # 5. Kesimpulan Debug
    accuracy = (correct_predictions / 10) * 100
    print(f"\n✅ DEBUG SELESAI")
    print(f"Total Sampel Benar: {correct_predictions} / 10")
    print(f"Akurasi Sesi Ini : {accuracy}%")
    print("\n💡 Tips: Jika ada status 'FAIL', cek apakah sensor melewati batas 'Safety Guardrail' di DecisionMaker.")
    print("="*60)

if __name__ == "__main__":
    run_debug_10_samples()