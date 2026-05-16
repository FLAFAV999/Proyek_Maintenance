import numpy as np
import pandas as pd

class DecisionMakerAgent:
    def __init__(self, classifier, localizer, diagnoser):
        self.classifier = classifier
        self.localizer = localizer
        self.diagnoser = diagnoser
        # Nilai rata-rata normal (sebagai referensi sederhana)
        self.normal_ref = {
            "Engine rpm": 1500,
            "Lub oil pressure": 4.5,
            "Fuel pressure": 5.0,
            "Coolant pressure": 3.5,
            "lub oil temp": 75,
            "Coolant temp": 80
        }

    def process(self, input_data, feature_names):
        # 1. Konversi input ke DataFrame untuk menghilangkan UserWarning
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # 2. Prediksi Status Utama
        pred = self.classifier.model.predict(input_df)[0]
        
        # 3. SAFETY GUARDRAIL (Veto Aturan Fisik)
        cool_t_val = input_data[5] 
        lub_p_val = input_data[1]

        if pred == 0:
            if cool_t_val > 95: pred = 2
            elif lub_p_val < 3.8: pred = 2

        # 4. LOCALIZATION (Mencari penyebab spesifik per kejadian)
        sub_class = "Normal"
        if pred != 0:
            # Hitung deviasi (selisih) tiap sensor dari nilai normalnya
            deviations = []
            for i, name in enumerate(feature_names):
                ref = self.normal_ref.get(name, 1)
                # Hitung seberapa jauh penyimpangannya dalam persen
                dev = abs(input_data[i] - ref) / ref
                deviations.append(dev)
            
            # Ambil index sensor dengan penyimpangan TERBESAR
            top_idx = np.argmax(deviations)
            culprit_sensor = feature_names[top_idx]
            
            # Jika suhu coolant tinggi, prioritaskan Cooling System
            if cool_t_val > 95:
                sub_class = "Cooling System"
            else:
                sub_class = self.localizer.identify_subclass(None, None).get(culprit_sensor, "General Engine")
        
        # 5. Ambil Diagnosa
        advice = self.diagnoser.get_advice(pred, sub_class)
        return pred, sub_class, advice