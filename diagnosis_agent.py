class DiagnosisAgent:
    def get_advice(self, main_class, sub_class_name):
        labels = {0: "SEHAT", 1: "RUSAK KRITIS", 2: "PERLU PERBAIKAN (WARNING)"}
        advice_map = {
            "Cooling System": "Cek Radiator, Water Pump, dan Water Jacket.",
            "Lubrication System": "Cek Oil Filter, Pompa Oli, dan Kebocoran Seal.",
            "Fuel System": "Cek Injektor dan Filter Bahan Bakar.",
            "Mechanical/Load": "Cek Governor dan Beban Operasional."
        }
        
        status = labels.get(main_class)
        msg = advice_map.get(sub_class_name, "Lakukan pengecekan menyeluruh.")
        
        if main_class == 0:
            return "Mesin dalam kondisi prima. Lanjutkan jadwal pemeliharaan rutin."
        return f"Status: {status}. Lokasi: {sub_class_name}. Saran: {msg}"