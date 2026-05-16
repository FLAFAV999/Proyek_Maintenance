class LocalizationAgent:
    def identify_subclass(self, input_data, feature_names):
        # Mapping sensor ke sistem (Sub-kelas)
        mapping = {
            "Coolant temp": "Cooling System",
            "Coolant pressure": "Cooling System",
            "Lub oil pressure": "Lubrication System",
            "lub oil temp": "Lubrication System",
            "Fuel pressure": "Fuel System",
            "Engine rpm": "Mechanical/Load"
        }
        
        # Cari fitur yang paling menyimpang dari rata-rata (simulasi)
        # Untuk tugas ini, kita pakai feature importance dari model sebagai proxy
        return mapping