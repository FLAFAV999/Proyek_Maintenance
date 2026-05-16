import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))

import pandas as pd
from sklearn.model_selection import train_test_split

class DataProvider:
    def __init__(self, url="https://huggingface.co/datasets/MohammedSohail/predictive-maintenance-dataset/resolve/main/data.csv"):
        self.url = url
        self.local_path = "data_engine_3class.csv"

    def fetch_and_process(self):
        if os.path.exists(self.local_path):
            df = pd.read_csv(self.local_path)
        else:
            df = pd.read_csv(self.url).dropna()
            
            # REKAYASA 3 KELAS: 0: Sehat, 1: Rusak, 2: Perlu Diperiksa (Warning)
            def create_label(row):
                if row['Engine Condition'] == 1: return 1
                # Threshold Warning: Suhu > 95 atau Tekanan Oli < 3.8
                if row['Coolant temp'] > 95 or row['Lub oil pressure'] < 3.8: return 2
                return 0
            
            df['Engine Condition'] = df.apply(create_label, axis=1)
            
            # Sampling < 2000 (Poin B.1)
            df, _ = train_test_split(df, train_size=1200/len(df), stratify=df['Engine Condition'], random_state=42)
            df.to_csv(self.local_path, index=False)
            
        X = df.drop(columns=['Engine Condition'])
        y = df['Engine Condition']
        return X, y, df