from data import DataProvider
from classifier_agent import ClassifierAgent
import joblib

def pretrain():
    print("Membangun Pre-trained Model...")
    dp = DataProvider()
    X, y, _ = dp.fetch_and_process()
    
    agent = ClassifierAgent()
    agent.model.fit(X, y)
    
    joblib.dump(agent.model, "model_3class.pkl")
    print("Berhasil! File model_3class.pkl telah dibuat.")

if __name__ == "__main__":
    pretrain()