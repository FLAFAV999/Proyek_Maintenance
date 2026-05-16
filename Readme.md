# 🛠️ Multi-Agent System: Engine Predictive Maintenance
**Programming Task for Group TI6043 - Machine Learning**

An advanced predictive maintenance system implemented as a Multi-Agent System (MAS). This project simulates an Industrial Digital Twin to monitor engine health, featuring 3-class classification, fault localization, and hybrid AI safety guardrails.

---
## 👥 Group TI6043 Members

Farrel Ballard Thomas - 71210679

Drestanta Dipta Jalu Prakasya - 71220895

---
## 🧠 System Workflow & Agent Mechanism
This system operates through a collaborative pipeline where each agent has a specific "intellectual" responsibility:

### 1. Data Provider Agent (`data.py`)
*   **Mechanism:** Ingests raw sensor data and performs **Label Engineering**. 
*   **Logic:** It transforms binary data into a 3-class system (0: Healthy, 1: Fault, 2: Warning) based on industrial thresholds. It also ensures **Requirement B.1 Compliance** by resampling the dataset to exactly 1,200 rows using Stratified Sampling to maintain class integrity.

### 2. Classifier Agent (`classifier_agent.py`)
*   **Mechanism:** The "Mathematical Brain" of the system. 
*   **ML Model:** Uses **Random Forest Classifier** for high-dimensional feature analysis.
*   **Validation Logic:** Implements **Manual Loop Iteration** for K-Fold, Stratified K-Fold, and LOOCV (Requirements B.4-B.6). It yields precision metrics for each individual fold to provide full transparency of model stability.

### 3. Localization Agent (`localization_agent.py`)
*   **Mechanism:** Focuses on **Fault Isolation**.
*   **Logic:** Instead of just predicting a failure, it analyzes sensor deviations from normal operational baselines to pinpoint which sub-system (Cooling, Lubrication, Fuel, or Mechanical) is causing the anomaly.

### 4. Diagnosis Agent (`diagnosis_agent.py`)
*   **Mechanism:** The "Technical Advisor".
*   **Logic:** Translates machine states into human-readable technical advice. It maps specific fault locations to actionable maintenance steps (e.g., "Check Oil Filter" if Lubrication System is identified).

### 5. Decision Maker Agent (`decision_maker_agent.py`)
*   **Mechanism:** The "System Orchestrator" and "Final Judge".
*   **Hybrid AI Logic:** It integrates ML predictions with **Safety Guardrails**. It holds "Veto Power"—if a critical sensor (like Coolant Temp) exceeds safety limits (> 95°C), it overrides the ML prediction to ensure a `WARNING` or `FAULT` status is triggered, prioritizing human and machine safety.

---

## 🚀 Getting Started

### 1. Environment Preparation
This project is designed with a **Portable Environment** for Windows. All dependencies are stored in the `./lib` folder. To replicate the environment:
```bash
pip install -t ./lib pandas scikit-learn streamlit joblib matplotlib seaborn fsspec huggingface_hub
```

### 2. Intellectual Initialization (Pre-training)
Before running the dashboard, you must train the agents' initial intelligence:
```bash
python model.py
```

### 3. Launching the Dashboard
```bash
streamlit run main.py
```
---

## 📊 Evaluation Metrics
In compliance with task requirements, the system prints:
 1. **Fold-by-Fold Precision**: Visible in the "Academic Validation" module.
 2. **Average Precision**: Calculated across all folds for the selected CV method.
 3. **Real-time Inference**: Tested via a randomized 10-sample debug script.