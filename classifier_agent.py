import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

class ClassifierAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def run_cv_task(self, X, y, cv_type):
        if cv_type == 'kfold': cv = KFold(n_splits=5, shuffle=True, random_state=42)
        elif cv_type == 'stratified': cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        else: cv = LeaveOneOut()

        precisions = []
        # Perulangan manual sesuai Poin B.4
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            # Poin B.5: Print presisi tiap fold (macro agar support 3-kelas)
            score = precision_score(y_test, y_pred, average='macro', zero_division=0)
            precisions.append(score)
            
            # Limit print untuk LOOCV agar tidak lag
            if cv_type != 'loocv' or i < 5:
                yield f"Fold {i+1} Precision: {score:.4f}"
            
            if cv_type == 'loocv' and i >= 99: break 
            
        avg_precision = np.mean(precisions)
        yield avg_precision