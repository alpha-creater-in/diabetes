# train_model.py

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("DDataset.csv")

IMPORTANT_COLUMNS = [
    'age', 'bmi', 'glucose', 'blood_pressure',
    'cholesterol', 'insulin',
    'family_history', 'stress_level',
    'sleep_hours'
]

df = df[IMPORTANT_COLUMNS].copy()
df.drop_duplicates(inplace=True)

# ==========================================
# Risk Score Generation
# ==========================================
risk_score = (
    0.04 * df['age'] +
    0.08 * df['bmi'] +
    0.09 * df['glucose'] +
    0.05 * df['blood_pressure'] +
    0.04 * df['cholesterol'] +
    0.07 * df['insulin'] +
    1.5  * df['family_history'] +
    0.30 * df['stress_level'] -
    0.35 * df['sleep_hours']
)

risk_score = (risk_score - risk_score.mean()) / risk_score.std()
probability = 1 / (1 + np.exp(-1.5 * risk_score))

df['diabetes'] = (np.random.rand(len(df)) < probability).astype(int)

# ==========================================
# Train/Test Split
# ==========================================
X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Model Pipeline
# ==========================================
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42
    ))
])

model_pipeline.fit(X_train, y_train)

# ==========================================
# Evaluation
# ==========================================
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

# ==========================================
# Save Model
# ==========================================
joblib.dump(model_pipeline, "diabetes_model.pkl")
print("✅ Model saved successfully!")
