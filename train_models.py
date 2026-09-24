"""
SmartPure AI - Model training script
Step 1: Create a synthetic dataset (realistic rules + random noise)
Step 2: Train Random Forest Classifier  -> water quality
Step 3: Train Random Forest Regressor   -> filter remaining life
Step 4: Save models so the dashboard can use them
"""
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

np.random.seed(42)
N = 3000  # number of samples

# ---------- 1. DATASET ----------
df = pd.DataFrame({
    "ph":          np.random.normal(7.3, 1.4, N).clip(4.5, 10).round(2),      # 0-14 scale
    "tds":         (np.random.gamma(2, 200, N) + 50).clip(50, 1500).round(0), # mg/L
    "turbidity":   np.random.gamma(1.5, 2.5, N).clip(0.1, 15).round(2),     # NTU
    "daily_usage": np.random.uniform(20, 300, N).round(0),        # litres/day
    "users":       np.random.randint(1, 10, N),                   # people in house
    "filter_age":  np.random.randint(0, 400, N),                  # days since install
})

# --- Water quality label (based on WHO/BIS-style limits) ---
# 0 = Safe, 1 = Moderate, 2 = Unsafe
def quality_label(row):
    score = 0
    if row.ph < 6.5 or row.ph > 8.5: score += 1
    if row.ph < 5.5 or row.ph > 9.5: score += 1
    if row.tds > 500:  score += 1
    if row.tds > 1000: score += 1
    if row.turbidity > 5:  score += 1
    if row.turbidity > 10: score += 1
    if score == 0: return 0
    if score <= 2: return 1
    return 2

df["quality"] = df.apply(quality_label, axis=1)

# --- Filter remaining life (days) ---
# Filter can clean ~ 12000 litres. Dirty water (high TDS/turbidity) uses it up faster.
capacity = 12000 / (1 + df.tds / 1500 + df.turbidity / 20)
litres_used = df.daily_usage * df.filter_age
remaining = (capacity - litres_used) / df.daily_usage
remaining += np.random.normal(0, 5, N)          # sensor/real-world noise
df["remaining_days"] = remaining.clip(lower=0).round(0)

df.to_csv("water_data.csv", index=False)
print("Dataset saved: water_data.csv", df.shape)

# ---------- 2. WATER QUALITY MODEL (Classifier) ----------
X1 = df[["ph", "tds", "turbidity"]]
y1 = df["quality"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X1_train, y1_train)
pred1 = clf.predict(X1_test)
print("\n=== Water Quality Classifier ===")
print("Accuracy:", round(accuracy_score(y1_test, pred1), 3))
print(classification_report(y1_test, pred1, target_names=["Safe", "Moderate", "Unsafe"]))

# ---------- 3. FILTER LIFE MODEL (Regressor) ----------
features2 = ["tds", "turbidity", "daily_usage", "users", "filter_age"]
X2 = df[features2]
y2 = df["remaining_days"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X2_train, y2_train)
pred2 = reg.predict(X2_test)
print("=== Filter Life Regressor ===")
print("MAE (avg error in days):", round(mean_absolute_error(y2_test, pred2), 2))
print("R2 score:", round(r2_score(y2_test, pred2), 3))

# ---------- 4. SAVE ----------
# Save the scores so the dashboard can show them
metrics = {
    "samples": N,
    "accuracy": round(float(accuracy_score(y1_test, pred1)) * 100, 1),
    "mae": round(float(mean_absolute_error(y2_test, pred2)), 1),
    "r2": round(float(r2_score(y2_test, pred2)), 3),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

joblib.dump(clf, "quality_model.pkl")
joblib.dump(reg, "filter_model.pkl")
print("\nModels saved: quality_model.pkl, filter_model.pkl")
