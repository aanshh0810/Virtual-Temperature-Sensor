


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('measures_v2.csv')

input_features = ['i_d', 'i_q', 'motor_speed', 'u_q', 'torque', 'coolant']
target = 'stator_winding'
X = df[input_features]
y = df[target]

# === Train/Validation/Test Split: 70/15/15 ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

hist_model = HistGradientBoostingRegressor(random_state=42)
hist_model.fit(X_train, y_train)
y_pred_hist = hist_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"\n📌 {name}")
    print(f"✅ R² Score: {r2:.5f}")
    print(f"✅ RMSE: {rmse:.5f}")
    return r2, rmse

evaluate_model("HistGradientBoosting", y_test, y_pred_hist)

models = {
  
    'HistGB': y_pred_hist
}

plt.figure(figsize=(10, 6))
for name, preds in models.items():
    plt.scatter(y_test, preds, alpha=0.5, label=name)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted - Model Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 HistGradientBoosting
# ✅ R² Score: 0.88008
# ✅ RMSE: 9.93436