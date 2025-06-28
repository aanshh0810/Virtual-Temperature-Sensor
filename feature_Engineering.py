import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# Load dataset
df = pd.read_csv('measures_v2.csv')

# Select input features and target variable
input_features = ['i_d', 'i_q', 'motor_speed', 'u_q', 'torque', 'coolant']
target = 'stator_winding'

X = df[input_features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rmse = np.sqrt(np.mean((y_test - y_pred_rf) ** 2))

print("🌲 Random Forest:")
print("✅ R² Score:", r2_score(y_test, y_pred_rf))
print("✅ RMSE:", rmse)
