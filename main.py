# === Install required packages (only once) ===
# pip install lightgbm catboost xgboost scikit-learn pandas matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# === Load Dataset (sample 50% to reduce memory use) ===
df = pd.read_csv('measures_v2.csv')

# === Define Features and Target ===
input_features = ['i_d', 'i_q', 'motor_speed', 'u_q', 'torque', 'coolant']
target = 'stator_winding'
X = df[input_features]
y = df[target]

print("model running")

# === Split Data ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Define Models ===
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42),
    'HistGB': HistGradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
}

results = {}
predictions = {}

# === Create output directory ===
os.makedirs("model_outputs", exist_ok=True)

# === Evaluate Function ===
def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    predictions[name] = y_pred
    return r2

# === Train, Evaluate, and Save Best Model ===
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = evaluate_model(name, y_test, y_pred)
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        joblib.dump(best_model, 'model_outputs/best_rf_model.pkl')

# === Save Metrics ===
pd.DataFrame(results).T.to_csv("model_outputs/model_metrics.csv")

# === Save Results to TXT ===
with open("model_outputs/model_results.txt", "w", encoding="utf-8") as f:
    f.write("Model Evaluation Results\n")
    f.write("=" * 40 + "\n\n")
    for name, metrics in results.items():
        f.write(f"{name}\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.5f}\n")
        f.write("\n")
    f.write(f"Best Model: {type(best_model).__name__}\n")
    f.write("Model saved as: best_rf_model.pkl\n")

# === Feature Importance and Cumulative Plot ===
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features_sorted = [input_features[i] for i in indices]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=features_sorted)
        plt.title(f"{name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"model_outputs/{name}_feature_importance.png")
        plt.close()

        # Cumulative
        cum_importance = np.cumsum(importances[indices])
        plt.figure()
        plt.plot(range(len(cum_importance)), cum_importance, marker='o')
        plt.title(f"{name} Cumulative Feature Importance")
        plt.xlabel("Number of Features")
        plt.ylabel("Cumulative Importance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"model_outputs/{name}_cumulative_importance.png")
        plt.close()

# === ROC Curve for Random Forest (Binarized Target) ===
y_binary = (y_test > y_test.median()).astype(int)
y_prob_rf = models['Random Forest'].predict(X_test)
fpr, tpr, _ = roc_curve(y_binary, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Binarized Target)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("model_outputs/roc_curve_rf.png")
plt.close()

# === Actual vs Predicted Line Plot ===
plt.figure(figsize=(10, 6))
for name, preds in predictions.items():
    plt.plot(y_test.values[:300], preds[:300], label=name, alpha=0.7)  # Subsampled for readability
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted - Model Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_outputs/actual_vs_predicted.png")
plt.close()


# === Actual vs Predicted Plot ===
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:300], label='Actual', color='black', linewidth=2)
plt.plot(y_pred[:300], label='Predicted (RF)', color='green', linewidth=2)
plt.title("Actual vs Predicted - Random Forest")
plt.xlabel("Sample Index")
plt.ylabel("Stator Winding Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_actual_vs_predicted_only.png")
plt.close()

# === Residuals Plot ===
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals Plot - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_residuals_plot.png")
plt.close()

# === Regression Line Plot ===
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3}, line_kws={"color": "red"})
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Regression Line - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_regression_line.png")
plt.close()

print("✅ RF-only plots saved to 'model_outputs/' as:\n- rf_actual_vs_predicted_only.png\n- rf_residuals_plot.png\n- rf_regression_line.png")


# === Line Residuals Plot (No Scatter) ===
residuals = y_test.values - y_pred
plt.figure(figsize=(10, 5))
plt.plot(residuals[:300], color='purple', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title("Residuals Over Samples - Random Forest")
plt.xlabel("Sample Index")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_residuals_line.png")
plt.close()





# === Heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(df[input_features + [target]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("model_outputs/correlation_matrix.png")
plt.close()

# === RMSE Comparison Bar Chart ===
metrics_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=metrics_df)
plt.title("RMSE Comparison Across Models")
plt.tight_layout()
plt.savefig("model_outputs/rmse_bar_comparison.png")
plt.close()

# === Final Message ===
print("✅ All models trained, evaluated, and visualizations saved in 'model_outputs/'.")
