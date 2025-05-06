from utils import db_connect
engine = db_connect()

Groundwater Level Prediction - Aquifer Luco (ACEA Smart Water Analytics)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# 1. Load and preprocess data
df = pd.read_csv("Aquifer_Luco.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date').set_index('Date')

# Filter from 2008 onwards
df = df.loc['2008-01-01':]

# Fill rainfall and temperature using forward-backward fill
rainfall_cols = [col for col in df.columns if 'Rainfall' in col]
temperature_cols = [col for col in df.columns if 'Temperature' in col]
volume_cols = [col for col in df.columns if 'Volume' in col]
target_col = 'Depth_to_Groundwater_Podere_Casetta'

df[rainfall_cols] = df[rainfall_cols].ffill().bfill()
df[temperature_cols] = df[temperature_cols].ffill().bfill()
df[volume_cols] = df[volume_cols].interpolate().ffill().bfill()
df[target_col] = df[target_col].interpolate().ffill().bfill()

# Create target lags
for lag in range(1, 4):
    df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

# Drop rows with NaNs created by lags
df_model = df.dropna()

# 2. Split data
X = df_model.drop(columns=[target_col])
y = df_model[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Train XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE (XGBoost): {mae:.4f}")
print(f"RMSE (XGBoost): {rmse:.4f}")

# 5. Plot predictions
plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test, label='Real', color='green')
plt.plot(y_test.index, y_pred, label='Prediction XGBoost', color='red', linestyle='--')
plt.title('XGBoost Prediction vs Real Values - Groundwater Depth')
plt.xlabel('Date')
plt.ylabel('Depth (m)')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# 6. Save model
joblib.dump(model, 'modelo_xgboost_luco.py.pkl')
print("Model saved as 'modelo_xgboost_luco.py.pkl'")