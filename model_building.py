import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Step 1: Load Data from MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['clv_database1']
collection = db['customer_clv2']
data = pd.DataFrame(list(collection.find()))

# Step 2: Limit Dataset to 50,000 Samples
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)

# Drop the MongoDB '_id' column if present
if '_id' in data:
    data.drop('_id', axis=1, inplace=True)

# Convert categorical features to numerical
data = pd.get_dummies(data, columns=['Country'], drop_first=True)

# Define Features and Target
features = ['Quantity', 'UnitPrice', 'Year', 'Month', 'DayOfWeek', 'TimePeriod']
target = 'Customer Lifetime Value'

X = data[features]
y = data[target]

# Handle Missing Values (Fixed SettingWithCopyWarning)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Required for SVR and Stacking)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optimize Random Forest
rf_params = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)
best_rf = rf_grid.best_estimator_

# Optimize XGBoost
xgb_params = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 9]
}

xgb = XGBRegressor(random_state=42)
xgb_grid = GridSearchCV(xgb, xgb_params, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
xgb_grid.fit(X_train_scaled, y_train)
best_xgb = xgb_grid.best_estimator_

# Define Base Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': best_rf,
    'XGBoost': best_xgb,
    'Linear SVR': LinearSVR(random_state=42, max_iter=10000),
    'SVR (RBF)': SVR(kernel='rbf')
}

results = {}

# Train Individual Models
for name, model in models.items():
    if isinstance(model, (SVR, LinearSVR)):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
    print(f"{name} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Create Meta-Features (Stacking)
train_meta = np.column_stack([
    best_rf.predict(X_train_scaled),
    best_xgb.predict(X_train_scaled)
])
test_meta = np.column_stack([
    best_rf.predict(X_test_scaled),
    best_xgb.predict(X_test_scaled)
])

# Implement Stacked Regressor with Gradient Boosting as Final Estimator
stacked_model = StackingRegressor(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lr', LinearRegression())
    ],
    final_estimator=GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
)

stacked_model.fit(train_meta, y_train)
y_pred_stacked = stacked_model.predict(test_meta)

# Evaluate Stacked Model
mae_stacked = mean_absolute_error(y_test, y_pred_stacked)
mse_stacked = mean_squared_error(y_test, y_pred_stacked)
rmse_stacked = np.sqrt(mse_stacked)

results["Stacked Regressor"] = {"MAE": mae_stacked, "MSE": mse_stacked, "RMSE": rmse_stacked}
print(f"\n🔥 Stacked Regressor - MAE: {mae_stacked:.2f}, MSE: {mse_stacked:.2f}, RMSE: {rmse_stacked:.2f} 🔥")

print("\n✅ Models trained successfully!")
