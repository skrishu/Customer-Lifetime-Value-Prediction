import pandas as pd
import numpy as np
import json
import difflib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load Schema JSON
schema_path = r'C:\Krisha\IPD\CLV Model Building\schema.json'
with open(schema_path, 'r') as f:
    schema = json.load(f)

required_columns = schema.get("required_columns", [])

# Connect to MongoDB and Fetch Data
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['clv_database1']
    collection = db['customer_clv7']  # Update with the correct collection name
    
    # Fetch data from MongoDB
    data = pd.DataFrame(list(collection.find()))
    
    # Drop MongoDB's default '_id' column if present
    if '_id' in data.columns:
        data.drop(columns=['_id'], inplace=True)
    print("✅ Data fetched from MongoDB")
except Exception as e:
    print(f"❌ MongoDB Fetch Error: {e}")
    exit()

# Standardize Column Names
data.columns = data.columns.str.strip()

# Function to map required schema columns dynamically
def map_columns(schema_columns, dataframe):
    column_mapping = {}
    for required_col in schema_columns:
        matched_col = difflib.get_close_matches(required_col, dataframe.columns, n=1, cutoff=0.5)
        if matched_col:
            column_mapping[required_col] = matched_col[0]
    return column_mapping

column_mapping = map_columns(required_columns, data)
data.rename(columns=column_mapping, inplace=True)

# Identify & rename customer ID column
customer_id_col = difflib.get_close_matches('CustomerID', data.columns, n=1, cutoff=0.5)
if customer_id_col:
    data.rename(columns={customer_id_col[0]: 'CustomerID'}, inplace=True)

# Debugging Step: Print Available Columns
print("✅ Available Columns:", data.columns)

# Check if necessary columns are available and compute them if not
if 'Repeat Count' not in data.columns:
    data['Repeat Count'] = data['CustomerID'].map(data['CustomerID'].value_counts())
    print("✅ 'Repeat Count' column created.")

if 'Days Since Last Purchase' not in data.columns:
    data['Days Since Last Purchase'] = 1  # Default if not available
    print("✅ 'Days Since Last Purchase' column created.")

if 'Total Revenue' not in data.columns:
    data['Total Revenue'] = data['Revenue']  # Assuming 'Revenue' is available
    print("✅ 'Total Revenue' column created.")

if 'Total Transactions' not in data.columns:
    # Use 'Repeat Count' to estimate the total transactions if missing
    data['Total Transactions'] = data['Repeat Count']
    print("✅ 'Total Transactions' column created.")

# Now aggregate data by CustomerID
data_aggregated = data.groupby('CustomerID', as_index=False).agg({
    'Total Revenue': 'sum',
    'Days Since Last Purchase': 'min',
    'Repeat Count': 'sum',
    'Total Transactions': 'sum'
})

# Print the result to verify the aggregation
print("✅ Aggregated Data:")
print(data_aggregated.head())

# Ensure 'Total Revenue' and 'Days Since Last Purchase' are filled
data_aggregated['Total Revenue'] = data_aggregated['Total Revenue'].fillna(0)
data_aggregated['Days Since Last Purchase'] = data_aggregated['Days Since Last Purchase'].fillna(0)

# Compute other necessary columns
data_aggregated['Retention Probability'] = 0.75  # Default value

# Calculate CLV (Customer Lifetime Value)
discount_rate = 0.05
data_aggregated["CLV_k(x)"] = np.where(
    (1 - discount_rate) == 0, 
    data_aggregated["Total Revenue"] * data_aggregated["Retention Probability"],  
    data_aggregated["Total Revenue"] * data_aggregated["Retention Probability"] / discount_rate
)

# Scale Features using MinMaxScaler
scaler = MinMaxScaler()
data_aggregated['Purchase Frequency Scaled'] = scaler.fit_transform(data_aggregated[['Total Transactions']].fillna(0))
data_aggregated['Recency Scaled'] = scaler.fit_transform(data_aggregated[['Days Since Last Purchase']].fillna(0))

# Load the pre-trained Random Forest model
model_path = r'C:\Krisha\IPD\CLV Model Building\stacked_regressor_model.pkl'  # Update the path as needed
model = joblib.load(model_path)

# Select the relevant features for prediction (adjust as needed based on your training)
model_features = data_aggregated[['Purchase Frequency Scaled', 'Recency Scaled']]  # Assuming these were the features used during training

# Predict retention probability using the trained model
predicted_retention_prob = model.predict(model_features)

# Add predicted retention probability to the aggregated data
data_aggregated['Predicted Retention Probability'] = predicted_retention_prob

# Compute Final CLV using the model-predicted retention probability
data_aggregated['Customer Lifetime Value'] = (
    data_aggregated['Purchase Frequency Scaled'] * data_aggregated['CLV_k(x)'] + data_aggregated['Total Transactions'] * data_aggregated['Predicted Retention Probability']
)

# Ensure CLV is float before storing in MongoDB
data_aggregated["Customer Lifetime Value"] = data_aggregated["Customer Lifetime Value"].astype(float)

# Debugging prints to check computed values
print("📊 CLV Calculation Summary:")
print(data_aggregated[['CustomerID', 'Total Revenue', 'Retention Probability', 'CLV_k(x)', 
                       'Purchase Frequency Scaled', 'Total Transactions', 'Predicted Retention Probability', 'Customer Lifetime Value']].head())

# Store in MongoDB
try:
    collection_output = db['customer_clv14']
    collection_output.insert_many(data_aggregated.to_dict('records'))
    print("✅ Data insertion into MongoDB complete! 🎉")
except Exception as e:
    print(f"❌ MongoDB Insertion Error: {e}")
