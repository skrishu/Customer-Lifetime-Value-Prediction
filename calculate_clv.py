import pandas as pd
import numpy as np
import json
import difflib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler

# Load Schema JSON
schema_path = r'C:\Krisha\IPD\CLV Model Building\schema.json'
with open(schema_path, 'r') as f:
    schema = json.load(f)

required_columns = schema.get("required_columns", [])

# Load Dataset
file_path = r"C:\Krisha\IPD\AutoInsurance.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
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

# Ensure required columns exist
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"❌ Missing Columns: {missing_cols}")

# Convert date columns safely
date_cols = ['Customer First Purchase Date']
for col in date_cols:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

# Compute CLV Metrics
if 'frequency' in data.columns:
    data['Total Transactions'] = data['frequency'] * 0.25
else:
    if 'Repeat Count' not in data.columns:
        data['Repeat Count'] = data['CustomerID'].map(data['CustomerID'].value_counts())
    data['Total Transactions'] = data['Repeat Count'] * 0.25

# Ensure 'Total Revenue' exists
if 'Total Revenue' not in data.columns:
    data['Total Revenue'] = 0  # Default to 0 if missing

# Handle missing 'Days Since Last Purchase'
if 'Days Since Last Purchase' in data.columns:
    data['Days Since Last Purchase'] = data['Days Since Last Purchase'].fillna(1)
else:
    data['Days Since Last Purchase'] = 1  # Default value

# Avoid division by zero
data['Average Order Value (AOV)'] = data['Total Revenue'] / data['Total Transactions'].replace(0, np.nan)
data['Purchase Frequency'] = data['Total Transactions'] / data['Days Since Last Purchase'].replace(0, np.nan)
data['Retention Probability'] = 0.75  # Default value

# Compute CLV Formula
discount_rate = 0.05
data["CLV_k(x)"] = np.where(
    (1 - discount_rate) == 0, 
    data["Total Revenue"] * data["Retention Probability"],  
    data["Total Revenue"] * data["Retention Probability"] / (1 - discount_rate)
)

# Scale Features using MinMaxScaler
scaler = MinMaxScaler()
data['Purchase Frequency Scaled'] = scaler.fit_transform(data[['Purchase Frequency']].fillna(0))
data['Recency Scaled'] = scaler.fit_transform(data[['Days Since Last Purchase']].fillna(0))

# Compute Final CLV
data['Customer Lifetime Value'] = (
    data['Purchase Frequency Scaled'] * data['CLV_k(x)'] + data['Total Transactions']
)

# Ensure CLV is float before storing in MongoDB
data["Customer Lifetime Value"] = data["Customer Lifetime Value"].astype(float)

# Debugging prints to check computed values
print("📊 CLV Calculation Summary:")
print(data[['CustomerID', 'Total Revenue', 'Retention Probability', 'CLV_k(x)', 
           'Purchase Frequency Scaled', 'Total Transactions', 'Customer Lifetime Value']].head())

# Store in MongoDB
try:
    client = MongoClient('mongodb://localhost:27017/')
    db = client['clv_database1']
    collection = db['customer_clv12']
    collection.insert_many(data.to_dict('records'))
    print("✅ Data insertion into MongoDB complete! 🎉")
except Exception as e:
    print(f"❌ MongoDB Insertion Error: {e}")
