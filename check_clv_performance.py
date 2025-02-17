import pandas as pd
from pymongo import MongoClient

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")  # Update if needed
db = client["clv_database1"]  # Update with your database name
collection = db["customer_clv3"]  # Update with your collection name

# Retrieve Data from MongoDB (Customer & Calculated CLV)
df_yours = pd.DataFrame(list(collection.find({}, {"_id": 0, "Customer": 1, "Customer Lifetime Value": 1})))

# Load Provided Dataset (CSV)
csv_path = r"C:\Krisha\IPD\WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
df_provided = pd.read_csv(csv_path)

# Convert Customer IDs to strings & strip spaces
df_yours['Customer'] = df_yours['Customer'].astype(str).str.strip()
df_provided['Customer'] = df_provided['Customer'].astype(str).str.strip()

# Convert CLV to float for sorting
df_yours['Customer Lifetime Value'] = df_yours['Customer Lifetime Value'].astype(float)
df_provided['Customer Lifetime Value'] = df_provided['Customer Lifetime Value'].astype(float)

# Sort both datasets in **descending order** of CLV
df_yours_sorted = df_yours.sort_values(by="Customer Lifetime Value", ascending=False)
df_provided_sorted = df_provided.sort_values(by="Customer Lifetime Value", ascending=False)

# Extract **Top 10 Customers** from both datasets
top_10_yours = df_yours_sorted[["Customer", "Customer Lifetime Value"]].head(20)
top_10_provided = df_provided_sorted[["Customer", "Customer Lifetime Value"]].head(20)

# Merge to check for matches (inner join on both Customer ID & CLV)
matched = pd.merge(top_10_yours, top_10_provided, on=["Customer", "Customer Lifetime Value"], how="inner")

# Count how many rows match
num_matches = len(matched)

# Print the results
print("\n✅ **Top 10 Customers from Calculated CLV Dataset:**")
print(top_10_yours)

print("\n✅ **Top 10 Customers from Provided CLV Dataset:**")
print(top_10_provided)

print(f"\n🔍 **Number of Exact Matches in Top 10: {num_matches}**")

# Close MongoDB Connection
client.close()
