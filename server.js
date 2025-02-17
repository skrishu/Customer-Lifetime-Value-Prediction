const express = require('express');
const mongoose = require('mongoose');
const multer = require('multer');
const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');
const cors = require('cors');
require('dotenv').config(); // Load environment variables

const app = express();

// Enable CORS (if needed)
app.use(cors());

// Set up MongoDB connection
mongoose.connect(process.env.MONGODB_URI, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('Connected to MongoDB'))
  .catch((error) => console.log('Error connecting to MongoDB:', error));

// Define the schema for your data
const customerSchema = new mongoose.Schema({
  customer_id: String,
  total_revenue: Number,
  total_transactions: Number,
  average_order_value: Number,
  purchase_frequency: Number,
  customer_first_purchase_date: Date,
  days_since_last_purchase: Number,
  retention_probability: Number,
});

const Customer = mongoose.model('Customer', customerSchema);

// Set up multer for file uploads
const storage = multer.memoryStorage(); // Store file in memory instead of a folder
const upload = multer({ storage: storage });

// Endpoint for handling the CSV upload and MongoDB storage
app.post('/upload', upload.single('dataset'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    const fileBuffer = req.file.buffer; // Get the file from memory
    const missingColumns = [];
    const customerData = []; // Array to hold data for MongoDB insertion

    // Parse CSV file directly from buffer
    const readableStream = require('stream').Readable.from(fileBuffer);
    
    readableStream
        .pipe(csv())
        .on('headers', (headers) => {
            console.log('CSV Headers:', headers);
            // Check if required columns are present
            const requiredColumnsMapping = {
                "CustomerID": ["Customer_Id","customer_id", "customer", "cust_id", "custid", "client_id", "user_id"],
        "Total Revenue": ["Total_Revenue","total_revenue", "revenue", "sales", "gross_revenue", "lifetime_value"],
        "Total Transactions": ["Total_Transactions","total_transactions", "num_transactions", "transaction_count", "orders", "purchases"],
        "Average Order Value (AOV)": ["Average_Order_Value","average_order_value", "aov", "mean_order_value", "order_avg"],
        "Purchase Frequency": ["Purchase_Frequency","purchase_frequency", "order_frequency", "buying_frequency", "repeat_rate"],
        "Customer First Purchase Date": ["Customer_First_Purchase_Date","customer_first_purchase_date", "first_order_date", "signup_date", "initial_purchase"],
        "Days Since Last Purchase": ["Days Since Last Purchase","days_since_last_purchase", "last_purchase_days", "recency", "time_since_last_order"],
        "Retention Probability": ["Retention_Probibility","retention_probability", "prob_alive", "customer_retention", "repeat_probability"]
  
            };

            for (const [key, columnNames] of Object.entries(requiredColumnsMapping)) {
                const columnFound = columnNames.some(col => headers.includes(col));
                if (!columnFound) {
                    missingColumns.push(key);
                }
            }
        })
        .on('data', (row) => {
            // Transform each row into a MongoDB document
            customerData.push({
                customer_id: row["customer_id"] || row["customer"],
                total_revenue: parseFloat(row["total_revenue"] || row["revenue"] || 0),
                total_transactions: parseInt(row["total_transactions"] || row["orders"] || 0),
                average_order_value: parseFloat(row["average_order_value"] || row["aov"] || 0),
                purchase_frequency: parseFloat(row["purchase_frequency"] || row["order_frequency"] || 0),
                customer_first_purchase_date: row["customer_first_purchase_date"] || row["signup_date"] || null,
                days_since_last_purchase: parseInt(row["days_since_last_purchase"] || 0),
                retention_probability: parseFloat(row["retention_probability"] || row["prob_alive"] || 0)
            });
        })
        .on('end', async () => {
            if (missingColumns.length > 0) {
                console.log('Missing columns:', missingColumns);
                return res.json({ error: `Missing required columns: ${missingColumns.join(', ')}` });
            }

            try {
                // Insert the parsed data into MongoDB
                const result = await Customer.insertMany(customerData);
                console.log('Data inserted into MongoDB:', result);
                res.json({ message: 'File uploaded and data inserted successfully' });
            } catch (err) {
                console.error('Error inserting data into MongoDB:', err);
                res.status(500).json({ error: 'Error inserting data into MongoDB.' });
            }
        })
        .on('error', (err) => {
            console.error('Error reading CSV:', err);
            res.status(500).json({ error: 'Error reading CSV file.' });
        });
});

// Start the server
app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});
