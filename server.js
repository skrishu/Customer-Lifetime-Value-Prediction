require('dotenv').config(); // Load environment variables at the beginning
const express = require('express');
const cors = require('cors'); 
const multer = require('multer');
const mongoose = require('mongoose');
const csv = require('csv-parser');
const fs = require('fs');
const path = require('path');

// Load required columns from schema.json
const schemaPath = path.join(__dirname, 'schema.json');
const schema = JSON.parse(fs.readFileSync(schemaPath, 'utf8'));

const app = express();
const PORT = process.env.PORT || 3000;
const MONGO_URI = process.env.MONGODB_URI; 

app.use(cors());
app.use(express.json());

// Ensure the uploads directory exists
const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR);
}

// MongoDB connection
mongoose.connect(MONGO_URI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Connected to MongoDB'))
    .catch(err => {
        console.error('Failed to connect to MongoDB:', err);
        process.exit(1); // Exit the application if MongoDB connection fails
    });

// Set up multer for file uploads
const upload = multer({ dest: UPLOAD_DIR });

// Define the MongoDB schema and model
const DatasetSchema = new mongoose.Schema({
    InvoiceNo: String,
    StockCode: String,
    Quantity: Number,
    InvoiceDate: Date,
    UnitPrice: Number,
    CustomerID: String,
    Country: String,
    CustomerLifetimeValue: Number,
});

const Dataset = mongoose.model('Dataset', DatasetSchema);

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint to handle file uploads
app.post('/upload', upload.single('dataset'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded.' });
    }

    const filePath = path.join(UPLOAD_DIR, req.file.filename);
    const results = [];

    // Process the uploaded CSV file
    fs.createReadStream(filePath)
        .pipe(csv())
        .on('data', (row) => {
            // Validate data using required columns from schema.json
            const isValid = schema.required_columns.every(col => row[col] !== undefined && row[col] !== null);
            if (isValid) {
                results.push(new Dataset({
                    InvoiceNo: row.InvoiceNo,
                    StockCode: row.StockCode,
                    Quantity: parseInt(row.Quantity, 10),
                    InvoiceDate: new Date(row.InvoiceDate),
                    UnitPrice: parseFloat(row.UnitPrice),
                    CustomerID: row.CustomerID,
                    Country: row.Country,
                    CustomerLifetimeValue: parseFloat(row["Customer Lifetime Value"]),
                }));
            }
        })
        .on('end', async () => {
            try {
                await Dataset.insertMany(results);
                res.json({ message: 'File uploaded and data saved to MongoDB successfully!' });
            } catch (err) {
                console.error('Error saving to MongoDB:', err);
                res.status(500).json({ error: 'Error saving data to MongoDB' });
            }
        })
        .on('error', (err) => {
            console.error('Error processing file:', err);
            res.status(500).json({ error: 'Error processing file' });
        });
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
