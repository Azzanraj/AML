# AML Fraud Detection Mini Project

A Flask-based **Anti-Money Laundering (AML) detection mini project** that predicts suspicious transactions from uploaded CSV files.  
The project combines **autoencoder-based feature learning**, **GraphSAGE embeddings**, **XGBoost classification**, and a simple **web interface** for batch prediction.

## Project Overview

This mini project is built to detect fraudulent or laundering-like transactions using a hybrid ML pipeline:

- **Stage A**: Autoencoder-based feature extraction
- **Stage B**: Multi-view GraphSAGE embedding learning
- **Stage C**: XGBoost fusion classification
- **Stage D**: End-to-end inference from raw CSV input

The web app allows login, CSV upload, and prediction result viewing in a table format.

## Features

- User login page
- CSV file upload for AML transactions
- Batch fraud prediction
- Fraud probability scoring
- Transaction-wise result table
- Clean Flask-based UI
- Git LFS support for large model artifacts

## Project Workflow

1. Upload an AML CSV file
2. Preprocess transaction data
3. Generate autoencoder features
4. Generate graph embeddings
5. Combine features in the fusion model
6. Predict fraud / non-fraud
7. Display results in the browser

## Model Pipeline

### Stage A — Autoencoder Feature Learning
This stage preprocesses AML transaction data, encodes categorical values, and trains an autoencoder to learn compact representations.

Generated artifacts:
- `ae_v2_features_0p2.pkl`
- `autoencoder_v2_0p2.pth`
- `ae_label_encoders_0p2.pkl`
- `ae_num_scaler_0p2.pkl`
- `ae_preproc_config_0p2.json`

### Stage B — Multi-View Graph Learning
This stage builds account and bank graphs and learns node embeddings using GraphSAGE.

Generated artifacts:
- `node_embeddings_mv_sage_0p2.npy`
- `account_embeddings_sage_0p2.npy`
- `bank_embeddings_sage_0p2.npy`
- `graph_accounts_0p2.pt`
- `graph_banks_0p2.pt`
- `graph_data_0p2.pt`

### Stage C — Fusion Classification
This stage combines AE features, graph embeddings, and uncertainty-related features to train the final classifier using XGBoost.

Generated artifacts:
- `xgboost_v4_ae_mvgnn_0p2.pkl`
- `xgb_fusion_schema_0p2.json`

### Stage D — Inference Pipeline
This stage loads the trained artifacts and performs predictions on new AML CSV files.

## Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: PyTorch, PyTorch Geometric, XGBoost, Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Frontend**: HTML, CSS
- **Graph Learning**: GraphSAGE, Multi-view Graph Networks

## Repository Structure

```text
AML/
│
├── app.py
├── aml_model.py
├── requirements.txt
│
├── artifacts/
│   ├── *.pth
│   ├── *.pkl
│   ├── *.npy
│   ├── *.pt
│   └── *.json
│
├── notebooks/
│   ├── Mini_2_StageA.ipynb
│   ├── Mini_2_StageB2(Mc__dropout).ipynb
│   ├── MIni_2_StageC2.ipynb
│   └── Mini_2_StageD2.ipynb
│
├── sample_data/
│   └── aml_sample_40_laund_0p25.csv
│
├── static/
│   └── style.css
│
└── templates/
    ├── login.html
    ├── upload.html
    └── results.html
```

## Web Interface

### Login Page
- Admin authentication
- Session-based access

Default credentials:
- **Username:** `admin`
- **Password:** `admin123`

### Upload Page
- Upload AML CSV file
- Run prediction pipeline

### Results Page
Shows:
- Total transactions
- Fraud count
- Fraud probability
- Transaction details
- Fraud prediction labels

Example result columns:
- Timestamp
- From Bank
- To Bank
- Amount Paid
- Currency
- Payment Format
- Is Laundering
- Fraud Prediction
- Fraud Probability

## Dataset

Sample file:
- `sample_data/aml_sample_40_laund_0p25.csv`

It contains:
- Transaction timestamps
- Bank accounts
- Currency types
- Payment formats
- Transaction amounts
- Laundering labels

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Azzanraj/AML.git
cd AML
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows**
```bash
.venv\Scripts\activate
```

**Linux / Mac**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

Start the Flask app:

```bash
python app.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## Git LFS

Large ML artifacts are tracked using **Git LFS**.

Install and enable LFS:

```bash
git lfs install
git lfs pull
```

Tracked file types:
- `*.npy`
- `*.pkl`
- `*.pt`
- `*.pth`

## Sample Workflow

1. Log in to the app
2. Upload the AML CSV file
3. Click **Predict**
4. View fraud results
5. Inspect suspicious transactions

## Notes

- This is a **mini project**, not a full production AML system.
- The project is designed for academic demonstration and portfolio use.
- Large model files are stored using Git LFS.

## Future Improvements

- Role-based authentication
- Real-time streaming input
- Explainable AI support
- Better dashboard UI
- REST API integration
- Cloud deployment

## Acknowledgement

This mini project was built using hybrid machine learning and graph-based techniques for AML fraud detection.
