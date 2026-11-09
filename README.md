# Intelligent Network Intrusion Detection System

A beginner-friendly coursework starter project using:
- Linear Regression (regression score converted to binary classification at 0.5)
- Random Forest Classifier
- Neural Network using Scikit-learn MLPClassifier
- Flask web dashboard
- TCP socket client/server simulation

## 1. Requirements
Install Python 3.10 or newer and VS Code. Open this folder in VS Code.

## 2. Create the dataset folder
Create a folder called `data` and place the two supplied CSV datasets inside it:
- `data/train.csv`
- `data/test.csv`

The CSV files should have column headers. They must include `label` (0 = normal, 1 = attack) or `attack_type` (normal versus attack name). If your original data is in another format, convert it to CSV first.

The model uses all suitable feature columns but excludes `label` and `attack_type` to prevent target leakage.

## 3. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

## 4. Install libraries
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Train and evaluate all models
```bash
python train_models.py
```
The terminal prints accuracy, precision, recall, F1-score, confusion matrix and classification report for each model. Trained pipelines are saved in `models/`.

## 6. Start the web dashboard
After training:
```bash
python app.py
```
Open this address in your browser:
`http://127.0.0.1:5000`

You can paste one JSON traffic record or upload a CSV. CSV upload predicts the first row.

## 7. Run the socket simulation
Open a second VS Code terminal:
```bash
python socket_server.py
```
Open a third terminal:
```bash
python socket_client.py
```
The client sends a JSON record to the local TCP server. The server returns predictions from all three models.

You can also send a JSON file:
```bash
python socket_client.py sample_record.json
```

## 8. Important notes for the coursework
- Linear Regression is not normally the preferred algorithm for a binary classification problem. It is included because the brief requests it; the 0.5 threshold turns its numeric output into a class.
- Random Forest and MLP are classification models.
- Use the supplied independent test dataset for final evaluation. Do not train on the test data.
- Compare precision, recall and F1-score as well as accuracy, especially if normal and attack records are imbalanced.
- This socket component is a local simulation: it classifies JSON records sent by a client. It does not capture live packets from a real network.
- Do not expose the Flask development server or socket listener to the public internet.
