# filepath: /pharma-flask-backend/app.py
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# --- 1. Initialize the Flask App ---
app = Flask(__name__)

# --- 2. Load the AqsolDB Dataset ---
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# --- 3. Convert SMILES to Fingerprint ---
def smiles_to_fingerprint(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
        return np.array(fp).reshape(1, -1)
    except:
        return None

# --- 4. Train the Random Forest Model ---
def train_model(data):
    X = []
    y = []
    
    for index, row in data.iterrows():
        smiles = row['smiles']  # Column name is lowercase 'smiles'
        solubility = row['solubility']  # Column name is lowercase 'solubility'
        
        fingerprint = smiles_to_fingerprint(smiles)
        if fingerprint is not None:
            X.append(fingerprint[0])
            y.append(solubility)
    
    X = np.array(X)
    y = np.array(y)
    
    model = RandomForestRegressor()
    model.fit(X, y)
    
    return model

# Load the dataset and train the model
data = load_data('aqsoldb.csv')
print("Available columns in the dataset:", list(data.columns))  # Debug print
model = train_model(data)

# Save the trained model to a file
joblib.dump(model, 'solubility_model.pkl')
print("Model trained and saved successfully!")

# --- 5. Define the API Endpoint for Prediction ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'smiles' not in data:
        return jsonify({'error': 'No SMILES string provided'}), 400

    smiles = data['smiles']
    fingerprint = smiles_to_fingerprint(smiles)

    if fingerprint is None:
        return jsonify({'error': 'Invalid SMILES string'}), 400

    prediction = model.predict(fingerprint)
    return jsonify({'smiles': smiles, 'predicted_solubility': prediction[0]})

# --- 6. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)