import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from PIL import Image
import io
import pandas as pd

# --- Functions and Model Loading ---

@st.cache_resource
def load_model():
    """Load the pre-trained model. Caching speeds this up."""
    return joblib.load('solubility_model.pkl')

model = load_model()

def smiles_to_molecule(smiles_string):
    """Convert SMILES to an RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(smiles_string)
    except:
        return None

def molecule_to_fingerprint(mol):
    """Generate a Morgan Fingerprint from a molecule."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
    return np.array(fp).reshape(1, -1)

def calculate_properties(mol):
    """Calculate key physicochemical properties."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    return {
        "Molecular Weight": f"{mw:.2f}",
        "LogP (Lipophilicity)": f"{logp:.2f}",
        "Hydrogen Bond Donors": h_donors,
        "Hydrogen Bond Acceptors": h_acceptors
    }

def molecule_to_image(mol):
    """Generate a PNG image of a molecule."""
    img = Draw.MolToImage(mol, size=(400, 400))
    return img


# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title('Pharma AI - Molecular Property Predictor 🔬')
# Examples section
st.write("### Try an Example")
col1, col2, col3 = st.columns(3)
if col1.button("Aspirin"):
    st.session_state.smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"
if col2.button("Caffeine"):
    st.session_state.smiles_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
if col3.button("Ibuprofen"):
    st.session_state.smiles_input = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"




# Input and Prediction section
smiles_input = st.text_input(
    'Enter a SMILES string:', 
    key='smiles_input',
    placeholder='Enter a SMILES string or click an example above'
)

if smiles_input:
    mol = smiles_to_molecule(smiles_input)
    
    if mol is None:
        st.error("❌ Invalid SMILES string. Please check your input.")
    else:
        # Create two columns for layout
        left_col, right_col = st.columns(2)
        
        # --- Left Column: Molecule Image and Properties ---
        with left_col:
            st.write("### Molecule Visualization")
            st.image(molecule_to_image(mol), use_container_width=True)
            
            st.write("### Physicochemical Properties")
            properties = calculate_properties(mol)
            st.json(properties)

        # --- Right Column: ML Prediction ---
        with right_col:
            st.write("### 🤖 AI-Powered Prediction")
            fingerprint = molecule_to_fingerprint(mol)
            prediction = model.predict(fingerprint)
            st.metric(
                label="Predicted Aqueous Solubility (logS)", 
                value=f"{prediction[0]:.4f}"
            )
            st.info("This prediction is from our AI model, indicating how well the compound dissolves in water.")

# --- Create Tabs for different functionalities ---
single_tab, batch_tab = st.tabs(["Single Molecule Prediction", "Batch Prediction"])


# --- Code for the Single Molecule Tab ---
with single_tab:
    # (All the code from your previous version goes here)
    # st.write("### Try an Example")
    # ...
    # if smiles_input:
    # ... and so on

# --- Code for the New Batch Prediction Tab (Corrected Version) -
 
 with batch_tab:
    st.header("Upload a CSV for Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Your CSV must have a column named 'smiles'", type="csv"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'smiles' not in df.columns:
            st.error("CSV file must contain a 'smiles' column.")
        else:
            with st.spinner('Processing batch... This may take a moment.'):
                # --- MODIFICATION START ---
                # Create lists to hold all our new data
                predictions = []
                mols = []
                properties_list = []
                # --- MODIFICATION END ---
                
                for smile in df['smiles']:
                    mol = smiles_to_molecule(smile)
                    mols.append(mol) # Save the molecule object
                    if mol:
                        fp = molecule_to_fingerprint(mol)
                        prediction = model.predict(fp)
                        predictions.append(prediction[0])
                        # Calculate properties for the valid molecule
                        properties_list.append(calculate_properties(mol))
                    else:
                        predictions.append(None)
                        properties_list.append({}) # Add empty dict for invalid SMILES
                
                # --- MODIFICATION START ---
                # Add all new data to the dataframe
                df['predicted_solubility'] = predictions
                
                # Create a temporary dataframe from the properties list
                props_df = pd.DataFrame(properties_list)
                
                # Join the new properties back to the original dataframe
                df = df.join(props_df)
                # --- MODIFICATION END ---
                
                st.success("Batch processing complete!")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                   label="Download results as CSV",
                   data=csv,
                   file_name='prediction_results.csv',
                   mime='text/csv',
                )
                
                # --- This plotting code will now work ---
                import plotly.express as px
                st.header("Visualize Results")
                # Drop rows with invalid SMILES before plotting
                plot_df = df.dropna().copy()
                # Convert columns to numeric for plotting
                plot_df['Molecular Weight'] = pd.to_numeric(plot_df['Molecular Weight'])

                fig = px.scatter(
                    plot_df, 
                    x="predicted_solubility", 
                    y="Molecular Weight",
                    hover_data=['smiles'],
                    title="Predicted Solubility vs. Molecular Weight"
                )
                st.plotly_chart(fig, use_container_width=True)
# --- End of Streamlit App ---
