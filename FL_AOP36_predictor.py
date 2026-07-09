import streamlit as st
import pandas as pd
from io import BytesIO
import joblib
from mordred import Calculator, descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
from rasar import calculate_descriptor
import numpy as np

files = joblib.load("input_files.joblib")
g_q = files[0]
g_r = files[1]
a_q = files[2]
a_r = files[3]
b_q = files[4]
b_r = files[5]

g_avg = 6.664
g_std = 1.170

a_avg = 6.055
a_std = 0.837

b_avg = 6.987
b_std = 1.143

models=joblib.load("model_list.joblib")
gr_model = models[0]
ar_model = models[1]
br_model = models[2]

#Descriptor Calculators
def FBT_atom_atom(mol):
    # Atom types for Fxx and Bxx
    atom_type = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "X"]
    # Atom types for Txx (exclude C, B, Si, X)
    atom_type_T = ["N", "O", "S", "P", "F", "Cl", "Br", "I"]

    def order_pair(a, b, types):
        """Return (a,b) ordered according to list 'types'"""
        return (a, b) if types.index(a) <= types.index(b) else (b, a)

    des_dict = {}
    if mol is None:
        return des_dict

    dist_mat = rdmolops.GetDistanceMatrix(mol)
    num_atoms = mol.GetNumAtoms()

    # --- initialize ---
    for d in range(1, 11):
        for j in atom_type:
            for k in atom_type:
                if atom_type.index(j) <= atom_type.index(k):
                    des_dict[f'F{d:02d}[{j}-{k}]'] = 0
                    des_dict[f'B{d:02d}[{j}-{k}]'] = 0

    # Initialize Txx descriptors (no distance bins)
    for j in atom_type_T:
        for k in atom_type_T:
            if atom_type_T.index(j) <= atom_type_T.index(k):
                des_dict[f'T({j}..{k})'] = 0

    # --- compute ---
    for i in range(num_atoms):
        sym_i = mol.GetAtomWithIdx(i).GetSymbol()
        if sym_i == "H":  # skip hydrogens
            continue
        if sym_i not in atom_type:
            sym_i = "X"

        for j in range(i + 1, num_atoms):
            sym_j = mol.GetAtomWithIdx(j).GetSymbol()
            if sym_j == "H":  # skip hydrogens
                continue
            if sym_j not in atom_type:
                sym_j = "X"

            d = int(dist_mat[i, j])

            # --- Fxx & Bxx ---
            if 1 <= d <= 10:
                a, b = order_pair(sym_i, sym_j, atom_type)
                des_dict[f'F{d:02d}[{a}-{b}]'] += 1
                des_dict[f'B{d:02d}[{a}-{b}]'] = 1

            # --- Txx ---
            if d > 1 and sym_i in atom_type_T and sym_j in atom_type_T:
                a, b = order_pair(sym_i, sym_j, atom_type_T)
                des_dict[f'T({a}..{b})'] += d

    return des_dict

def mordred_descriptors(mol):
    calc = Calculator(descriptors, ignore_3D=True)
    desc = calc(mol)
    return desc.asdict()

def main_des_cal(input_data):
    def process_smiles(smiles_list, add_hs=True, kekulize=False):
        mols = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                mols.append(None)
                continue

            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    mols.append(None)
                    continue
            if add_hs:
                mol = Chem.AddHs(mol)

            mols.append(mol)

        return mols


    input_data["SMILES"] = input_data["SMILES"].apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), canonical=True))

    smiles = input_data["SMILES"].values.tolist()

    smiles1 = process_smiles(smiles_list=smiles)

    calc_desc = []

    for mol in smiles1:
        if mol is None:
            calc_desc.append(None)
            continue

        des = FBT_atom_atom(mol)
        calc_desc.append(des)
    
    calc_desc1 = []

    for mol in smiles1:
        if mol is None:
            calc_desc1.append(None)
            continue

        des1 = mordred_descriptors(mol)
        calc_desc1.append(des1)

    odf1 = pd.DataFrame(calc_desc, index=input_data.index)
    odf2 = pd.DataFrame(calc_desc1, index=input_data.index)
    odf2 = odf2.apply(pd.to_numeric, errors="coerce").fillna(0)
    odf2["Lipinski"] = odf2["Lipinski"].astype(int)
    odf2["GhoseFilter"] = odf2["GhoseFilter"].astype(int)

    fop = pd.concat([odf1, odf2], axis=1)

    return fop

#standardization
def stand(df1, df2):
    avg = df1.mean()
    stdev = df1.std()
    std_df1 = (df1-avg)/stdev
    std_df2 = (df2-avg)/stdev
    return std_df1, std_df2


#applicability domain analysis

def leverage_calculator(data1: pd.DataFrame, data2: pd.DataFrame):
    '''
    This function calculates the leverage of training and test sets. \n
    data1: training set descriptor matrix (n_samples x n_features) \n
    data2: test set descriptor matrix (m_samples x n_features)\n
    Returns:
        leverage_tr_df: Leverage values for training data \n
        leverage_te_df: Leverage values for test data 
    '''

    data3 = data1.copy()
    data4 = data2.copy()

    if data3.shape[1] != data4.shape[1]:
        raise ValueError("Input files must have the same number of columns/features.")
    
    p_val = len(data3.columns)
    n_val = len(data3)
    h_star = 3*(p_val+1)/(n_val)
    print(h_star)
    
    data3.insert(0, "dummey", 1)
    data4.insert(0, "dummey", 1)
    X = data3.values  # shape (n, p)
    X_test = data4.values  # shape (m, p)

    # (X^T X)^-1
    XtX_inv = np.linalg.inv(X.T @ X)

    # Leverage for training data: diag(X @ (X^T X)^-1 @ X^T)
    H_train = X @ XtX_inv @ X.T
    leverage_tr = np.diag(H_train)

    # Leverage for test data: row-wise x_i @ (X^T X)^-1 @ x_i^T
    leverage_te = np.array([x @ XtX_inv @ x.T for x in X_test])

    leverage_tr_df = pd.DataFrame(leverage_tr, columns=["Leverage Value"], index=data3.index)
    leverage_te_df = pd.DataFrame(leverage_te, columns=["Leverage Value"], index=data4.index)

    leverage_tr_df["AD Status"] = np.where(leverage_tr_df["Leverage Value"]>=h_star, "Outside AD", "Inside AD")
    leverage_te_df["AD Status"] = np.where(leverage_te_df["Leverage Value"]>=h_star, "Outside AD", "Inside AD")

    return leverage_te_df

# Page configuration
st.set_page_config(
    page_title="FL_AOP36_Predictor",
    page_icon="dtc.jpg",
    layout="wide"
)

# Custom CSS for footer and styling
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
options = ["Home", "Manual", "Contact Us", "References"]
selected = st.sidebar.radio("Go to", options)

# Sample data for download
def get_sample_excel():
    sample_df = files[-1]
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        sample_df.to_excel(writer, index=True, sheet_name='SampleData')
    output.seek(0)
    return output

# Main content based on selection
if selected == "Home":
    st.title("FL_AOP36_Predictor")
    st.markdown("Predict the potential of chemicals for the development of NAFLD")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an .xlsx file",
        type=["xlsx"],
        help="Upload an Excel file to process"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    
    if submit_button:
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file, index_col=0)
                st.success("File uploaded successfully!")
                st.subheader("Preview of Uploaded Data")
                st.dataframe(df, use_container_width=True)
                calc_desc = main_des_cal(input_data=df)
                #gamma
                gamma_desc  = calc_desc[g_q.iloc[:,:-1].columns.tolist()].copy()
                rasar_desc_tr, rasar_desc_te = calculate_descriptor(df1=g_q, df2=gamma_desc, method="Laplacian Kernel", ctc=10, gamma=0.75, data_fusion=True)
                rasra_desc_sel_te = rasar_desc_te[['RA_function']].copy()
                __, rasra_desc_sel_te_gamma = stand(df1=g_r.iloc[:,:-1], df2=rasra_desc_sel_te)
                gr_pred = gr_model.predict(rasra_desc_sel_te_gamma)
                gr_pred1 = np.round((gr_pred*g_std)+g_avg,3)
                gr_pred_df = pd.DataFrame(gr_pred1, columns=["Predicted Value (gamma)"], index= df.index)
                gamma_AD = leverage_calculator(data1=g_r.iloc[:,:-1], data2=rasra_desc_sel_te)
                gr_pred_df["AD Status (gamma)"] = gamma_AD["AD Status"]
                #alpha
                alpha_desc = calc_desc[a_q.iloc[:,:-1].columns.tolist()].copy()
                rasar_desc_tr1, rasar_desc_te1 = calculate_descriptor(df1=a_q, df2=alpha_desc, method="Laplacian Kernel", ctc=7, gamma=0.5, data_fusion=True)
                rasra_desc_sel_te1 = rasar_desc_te1[a_r.iloc[:,:-1].columns.tolist()].copy()
                __, rasra_desc_sel_te_alpha= stand(df1=a_r.iloc[:,:-1], df2=rasra_desc_sel_te1)
                ar_pred = ar_model.predict(rasra_desc_sel_te_alpha)
                ar_pred1 = np.round((ar_pred*a_std)+a_avg,3)
                ar_pred_df = pd.DataFrame(ar_pred1, columns=["Predicted Value (alpha)"], index= df.index)
                alpha_AD = leverage_calculator(data1=a_r.iloc[:,:-1], data2=rasra_desc_sel_te1)
                ar_pred_df["AD Status (alpha)"] = alpha_AD["AD Status"]
                #beta
                beta_desc = calc_desc[b_q.iloc[:,:-1].columns.tolist()].copy()
                rasar_desc_tr2, rasar_desc_te2 = calculate_descriptor(df1=b_q, df2=beta_desc, method="Laplacian Kernel", ctc=7, gamma=0.5, data_fusion=True)
                rasra_desc_sel_te2 = rasar_desc_te2[['RA_function']].copy()
                __, rasra_desc_sel_te_beta = stand(df1=b_r.iloc[:,:-1], df2=rasra_desc_sel_te2)
                br_pred = br_model.predict(rasra_desc_sel_te_beta)
                br_pred1 = np.round((br_pred*b_std)+b_avg,3)
                br_pred_df = pd.DataFrame(br_pred1, columns=["Predicted Value (beta)"], index= df.index)
                beta_AD = leverage_calculator(data1=b_r.iloc[:,:-1], data2=rasra_desc_sel_te2)
                br_pred_df["AD Status (beta)"] = beta_AD["AD Status"]
                #final
                final_df = pd.concat([gr_pred_df, ar_pred_df, br_pred_df], axis=1)
                st.subheader("Results")
                st.dataframe(final_df, use_container_width=True)
                
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.warning("Please upload an .xlsx file before submitting.")


    # Footer with sample download - appears on home page
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown("### Sample File")
        st.caption("Download a sample Excel file to test the app")

    with footer_col2:
        sample_file = get_sample_excel()
        st.download_button(
            label="📥 Download Sample.xlsx",
            data=sample_file,
            file_name="sample_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

elif selected == "Manual":
    st.title("📖 User Manual")
    st.markdown("""
    ### How to Use This App
    
    1. **Input File**
       - Upload your .xlsx file using the file uploader.
       - Click the Submit button to process and preview your data.
       - The input file should containt ID column (first column) and SMILES column (second column).
       - The column contains SMILES which should also be named as "SMILES".
    
    2. **Prediction**
       - Here Prediction for a compound is made using three q-RASAR models (gamma, alpha, beta).
       - gamma: Linear Support Vector Machine (C = 15)
       - alpha: Linear Support Vector Machine (C = 1)
       - beta: Linear Support Vector Machine (C = 5)
    
    3. **Applicability Domain**
       - This app also perform applicability domain analysis for a new compound using leverage approach. 
                
    4. **Download Sample**
       - Use the sample file from the footer to test the app.
    
    **Tips:**
    - Ensure your Excel file has proper structure.
    - Large files may take longer to process.
                
    ### Data Processing Best Practices
    1. Clean your data before upload.
    2. Use consistent column naming.
    3. Check for missing values.
    """)

elif selected == "Contact Us":
    st.markdown("## 📌 Contact & Support")
    st.write("Get in touch with the development team.")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### 👩‍🔬 Principal Investigator")
        st.write("Prof. Kunal Roy")
        st.write("Drug Theoretics and Cheminformatics (DTC) Laboratory")
        st.write("Department of Pharmaceutical Technology")
        st.write("Email: kunal.roy@jadavpuruniversity.in")

    with col_2:
        st.markdown("### 💻 Software Developer")
        st.write("Souvik Pore")
        st.write("Drug Theoretics and Cheminformatics (DTC) Laboratory")
        st.write("Department of Pharmaceutical Technology")
        st.write("Email: souvikpore123@gmail.com")
    
    other_info = [
            ("University",    "Jadavpur University"),
            ("Department",    "Department of Pharmaceutical Technology"),
            ("Address",       "Jadavpur University, Kolkata 700 032, India"),
            ("Website",       "https://sites.google.com/jadavpuruniversity.in/dtc-lab-software/home"),
        ]
    
    for title1, content1 in other_info:
        with st.expander(f"{title1}", expanded=True):
            st.markdown(content1)

elif selected == "References":
    st.title("📚 References")
    st.markdown("""
    ### Useful References:
    
    - Pore, S. and Roy, K., 2025.
        “intelligent Read Across (iRA)” — A tool for read-across-based toxicity prediction of nanoparticles.
        Computational and Structural Biotechnology Journal.
        https://doi.org/10.1016/j.csbj.2025.07.032
        
    - Banerjee, A. and Roy, K., 2022. 
        First report of q-RASAR modeling toward an approach of easy interpretability and efficient transferability. 
        Molecular Diversity.
        https://doi.org/10.1007/s11030-022-10478-6
    """)


