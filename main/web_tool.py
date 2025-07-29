import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import base64
from pathlib import Path
import io
import pickle

# Set page config
st.set_page_config(
    page_title="CT-Index & Rut Depth Predictor",
    page_icon="🛣️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding-top: 2rem;}
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }
    .results-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    h1 {
        color: #1e3a8a;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛣️ CT-Index & Average Rut Depth Prediction Tool")
st.markdown("""
Upload your Excel file to get instant predictions for **CT-Index** and **Average Rut Depth** values.
This tool uses pre-trained machine learning models optimized for accurate predictions.
""")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.ct_model = None
    st.session_state.rut_model = None
    st.session_state.minmax_scaler = None
    st.session_state.feature_columns = None
    st.session_state.predictions_made = False
    st.session_state.metadata = {}

# Function to load model from uploaded file
def load_model_from_file(uploaded_model_file):
    """Load model from uploaded file"""
    try:
        # Try different loading methods
        try:
            # First try joblib
            models_data = joblib.load(uploaded_model_file)
        except:
            # If joblib fails, try pickle
            uploaded_model_file.seek(0)  # Reset file pointer
            models_data = pickle.load(uploaded_model_file)
        
        # Extract model components
        ct_model = models_data.get('ct_model')
        rut_model = models_data.get('rut_model')
        minmax_scaler = models_data.get('minmax_scaler')
        feature_columns = models_data.get('feature_columns')
        metadata = models_data.get('metadata', {})
        
        # Validate all required components exist
        if ct_model is None or rut_model is None or feature_columns is None:
            st.error("Model file is missing required components (ct_model, rut_model, or feature_columns)")
            return None, None, None, None, None
            
        return ct_model, rut_model, minmax_scaler, feature_columns, metadata
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure your model file is a valid .pkl file created with joblib or pickle")
        return None, None, None, None, None

# Model upload section
st.header("📤 Step 1: Upload Model File")

uploaded_model = st.file_uploader(
    "Choose your model file (.pkl)",
    type=['pkl'],
    help="Upload the pre-trained model file containing CT-Index and Rut Depth models"
)

if uploaded_model is not None:
    with st.spinner("Loading model..."):
        ct_model, rut_model, minmax_scaler, feature_columns, metadata = load_model_from_file(uploaded_model)
        
        if ct_model is not None:
            st.session_state.ct_model = ct_model
            st.session_state.rut_model = rut_model
            st.session_state.minmax_scaler = minmax_scaler
            st.session_state.feature_columns = feature_columns
            st.session_state.metadata = metadata
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")


# Display model information in sidebar
if st.session_state.model_loaded:
    with st.sidebar:
        st.header("📊 Model Information")
        st.success("✅ Model loaded successfully!")
        
        if metadata:
            st.write(f"**Model Type:** {metadata.get('model_type', 'Random Forest')}")
            # st.write(f"**Training Date:** {metadata.get('training_date', 'N/A')}")
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            ct_metrics = metadata.get('ct_metrics', {})
            rut_metrics = metadata.get('rut_metrics', {})
            
            st.write("**CT-Index Model:**")
            st.write(f"- R² Score: {ct_metrics.get('r2_test', 'N/A'):.4f}")
            st.write(f"- MAE: {ct_metrics.get('mae_test', 'N/A'):.4f}")
            
            st.write("**Rut Depth Model:**")
            st.write(f"- R² Score: {rut_metrics.get('r2_test', 'N/A'):.4f}")
            st.write(f"- MAE: {rut_metrics.get('mae_test', 'N/A'):.4f}")
        
        st.subheader("Required Features")
        for i, feature in enumerate(feature_columns, 1):
            st.write(f"{i}. {feature}")

# Main prediction interface
if st.session_state.model_loaded:
    # File upload section
    st.header("📁 Upload Data for Prediction")
    
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help=f"File must contain these columns: {', '.join(feature_columns)}"
    )
    
    if uploaded_file is not None:
        # Load the Excel file
        try:
            df = pd.read_excel(uploaded_file)
            
            # Display file information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Check for required features
            missing_features = [f for f in feature_columns if f not in df.columns]
            
            if missing_features:
                st.error(f"❌ Missing required columns: {', '.join(missing_features)}")
                st.info("Please ensure your Excel file contains all the required features listed in the sidebar.")
                
                # Show available columns
                with st.expander("Available columns in your file"):
                    st.write(list(df.columns))
            else:
                st.success("✅ All required features found!")
                
                # Display data preview
                with st.expander("📋 Data Preview (First 10 rows)"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Prediction button
                if st.button("🔮 Generate Predictions", type="primary"):
                    with st.spinner("Making predictions..."):
                        # Prepare data
                        X = df[feature_columns].copy()
                        X['Additives'] = df['Additives'].copy().fillna(0)

                        PG_binder_grade_dict = {'70-28':0, '58E - 34':1}
                        Additives_dict = {0:0, '0.1% Zycotherm': 1, "0.3% Evotherm": 2, '0.5% Evotherm': 3, '0.5% Rediset LQ':4, '0.5% SonneWarmix':5, '0.75% Evotherm':6, '1-1.5% Double Barrel Green Foaming':7, '1.0 - 1.5% water' :8, '1.5 - 2.0% water':9}
                        st.error(f"❌ No records for this PG binder grade: {', '.join(list(set(X['PG binder grade']) - set(PG_binder_grade_dict.keys())))}") 
                        st.error(f"❌ No records for this additive: {', '.join(list(set(X['Additives']) - set(Additives_dict.keys())))}") 

                        diff_binder_index = np.where(X['PG binder grade'].isin(list(set(X['PG binder grade']) - set(PG_binder_grade_dict.keys()))))[0]
                        diff_additive_index = np.where(X['Additives'].isin(list(set(X['Additives']) - set(Additives_dict.keys()))))[0]
                        invalid_ct_index = np.where(pd.to_numeric(X['Avg. CTindex'], errors='coerce').isnull())[0]
                        invalid_ct_index = np.where(pd.to_numeric(X['Avg. Rut Depth'], errors='coerce').isnull())[0]
                        invalid_index_list = list(set(diff_binder_index).union(set(diff_additive_index), set(invalid_ct_index), set(invalid_ct_index)))
                        X = X.drop(index=invalid_index_list)   ### remove the rows with invalid PG binder grade, Additives, Avg. CTindex, Avg. Rut Depth

                        X['PG binder grade Type'] = X['PG binder grade'].apply(lambda x: PG_binder_grade_dict[x])
                        X['Additives Type'] = X['Additives'].apply(lambda x: Additives_dict[x])
                        
                        final_all_columns = ['RAP ', 'PG binder grade Type', 'Design Gyration', 'Additives Type','BSG (field)', 'Air Voids (field)', 'VMA (field)','Dust/Binder (Field)', 'Ignition Oven AC (%) (Field)','Slip AC Content (%) (Field)', 'Avg. CTindex', 'Avg. Rut Depth']
                        X = X[final_all_columns]
                        nan_row_index = np.where(X.isnull())[0]
                        X = X.drop(index=nan_row_index).reset_index(drop=True) ### remove the rows with NaN

                        drop_index_list = list(set(invalid_index_list).union(nan_row_index))
                        st.warning(f"Wrong values detected. Deleting these rows: {', '.join(map(str, drop_index_list))}")

                        input_x = np.array(X[final_all_columns[:-2]].astype(float), dtype=np.float32)
                        input_x = st.session_state.minmax_scaler.transform(input_x)

                        # Make predictions
                        ct_predictions = st.session_state.ct_model.predict(input_x)
                        rut_predictions = st.session_state.rut_model.predict(input_x)
                        
                        # Add predictions to dataframe
                        results_df = df.copy().drop(index=drop_index_list).reset_index(drop=True)
                        results_df['CT_Index_Predicted'] = ct_predictions
                        results_df['Average_Rut_Depth_Predicted'] = rut_predictions
                        
                        # Store results in session state
                        st.session_state.results_df = results_df
                        st.session_state.predictions_made = True
                        
                    # Display results
                    st.header("📊 Prediction Results")
                    
                    # Summary statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("CT-Index Predictions")
                        st.metric("Mean", f"{ct_predictions.mean():.4f}")
                        st.metric("Std Dev", f"{ct_predictions.std():.4f}")
                        st.metric("Min", f"{ct_predictions.min():.4f}")
                        st.metric("Max", f"{ct_predictions.max():.4f}")
                    
                    with col2:
                        st.subheader("Average Rut Depth Predictions")
                        st.metric("Mean", f"{rut_predictions.mean():.4f}")
                        st.metric("Std Dev", f"{rut_predictions.std():.4f}")
                        st.metric("Min", f"{rut_predictions.min():.4f}")
                        st.metric("Max", f"{rut_predictions.max():.4f}")
                    
                    # Display results table
                    st.subheader("📋 Detailed Results")
                    
                    # Show only original columns + predictions
                    display_cols = list(df.columns) + ['CT_Index_Predicted', 'Average_Rut_Depth_Predicted']
                    st.dataframe(
                        results_df[display_cols],
                        use_container_width=True,
                        height=400
                    )
                    
                    # Visualizations
                    st.subheader("📊 Actual vs Predicted Comparison")
                        
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # CT-Index scatter
                    ax1.scatter(results_df['Avg. CTindex'], ct_predictions, alpha=0.6, edgecolor='black', linewidth=0.5)
                    min_val = min(results_df['Avg. CTindex'].min(), ct_predictions.min())
                    max_val = max(results_df['Avg. CTindex'].max(), ct_predictions.max())
                    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    ax1.set_xlabel('Actual CT-Index', fontsize=12)
                    ax1.set_ylabel('Predicted CT-Index', fontsize=12)
                    ax1.set_title('CT-Index: Predicted vs Actual', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    # Rut Depth scatter
                    ax2.scatter(results_df['Avg. Rut Depth'], rut_predictions, alpha=0.6, edgecolor='black', linewidth=0.5)
                    min_val = min(results_df['Avg. Rut Depth'].min(), rut_predictions.min())
                    max_val = max(results_df['Avg. Rut Depth'].max(), rut_predictions.max())
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    ax2.set_xlabel('Actual Average Rut Depth', fontsize=12)
                    ax2.set_ylabel('Predicted Average Rut Depth', fontsize=12)
                    ax2.set_title('Average Rut Depth: Predicted vs Actual', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Download results
                    st.subheader("💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv = results_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">📥 Download as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                    with col2:
                        # Excel download
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Predictions')
                        excel_data = output.getvalue()
                        b64 = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="prediction_results.xlsx">📥 Download as Excel</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    
                        
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            st.info("Please ensure your file is a valid Excel file (.xlsx or .xls)")

else:
    st.error("❌ Model not loaded. Please check the model file.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with Streamlit • CT-Index & Rut Depth Prediction Tool
    </div>
    """, 
    unsafe_allow_html=True
)