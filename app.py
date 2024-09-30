import streamlit as st
import pandas as pd
import numpy as np
import io
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="ML AutoPilot", layout="wide")

# Custom CSS to improve look and feel
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #262730
    }
    .Widget>label {
        color: #262730;
        font-family: sans-serif;
    }
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 ML AutoPilot")

@st.cache(allow_output_mutation=True)
def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def display_eda(data):
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Info")
        buffer = io.StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
    
    with col2:
        st.subheader("Data Types")
        st.write(data.dtypes)
    
    st.subheader("Data Description")
    st.write(data.describe())
    
    st.subheader("Missing Values")
    missing = data.isnull().sum()
    st.write(missing[missing > 0])
    
    # Interactive visualizations
    st.subheader("Data Visualization")
    
    # Correlation heatmap
    if len(data.select_dtypes(include=[np.number]).columns) > 1:
        fig = px.imshow(data.corr(), color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig)
    
    # Distribution plots
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
    fig = px.histogram(data, x=selected_col, marginal="box")
    st.plotly_chart(fig)

def handle_missing_values(data):
    st.header("🔧 Handle Missing Values")
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    
    col1, col2 = st.columns(2)
    
    with col1:
        if data[numeric_columns].isnull().sum().sum() > 0:
            st.subheader("Numeric Columns")
            numeric_impute_method = st.selectbox(
                "Choose imputation method for numeric columns",
                options=["mean", "median", "mode"]
            )
            if st.button("Impute Numeric Missing Values"):
                if numeric_impute_method == "mean":
                    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
                elif numeric_impute_method == "median":
                    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
                else:
                    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mode().iloc[0])
                st.success("Missing numeric values imputed successfully.")
    
    with col2:
        if data[categorical_columns].isnull().sum().sum() > 0:
            st.subheader("Categorical Columns")
            categorical_impute_method = st.selectbox(
                "Choose imputation method for categorical columns",
                options=["mode", "additional class"]
            )
            if st.button("Impute Categorical Missing Values"):
                if categorical_impute_method == "mode":
                    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
                else:
                    data[categorical_columns] = data[categorical_columns].fillna("Unknown")
                st.success("Missing categorical values imputed successfully.")
    
    return data

def feature_engineering(data):
    st.header("🛠️ Feature Engineering")
    
    # Add option to create new features
    st.subheader("Create New Features")
    col1, col2 = st.columns(2)
    with col1:
        feature1 = st.selectbox("Select first feature", data.columns)
    with col2:
        feature2 = st.selectbox("Select second feature", data.columns)
    
    operation = st.selectbox("Select operation", ["Add", "Subtract", "Multiply", "Divide"])
    new_feature_name = st.text_input("Enter new feature name")
    
    if st.button("Create New Feature"):
        if operation == "Add":
            data[new_feature_name] = data[feature1] + data[feature2]
        elif operation == "Subtract":
            data[new_feature_name] = data[feature1] - data[feature2]
        elif operation == "Multiply":
            data[new_feature_name] = data[feature1] * data[feature2]
        else:
            data[new_feature_name] = data[feature1] / data[feature2]
        st.success(f"New feature '{new_feature_name}' created successfully.")
    
    return data

def select_features_target(data):
    st.header("🎯 Select Features and Target")
    
    target_column = st.selectbox("Select the target variable", data.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column])
    
    return feature_columns, target_column

def train_model(data, target_column, task_type):
    st.header("🤖 Model Training")
    
    if st.button("Train Models"):
        with st.spinner("Training models... This may take a while."):
            try:
                if task_type == 'classification':
                    clf_setup(data=data, target=target_column, silent=True, verbose=False)
                    best_model = clf_compare_models()
                    model_details = clf_pull()
                    st.subheader("Best Classification Model:")
                    st.write(model_details)
                else:
                    reg_setup(data=data, target=target_column, silent=True, verbose=False)
                    best_model = reg_compare_models()
                    model_details = reg_pull()
                    st.subheader("Best Regression Model:")
                    st.write(model_details)
                st.success("Model Training Complete!")
            except Exception as e:
                st.error(f"Error during model training: {e}")

def main():
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.sidebar.success("Data loaded successfully!")
            
            # Data Preview
            if st.sidebar.checkbox("Show Data Preview"):
                st.write(data.head())
            
            # EDA
            if st.sidebar.checkbox("Perform EDA"):
                display_eda(data)
            
            # Handle Missing Values
            if st.sidebar.checkbox("Handle Missing Values"):
                data = handle_missing_values(data)
            
            # Feature Engineering
            if st.sidebar.checkbox("Perform Feature Engineering"):
                data = feature_engineering(data)
            
            # Select Features and Target
            feature_columns, target_column = select_features_target(data)
            
            # Detect task type
            y = data[target_column]
            task_type = 'classification' if y.dtype == 'object' or y.nunique() < 10 else 'regression'
            st.write(f"**Detected task type:** {task_type}")
            
            # Train Model
            train_model(data[feature_columns + [target_column]], target_column, task_type)

if __name__ == "__main__":
    main()