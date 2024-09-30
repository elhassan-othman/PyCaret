import streamlit as st
import pandas as pd
import numpy as np
import io
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, pull as clf_pull
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML AutoPilot", layout="wide")

st.title("ML AutoPilot")

def load_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def display_eda(data):
    st.subheader("Exploratory Data Analysis")
    st.write("**Data Info:**")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("**Data Description:**")
    st.write(data.describe())

    st.write("**Missing Values:**")
    missing = data.isnull().sum()
    st.write(missing[missing > 0])

    # Optional: Add visualizations
    st.write("**Missing Values Heatmap:**")
    fig, ax = plt.subplots()
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    st.pyplot(fig)

def handle_missing_values(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns

    # Handle numeric missing values
    if data[numeric_columns].isnull().sum().sum() > 0:
        st.subheader("Handle Missing Values in Numeric Columns")
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

    # Handle categorical missing values
    if data[categorical_columns].isnull().sum().sum() > 0:
        st.subheader("Handle Missing Values in Categorical Columns")
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

def drop_columns(data):
    st.subheader("Drop Columns")
    columns_to_drop = st.multiselect("Select columns to drop", data.columns)
    if columns_to_drop:
        if st.button("Drop Selected Columns"):
            data = data.drop(columns=columns_to_drop)
            st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
            st.write("Updated Data Preview:")
            st.write(data.head())
    return data

def select_features_target(data):
    st.subheader("Select Features and Target")
    target_column = st.selectbox("Select the target variable", data.columns)
    feature_columns = [col for col in data.columns if col != target_column]

    X = data[feature_columns]
    y = data[target_column]

    return feature_columns, target_column

def detect_task_type(y):
    if y.dtype == 'object' or y.nunique() < 10:
        return 'classification'
    else:
        return 'regression'

def train_model(data, target_column, task_type):
    st.subheader("Model Training")
    if st.button("Train Models"):
        st.write("Training models... This may take a while.")
        try:
            if task_type == 'classification':
                clf_setup(data=data, target=target_column, silent=True, verbose=False)
                best_model = clf_compare_models()
                model_details = clf_pull()
                st.write("**Best Classification Model:**")
                st.write(model_details)
            else:
                reg_setup(data=data, target=target_column, silent=True, verbose=False)
                best_model = reg_compare_models()
                model_details = reg_pull()
                st.write("**Best Regression Model:**")
                st.write(model_details)
            st.success("Model Training Complete!")
        except Exception as e:
            st.error(f"Error during model training: {e}")

def main():
    # 1. Ask user to input data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("**Data Preview:**")
            st.write(data.head())

            # 2. Perform EDA
            display_eda(data)

            # 3 & 4. Handle missing values
            data = handle_missing_values(data)

            # 5. Drop columns
            data = drop_columns(data)

            # 6. Select X and y
            feature_columns, target_column = select_features_target(data)

            # 7. Detect task type
            y = data[target_column]
            task_type = detect_task_type(y)
            st.write(f"**Detected task type:** {task_type}")

            # 8. Apply PyCaret and show the best model
            train_model(data, target_column, task_type)

if __name__ == "__main__":
    main()
