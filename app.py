
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# Page configuration
st.set_page_config(
    page_title="Student Success Classifier",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Dropout and Academic Success Classifier")
st.subheader("M.Tech AIML - Assignment 2")

st.write(
    "Upload a labelled student dataset and evaluate it "
    "using five trained machine-learning classification models."
)

st.divider()

# Expected feature columns
reference_file = BASE_DIR / "test_data.csv"

if not reference_file.exists():
    st.error("test_data.csv is missing.")
    st.stop()

expected_columns = [
    col for col in pd.read_csv(reference_file, nrows=0).columns
    if col != "Target"
]

# Saved models
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Model selection
st.header("1. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model",
    list(model_files.keys())
)

# Dataset upload
st.header("2. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your labelled test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    # Validate target column
    if "Target" not in df.columns:
        st.error("The uploaded CSV must contain a Target column.")
        st.stop()

    # Validate feature columns
    missing_columns = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(f"Missing columns: {missing_columns}")
        st.stop()

    X = df[expected_columns]
    y = df["Target"]

    # Load saved model
    model_path = MODEL_DIR / model_files[selected_model]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}")
        st.stop()

    model = joblib.load(model_path)

    # Predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Evaluation metrics
    accuracy = accuracy_score(y, predictions)

    auc = roc_auc_score(
        y,
        probabilities,
        labels=model.classes_,
        multi_class="ovr",
        average="macro"
    )

    precision = precision_score(
        y, predictions, average="macro", zero_division=0
    )

    recall = recall_score(
        y, predictions, average="macro", zero_division=0
    )

    f1 = f1_score(
        y, predictions, average="macro", zero_division=0
    )

    mcc = matthews_corrcoef(y, predictions)

    # Display results
    st.divider()
    st.header("3. Model Evaluation Results")
    st.write("Selected Model:", selected_model)

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("AUC Score", f"{auc:.3f}")
    col3.metric("Precision", f"{precision:.3f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC Score", f"{mcc:.3f}")

    # Confusion matrix
    st.header("4. Confusion Matrix")

    cm = confusion_matrix(
        y, predictions, labels=model.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    )

    st.dataframe(cm_df)

    # Classification report
    st.header("5. Classification Report")

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose())

    st.success("Model evaluation completed successfully!")
