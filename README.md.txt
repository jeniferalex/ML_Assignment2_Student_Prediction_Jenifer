# Student Dropout and Academic Success Prediction Using Machine Learning

## 1. Introduction

This project focuses on predicting student academic outcomes using machine learning classification algorithms. The main objective is to analyse student-related information and classify students into three possible categories: Dropout, Enrolled, or Graduate.

Student dropout is an important concern for educational institutions. Analysing student information through machine learning can help identify patterns associated with different academic outcomes and support better decision-making.

As part of this project, five classification algorithms are implemented, evaluated and compared. A Streamlit application is also developed to make the trained models accessible through a simple user interface.

## 2. Dataset Description

The **Student Dropout and Academic Success** dataset is obtained from the UCI Machine Learning Repository.

The dataset contains student information related to personal background, admission details, previous qualifications and other academic and socio-economic factors.

**Dataset details:**

- Total number of instances: 4,424
- Number of input features: 36
- Target variable: `Target`
- Number of target classes: 3
- Data source: - Data source: [UCI Machine Learning Repository – Student Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The target variable contains three categories:

1. Dropout
2. Enrolled
3. Graduate

The classification task is to predict the student's outcome based on the available input features.

## 3. Data Preparation and Preprocessing

Data inspection and preprocessing were performed before training the classification models.

The following steps were carried out:

- Loaded the dataset using Pandas.
- Examined the number of rows, columns and data types.
- Checked for missing values and duplicate records.
- Separated the input features and target variable.
- Identified categorical and numerical features.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features where required.

A ColumnTransformer was used to manage the preprocessing steps, and machine learning pipelines were created to maintain consistency during model training and prediction.

The dataset was divided into training and testing sets using an 80:20 ratio. Stratified splitting was used to maintain the target-class distribution, with `random_state=42`.

**Dataset split:**

- Training records: 3,539
- Testing records: 885

## 4. Machine Learning Algorithms

Five classification algorithms were implemented and compared.

### 4.1 Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to different classes. It was used to predict the academic outcome of students.

### 4.2 Decision Tree

Decision Tree is a tree-based algorithm that makes predictions by splitting the dataset based on feature values. It is relatively easy to understand and interpret.

### 4.3 K-Nearest Neighbors (KNN)

KNN is a distance-based classification algorithm. It predicts the class of a new observation based on the classes of its nearest neighbours. Feature scaling is important for this algorithm.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It assumes conditional independence between features and models numerical feature distributions using Gaussian distributions.

### 4.5 Support Vector Machine (SVM)

SVM is a classification algorithm that attempts to identify a decision boundary that separates different classes. It is useful for handling classification problems with complex decision boundaries.

## 5. Model Evaluation Metrics

All five models were evaluated using the same test dataset.

The following metrics were used:

- **Accuracy:** Measures the overall proportion of correctly classified observations.
- **AUC:** Measures the ability of the model to distinguish between classes. A macro one-vs-rest approach was used for this multiclass problem.
- **Precision:** Measures the proportion of correct positive predictions.
- **Recall:** Measures the proportion of actual class observations correctly identified.
- **F1-score:** Combines precision and recall into a single measure.
- **Matthews Correlation Coefficient (MCC):** Provides an overall measure of classification quality by considering correct and incorrect predictions.

Confusion matrices and classification reports were also generated to examine the performance of each model across the three target categories.

## 6. Comparison of Model Results

The evaluation results obtained from the experiments are presented below.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.767 | 0.881 | 0.712 | 0.685 | 0.693 | 0.614 |
| Decision Tree | 0.685 | 0.731 | 0.619 | 0.622 | 0.620 | 0.492 |
| KNN | 0.703 | 0.798 | 0.632 | 0.613 | 0.618 | 0.505 |
| Naive Bayes | 0.255 | 0.683 | 0.483 | 0.403 | 0.232 | 0.138 |
| SVM | 0.760 | 0.855 | 0.709 | 0.673 | 0.684 | 0.602 |

## 7. Observations and Comparative Analysis

The following observations were made from the model evaluation results:

1. **Logistic Regression:** Achieved the highest accuracy of 76.7% and AUC of 0.881. It also obtained the highest F1-score and MCC, making it the best-performing model overall in this experiment.

2. **Support Vector Machine:** Achieved 76.0% accuracy and an AUC of 0.855. Its performance was close to Logistic Regression.

3. **Decision Tree:** Achieved 68.5% accuracy. Its performance was lower than Logistic Regression and SVM.

4. **K-Nearest Neighbors:** Achieved 70.3% accuracy and an AUC of 0.798, showing moderate classification performance.

5. **Naive Bayes:** Achieved 25.5% accuracy, which was considerably lower than the other models. The classification report showed that it identified the Enrolled class much more frequently than the Graduate class. This may be related to the distributional and independence assumptions of Gaussian Naive Bayes.

Overall, Logistic Regression performed best across the reported evaluation metrics, followed closely by SVM.

The comparison also demonstrates that accuracy alone is not sufficient for evaluating a multiclass classification problem. Precision, recall, F1-score, MCC, confusion matrices and classification reports provide additional information about class-wise performance.

## 8. Best Performing Model

Based on the experimental results, Logistic Regression was selected as the best-performing model.

Its evaluation results were:

- Accuracy: 76.7%
- AUC: 0.881
- Precision: 0.712
- Recall: 0.685
- F1-score: 0.693
- MCC: 0.614

Although SVM produced comparable results, Logistic Regression achieved slightly better performance across the reported metrics.

## 9. Streamlit Application

A Streamlit application was developed to provide an interactive interface for evaluating the trained classification models.

The application includes the following features:

1. Upload a labelled CSV test dataset.
2. Select one of the five trained classification models.
3. Evaluate the uploaded dataset using the selected model.
4. Display accuracy, AUC, precision, recall, F1-score and MCC.
5. Display the confusion matrix.
6. Display the classification report.

The application loads the previously saved models and their preprocessing pipelines from the model folder. It does not retrain the models when a test dataset is uploaded.

The uploaded CSV file must contain all required input features along with the `Target` column.

## 10. Saved Models and Project Structure

The trained models were saved using Joblib so that they could be reused by the Streamlit application.

The project directory is organised as follows:

```text
ML_Assignment2/
│
├── app.py
├── ML_Assignment2.ipynb
├── data.csv
├── test_data.csv
├── requirements.txt
├── README.md
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

**File descriptions:**

- `app.py` – Source code for the Streamlit application.
- `ML_Assignment2.ipynb` – Notebook containing data analysis,