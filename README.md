# Parkinson's Disease Prediction Using Artificial Neural Network (ANN)

## 1. Project Overview

Parkinson's Disease Prediction Using ANN is a Machine Learning project that uses an Artificial Neural Network (ANN) to predict Parkinson's Disease based on biomedical voice measurements.

The project uses the **Oxford Parkinson's Disease Detection Dataset** available through the UCI Machine Learning Repository.

The main objective is to develop an ANN-based binary classification model that identifies patterns in voice-related features and predicts whether a record belongs to a healthy individual or an individual with Parkinson's Disease.

This project demonstrates an end-to-end Machine Learning workflow, from dataset collection and exploratory data analysis to model development, evaluation, and saving trained model artifacts.

## 2. Project Objectives

- Collect and analyze the Oxford Parkinson's Disease Detection Dataset.
- Perform Exploratory Data Analysis (EDA).
- Visualize feature distributions and relationships.
- Apply appropriate data preprocessing techniques.
- Prevent participant-level data leakage using group-based splitting.
- Develop an Artificial Neural Network using TensorFlow and Keras.
- Train and evaluate the ANN model.
- Analyze model performance using multiple evaluation metrics.
- Save the trained model and supporting preprocessing files.

## 3. Dataset Information

**Dataset Name:** Oxford Parkinson's Disease Detection Dataset

**Source:** UCI Machine Learning Repository

**Dataset ID:** 174

**Dataset Link:** https://archive.ics.uci.edu/dataset/174/parkinsons

The dataset contains biomedical voice measurements used for Parkinson's Disease classification.

The target variable is:

- `status = 0`: Healthy
- `status = 1`: Parkinson's Disease

The notebook uses 22 numerical voice features for model development. The participant identifier is retained for group-based data splitting and excluded from model input features.

## 4. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| TensorFlow | Deep Learning framework |
| Keras | ANN model development |
| NumPy | Numerical computation |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing and evaluation |
| UCI ML Repository | Dataset collection |
| Joblib | Saving preprocessing artifacts |

## 5. Project Workflow

The project is divided into 14 stages:

1. Project Introduction
2. Google Colab Environment Setup
3. Dataset Collection and Loading
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. ANN Model Architecture
7. ANN Model Training
8. Model Evaluation
9. Required Model Evaluation Graphs
10. Final Model Analysis
11. Sample Prediction Demonstration
12. Saving the Model and Supporting Files
13. Final Conclusion and Future Scope
14. BCA Final-Year Academic Project Documentation Outline

## 6. Data Preprocessing

The following preprocessing techniques are implemented:

- Separation of input features and target variable.
- Extraction of participant identifiers.
- Group-based dataset splitting using `GroupShuffleSplit`.
- Dataset partitioning into 60% training, 20% validation, and 20% testing.
- Feature standardization using `StandardScaler`.
- Class imbalance handling using balanced class weights.

The scaler is fitted only on the training data to prevent data leakage.

## 7. ANN Model Architecture

The project implements a feedforward Artificial Neural Network using TensorFlow and Keras.

**Architecture:**

- Input Layer: 22 features
- Hidden Layer 1: 32 neurons with ReLU activation
- Dropout Layer: 30%
- Hidden Layer 2: 16 neurons with ReLU activation
- Dropout Layer: 20%
- Output Layer: 1 neuron with Sigmoid activation

**Model Configuration:**

- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Binary Crossentropy
- Training Epochs: Up to 200
- Batch Size: 16
- Early Stopping: Patience of 20 epochs
- Class Weighting: Balanced class weights

## 8. Model Evaluation

The model is evaluated using the unseen test dataset.

The following evaluation metrics are calculated:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

These metrics provide a more detailed understanding of the model's classification performance.

## 9. Data Visualization

The project includes exploratory and model evaluation visualizations to understand the dataset and monitor model performance.

These include:

- Missing Value Analysis
- Duplicate Record Analysis
- Target Class Distribution
- Feature Distribution Histograms
- Correlation Analysis
- Class-Based Feature Comparisons
- ANN Training and Validation Performance
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

## 10. Saved Model and Output Files

The notebook includes functionality to save the following files:

| File Name | Description |
|---|---|
| `parkinsons_ann_model.keras` | Trained ANN model |
| `parkinsons_scaler.pkl` | Fitted StandardScaler |
| `parkinsons_features.pkl` | Feature information |
| `ANN_Performance_Results.csv` | Model performance metrics |
| `ANN_Classification_Report.csv` | Classification report |
| `ANN_Training_History.csv` | Training history |

These artifacts can be used for future model inference and analysis.

## 11. How to Run the Project

This project is developed in Google Colab.

**Step 1:** Clone or download this GitHub repository.

**Step 2:** Open Google Colab.

https://colab.research.google.com/

**Step 3:** Upload `Machine_Learning.ipynb` to Google Colab.

**Step 4:** Install the required dependencies if they are not already available.

```python
!pip install ucimlrepo tensorflow numpy pandas matplotlib seaborn scikit-learn joblib
```

**Step 5:** Run the notebook cells sequentially, starting from the first stage.

**Step 6:** Allow the notebook to retrieve the dataset from the UCI Machine Learning Repository.

**Step 7:** Complete data preprocessing, ANN training, evaluation, and visualization.

**Step 8:** Execute the final cells to save and download the model artifacts.

## 12. Future Scope

The project can be extended through:

- Comparing ANN performance with Random Forest and Support Vector Machine algorithms.
- Using deeper neural network architectures.
- Integrating Explainable AI techniques such as SHAP and LIME.
- Expanding the dataset with more diverse voice recordings.
- Developing a web-based prediction interface.
- Deploying the trained model as a prediction service.

## 13. Project Limitations

- The model is developed using a specific biomedical voice dataset.
- Model performance depends on dataset quality and representativeness.
- The project is an academic Machine Learning prototype and is not a clinically validated diagnostic system.
- Predictions should not be used as a substitute for professional medical diagnosis.
