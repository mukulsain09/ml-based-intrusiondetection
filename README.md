# ML-Based Intrusion Detection System

## Description

This project implements a machine learning-based intrusion detection system to classify network connections as either "normal" or an "intrusion/attack". The system is built using a `RandomForestClassifier` and the NSL-KDD dataset.

The primary goal is to create a predictive model that can be used to identify and flag malicious network activity, forming a critical component of a network security infrastructure.

## Dataset

The dataset used for training and testing this model is the **NSL-KDD dataset**, which is a refined version of the original KDD'99 dataset. It's a widely used benchmark for intrusion detection systems.

-   `Train_data.csv`: The dataset used for training the model.
-   `Test_data.csv`: The dataset used for evaluating the model's performance on unseen data.

## Model

The core of this intrusion detection system is a `RandomForestClassifier` from the scikit-learn library. Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) of the individual trees.

## Getting Started

To get started with this project, you will need to have Python and the following libraries installed:

-   pandas
-   scikit-learn
-   joblib
-   matplotlib
-   seaborn

You can install these dependencies using pip:
```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

### Usage

The main logic of the project is contained in the `intrusion_detection_system.ipynb` Jupyter Notebook. You can open and run this notebook to see the entire process, from data loading and preprocessing to model training, evaluation, and saving.

The notebook is structured as follows:
1.  **Data Loading:** Loads the `Train_data.csv` and `Test_data.csv` files.
2.  **Exploratory Data Analysis (EDA):** Visualizes the distribution of connection types.
3.  **Data Preprocessing:**
    -   Handles categorical and numerical features.
    -   Uses one-hot encoding for categorical features.
    -   Scales numerical features using `StandardScaler`.
    -   Encodes the target variable using `LabelEncoder`.
4.  **Model Training:** Trains a `RandomForestClassifier` on the preprocessed training data.
5.  **Model Evaluation:** Evaluates the model's performance on a validation set using metrics like accuracy, a classification report, and a confusion matrix.
6.  **Saving the Model:** Saves the trained model and the preprocessors (`StandardScaler`, `LabelEncoder`, and model columns) to `.pkl` files for later use.
7.  **Prediction on Test Data:** Demonstrates how to load the saved model and preprocessors to make predictions on new, unseen data from the test set.

## Files in this Repository

-   `intrusion_detection_system.ipynb`: The main Jupyter Notebook containing the code for the project.
-   `Train_data.csv`: The training dataset.
-   `Test_data.csv`: The testing dataset.
-   `intrusion_detection_model.pkl`: The saved, pre-trained `RandomForestClassifier` model.
-   `scaler.pkl`: The saved `StandardScaler` object used for scaling numerical features.
-   `label_encoder.pkl`: The saved `LabelEncoder` object used for encoding the target variable.
-   `model_columns.pkl`: The saved list of column names used during model training. This is important for ensuring that the input to the model has the same features in the same order.
-   `README.md`: This file, providing an overview of the project.

## Working of the Code

The `intrusion_detection_system.ipynb` notebook provides a step-by-step guide to building the intrusion detection system.

1.  **Imports:** The necessary libraries are imported.
2.  **Data Loading and Initial Cleaning:** The training and test datasets are loaded into pandas DataFrames. The column names of the test data are corrected.
3.  **Preprocessing:**
    -   The features (`X`) and the target (`y`) are separated.
    -   Categorical features are identified and one-hot encoded. This converts categorical string values into a numerical format that the model can understand.
    -   Numerical features are scaled using `StandardScaler`. This is important for many machine learning algorithms as it ensures that all features have the same scale.
    -   The target labels (e.g., 'normal', 'anomaly') are converted into numerical form using `LabelEncoder`.
4.  **Train-Validation Split:** The training data is split into training and validation sets. This allows for an unbiased evaluation of the model's performance.
5.  **Model Training:** The `RandomForestClassifier` is trained on the preprocessed training data.
6.  **Evaluation:** The trained model is used to make predictions on the validation set. The accuracy, classification report (with precision, recall, and F1-score), and a confusion matrix are generated to assess the model's performance.
7.  **Saving and Loading the Model:** The trained model and all the necessary preprocessing objects are saved to disk using `joblib`. The notebook also demonstrates how to load these objects back into memory to make predictions on new data. This is crucial for deploying the model in a real-world application without needing to retrain it every time.
8.  **Prediction on New Data:** The notebook shows a simple example of how to use the loaded model and preprocessors to predict the class of a single sample from the test set.

This project provides a solid foundation for understanding and building a machine learning-powered intrusion detection system.
