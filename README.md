# Diabetes Detection using Machine Learning
This project applies various supervised machine learning algorithms to predict diabetes based on patient medical data.
The dataset is sourced from the Pima Indians Diabetes Database.

## Project Structure
### Data Preprocessing:
  Loaded the dataset.
  Identified and handled missing or zero values in key medical measurements (like Blood Pressure, BMI, Glucose).
  Filtered out invalid entries for better model accuracy.

### Exploratory Data Analysis:
Visualized data distributions across diabetic and non-diabetic groups.
Checked for imbalances and patterns in the dataset.

### Model Training and Evaluation:
Tested 7 different machine learning models:
  K-Nearest Neighbors (KNN)
  Support Vector Classifier (SVC)
  Logistic Regression (LR)
  Decision Tree (DT)
  Gaussian Naive Bayes (GNB)
  Random Forest (RF)
  Gradient Boosting (GB)

Performed train/test split with stratification.

Evaluated models using accuracy scores.

### Results

Model	Accuracy                 (%)
K-Nearest Neighbors	          72.92
Support Vector Classifier     74.03
Logistic Regression	          76.24
Decision Tree	                71.82
Gaussian Naive Bayes	        73.48
Random Forest	                78.45
Gradient Boosting	            77.34

Best Model: Random Forest Classifier with an accuracy of 78.45%.
