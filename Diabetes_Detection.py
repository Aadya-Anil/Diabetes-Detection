import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Set page config
st.set_page_config(page_title="Diabetes Detection", layout="wide")

st.title("🩺 Diabetes Detection Dashboard")

# Sidebar navigation
section = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling"])

# Show project summary toggle
if st.sidebar.checkbox("\ud83d\udcd8 Show Project Summary"):
    st.markdown("## \ud83e\ude7a Diabetes Detection using Machine Learning")
    st.markdown("""
This project applies a range of supervised machine learning algorithms to predict diabetes based on medical diagnostic features. 
The dataset used is from the **Pima Indians Diabetes Database**.
""")

    st.markdown("### \ud83e\uddf1\u200d\ud83e\uddf2 Project Structure")
    st.markdown("""
- **Data Preprocessing**: 
    - Loaded the dataset and removed invalid zero values from key medical features like *Glucose*, *Blood Pressure*, *BMI*, etc.
- **Exploratory Data Analysis**:
    - Visualized distributions and compared features across diabetic and non-diabetic groups.
- **Model Training and Evaluation**:
    - Tested 7 ML algorithms:
        - K-Nearest Neighbors (KNN)  
        - Support Vector Classifier (SVC)  
        - Logistic Regression (LR)  
        - Decision Tree (DT)  
        - Gaussian Naive Bayes (GNB)  
        - Random Forest (RF)  
        - Gradient Boosting (GB)
    - Applied stratified train-test split.
    - Evaluated using accuracy scores.
""")

    st.markdown("### \ud83d\udcca Results")
    result_table = pd.DataFrame({
        "Model": ["K-Nearest Neighbors", "SVC", "Logistic Regression", "Decision Tree", 
                  "Naive Bayes", "Random Forest", "Gradient Boosting"],
        "Accuracy (%)": [72.92, 74.03, 76.24, 71.82, 73.48, 78.45, 77.34]
    }).set_index("Model")
    st.dataframe(result_table)

    st.markdown("\u2705 **Best Performing Model:** Random Forest with **78.45% accuracy**.")

# Load and clean data
url = "https://raw.githubusercontent.com/Aadya-Anil/Diabetes-Detection/main/Diabetes%20data.csv"
df = pd.read_csv(url, encoding='Windows-1252')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Filter out invalid zero values
df = df[(df['Glucose'] != 0) & (df['BloodPressure'] != 0) & (df['BMI'] != 0)]

if section == "Data Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head().reset_index(drop=True))

    st.subheader("Zero Value Check (Invalid Entries)")
    zero_values = (df[["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]] == 0).sum()
    st.dataframe(zero_values.to_frame(name='Count'))

elif section == "EDA":
    st.subheader("Glucose Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Glucose"], kde=True, bins=30, ax=ax1, color="skyblue")
    st.pyplot(fig1)

    st.subheader("Outcome Count")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Outcome", data=df, ax=ax2)
    st.pyplot(fig2)

elif section == "Modeling":
    st.subheader("Model Training & Evaluation")

    X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    models = [
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC()),
        ("LR", LogisticRegression()),
        ("DT", DecisionTreeClassifier()),
        ("GNB", GaussianNB()),
        ("RF", RandomForestClassifier()),
        ("GB", GradientBoostingClassifier())
    ]

    names = []
    scores = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred) * 100
        names.append(name)
        scores.append(score)

    tr_split = pd.DataFrame({'Model': names, 'Accuracy (%)': scores})

    fig3, ax3 = plt.subplots()
    sns.barplot(data=tr_split, x='Model', y='Accuracy (%)', ax=ax3)
    ax3.set_title("Model Accuracy Comparison")
    st.pyplot(fig3)

    st.dataframe(tr_split.set_index('Model'))

    best_model = tr_split.sort_values("Accuracy (%)", ascending=False).iloc[0]
    st.success(f"\ud83c\udfc6 Best Model: {best_model['Model']} with {best_model['Accuracy (%)']:.2f}% accuracy")
