import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
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
section = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling", "Live Prediction"])

# Show project summary toggle
if st.sidebar.checkbox("Show Project Summary"):
    st.markdown("## 🩺 Diabetes Detection using Machine Learning")
    st.markdown("""
This project applies a range of supervised machine learning algorithms to predict diabetes based on medical diagnostic features. 
The dataset used is from the **Pima Indians Diabetes Database**.
""")

    st.markdown("### 🧱 Project Structure")
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

    st.markdown("### 📊 Results")
    result_table = pd.DataFrame({
        "Model": ["K-Nearest Neighbors", "SVC", "Logistic Regression", "Decision Tree", 
                  "Naive Bayes", "Random Forest", "Gradient Boosting"],
        "Accuracy (%)": [72.92, 74.03, 76.24, 71.82, 73.48, 78.45, 77.34]
    }).set_index("Model")
    st.dataframe(result_table)

    st.markdown("✅ **Best Performing Model:** Random Forest with **78.45% accuracy**.")

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
        ("SVC", SVC(probability=True)),
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

    best_model_name = tr_split.sort_values("Accuracy (%)", ascending=False).iloc[0]['Model']
    best_model = [m for n, m in models if n == best_model_name][0]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    st.success(f"🏆 Best Model: {best_model_name} with {max(scores):.2f}% accuracy")

elif section == "Live Prediction":
    st.subheader("Try Your Own Prediction")
    st.markdown("Enter values below to predict diabetes risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 120)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    with col2:
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 10, 100, 33)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    if st.button("Predict"):
        prediction = best_model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ The model predicts a high risk of diabetes.")
        else:
            st.success("✅ The model predicts a low risk of diabetes.")
