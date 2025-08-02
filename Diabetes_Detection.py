import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Diabetes Detection", layout="wide")

st.title("🩺 Diabetes Detection Dashboard")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Aadya-Anil/Diabetes-Detection/main/Diabetes%20data.csv"
    return pd.read_csv(url, encoding='Windows-1252')

df = load_data()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = [col.strip() for col in df.columns]

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling"])

# Shared data cleaning
df_cleaned = df[(df.BloodPressure != 0) & (df.BMI != 0) & (df.Glucose != 0)]

# Section: Data Overview
if section == "Data Overview":
    st.subheader("Dataset Preview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())

    st.subheader("Zero Value Check (Invalid Entries)")

    zero_values = (df[["Glucose", "BloodPressure", "SkinThickness", "BMI", "Insulin"]] == 0).sum()
    st.dataframe(zero_values.to_frame(name='Count'))


    st.subheader("Outcome Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Outcome", data=df, palette="Set2", ax=ax1)
    ax1.set_title("Count of Diabetes Outcomes (0 = No, 1 = Yes)")
    st.pyplot(fig1)

# Section: EDA
elif section == "EDA":
    st.subheader("Exploratory Data Analysis")

    col = st.selectbox("Select a feature to visualize", df.columns[:-1])
    
    fig2, ax2 = plt.subplots()
    sns.histplot(df[col], kde=True, bins=30, ax=ax2, color='steelblue')
    ax2.set_title(f"Distribution of {col}")
    st.pyplot(fig2)

    if col in ['Glucose', 'BMI', 'BloodPressure', 'Age']:
        st.subheader(f"{col} vs Outcome")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='Outcome', y=col, data=df, ax=ax3, palette="pastel")
        ax3.set_title(f"{col} by Diabetes Outcome")
        st.pyplot(fig3)

# Section: Modeling
elif section == "Modeling":
    st.subheader("ML Model Comparison")

    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df_cleaned[features]
    y = df_cleaned.Outcome

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    models = [
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC()),
        ("Logistic Regression", LogisticRegression()),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Naive Bayes", GaussianNB()),
        ("Random Forest", RandomForestClassifier()),
        ("Gradient Boosting", GradientBoostingClassifier())
    ]

    results = []
    for name, model in models:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        results.append((name, acc))

    result_df = pd.DataFrame(results, columns=["Model", "Accuracy (%)"]).sort_values(by="Accuracy (%)", ascending=False)

    st.dataframe(result_df.set_index("Model"))

    st.subheader("Accuracy Comparison")
    fig4, ax4 = plt.subplots()
    sns.barplot(data=result_df, x="Model", y="Accuracy (%)", palette="viridis", ax=ax4)
    ax4.set_title("Model Accuracy")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    st.pyplot(fig4)

