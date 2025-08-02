import pandas as pd
import numpy as np
import seaborn as sns

url = "https://raw.githubusercontent.com/Aadya-Anil/Diabetes-Detection/main/Diabetes%20data.csv"
dt = pd.read_csv(url, encoding = 'Windows-1252')
dt.columns 


dt.head()

print("Diabetes data set dimensions : {}".format(dt.shape))

dt.groupby('Outcome').size()

dt.isnull().sum()
dt.isna().sum()
     

print("Total : ", dt[dt.BloodPressure == 0].shape[0])
     

print(dt[dt.BloodPressure == 0].groupby('Outcome')['Age'].count())


print("Total : ", dt[dt.Glucose == 0].shape[0])
     

print(dt[dt.Glucose == 0].groupby('Outcome')['Age'].count())
     
print("Total : ", dt[dt.SkinThickness == 0].shape[0])
     

print(dt[dt.SkinThickness == 0].groupby('Outcome')['Age'].count())
     


print("Total : ", dt[dt.BMI == 0].shape[0])
print(dt[dt.BMI == 0].groupby('Outcome')['Age'].count())
     
print("Total : ", dt[dt.Insulin == 0].shape[0])
print(dt[dt.Insulin == 0].groupby('Outcome')['Age'].count())


diabetes_mod = dt[(dt.BloodPressure != 0) & (dt.BMI != 0) & (dt.Glucose != 0)]
print(diabetes_mod.shape)
     


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome
     

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
     
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
     

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

     

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)

names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred)*100)
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print(tr_split)
