import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns

def train():
    df = pd.read_csv('./data/processed/diabetes.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    y.head()
    
    # # Visualisasi sebelum normalisasi
    # plt.figure(figsize=(15, 5))
    # plt.suptitle('Sebelum Normalisasi')

    # for i, col in enumerate(X.columns):
    #     plt.subplot(1, len(X.columns), i+1)
    #     sns.histplot(X[col], kde=True, color='skyblue')
    #     plt.title(col)

    # plt.show()
    
    # Normalisasi
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Visualisasi setelah normalisasi
    # plt.figure(figsize=(15, 5))
    # plt.suptitle('Setelah Normalisasi')

    # for i, col in enumerate(X_normalized.columns):
    #     plt.subplot(1, len(X_normalized.columns), i+1)
    #     sns.histplot(X_normalized[col], kde=True, color='green')
    #     plt.title(col)

    # plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    print("Jumlah data latih:", len(X_train))
    print("Jumlah data uji:", len(X_test))
    
    model_svm = SVC(kernel='linear', random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred_svm)
    print("SVM:\n", classification_report(y_test, y_pred_svm))
    print(f'Accuracy : {accuracy}')
    print('Update')
    
    mlflow.set_experiment('diabetes_svc')
    with mlflow.start_run():
        mlflow.log_metric('accurracy', accuracy)
        mlflow.sklearn.log_model(model_svm, 'model')
    
    joblib.dump(model_svm, 'models/svc_model.pkl')
    print(f'Model saved with accuracy  : {accuracy}')
        

    

train()