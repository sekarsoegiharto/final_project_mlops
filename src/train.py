import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
import joblib

def train():
    # Load dataset
    df = pd.read_csv('./data/processed/diabetes.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # Visualisasi sebelum normalisasi
    plt.figure(figsize=(15, 5))
    plt.suptitle('Sebelum Normalisasi')
    for i, col in enumerate(X.columns):
        plt.subplot(1, len(X.columns), i+1)
        sns.histplot(X[col], kde=True, color='skyblue')
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Normalisasi
    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Visualisasi setelah normalisasi
    plt.figure(figsize=(15, 5))
    plt.suptitle('Setelah Normalisasi')
    for i, col in enumerate(X_normalized.columns):
        plt.subplot(1, len(X_normalized.columns), i+1)
        sns.histplot(X_normalized[col], kde=True, color='green')
        plt.title(col)
    plt.tight_layout()
    plt.show()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    print("Jumlah data latih:", len(X_train))
    print("Jumlah data uji:", len(X_test))

    # Train SVM model
    model_svm = SVC(kernel='linear', probability=True, random_state=42)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    y_pred_proba = model_svm.predict_proba(X_test)[:, 1]

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred_svm)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print("SVM:\n", classification_report(y_test, y_pred_svm))
    print(f'Accuracy : {accuracy}')
    print(f'ROC-AUC : {roc_auc}')

    # Visualize ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    # Log to MLflow
    mlflow.set_experiment('diabetes_svc')
    with mlflow.start_run():
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.sklearn.log_model(model_svm, 'model')

    # Save model
    joblib.dump(model_svm, 'models/svc_model.pkl')
    print(f'Model saved with accuracy : {accuracy}')

# Run the function
train()
