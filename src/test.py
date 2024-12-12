import pandas as pd
import joblib

model = joblib.load("models/svc_model.pkl")


data = {
    "Pregnancies": [1],
    "Glucose": [85],
    "BloodPressure": [66],
    "SkinThickness": [29],
    "Insulin": [0],
    "BMI": [26.6],
    "DiabetesPedigreeFunction": [0.351],
    "Age": [31],
}
df = pd.DataFrame(data)


predictions = model.predict(df)
print(f"Predictions: {predictions}")