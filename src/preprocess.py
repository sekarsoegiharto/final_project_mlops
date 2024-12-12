import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocessing():
    df = pd.read_csv('./data/diabetes.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    scaler = MinMaxScaler()
    X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    processed_data = pd.concat([X_normalized, y], axis=1)
    processed_data.to_csv('data/processed/diabetes.csv')
    
preprocessing()

    