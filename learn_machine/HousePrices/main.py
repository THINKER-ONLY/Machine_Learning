import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def read_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    train_data = train_data.drop(['Id'], axis=1)
    test_data = test_data.drop(['Id'], axis=1)
    return train_data, test_data

def preprocess_data(data):
    processed_data = data.copy()
    for column in processed_data.columns:
        if processed_data[column].dtype == np.object_:
            le = LabelEncoder()
            processed_data[column] = le.fit_transform(processed_data[column].astype(str))
    return processed_data

def test(model, X_test, y_test):
    predictions = model.predict(X_test)
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.show()

def train(train_data):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    features = train_data.drop(['SalePrice'], axis=1)
    target = train_data['SalePrice']
    features_processed = preprocess_data(features)
    X_train, X_val, y_train, y_val = train_test_split(
        features_processed, target, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_val, y_val

def predict(test_data, model):
    test_data_processed = preprocess_data(test_data)
    predictions = model.predict(test_data_processed)
    submission = pd.DataFrame({'Id': range(1461, 2920), 'SalePrice': predictions})
    submission.to_csv('data/submission.csv', index=False)
    
def main():
    train_data, test_data = read_data()
    model, X_test, y_test = train(train_data)
    test(model, X_test, y_test)
    predict(test_data, model)

if __name__ == '__main__':
    main()


