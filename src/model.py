import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    """Load training and test data."""
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')
    return train_data, test_data

def preprocess_data(df):
    """Preprocess the data with feature engineering."""
    # Create a copy of the dataframe
    data = df.copy()
    
    # Handle missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical features
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    
    return data[features]

def train_model(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    # Load data
    train_data, test_data = load_data()
    
    # Preprocess data
    X_train = preprocess_data(train_data)
    X_test = preprocess_data(test_data)
    y_train = train_data['Survived']
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('../submissions/submission.csv', index=False)
    print("Submission file created successfully!")

if __name__ == "__main__":
    main() 