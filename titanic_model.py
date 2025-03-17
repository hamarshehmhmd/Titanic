#!/usr/bin/env python
# Titanic: Machine Learning from Disaster
# A comprehensive approach to the Kaggle Titanic competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

# Function to load data
def load_data():
    """
    Load train and test datasets from the data directory
    """
    try:
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure train.csv and test.csv are in the data directory.")
        return None, None

# Data exploration function
def explore_data(df, title="Dataset"):
    """
    Perform initial data exploration
    """
    print(f"\n{title} Overview:")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print("\nInfo:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all').T)
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    print(missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Feature engineering function
def engineer_features(train_df, test_df):
    """
    Create new features and prepare data for modeling
    """
    # Combine datasets for consistent feature engineering
    test_df['Survived'] = np.nan
    combined = pd.concat([train_df, test_df], axis=0)
    
    # Extract titles from names
    combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Countess": "Rare",
        "Ms": "Miss",
        "Lady": "Rare",
        "Jonkheer": "Rare",
        "Don": "Rare",
        "Dona": "Rare",
        "Mme": "Mrs",
        "Capt": "Rare",
        "Sir": "Rare"
    }
    combined['Title'] = combined['Title'].map(title_mapping)
    
    # Create family size feature
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Create age bands
    combined['AgeBand'] = pd.cut(combined['Age'], 5)
    
    # Create fare bands
    combined['FareBand'] = pd.qcut(combined['Fare'].fillna(combined['Fare'].median()), 4)
    
    # Create embarked_num and cabin features
    combined['EmbarkedNum'] = combined['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    combined['CabinLetter'] = combined['Cabin'].str[0]
    
    # Split back into train and test
    train_df = combined[~combined['Survived'].isna()].copy()
    test_df = combined[combined['Survived'].isna()].copy()
    test_df = test_df.drop('Survived', axis=1)
    
    return train_df, test_df

# Build model pipeline
def build_model_pipeline():
    """
    Create a preprocessing and modeling pipeline
    """
    # Define preprocessing for numeric columns
    numeric_features = ['Age', 'Fare', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical columns
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create full pipeline with preprocessing and model
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return full_pipeline

# Train and evaluate model
def train_evaluate_model(model, X_train, y_train, X_val=None, y_val=None):
    """
    Train model and evaluate performance
    """
    # Train model
    model.fit(X_train, y_train)
    
    # If validation set is provided, evaluate on it
    if X_val is not None and y_val is not None:
        val_predictions = model.predict(X_val)
        print("\nValidation Set Performance:")
        print(f"Accuracy: {accuracy_score(y_val, val_predictions)}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, val_predictions))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model

# Generate predictions for test set
def generate_predictions(model, test_data, test_ids):
    """
    Generate predictions for test set and create submission file
    """
    predictions = model.predict(test_data)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions.astype(int)
    })
    
    # Save submission to CSV
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created.")
    
    return submission

# Main function
def main():
    print("Titanic: Machine Learning from Disaster")
    print("=" * 50)
    
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Explore train data
    explore_data(train_data, "Training Data")
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    train_processed, test_processed = engineer_features(train_data, test_data)
    
    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = train_processed[features]
    y = train_processed['Survived']
    test_X = test_processed[features]
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train model
    print("\nBuilding and training model...")
    model_pipeline = build_model_pipeline()
    trained_model = train_evaluate_model(model_pipeline, X_train, y_train, X_val, y_val)
    
    # Generate predictions
    print("\nGenerating predictions for test set...")
    submission = generate_predictions(trained_model, test_X, test_processed['PassengerId'])
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 