#!/usr/bin/env python3
"""
Titanic: Machine Learning from Disaster
Robust model with fundamental features and strong regularization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

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
        print("Error: Dataset files not found.")
        return None, None

def basic_feature_engineering(train_df, test_df):
    """
    Focused feature engineering with only the most reliable predictors
    """
    # Create copies to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # Combine datasets for consistent feature engineering
    test['Survived'] = np.nan
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Extract titles from names - a proven strong predictor
    combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Simplify to just a few title categories
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Mlle': 'Miss',
        'Ms': 'Miss',
        'Mme': 'Mrs',
        'Dr': 'Other',
        'Rev': 'Other',
        'Col': 'Other',
        'Major': 'Other',
        'Capt': 'Other',
        'Sir': 'Other',
        'Lady': 'Other',
        'Don': 'Other',
        'Dona': 'Other',
        'Jonkheer': 'Other',
        'Countess': 'Other'
    }
    
    combined['Title'] = combined['Title'].map(title_mapping)
    combined['Title'] = combined['Title'].fillna('Other')
    
    # Create family size feature
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    
    # Create is_alone feature
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Fill missing embarked values with the most common value (S)
    combined['Embarked'] = combined['Embarked'].fillna('S')
    
    # Fill missing fares with the median fare for that passenger class
    combined['Fare'] = combined.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Create categorical fare feature
    combined['FareBand'] = pd.qcut(
        combined['Fare'], 
        q=4, 
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )
    
    # Age imputation (using median age by title and class)
    combined['Age'] = combined.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # If still missing, use title median
    combined['Age'] = combined.groupby(['Title'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # If still missing, use global median
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    
    # Create age bands
    combined['AgeBand'] = pd.cut(
        combined['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Elderly']
    )
    
    # Convert categorical features to string for proper handling
    for col in ['Title', 'FareBand', 'AgeBand']:
        combined[col] = combined[col].astype(str)
    
    # Split back into train and test
    train_processed = combined.iloc[:train.shape[0]].copy()
    test_processed = combined.iloc[train.shape[0]:].copy()
    test_processed = test_processed.drop('Survived', axis=1)
    
    return train_processed, test_processed

def build_robust_model():
    """
    Create a robust model that generalizes well
    """
    # Define feature categories for preprocessing
    numeric_features = ['Age', 'Fare', 'FamilySize']
    
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'FareBand', 'AgeBand']
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly listed
    )

    # Gradient Boosting with strong regularization to prevent overfitting
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,          # Shallow trees to prevent overfitting
        min_samples_split=8,  # Require more samples to split nodes
        min_samples_leaf=4,   # Require more samples in leaf nodes
        subsample=0.8,        # Use only 80% of data for each tree
        max_features=0.8,     # Use only 80% of features for each tree
        learning_rate=0.05,   # Slow learning rate
        random_state=42
    )
    
    # Create full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return model_pipeline

def evaluate_model(model, X, y):
    """
    Evaluate model using stratified cross-validation
    """
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='accuracy'
    )
    
    print(f"\nCross-Validation Results (5 folds):")
    print(f"Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Individual fold scores: {cv_scores}")
    
    return cv_scores.mean()

def generate_submission(model, X_train, y_train, X_test, test_ids):
    """
    Train model on all training data and generate predictions
    """
    # Train the model on all available training data
    print("\nTraining final model on all training data...")
    model.fit(X_train, y_train)
    
    # Generate predictions
    print("Generating predictions for test set...")
    predictions = model.predict(X_test)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions.astype(int)
    })
    
    # Save submission
    submission_file = 'robust_submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    
    return submission

def main():
    print("=" * 60)
    print("Titanic: Robust Model")
    print("=" * 60)
    
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Perform feature engineering
    print("\nPerforming feature engineering...")
    train_processed, test_processed = basic_feature_engineering(train_data, test_data)
    
    # Define features for the model
    exclude_columns = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']
    feature_columns = [col for col in train_processed.columns if col not in exclude_columns]
    
    # Create feature matrix and target vector
    X = train_processed[feature_columns]
    y = train_processed['Survived']
    X_test = test_processed[feature_columns]
    
    # Build and evaluate model
    print("\nBuilding and evaluating robust model...")
    model = build_robust_model()
    
    # Evaluate with cross-validation
    cv_accuracy = evaluate_model(model, X, y)
    
    # Generate submission file
    submission = generate_submission(model, X, y, X_test, test_processed['PassengerId'])
    
    print("\nProcess completed successfully!")
    print("=" * 60)
    print(f"Expected accuracy (based on CV): {cv_accuracy:.4f}")
    print("Submit the 'robust_submission.csv' file to Kaggle for a better score.")
    print("=" * 60)

if __name__ == "__main__":
    main() 