#!/usr/bin/env python3
"""
Titanic: Machine Learning from Disaster
Improved model with enhanced feature engineering and optimized Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import warnings

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set(font_scale=1.2)

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

def advanced_feature_engineering(train_df, test_df):
    """
    Advanced feature engineering with improved handling of missing values
    """
    # Create copies to avoid modifying originals
    train = train_df.copy()
    test = test_df.copy()
    
    # Combine datasets for consistent feature engineering
    test['Survived'] = np.nan
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Extract titles from names with improved regex
    combined['Title'] = combined['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group titles more effectively
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Officer',
        'Rev': 'Officer',
        'Col': 'Officer',
        'Major': 'Officer',
        'Mlle': 'Miss',
        'Countess': 'Royalty',
        'Ms': 'Miss',
        'Lady': 'Royalty',
        'Jonkheer': 'Royalty',
        'Don': 'Royalty',
        'Dona': 'Royalty',
        'Mme': 'Mrs',
        'Capt': 'Officer',
        'Sir': 'Royalty'
    }
    combined['Title'] = combined['Title'].map(title_mapping)
    
    # Fill in any missing titles with most common
    combined['Title'] = combined['Title'].fillna('Mr')
    
    # Create family size feature and family groups
    combined['FamilySize'] = combined['SibSp'] + combined['Parch'] + 1
    combined['IsAlone'] = (combined['FamilySize'] == 1).astype(int)
    
    # Create family groups
    combined['FamilyGroup'] = pd.cut(
        combined['FamilySize'],
        bins=[0, 1, 4, 11],
        labels=['Alone', 'Small', 'Large']
    )
    
    # Extract surname from the Name column
    combined['Surname'] = combined['Name'].apply(lambda x: x.split(',')[0].strip())
    
    # Create family survival rate feature
    # This is more sophisticated and captures family correlations
    family_survival = {}
    for i, row in combined.iloc[:train.shape[0]].iterrows():
        if row['Survived'] != np.nan:
            family_survival[(row['Surname'], row['Parch'], row['SibSp'])] = row['Survived']
    
    # Apply family survival rates to each passenger
    def get_family_survival(row):
        key = (row['Surname'], row['Parch'], row['SibSp'])
        if key in family_survival:
            return family_survival[key]
        else:
            return 0.5  # No information
    
    combined['FamilySurvived'] = combined.apply(get_family_survival, axis=1)
    
    # Extract cabin deck information (first letter of cabin)
    combined['Cabin_Deck'] = combined['Cabin'].astype(str).str[0]
    combined.loc[combined['Cabin_Deck'] == 'n', 'Cabin_Deck'] = 'U'  # Unknown
    
    # Create a feature for ticket frequency
    ticket_counts = combined['Ticket'].value_counts()
    combined['TicketFreq'] = combined['Ticket'].map(ticket_counts)
    
    # Extract ticket prefix
    combined['TicketPrefix'] = combined['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]) 
                                               if len(x.split(' ')) > 1 else 'XXX')
    
    # Extract numeric part of ticket and create fare per person
    combined['TicketNumber'] = combined['Ticket'].apply(lambda x: x.split(' ')[-1] if x.split(' ')[-1].isdigit() else 0)
    combined['TicketNumber'] = combined['TicketNumber'].astype(int)
    combined['FarePerPerson'] = combined['Fare'] / combined['FamilySize']
    
    # Fill missing embarked values with most common (S)
    combined['Embarked'] = combined['Embarked'].fillna('S')
    
    # Fill missing fare values with median by Pclass and Embarked
    combined['Fare'] = combined.groupby(['Pclass', 'Embarked'])['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # For any remaining missing fares, use the median for the passenger class
    combined['Fare'] = combined.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Create fare categories
    combined['FareCategory'] = pd.qcut(
        combined['Fare'], 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Age imputation using Title, Pclass, and Sex
    # First fill with group medians
    combined['Age'] = combined.groupby(['Title', 'Pclass', 'Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # For any remaining missing ages, use Title median
    combined['Age'] = combined.groupby(['Title'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # If still any missing ages, use overall median
    combined['Age'] = combined['Age'].fillna(combined['Age'].median())
    
    # Create age categories and child/adult indicators
    combined['AgeGroup'] = pd.cut(
        combined['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teenager', 'Young Adult', 'Adult', 'Elderly']
    )
    combined['IsChild'] = (combined['Age'] < 16).astype(int)
    
    # Create interaction features
    combined['Pclass_Sex'] = combined['Pclass'].astype(str) + '_' + combined['Sex']
    combined['Pclass_Age'] = combined['Pclass'].astype(str) + '_' + combined['AgeGroup'].astype(str)
    combined['Sex_Age'] = combined['Sex'] + '_' + combined['AgeGroup'].astype(str)
    combined['Embarked_Pclass'] = combined['Embarked'] + '_' + combined['Pclass'].astype(str)
    
    # Split back into train and test first to calculate survival rates properly
    train_processed = combined.iloc[:train.shape[0]].copy()
    test_processed = combined.iloc[train.shape[0]:].copy()
    
    # Create survival chance by sex, pclass, and age group
    for col in ['Sex', 'Pclass', 'Embarked']:
        survival_rate = train_processed.groupby(col)['Survived'].mean()
        # Map back to both train and test
        train_processed[f'{col}_SurvivalRate'] = train_processed[col].map(survival_rate).fillna(0.5)
        test_processed[f'{col}_SurvivalRate'] = test_processed[col].map(survival_rate).fillna(0.5)
    
    # Handle AgeGroup separately since it's a categorical variable
    # Convert to string first to avoid issues with categorical data
    train_processed['AgeGroup'] = train_processed['AgeGroup'].astype(str)
    test_processed['AgeGroup'] = test_processed['AgeGroup'].astype(str)
    
    survival_rate = train_processed.groupby('AgeGroup')['Survived'].mean()
    train_processed['AgeGroup_SurvivalRate'] = train_processed['AgeGroup'].map(survival_rate).fillna(0.5)
    test_processed['AgeGroup_SurvivalRate'] = test_processed['AgeGroup'].map(survival_rate).fillna(0.5)
    
    # Drop the Survived column from test data
    test_processed = test_processed.drop('Survived', axis=1)
    
    return train_processed, test_processed

def build_optimized_model():
    """
    Create an optimized gradient boosting model
    """
    # Define feature categories for preprocessing
    numeric_features = [
        'Age', 'Fare', 'FamilySize', 'TicketFreq', 'FarePerPerson', 'FamilySurvived',
        'Sex_SurvivalRate', 'Pclass_SurvivalRate', 'AgeGroup_SurvivalRate', 'Embarked_SurvivalRate'
    ]
    
    categorical_features = [
        'Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'FamilyGroup', 
        'Cabin_Deck', 'FareCategory', 'AgeGroup', 'IsChild'
    ]
    
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
        ])

    # Gradient Boosting with optimized hyperparameters
    # These parameters are based on common best practices for Titanic
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        subsample=0.8,
        random_state=42
    )
    
    # Create a voting classifier with other strong models
    ensemble_model = VotingClassifier(estimators=[
        ('gb', gb_model),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    ], voting='soft')
    
    # Create full pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble_model)
    ])
    
    return model_pipeline

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate model using cross-validation with stratification
    """
    # Define cross-validation strategy with stratification
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(
        model, X, y, 
        cv=cv_strategy, 
        scoring='accuracy'
    )
    
    print(f"\nCross-Validation Results ({cv} folds):")
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
    submission_file = 'improved_submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission file created: {submission_file}")
    
    return submission

def main():
    print("=" * 60)
    print("Titanic: Improved Model")
    print("=" * 60)
    
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Perform advanced feature engineering
    print("\nPerforming advanced feature engineering...")
    train_processed, test_processed = advanced_feature_engineering(train_data, test_data)
    
    # Define features for the model
    # Exclude unnecessary columns like Name, Ticket, Cabin, etc.
    exclude_columns = ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Surname', 'TicketNumber', 'TicketPrefix']
    feature_columns = [col for col in train_processed.columns if col not in exclude_columns]
    
    # Create feature matrix and target vector
    X = train_processed[feature_columns]
    y = train_processed['Survived']
    X_test = test_processed[feature_columns]
    
    # Build and evaluate model
    print("\nBuilding and evaluating optimized model...")
    model = build_optimized_model()
    
    # Evaluate with cross-validation
    cv_accuracy = evaluate_model(model, X, y, cv=10)
    
    # Generate submission file
    submission = generate_submission(model, X, y, X_test, test_processed['PassengerId'])
    
    print("\nProcess completed successfully!")
    print("=" * 60)
    print(f"Expected accuracy (based on CV): {cv_accuracy:.4f}")
    print("Submit the 'improved_submission.csv' file to Kaggle for a better score.")
    print("=" * 60)

if __name__ == "__main__":
    main() 