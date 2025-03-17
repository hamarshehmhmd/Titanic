#!/usr/bin/env python
"""
Titanic Model Comparison

This script compares different machine learning models on the Titanic dataset.
It evaluates multiple algorithms and reports their performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

# Import feature engineering functions from titanic_model.py
from titanic_model import load_data, engineer_features

warnings.filterwarnings('ignore')

def compare_models(X, y, models, cv=5):
    """
    Compare multiple models using cross-validation.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target vector
    models : dict
        Dictionary of models to compare
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    results_df : DataFrame
        DataFrame with model performance metrics
    """
    results = []
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Define preprocessing pipeline
    numeric_features = ['Age', 'Fare', 'FamilySize']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Cross-validate each model
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            pipeline, X, y, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True
        )
        
        # Collect results
        model_results = {
            'Model': name,
            'Accuracy': cv_results['test_accuracy'].mean(),
            'Accuracy Std': cv_results['test_accuracy'].std(),
            'Precision': cv_results['test_precision'].mean(),
            'Recall': cv_results['test_recall'].mean(),
            'F1 Score': cv_results['test_f1'].mean(),
            'ROC AUC': cv_results['test_roc_auc'].mean(),
            'Training Accuracy': cv_results['train_accuracy'].mean(),
            'Training ROC AUC': cv_results['train_roc_auc'].mean(),
            'Fit Time': cv_results['fit_time'].mean(),
            'Score Time': cv_results['score_time'].mean()
        }
        
        results.append(model_results)
    
    # Convert to DataFrame and sort by accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    return results_df

def plot_model_comparison(results_df):
    """
    Plot model comparison results.
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame with model performance metrics
    """
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        x='Model', 
        y='Accuracy', 
        data=results_df,
        palette='viridis'
    )
    g.set_title('Model Accuracy Comparison', fontsize=15)
    g.set_xlabel('Model', fontsize=12)
    g.set_ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add accuracy values on top of bars
    for i, row in results_df.iterrows():
        g.text(
            i, row['Accuracy'] + 0.01, 
            f"{row['Accuracy']:.4f}", 
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.show()
    
    # Training vs Test accuracy (checking for overfitting)
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    plot_data = pd.melt(
        results_df, 
        id_vars=['Model'], 
        value_vars=['Accuracy', 'Training Accuracy'],
        var_name='Dataset',
        value_name='Accuracy_Score'
    )
    
    g = sns.barplot(
        x='Model', 
        y='Accuracy_Score', 
        hue='Dataset',
        data=plot_data,
        palette=['#66b3ff', '#ff9999']
    )
    g.set_title('Training vs. Test Accuracy Comparison', fontsize=15)
    g.set_xlabel('Model', fontsize=12)
    g.set_ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dataset')
    
    plt.tight_layout()
    plt.savefig('train_test_accuracy_comparison.png')
    plt.show()
    
    # ROC AUC comparison
    plt.figure(figsize=(12, 6))
    g = sns.barplot(
        x='Model', 
        y='ROC AUC', 
        data=results_df,
        palette='viridis'
    )
    g.set_title('Model ROC AUC Comparison', fontsize=15)
    g.set_xlabel('Model', fontsize=12)
    g.set_ylabel('ROC AUC', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add ROC AUC values on top of bars
    for i, row in results_df.iterrows():
        g.text(
            i, row['ROC AUC'] + 0.01, 
            f"{row['ROC AUC']:.4f}", 
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig('model_roc_auc_comparison.png')
    plt.show()
    
    # Time performance comparison
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    time_data = pd.melt(
        results_df, 
        id_vars=['Model'], 
        value_vars=['Fit Time', 'Score Time'],
        var_name='Time Type',
        value_name='Time_Seconds'
    )
    
    g = sns.barplot(
        x='Model', 
        y='Time_Seconds', 
        hue='Time Type',
        data=time_data,
        palette=['#66b3ff', '#ff9999']
    )
    g.set_title('Model Training and Scoring Time Comparison', fontsize=15)
    g.set_xlabel('Model', fontsize=12)
    g.set_ylabel('Time (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Time Type')
    
    plt.tight_layout()
    plt.savefig('model_time_comparison.png')
    plt.show()

def main():
    """
    Main function to run the model comparison.
    """
    print("=" * 60)
    print("Titanic Model Comparison")
    print("=" * 60)
    
    # Load data
    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return
    
    # Engineer features
    print("\nPerforming feature engineering...")
    train_processed, _ = engineer_features(train_data, test_data)
    
    # Define features and target
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    X = train_processed[features]
    y = train_processed['Survived']
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Compare models
    print("\nComparing models using 5-fold cross-validation...")
    results = compare_models(X, y, models, cv=5)
    
    # Display results
    print("\nModel Comparison Results:")
    print("-" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results.to_string(index=False))
    
    # Save results to CSV
    results.to_csv('model_comparison_results.csv', index=False)
    print("\nResults saved to model_comparison_results.csv")
    
    # Plot comparison
    print("\nPlotting model comparison charts...")
    plot_model_comparison(results)
    
    print("\nModel comparison completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 