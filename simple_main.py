# simple_main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== Titanic Survival Prediction ===")
print("Loading data...")

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Basic preprocessing function
def preprocess_data(df, is_train=True, age_median=None, fare_median=None):
    df = df.copy()
    
    # Extract title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = np.where(df['FamilySize'] > 1, 0, 1)
    
    # Fill missing values
    if age_median is None:
        age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    
    df['Embarked'].fillna('S', inplace=True)
    
    if fare_median is None:
        fare_median = df['Fare'].median()
    df['Fare'].fillna(fare_median, inplace=True)
    
    # Convert categorical to numerical
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Title mapping (simplified)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'].fillna(0, inplace=True)
    
    # Drop columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    if 'Survived' in df.columns and not is_train:
        columns_to_drop.append('Survived')
    
    if is_train:
        passenger_ids = None
        X = df.drop(columns_to_drop + ['Survived'], axis=1)
        y = df['Survived']
    else:
        passenger_ids = df['PassengerId']
        X = df.drop(columns_to_drop, axis=1)
        y = None
    
    return X, y, passenger_ids, age_median, fare_median

print("\nPreprocessing training data...")
X_train, y_train, _, age_median, fare_median = preprocess_data(train_df, is_train=True)

print(f"Training features: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")

# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining set: {X_train_split.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_split, y_train_split)

lr_pred = lr_model.predict(X_val)
lr_accuracy = accuracy_score(y_val, lr_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Train Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)

rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Train on full training data with best model
print("\nTraining final model on all training data...")
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# Make predictions on test data
print("\nMaking predictions on test data...")
X_test, _, passenger_ids, _, _ = preprocess_data(
    test_df, is_train=False, age_median=age_median, fare_median=fare_median
)

test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission saved to 'submission.csv'")
print(f"First few predictions:")
print(submission.head(10))

# Feature importance
print("\nTop 10 Feature Importances:")
importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))