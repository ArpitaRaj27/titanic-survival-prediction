# simple_titanic.py - A minimal working version
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== Titanic Survival Prediction (Simplified) ===")

# Load data
try:
    train_df = pd.read_csv('data/train.csv')
    print(f"✓ Loaded training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: data/train.csv not found")
    print("Please download from: https://www.kaggle.com/c/titanic/data")
    exit()

# Show first few rows
print("\nFirst few rows of data:")
print(train_df.head())

# Basic preprocessing
print("\nPreprocessing data...")
# Fill missing values
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Convert categorical to numerical
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Create family size feature
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
X = train_df[features]
y = train_df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining data: {X_train.shape[0]} samples")
print(f"Test data: {X_test.shape[0]} samples")

# Train model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Show feature importance
print("\nFeature coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"  {feature:15s}: {coef:8.4f}")

# Make predictions on first 5 test samples
print("\nPredictions on first 5 test samples:")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    passenger_features = X_test.iloc[i].values
    print(f"  Sample {i+1}: Actual={actual}, Predicted={predicted}, "
          f"{'✓' if actual == predicted else '✗'}")