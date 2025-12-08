# src/data_preprocessing.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.age_imputer = SimpleImputer(strategy='median')
        self.embarked_imputer = SimpleImputer(strategy='most_frequent')
        self.fare_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.feature_order = None  # To store feature order
        
    def preprocess_train(self, df):
        """Preprocess training data"""
        df = df.copy()
        
        # Extract title from name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                           'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fill missing values
        df['Age'] = self.age_imputer.fit_transform(df[['Age']])
        df['Embarked'] = self.embarked_imputer.fit_transform(df[['Embarked']])
        
        # Drop unnecessary columns
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        
        # Convert categorical to numerical
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Title mapping
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0).astype(int)
        
        # Separate features and target
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        # Scale numerical features
        numerical_features = ['Age', 'Fare', 'FamilySize']
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        # Store feature order for consistency
        self.feature_order = X.columns.tolist()
        
        return X, y
    
    def preprocess_test(self, df):
        """Preprocess test data"""
        df = df.copy()
        
        # Extract title
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                           'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Fill missing values
        df['Age'] = self.age_imputer.transform(df[['Age']])
        df['Fare'] = self.fare_imputer.fit_transform(df[['Fare']])
        df['Embarked'] = self.embarked_imputer.transform(df[['Embarked']])
        
        # Store PassengerId for submission
        passenger_ids = df['PassengerId'].copy()
        
        # Drop columns
        df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
        
        # Convert categorical to numerical (SAME MAPPING)
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        
        # Title mapping
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping).fillna(0).astype(int)
        
        # Scale features
        numerical_features = ['Age', 'Fare', 'FamilySize']
        df[numerical_features] = self.scaler.transform(df[numerical_features])
        
        # Ensure same feature order as training
        if self.feature_order:
            # Add any missing columns with 0
            for col in self.feature_order:
                if col not in df.columns:
                    df[col] = 0
            
            # Reorder columns to match training
            df = df[self.feature_order]
        
        return df, passenger_ids