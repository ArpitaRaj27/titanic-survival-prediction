# titanic_minimal.py - Absolutely minimal version
import pandas as pd

print("=== Minimal Titanic Survival Prediction ===")

# Load data
df = pd.read_csv('data/train.csv')
print(f"Data loaded: {len(df)} passengers")

# Show basic info
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 passengers:")
print(df[['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age']].head())

# Basic survival statistics
survival_rate = df['Survived'].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.1f}%")

# Survival by gender
male_survival = df[df['Sex'] == 'male']['Survived'].mean() * 100
female_survival = df[df['Sex'] == 'female']['Survived'].mean() * 100
print(f"Male survival rate: {male_survival:.1f}%")
print(f"Female survival rate: {female_survival:.1f}%")

# Survival by class
for pclass in [1, 2, 3]:
    class_survival = df[df['Pclass'] == pclass]['Survived'].mean() * 100
    print(f"Class {pclass} survival rate: {class_survival:.1f}%")