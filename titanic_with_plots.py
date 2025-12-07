# titanic_with_plots.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os

warnings.filterwarnings('ignore')

# Create a directory for plots
os.makedirs('plots', exist_ok=True)

print("=== Titanic Survival Prediction with Visualizations ===")
print("Plots will be saved in the 'plots' folder\n")

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# 1. EXPLORATORY DATA ANALYSIS PLOTS
print("\n1. Creating exploratory data analysis plots...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Plot 1: Survival Distribution
plt.figure(figsize=(10, 6))
survival_counts = train_df['Survived'].value_counts()
plt.bar(['Not Survived (0)', 'Survived (1)'], survival_counts.values, 
        color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
plt.title('Titanic Survival Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Survival Status', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add count labels on bars
for i, count in enumerate(survival_counts.values):
    plt.text(i, count + 10, str(count), ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/1_survival_distribution.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/1_survival_distribution.png")
plt.close()

# Plot 2: Survival by Gender
plt.figure(figsize=(10, 6))
gender_survival = pd.crosstab(train_df['Sex'], train_df['Survived'])
gender_survival.columns = ['Not Survived', 'Survived']
ax = gender_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
plt.title('Survival by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# Add count labels
for container in ax.containers:
    ax.bar_label(container, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/2_survival_by_gender.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/2_survival_by_gender.png")
plt.close()

# Plot 3: Survival by Passenger Class
plt.figure(figsize=(10, 6))
class_survival = pd.crosstab(train_df['Pclass'], train_df['Survived'])
class_survival.columns = ['Not Survived', 'Survived']
ax = class_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], edgecolor='black')
plt.title('Survival by Passenger Class', fontsize=16, fontweight='bold')
plt.xlabel('Passenger Class', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# Add count labels
for container in ax.containers:
    ax.bar_label(container, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/3_survival_by_class.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/3_survival_by_class.png")
plt.close()

# Plot 4: Age Distribution by Survival
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
survived_age = train_df[train_df['Survived'] == 1]['Age'].dropna()
not_survived_age = train_df[train_df['Survived'] == 0]['Age'].dropna()

plt.hist(survived_age, bins=30, alpha=0.7, color='#4ecdc4', edgecolor='black', label='Survived')
plt.title('Age Distribution - Survived', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(not_survived_age, bins=30, alpha=0.7, color='#ff6b6b', edgecolor='black', label='Not Survived')
plt.title('Age Distribution - Not Survived', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.legend()

plt.suptitle('Age Distribution by Survival Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/4_age_distribution.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/4_age_distribution.png")
plt.close()

# Plot 5: Fare Distribution by Survival
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
survived_fare = train_df[train_df['Survived'] == 1]['Fare'].dropna()
plt.hist(survived_fare, bins=30, alpha=0.7, color='#4ecdc4', edgecolor='black', label='Survived')
plt.title('Fare Distribution - Survived', fontsize=14)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
not_survived_fare = train_df[train_df['Survived'] == 0]['Fare'].dropna()
plt.hist(not_survived_fare, bins=30, alpha=0.7, color='#ff6b6b', edgecolor='black', label='Not Survived')
plt.title('Fare Distribution - Not Survived', fontsize=14)
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.legend()

plt.suptitle('Fare Distribution by Survival Status', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/5_fare_distribution.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/5_fare_distribution.png")
plt.close()

# Plot 6: Correlation Heatmap
plt.figure(figsize=(10, 8))
# Prepare data for correlation
plot_data = train_df.copy()
plot_data['Sex'] = plot_data['Sex'].map({'male': 0, 'female': 1})
plot_data['Embarked'] = plot_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
plot_data = plot_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()

correlation = plot_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/6_correlation_heatmap.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/6_correlation_heatmap.png")
plt.close()

# 2. DATA PREPROCESSING
print("\n2. Preprocessing data...")

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
    
    # Title mapping
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

# Preprocess data
X_train, y_train, _, age_median, fare_median = preprocess_data(train_df, is_train=True)
print(f"   Training features: {X_train.shape}")
print(f"   Features: {list(X_train.columns)}")

# 3. MODEL TRAINING AND EVALUATION
print("\n3. Training models...")

# Split data
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"   Training set: {X_train_split.shape[0]} samples")
print(f"   Validation set: {X_val.shape[0]} samples")

# Train Logistic Regression
print("\n   Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_split, y_train_split)
lr_pred = lr_model.predict(X_val)
lr_accuracy = accuracy_score(y_val, lr_pred)
print(f"   ✓ Logistic Regression Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")

# Train Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_split, y_train_split)
rf_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_pred)
print(f"   ✓ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# 4. MODEL VISUALIZATIONS
print("\n4. Creating model evaluation plots...")

# Plot 7: Model Comparison
plt.figure(figsize=(10, 6))
models = ['Logistic Regression', 'Random Forest']
accuracies = [lr_accuracy, rf_accuracy]
colors = ['#ff9f43', '#36a2eb']

bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.0)
plt.grid(axis='y', alpha=0.3)

# Add accuracy labels on bars
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{accuracy:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/7_model_comparison.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/7_model_comparison.png")
plt.close()

# Plot 8: Confusion Matrix for Random Forest (better model)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_val, rf_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Not Survived', 'Predicted Survived'],
            yticklabels=['Actual Not Survived', 'Actual Survived'])
plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('plots/8_confusion_matrix.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/8_confusion_matrix.png")
plt.close()

# Plot 9: Feature Importance for Random Forest
plt.figure(figsize=(12, 6))
feature_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(range(len(feature_importances)), feature_importances['importance'], 
         color='#36a2eb', edgecolor='black')
plt.yticks(range(len(feature_importances)), feature_importances['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()  # Most important on top
plt.grid(axis='x', alpha=0.3)

# Add importance values
for i, v in enumerate(feature_importances['importance']):
    plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('plots/9_feature_importance.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: plots/9_feature_importance.png")
plt.close()

# 5. MAKE PREDICTIONS
print("\n5. Making predictions on test data...")

# Train final model on all data
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# Preprocess test data
X_test, _, passenger_ids, _, _ = preprocess_data(
    test_df, is_train=False, age_median=age_median, fare_median=fare_median
)

# Make predictions
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Survived': test_predictions
})

submission.to_csv('titanic_predictions.csv', index=False)
print(f"   ✓ Predictions saved to 'titanic_predictions.csv'")

# Show first few predictions
print("\n   First 10 predictions:")
print(submission.head(10))

# 6. SUMMARY STATISTICS
print("\n" + "="*50)
print("PROJECT SUMMARY")
print("="*50)

# Survival statistics
total_passengers = len(train_df)
survived_count = train_df['Survived'].sum()
survival_rate = (survived_count / total_passengers) * 100

print(f"\nTraining Data Statistics:")
print(f"  • Total passengers: {total_passengers}")
print(f"  • Survived: {survived_count}")
print(f"  • Not survived: {total_passengers - survived_count}")
print(f"  • Overall survival rate: {survival_rate:.1f}%")

# Gender statistics
male_survival = train_df[train_df['Sex'] == 'male']['Survived'].mean() * 100
female_survival = train_df[train_df['Sex'] == 'female']['Survived'].mean() * 100
print(f"\nSurvival by Gender:")
print(f"  • Male survival rate: {male_survival:.1f}%")
print(f"  • Female survival rate: {female_survival:.1f}%")

# Class statistics
for pclass in [1, 2, 3]:
    class_data = train_df[train_df['Pclass'] == pclass]
    class_survival = class_data['Survived'].mean() * 100
    print(f"  • Class {pclass} survival rate: {class_survival:.1f}%")

print(f"\nModel Performance:")
print(f"  • Logistic Regression: {lr_accuracy*100:.2f}%")
print(f"  • Random Forest: {rf_accuracy*100:.2f}%")

print(f"\nFiles Created:")
print(f"  • Predictions: titanic_predictions.csv")
print(f"  • Plots: 9 plots saved in 'plots/' folder")
print(f"    1. plots/1_survival_distribution.png")
print(f"    2. plots/2_survival_by_gender.png")
print(f"    3. plots/3_survival_by_class.png")
print(f"    4. plots/4_age_distribution.png")
print(f"    5. plots/5_fare_distribution.png")
print(f"    6. plots/6_correlation_heatmap.png")
print(f"    7. plots/7_model_comparison.png")
print(f"    8. plots/8_confusion_matrix.png")
print(f"    9. plots/9_feature_importance.png")

print("\n✅ Project completed successfully!")
print("Open the 'plots' folder to see all visualizations.")