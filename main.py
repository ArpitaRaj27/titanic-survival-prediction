# main.py - UPDATED WITH BETTER FONT SIZES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

def create_eda_plots(train_df):
    """Create Exploratory Data Analysis plots with readable font sizes"""
    print("\n📊 Creating Exploratory Data Analysis plots...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 18))
    
    # 1. Survival Distribution
    ax1 = plt.subplot(3, 3, 1)
    survival_counts = train_df['Survived'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(['Not Survived', 'Survived'], survival_counts.values, 
                   color=colors, edgecolor='black')
    ax1.set_title('Survival Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='both', labelsize=11)
    
    # Adding count labels on bars
    for bar, count in zip(bars, survival_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                str(count), ha='center', fontsize=12, fontweight='bold')
    
    # 2. Survival by Gender
    ax2 = plt.subplot(3, 3, 2)
    gender_survival = pd.crosstab(train_df['Sex'], train_df['Survived'])
    gender_survival.columns = ['Not Survived', 'Survived']
    gender_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax2)
    ax2.set_title('Survival by Gender', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Gender', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xticklabels(['Male', 'Female'], rotation=0)
    
    # 3. Survival by Class
    ax3 = plt.subplot(3, 3, 3)
    class_survival = pd.crosstab(train_df['Pclass'], train_df['Survived'])
    class_survival.columns = ['Not Survived', 'Survived']
    class_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax3)
    ax3.set_title('Survival by Passenger Class', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Passenger Class', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.tick_params(axis='both', labelsize=11)
    ax3.set_xticklabels(['1st', '2nd', '3rd'], rotation=0)
    
    # 4. Age Distribution
    ax4 = plt.subplot(3, 3, 4)
    survived_age = train_df[train_df['Survived'] == 1]['Age'].dropna()
    not_survived_age = train_df[train_df['Survived'] == 0]['Age'].dropna()
    ax4.hist([not_survived_age, survived_age], bins=20, 
             label=['Not Survived', 'Survived'], 
             color=['#ff6b6b', '#4ecdc4'], alpha=0.7, edgecolor='black')
    ax4.set_title('Age Distribution by Survival', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Age', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.legend(fontsize=11)
    ax4.tick_params(axis='both', labelsize=11)
    
    # 5. Fare Distribution
    ax5 = plt.subplot(3, 3, 5)
    survived_fare = train_df[train_df['Survived'] == 1]['Fare'].dropna()
    not_survived_fare = train_df[train_df['Survived'] == 0]['Fare'].dropna()
    ax5.hist([not_survived_fare, survived_fare], bins=30, 
             label=['Not Survived', 'Survived'],
             color=['#ff6b6b', '#4ecdc4'], alpha=0.7, edgecolor='black')
    ax5.set_title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Fare', fontsize=12)
    ax5.set_ylabel('Frequency', fontsize=12)
    ax5.legend(fontsize=11)
    ax5.tick_params(axis='both', labelsize=11)
    
    # 6. Embarkment Port
    ax6 = plt.subplot(3, 3, 6)
    embarked_survival = pd.crosstab(train_df['Embarked'], train_df['Survived'])
    embarked_survival.columns = ['Not Survived', 'Survived']
    embarked_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax6)
    ax6.set_title('Survival by Embarkment Port', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Embarkment Port', fontsize=12)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.legend(fontsize=11)
    ax6.tick_params(axis='both', labelsize=11)
    ax6.set_xticklabels(['Southampton', 'Cherbourg', 'Queenstown'], rotation=0)
    
    # 7. Family Size
    ax7 = plt.subplot(3, 3, 7)
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    family_survival = pd.crosstab(train_df['FamilySize'], train_df['Survived'])
    family_survival.columns = ['Not Survived', 'Survived']
    family_survival.plot(kind='bar', color=['#ff6b6b', '#4ecdc4'], ax=ax7)
    ax7.set_title('Survival by Family Size', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Family Size', fontsize=12)
    ax7.set_ylabel('Count', fontsize=12)
    ax7.legend(fontsize=11)
    ax7.tick_params(axis='both', labelsize=11)
    
    # 8. Correlation Heatmap
    ax8 = plt.subplot(3, 3, 8)
    plot_data = train_df.copy()
    plot_data['Sex'] = plot_data['Sex'].map({'male': 0, 'female': 1})
    plot_data['Embarked'] = plot_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    plot_data = plot_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']].dropna()
    correlation = plot_data.corr()
    
    # Create heatmap with larger text
    heatmap = sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                         square=True, ax=ax8, fmt='.2f',
                         annot_kws={'size': 11, 'weight': 'bold'},  # Annotation font
                         cbar_kws={'shrink': 0.8})
    
    ax8.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # Adjusting tick labels
    ax8.set_xticklabels(ax8.get_xticklabels(), fontsize=11, rotation=45, ha='right')
    ax8.set_yticklabels(ax8.get_yticklabels(), fontsize=11, rotation=0)
    
    # 9. Missing Values Heatmap
    ax9 = plt.subplot(3, 3, 9)
    missing_data = train_df.isnull()
    sns.heatmap(missing_data, cbar=False, cmap='viridis', ax=ax9)
    ax9.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Features', fontsize=12)
    ax9.set_ylabel('Passengers', fontsize=12)
    
    # Adjusting x-tick labels for missing values heatmap
    ax9.set_xticklabels(ax9.get_xticklabels(), fontsize=11, rotation=45, ha='right')
    ax9.set_yticklabels([])  # Hiding y-tick labels for clarity
    
    plt.suptitle('Titanic Dataset - Exploratory Data Analysis', 
                fontsize=18, fontweight='bold', y=1.02)
    
    # Adjusting layout with more padding
    plt.tight_layout(pad=3.0)
    plt.savefig('titanic_eda_large.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: titanic_eda_large.png (with better font sizes)")
    
    # Also creating a simplified version with just the correlation heatmap
    create_correlation_plot(train_df)

def create_correlation_plot(train_df):
    """Creating a standalone correlation heatmap with large, clear text"""
    plt.figure(figsize=(14, 10))
    
    # Preparing data
    plot_data = train_df.copy()
    plot_data['Sex'] = plot_data['Sex'].map({'male': 0, 'female': 1})
    plot_data['Embarked'] = plot_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    plot_data['FamilySize'] = plot_data['SibSp'] + plot_data['Parch'] + 1
    plot_data = plot_data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']].dropna()
    
    correlation = plot_data.corr()
    
    # Creating heatmap with very clear text
    ax = sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f',
                    annot_kws={'size': 14, 'weight': 'bold'},
                    cbar_kws={'shrink': 0.8})
    
    plt.title('Feature Correlation Heatmap - Titanic Dataset', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Setting axis labels with readable font size
    ax.set_xlabel('Features', fontsize=14, labelpad=15)
    ax.set_ylabel('Features', fontsize=14, labelpad=15)
    
    # Adjusting tick labels
    feature_names = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 
                     'Parch', 'Fare', 'Embarked', 'FamilySize']
    ax.set_xticklabels(feature_names, fontsize=13, rotation=45, ha='right')
    ax.set_yticklabels(feature_names, fontsize=13, rotation=0)
    
    # Adding an interpretation note 
    plt.figtext(0.5, 0.01, 
                'Note: Values close to +1 indicate strong positive correlation,\n'
                'values close to -1 indicate strong negative correlation,\n'
                'values near 0 indicate little to no correlation.',
                ha='center', fontsize=11, style='italic', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
    
    plt.tight_layout(pad=2.0)
    plt.savefig('correlation_heatmap_large.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: correlation_heatmap_large.png (standalone with large text)")

def main():
    print("🚢 Titanic Survival Prediction Project")
    print("=" * 50)
    
    # Loading data
    print("\n1. 📥 Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"   Training data shape: {train_df.shape}")
    print(f"   Test data shape: {test_df.shape}")
    
    # Creating EDA plots with better font sizes
    create_eda_plots(train_df)
    
    # Preprocessing data
    print("\n2. 🔧 Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Preprocessing training data
    X_train, y_train = preprocessor.preprocess_train(train_df)
    print(f"   Preprocessed training features shape: {X_train.shape}")
    print(f"   Features: {list(X_train.columns)}")
    
    # Training and evaluating models
    print("\n3. 🤖 Training models...")
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(X_train, y_train)
    
    # Creating model comparison plot
    trainer.create_model_comparison_plot(results)
    
    # Using the best model from evaluation
    best_model_name = trainer.best_model
    best_model = results[best_model_name]['model']
    trainer.save_model(best_model, 'titanic_model.pkl')
    
    # Making predictions on test data
    print("\n4. 🔮 Making predictions on test data...")
    
    # Preprocess test data USING THE SAME PREPROCESSOR
    X_test, passenger_ids = preprocessor.preprocess_test(test_df)
    
    # To Check if features match
    print(f"   Training features: {X_train.columns.tolist()}")
    print(f"   Test features: {X_test.columns.tolist()}")
    
    # Ensuring same number of features
    if X_train.shape[1] != X_test.shape[1]:
        print(f"   WARNING: Feature mismatch! Train: {X_train.shape[1]}, Test: {X_test.shape[1]}")
        print("   Attempting to fix...")
        
        # Finding missing features
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        missing_in_train = set(X_test.columns) - set(X_train.columns)
        
        if missing_in_test:
            print(f"   Adding missing features to test: {missing_in_test}")
            for col in missing_in_test:
                X_test[col] = 0
        
        if missing_in_train:
            print(f"   Removing extra features from test: {missing_in_train}")
            X_test = X_test.drop(columns=list(missing_in_train))
        
        # Reorder to match
        X_test = X_test[X_train.columns]
    
    # Loading the best model and make predictions
    loaded_model = trainer.load_model('titanic_model.pkl')
    predictions = loaded_model.predict(X_test)
    
    # Creating submission file
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print("   ✓ Submission file saved as 'submission.csv'")
    
    # Displaying results
    print("\n" + "=" * 50)
    print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    print("\n📁 FILES CREATED:")
    print("   1. titanic_eda_large.png - EDA plots (large text)")
    print("   2. correlation_heatmap_large.png - Just correlation (very large text)")
    print("   3. confusion_matrix_logistic_regression.png")
    print("   4. confusion_matrix_random_forest.png")
    print("   5. model_comparison.png - Model performance")
    print("   6. titanic_model.pkl - Trained model")
    print("   7. submission.csv - Kaggle submission file")
    
    print(f"\n📊 MODEL PERFORMANCE:")
    for model_name, result in results.items():
        name = model_name.replace('_', ' ').title()
        print(f"   • {name}: {result['accuracy']:.4f} accuracy")
    
    print(f"\n🎯 BEST MODEL: {trainer.best_model.replace('_', ' ').title()}")
    
    print("\n🔍 KEY FINDINGS FROM VISUALIZATIONS:")
    print("   • Correlation Heatmap shows relationships between features")
    print("   - Survived vs Sex: Strong correlation (women more likely to survive)")
    print("   - Survived vs Pclass: Negative correlation (higher class = better survival)")
    print("   - Survived vs Fare: Positive correlation (higher fare = better survival)")
    
    print("\n✅ Done! Open 'titanic_eda_large.png' and 'correlation_heatmap_large.png' to see clear visualizations.")
    
if __name__ == "__main__":
    main()
