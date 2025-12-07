# src/model_training.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        self.best_model = None
        
    def train_and_evaluate(self, X, y):
        """Train and evaluate multiple models - SAVES plots instead of showing"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X, y, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
            
            # Print classification report
            print(f"\nClassification Report for {name}:")
            print(classification_report(y_test, y_pred))
            
            # SAVE confusion matrix (NOT show)
            self.save_confusion_matrix(y_test, y_pred, name)
        
        # Select best model
        self.best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\n{'='*50}")
        print(f"Best model: {self.best_model}")
        
        return results
    
    def save_confusion_matrix(self, y_true, y_pred, model_name):
        """Save confusion matrix as image file with better font sizes"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    annot_kws={'size': 14, 'weight': 'bold'})
        plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, labelpad=15)
        plt.xlabel('Predicted Label', fontsize=14, labelpad=15)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=100)
        plt.close()
        print(f"   ✓ Saved: confusion_matrix_{model_name}.png")
    
    def create_model_comparison_plot(self, results):
        """Create and save model comparison plot"""
        plt.figure(figsize=(10, 6))
        
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        cv_means = [results[m]['cv_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Test Accuracy', color='#36a2eb', edgecolor='black')
        plt.bar(x + width/2, cv_means, width, label='CV Mean Accuracy', color='#ff6384', edgecolor='black')
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, [m.replace('_', ' ').title() for m in models])
        plt.legend()
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (acc, cv) in enumerate(zip(accuracies, cv_means)):
            plt.text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
            plt.text(i + width/2, cv + 0.01, f'{cv:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: model_comparison.png")
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning for Random Forest"""
        print("\nPerforming hyperparameter tuning for Random Forest...")
        
        # Simplified param grid for faster tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1  # Reduced cv for speed
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def save_model(self, model, filename):
        """Save trained model"""
        joblib.dump(model, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        return joblib.load(filename)