# Titanic Survival Prediction 🚢

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms.

![Titanic](https://img.shields.io/badge/Project-Titanic-blue)
![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Project Overview

This project implements a binary classification model to predict whether a passenger survived the Titanic disaster based on features like:
- Passenger class
- Gender
- Age
- Fare
- Family size
- Embarkment port

## 🎯 Project Goals

1. Perform Exploratory Data Analysis (EDA) on Titanic dataset
2. Clean and preprocess the data
3. Engineer new features
4. Train multiple machine learning models
5. Evaluate and compare model performance
6. Make predictions on test data for Kaggle submission

## 📊 Dataset

The dataset comes from Kaggle's [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data) competition.

**Files:**
- `train.csv` - Training data (891 passengers with survival labels)
- `test.csv` - Test data (418 passengers without survival labels)

## 🏗️ Project Structure
titanic-survival-prediction/
├── data/ # Dataset files
│ ├── train.csv # Training data
│ └── test.csv # Test data
├── src/ # Source code
│ ├── data_preprocessing.py # Data cleaning and preprocessing
│ └── model_training.py # Model training and evaluation
├── notebooks/ # Jupyter notebooks for exploration
├── main.py # Main execution script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Files to ignore in Git

text

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager

### Steps
1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
Create virtual environment:

bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# venv\Scripts\activate   # On Windows
Install dependencies:

bash
pip install -r requirements.txt
Download the dataset from Kaggle and place in data/ folder.

🚀 Usage
Run the complete project:

bash
python main.py

This will:
- Load and analyze the data
- Create visualizations
- Train machine learning models
- Generate predictions for test data

🤖 Machine Learning Models Implemented

1. Logistic Regression - Baseline model
2. Random Forest Classifier - Ensemble method

Model Performance:

1. Logistic Regression: ~81% accuracy
2. Random Forest: ~82-83% accuracy

📈 Key Findings

Exploratory Data Analysis:
1. Overall survival rate: 38.4%
2. Gender disparity: 74.2% of women survived vs 18.9% of men
3. Class advantage: 1st class had 63% survival vs 24% for 3rd class
4. Age factor: Children under 10 had higher survival rates

Feature Importance:
1. Gender (most important predictor)
2. Passenger Class
3. Age
4. Fare
5. Title (extracted from name)

📁 Output Files
The project generates several files:

Visualizations:

- titanic_eda_large.png - 9 EDA plots
- correlation_heatmap_large.png - Feature correlations
- confusion_matrix_*.png - Model performance
- model_comparison.png - Model accuracy comparison

Predictions:

- submission.csv - Kaggle submission file

Model:

- titanic_model.pkl - Trained model file

### What I Learned
This project helped me understand:
Data preprocessing and cleaning techniques
Feature engineering (creating new features from existing data)
Handling missing values in datasets
Training and evaluating classification models
Hyperparameter tuning
Creating informative data visualizations
Version control with Git and GitHub

🔧 Technologies Used
Python - Primary programming language
Pandas & NumPy - Data manipulation
Scikit-learn - Machine learning algorithms
Matplotlib & Seaborn - Data visualization
Jupyter - Interactive exploration


📄 License
This project is open source and available under the MIT License.

Acknowledgments:
Kaggle for hosting the competition and providing the dataset
The Titanic dataset from the National Archives
All open-source libraries used in this project