# Would you have survived the Titanic?

**A binary classification model trained on 891 real passenger records — with a live "try it yourself" interface.**
Logistic Regression and Random Forest models predicting survival from passenger class, age, sex, fare, family size, and inferred social title.

🔗 **[Try the live demo →](https://your-app.streamlit.app)** &nbsp;·&nbsp;  [Source on GitHub](https://github.com/ArpitaRaj27/titanic-survival-prediction)

---

## What you can do with it

- **Try the model on yourself or a famous passenger** — sliders for age/fare/family size, dropdowns for class/sex/port, get a live survival probability
- **See the patterns** — interactive charts showing how class, sex, age, and fare drove survival
- **Inspect the model** — confusion matrix, feature importances, and accuracy metrics for both models

## Sanity check: it gets the famous passengers right

| Passenger | Predicted survival probability | Reality |
|---|---|---|
| Jack Dawson (3rd class, 20yo male, alone) | 4.6% | died |
| Rose DeWitt Bukater (1st class, 17yo female, family) | 99.5% | survived |
| Captain Edward Smith | 10.0% | died |
| Molly Brown ("the Unsinkable") | 90.0% | survived |

## Performance

- **Random Forest:** 83.2% test accuracy, 80.5% 5-fold CV
- **Logistic Regression:** 81.0% test accuracy, 81.3% 5-fold CV

---

## How it was built

1. **Data:** 891-passenger Kaggle Titanic dataset, 12 raw features
2. **Feature engineering:** extracted `Title` from passenger names (Mr/Mrs/Miss/Master/Rare), computed `FamilySize` and `IsAlone`
3. **Preprocessing:** median imputation for `Age`/`Fare`, mode imputation for `Embarked`, label encoding for categoricals, `StandardScaler` on numerical features
4. **Models:** Logistic Regression (interpretable baseline) + Random Forest (200 trees)
5. **Evaluation:** 80/20 stratified train/test split + 5-fold cross-validation

**Stack:** Python · pandas · numpy · scikit-learn · Streamlit · Altair

---

## Run locally

```bash
git clone https://github.com/ArpitaRaj27/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Project structure

```
streamlit_app.py            # Dashboard + prediction UI
requirements.txt
data/
  train.csv                 # 891 labeled passengers
  test.csv                  # Unlabeled holdout
src/
  data_preprocessing.py     # Feature engineering + scaling pipeline
  __init__.py
.streamlit/
  config.toml               # Theme
```

Models train at app startup (cached) so there's no `.pkl` file to manage — avoids sklearn version mismatches across environments.

---

## What I learned

- Feature engineering matters more than model choice — `Title` and `FamilySize` (both engineered) outweigh most raw features in the RF importance ranking
- Class × sex interaction matters: Random Forest captures it, Logistic Regression underestimates it
- A well-presented model demo beats a notebook every time — recruiters can play with the interactive form in 30 seconds
