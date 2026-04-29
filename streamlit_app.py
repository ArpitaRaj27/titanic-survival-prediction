"""
Titanic Survival Predictor - Streamlit dashboard
Wraps the existing src/data_preprocessing.py with a UI and an interactive
"would I have survived?" prediction form.
"""
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# Allow `from src.data_preprocessing import DataPreprocessor` to work
sys.path.insert(0, str(Path(__file__).parent))
from src.data_preprocessing import DataPreprocessor  # noqa: E402

# ---------- Page setup ----------
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide", page_icon="🚢")

st.title("Would you have survived the Titanic?")
st.markdown(
    "##### A binary classification model trained on 891 real passenger records. Try it on yourself."
)
st.caption(
    "Logistic Regression + Random Forest · scikit-learn · pandas · Streamlit  ·  "
    "[GitHub repo →](https://github.com/ArpitaRaj27/titanic-survival-prediction)"
)
st.divider()


# ---------- Load data + train models (cached) ----------
DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv")


@st.cache_resource(show_spinner="Training models on 891 passengers...")
def fit_pipeline():
    """Fit preprocessor + both models. Cached so it only runs once per session."""
    df = load_train()
    pre = DataPreprocessor()
    X, y = pre.preprocess_train(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr, y_tr)
    rf = RandomForestClassifier(random_state=42, n_estimators=200).fit(X_tr, y_tr)

    return {
        "pre": pre,
        "models": {"Logistic Regression": lr, "Random Forest": rf},
        "X": X,
        "y": y,
        "X_test": X_te,
        "y_test": y_te,
        "lr_acc": accuracy_score(y_te, lr.predict(X_te)),
        "rf_acc": accuracy_score(y_te, rf.predict(X_te)),
        "lr_cv": cross_val_score(lr, X, y, cv=5).mean(),
        "rf_cv": cross_val_score(rf, X, y, cv=5).mean(),
        "feature_order": pre.feature_order,
    }


train_df = load_train()
state = fit_pipeline()


# ---------- Prediction helper ----------
def predict_one(model, *, pclass, sex, age, sibsp, parch, fare, embarked, title):
    """Run a single user-input row through the same preprocessing as training."""
    pre = state["pre"]
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    sex_int = {"male": 0, "female": 1}[sex]
    embarked_int = {"S": 0, "C": 1, "Q": 2}[embarked]
    title_int = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}[title]

    # Use the SAME scaler that was fit on training data
    scaled = pre.scaler.transform([[age, fare, family_size]])[0]
    age_s, fare_s, fs_s = scaled

    row = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex": sex_int,
                "Age": age_s,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare_s,
                "Embarked": embarked_int,
                "Title": title_int,
                "FamilySize": fs_s,
                "IsAlone": is_alone,
            }
        ]
    )
    row = row[pre.feature_order]  # match training column order

    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1])
    return pred, proba


# ============================================================
# 1. INTERACTIVE PREDICTION — the star of the show
# ============================================================
st.subheader("Try the model")
st.caption(
    "Enter passenger details below (or pick a famous passenger) and the trained "
    "model will predict survival probability."
)

# Famous-passenger presets
PRESETS = {
    "Custom": None,
    "Jack Dawson (3rd class, age 20, alone)": dict(
        pclass=3, sex="male", age=20, sibsp=0, parch=0, fare=8.0, embarked="S", title="Mr"
    ),
    "Rose DeWitt Bukater (1st class, age 17, with mother & fiancé)": dict(
        pclass=1, sex="female", age=17, sibsp=1, parch=1, fare=80.0, embarked="S", title="Miss"
    ),
    "Captain Edward Smith (crew)": dict(
        pclass=1, sex="male", age=62, sibsp=0, parch=0, fare=0.0, embarked="S", title="Rare"
    ),
    "Molly Brown (1st class, age 44)": dict(
        pclass=1, sex="female", age=44, sibsp=0, parch=0, fare=27.0, embarked="C", title="Mrs"
    ),
}

preset_name = st.selectbox("Pick a preset (optional)", list(PRESETS.keys()))
preset = PRESETS[preset_name] or {}

col1, col2, col3 = st.columns(3)
with col1:
    pclass = st.selectbox("Passenger class", [1, 2, 3], index=[1, 2, 3].index(preset.get("pclass", 3)))
    sex = st.radio("Sex", ["male", "female"], index=["male", "female"].index(preset.get("sex", "male")), horizontal=True)
    title = st.selectbox(
        "Title",
        ["Mr", "Miss", "Mrs", "Master", "Rare"],
        index=["Mr", "Miss", "Mrs", "Master", "Rare"].index(preset.get("title", "Mr")),
        help="'Master' was used for young boys. 'Rare' covers Dr/Rev/Capt/etc.",
    )
with col2:
    age = st.slider("Age", 0, 80, preset.get("age", 30))
    fare = st.slider("Fare paid (£)", 0.0, 250.0, float(preset.get("fare", 15.0)), step=0.5)
    embarked = st.selectbox(
        "Port of embarkation",
        ["S", "C", "Q"],
        index=["S", "C", "Q"].index(preset.get("embarked", "S")),
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x],
    )
with col3:
    sibsp = st.slider("Siblings/spouses aboard", 0, 8, preset.get("sibsp", 0))
    parch = st.slider("Parents/children aboard", 0, 6, preset.get("parch", 0))
    model_choice = st.selectbox("Model", list(state["models"].keys()), index=1)

# Run prediction
pred, proba = predict_one(
    state["models"][model_choice],
    pclass=pclass, sex=sex, age=age, sibsp=sibsp, parch=parch,
    fare=fare, embarked=embarked, title=title,
)

st.markdown("")
result_col1, result_col2 = st.columns([1, 2])
with result_col1:
    if pred == 1:
        st.success(f"### ✅ Predicted: **SURVIVED**\n#### {proba:.1%} probability")
    else:
        st.error(f"### ❌ Predicted: **DID NOT SURVIVE**\n#### {proba:.1%} survival probability")
with result_col2:
    # Show probability as a horizontal gauge
    gauge_df = pd.DataFrame({"label": ["Survival probability"], "value": [proba]})
    gauge = (
        alt.Chart(gauge_df)
        .mark_bar(height=40, color="#3B82F6")
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Survival probability"),
            y=alt.Y("label:N", title=None, axis=None),
        )
        .properties(height=80)
    )
    threshold = (
        alt.Chart(pd.DataFrame({"x": [0.5]}))
        .mark_rule(color="white", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    st.altair_chart(gauge + threshold, use_container_width=True)
    st.caption("Dashed line = 50% decision boundary.")

st.divider()


# ============================================================
# 2. AUTO-INSIGHTS from the data
# ============================================================
st.subheader("What the data says")

female_rate = train_df[train_df["Sex"] == "female"]["Survived"].mean()
male_rate = train_df[train_df["Sex"] == "male"]["Survived"].mean()
class1_rate = train_df[train_df["Pclass"] == 1]["Survived"].mean()
class3_rate = train_df[train_df["Pclass"] == 3]["Survived"].mean()
child_rate = train_df[train_df["Age"] < 12]["Survived"].mean()
adult_rate = train_df[(train_df["Age"] >= 18) & (train_df["Age"] < 60)]["Survived"].mean()
high_fare_rate = train_df[train_df["Fare"] > 50]["Survived"].mean()
low_fare_rate = train_df[train_df["Fare"] < 10]["Survived"].mean()

insights = [
    f"➡️ **Sex was the strongest predictor.** Women survived at **{female_rate:.0%}**, men at only **{male_rate:.0%}**, about **{female_rate/male_rate:.1f}× more likely**.",
    f"➡️ **Class mattered enormously.** First-class passengers survived at **{class1_rate:.0%}** vs **{class3_rate:.0%}** for third class.",
    f"➡️ **'Women and children first' was real.** Children under 12 survived at **{child_rate:.0%}** vs **{adult_rate:.0%}** for working-age adults.",
    f"➡️ **Money helped.** Passengers paying >£50 survived at **{high_fare_rate:.0%}** vs **{low_fare_rate:.0%}** for those paying <£10.",
    f"➡️ **Overall survival rate: {train_df['Survived'].mean():.0%}** across {len(train_df)} passengers in the training set.",
]
for line in insights:
    st.markdown(f"- {line}")

st.divider()


# ============================================================
# 3. EDA CHARTS — interactive Altair
# ============================================================
st.subheader("The patterns visualized")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Survival rate by sex & class**")
    grouped = (
        train_df.groupby(["Pclass", "Sex"])["Survived"]
        .mean()
        .reset_index()
        .rename(columns={"Survived": "survival_rate"})
    )
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            x=alt.X("Pclass:O", title="Passenger class"),
            y=alt.Y("survival_rate:Q", title="Survival rate", axis=alt.Axis(format=".0%")),
            color=alt.Color("Sex:N", scale=alt.Scale(domain=["female", "male"], range=["#EC4899", "#3B82F6"])),
            xOffset="Sex:N",
            tooltip=[
                alt.Tooltip("Pclass:O", title="Class"),
                alt.Tooltip("Sex:N", title="Sex"),
                alt.Tooltip("survival_rate:Q", title="Survival rate", format=".1%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

with c2:
    st.markdown("**Survival rate by age group**")
    bins = [0, 12, 18, 30, 45, 60, 100]
    labels = ["0-11 (child)", "12-17 (teen)", "18-29", "30-44", "45-59", "60+"]
    train_df_age = train_df.dropna(subset=["Age"]).copy()
    train_df_age["age_group"] = pd.cut(train_df_age["Age"], bins=bins, labels=labels, right=False)
    age_rate = (
        train_df_age.groupby("age_group", observed=True)["Survived"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "survival_rate", "count": "n"})
    )
    chart = (
        alt.Chart(age_rate)
        .mark_bar(color="#3B82F6")
        .encode(
            x=alt.X("age_group:N", title="Age group", sort=labels),
            y=alt.Y("survival_rate:Q", title="Survival rate", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("age_group:N", title="Age"),
                alt.Tooltip("survival_rate:Q", title="Survival rate", format=".1%"),
                alt.Tooltip("n:Q", title="Passengers"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()


# ============================================================
# 4. MODEL PERFORMANCE
# ============================================================
st.subheader("Model performance")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Logistic Regression - test acc", f"{state['lr_acc']:.1%}")
m2.metric("Logistic Regression - 5-fold CV", f"{state['lr_cv']:.1%}")
m3.metric("Random Forest - test acc", f"{state['rf_acc']:.1%}")
m4.metric("Random Forest - 5-fold CV", f"{state['rf_cv']:.1%}")

cm_col, fi_col = st.columns(2)

with cm_col:
    st.markdown(f"**Confusion matrix - {model_choice}**")
    model = state["models"][model_choice]
    y_pred = model.predict(state["X_test"])
    cm = confusion_matrix(state["y_test"], y_pred)
    cm_df = pd.DataFrame(
        [
            {"Actual": "Did not survive", "Predicted": "Did not survive", "count": int(cm[0, 0])},
            {"Actual": "Did not survive", "Predicted": "Survived", "count": int(cm[0, 1])},
            {"Actual": "Survived", "Predicted": "Did not survive", "count": int(cm[1, 0])},
            {"Actual": "Survived", "Predicted": "Survived", "count": int(cm[1, 1])},
        ]
    )
    heat = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", title="Predicted"),
            y=alt.Y("Actual:N", title="Actual"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Actual", "Predicted", "count"],
        )
        .properties(height=300)
    )
    text = heat.mark_text(fontSize=20, fontWeight="bold").encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > cm.max() / 2, alt.value("white"), alt.value("black")),
    )
    st.altair_chart(heat + text, use_container_width=True)

with fi_col:
    st.markdown("**Feature importance (Random Forest)**")
    rf = state["models"]["Random Forest"]
    fi_df = (
        pd.DataFrame({"feature": state["feature_order"], "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    fi_chart = (
        alt.Chart(fi_df)
        .mark_bar(color="#3B82F6")
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(fi_chart, use_container_width=True)

st.divider()


# ============================================================
# 5. METHODOLOGY
# ============================================================
with st.expander("How it was built (methodology)"):
    st.markdown(
        """
        **Pipeline:**
        1. **Data:** 891-passenger Titanic dataset from Kaggle, with 12 raw features per passenger.
        2. **Feature engineering:** extracted `Title` from passenger names (Mr/Mrs/Miss/Master/Rare),
           computed `FamilySize` and `IsAlone` from siblings/parents counts.
        3. **Preprocessing:** median imputation for missing `Age`/`Fare`, mode imputation for `Embarked`,
           label encoding for categoricals, `StandardScaler` on numerical features.
        4. **Models:** Logistic Regression (interpretable baseline) and Random Forest (200 trees).
        5. **Evaluation:** 80/20 stratified train/test split + 5-fold cross-validation.

        **Why two models?** LR gives a simple, interpretable linear baseline; RF captures nonlinear
        interactions (e.g. third-class women survived less than first-class women, class and sex
        interact). The RF beats LR by a few points and is the default selection above.
        """
    )
