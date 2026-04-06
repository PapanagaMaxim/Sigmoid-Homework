import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Wine Quality App", layout="wide")


# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "Datasets", "StreamLit", "wines.csv")
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)

    # Encode 'type' once here (safe for cache)
    df['type'] = df['type'].map({'red': 0, 'white': 1})

    return df


df = load_data()


# ------------------ SIDEBAR ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model & Analysis"])


# ------------------ HOME ------------------
if page == "Home":
    st.title("Wine Quality Prediction App 🍷")

    st.header("Dataset Description")

    st.markdown("""
    This dataset contains physicochemical properties of **red and white wines**.
    
    The goal is to predict the **quality** of the wine based on features such as:
    - acidity
    - sugar
    - pH
    - alcohol
    
    The quality score ranges from **0 to 10**.
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.code("""
df = pd.read_csv("wines.csv")
df.head()
""", language='python')

    st.latex(r'Quality = f(features)')


# ------------------ MODEL ------------------
elif page == "Model & Analysis":

    st.title("Model Configuration")

    st.header("🔧 Parameters")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    max_iter = st.slider("Max Iterations", 100, 1000, 200)
    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)

    # ✅ Fixed target (no more crashes)
    target = "quality"
    st.info(f"Target column is fixed: {target}")

    st.divider()

    st.subheader("Current Dataset Head")
    st.dataframe(df.head())


    if st.button("Train Model"):

        # Work on a copy (safe with cache)
        working_df = df.copy()

        X = working_df.drop(columns=[target])
        y = working_df[target]

        # Binary classification
        y = (y >= 6).astype(int)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Stable model
        model = LogisticRegression(max_iter=max_iter, C=C, solver='liblinear')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # ------------------ PARAMETERS ------------------
        with st.container():
            st.subheader("Selected Parameters")
            st.json({
                "test_size": test_size,
                "max_iter": max_iter,
                "C": C,
                "target": target
            })

        st.metric("Accuracy", round(acc, 3))

        # ------------------ HEATMAP ------------------
        st.subheader("Correlation Heatmap")

        fig1, ax1 = plt.subplots()
        sns.heatmap(working_df.corr(), annot=False, cmap="coolwarm", ax=ax1)
        st.pyplot(fig1)

        # ------------------ CONFUSION MATRIX ------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, preds)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)

        # ------------------ FEATURE IMPORTANCE ------------------
        st.subheader("Feature Importance")

        importance = pd.Series(model.coef_[0], index=X.columns)

        fig3, ax3 = plt.subplots()
        importance.sort_values().plot(kind='barh', ax=ax3)
        ax3.set_title("Feature Importance")
        st.pyplot(fig3)

        # ------------------ PREDICTIONS ------------------
        st.subheader("Prediction Distribution")

        fig4, ax4 = plt.subplots()
        sns.histplot(preds, bins=2, ax=ax4)
        ax4.set_title("Predicted Class Distribution")
        st.pyplot(fig4)