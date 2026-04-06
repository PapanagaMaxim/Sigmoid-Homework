import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Wine Quality App", layout="wide")


@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "..", "Datasets", "StreamLit", "wines.csv")
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)
    return df

df = load_data()


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Model & Analysis"])


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
df = pd.read_csv("wine-quality-white-and-red.csv")
df.head()
""", language='python')

    st.latex(r'Quality = f(features)')


elif page == "Model & Analysis":

    st.title(" Model Configuration")


    st.header("🔧 Parameters")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    max_iter = st.slider("Max Iterations", 100, 1000, 200)
    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)

    target = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)

    st.divider()

    st.subheader(" Current Dataset Head")
    st.dataframe(df.head())


    if st.button(" Train Model"):

        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        y = (y >= 6).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = LogisticRegression(max_iter=max_iter, C=C)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        st.container()
        st.subheader(" Selected Parameters")

        st.json({
            "test_size": test_size,
            "max_iter": max_iter,
            "C": C,
            "target": target
        })

        st.metric("Accuracy", round(acc, 3))

        st.subheader(" Correlation Heatmap")

        df['type_encoded'] = df['type'].map({'red': 0, 'white': 1})
        numeric_df = df.select_dtypes(include=[np.number])

        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.subheader(" Confusion Matrix")

        cm = confusion_matrix(y_test, preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2)
        st.pyplot(fig2)

        st.subheader(" Feature Importance")

        importance = pd.Series(model.coef_[0], index=X.columns)
        importance.sort_values().plot(kind='barh')
        st.pyplot(plt)

        st.subheader(" Prediction Distribution")

        fig3, ax3 = plt.subplots()
        sns.histplot(preds, bins=2, ax=ax3)
        st.pyplot(fig3)