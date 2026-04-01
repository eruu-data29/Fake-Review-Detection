# ===============================
# Fake Review Detection Dashboard
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import re
import string

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import plotly.express as px

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fake Review Detection Dashboard", layout="wide")

st.title("🕵️ Fake Review Detection Dashboard")
st.markdown("Detect fake product reviews using Machine Learning & NLP")

# ===============================
# TEXT CLEANING FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ===============================
# MAIN APP
# ===============================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head(), use_container_width=True)

    if "review" not in df.columns or "label" not in df.columns:
        st.error("Dataset must contain 'review' and 'label' columns")

    else:
        # Clean text
        df['clean_review'] = df['review'].apply(clean_text)

        # Train Model
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['clean_review'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        st.success(f"Model Accuracy: {accuracy:.2f}")

        # USER INPUT
        st.sidebar.subheader("🔍 Test a Review")
        user_input = st.sidebar.text_area("Enter Review")

        if st.sidebar.button("Predict"):
            if user_input.strip() != "":
                cleaned = clean_text(user_input)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]

                if pred == 1:
                    st.sidebar.error("🚨 Fake Review Detected")
                else:
                    st.sidebar.success("✅ Genuine Review")
            else:
                st.sidebar.warning("Please enter a review")

        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Overview", "📈 Visualizations", "☁️ WordCloud", "🔎 Search"]
        )

        with tab1:
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Reviews", len(df))

            with col2:
                fake_count = df['label'].sum()
                st.metric("Fake Reviews", fake_count)

        with tab2:
            st.subheader("Review Distribution")
            fig = px.pie(df, names='label', title="Fake vs Real Reviews")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Review Length Analysis")
            df['length'] = df['review'].apply(len)
            fig2 = px.histogram(df, x="length", nbins=50)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("WordCloud of Reviews")
            text = " ".join(df['clean_review'])

            wc = WordCloud(width=800, height=400).generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

        with tab4:
            st.subheader("Search Reviews")
            query = st.text_input("Enter keyword")

            if query:
                results = df[df['review'].str.contains(query, case=False, na=False)]

                if not results.empty:
                    st.success(f"Found {len(results)} reviews")
                    st.dataframe(results.head(), use_container_width=True)
                else:
                    st.warning("No results found")

else:
    st.info("Please upload a dataset to start")
