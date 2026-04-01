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
# SAMPLE DATA (Fallback)
# ===============================
sample_data = {
    "review": [
        "This product is amazing and works perfectly",
        "Worst product ever waste of money",
        "Fake product do not buy",
        "Very good product satisfied",
        "Totally useless and scam",
        "Loved it great purchase",
        "This is fake and terrible",
        "Excellent product worth buying",
        "Not original product fake seller",
        "Bad quality waste of money"
    ],
    "label": [0,1,1,0,1,0,1,0,1,1]
}
sample_df = pd.DataFrame(sample_data)

# ===============================
# USER CHOICE
# ===============================
option = st.sidebar.radio("Choose Option", ["Upload CSV", "Type Review Only"])

# ===============================
# MODEL VARIABLES
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=200)

# ===============================
# OPTION 1: CSV UPLOAD
# ===============================
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Raw Data")
        st.dataframe(df.head(), use_container_width=True)

        if "review" not in df.columns or "label" not in df.columns:
            st.error("Dataset must contain 'review' and 'label'")
        else:
            df['clean_review'] = df['review'].apply(clean_text)

            X = vectorizer.fit_transform(df['clean_review'])
            y = df['label']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

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

            # TABS (same as your original)
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📊 Overview", "📈 Visualizations", "☁️ WordCloud", "🔎 Search"]
            )

            with tab1:
                col1, col2 = st.columns(2)
                col1.metric("Total Reviews", len(df))
                col2.metric("Fake Reviews", df['label'].sum())

            with tab2:
                fig = px.pie(df, names='label', title="Fake vs Real Reviews")
                st.plotly_chart(fig, use_container_width=True)

                df['length'] = df['review'].apply(len)
                fig2 = px.histogram(df, x="length")
                st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                text = " ".join(df['clean_review'])
                wc = WordCloud().generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc)
                ax.axis("off")
                st.pyplot(fig)

            with tab4:
                query = st.text_input("Enter keyword")
                if query:
                    results = df[df['review'].str.contains(query, case=False, na=False)]
                    st.dataframe(results.head())

    else:
        st.info("Please upload a dataset")

# ===============================
# OPTION 2: TYPE ONLY
# ===============================
else:
    st.subheader("🔍 Try Fake Review Detection")

    # Train on sample data
    sample_df['clean_review'] = sample_df['review'].apply(clean_text)
    X = vectorizer.fit_transform(sample_df['clean_review'])
    y = sample_df['label']
    model.fit(X, y)

    user_input = st.text_area("Enter your review")

    if st.button("Predict"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]

            if pred == 1:
                st.error("🚨 Fake Review")
            else:
                st.success("✅ Genuine Review")
        else:
            st.warning("Please enter a review")
