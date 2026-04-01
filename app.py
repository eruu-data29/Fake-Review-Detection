import streamlit as st
import pandas as pd
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ===============================
# SAMPLE DATA (Fallback Dataset)
# ===============================
sample_data = {
    "review": [
        "This product is amazing and works perfectly",
        "Worst product ever waste of money",
        "Absolutely fantastic quality highly recommend",
        "Fake product do not buy",
        "Very good product satisfied",
        "Totally useless and scam",
        "Loved it great purchase",
        "This is fake and terrible",
        "Excellent product worth buying",
        "Not original product fake seller"
    ],
    "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

sample_df = pd.DataFrame(sample_data)

# ===============================
# CLEAN TEXT FUNCTION
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Fake Review Detector", layout="wide")

st.title("🕵️ Fake Review Detection Dashboard")

# ===============================
# USER CHOICE
# ===============================
option = st.radio("Choose Input Method:", ["Upload CSV", "Type Review"])

# ===============================
# OPTION 1: CSV UPLOAD
# ===============================
if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns or "label" not in df.columns:
            st.error("Dataset must contain 'review' and 'label'")
        else:
            df["clean_review"] = df["review"].apply(clean_text)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(df["clean_review"])
            y = df["label"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)

            st.success(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

# ===============================
# OPTION 2: TYPE REVIEW
# ===============================
else:
    # Use sample dataset
    sample_df["clean_review"] = sample_df["review"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sample_df["clean_review"])
    y = sample_df["label"]

    model = LogisticRegression()
    model.fit(X, y)

    user_input = st.text_area("Enter your review:")

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
