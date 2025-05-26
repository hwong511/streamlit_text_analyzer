import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np

# --- Title ---
st.title("🧠 Smart Text Analyzer")

# --- User Input ---
text = st.text_area("Enter some text for analysis:")

# --- Sentiment Analysis ---
if st.button("Run Sentiment Analysis"):
    if text.strip():
        st.subheader("Sentiment Analysis")
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        st.write(f"**Polarity:** {polarity:.2f}")
        st.write(f"**Subjectivity:** {subjectivity:.2f}")

        # Sentiment Bar Chart
        st.subheader("Sentiment Scores")
        fig, ax = plt.subplots()
        ax.bar(["Polarity", "Subjectivity"], [polarity, subjectivity], color=["green", "blue"])
        ax.set_ylim(-1, 1)
        st.pyplot(fig)
    else:
        st.warning("Please enter some text.")

# --- Topic Modeling ---
if st.button("Run Topic Modeling"):
    if text.strip():
        st.subheader("Topic Modeling (LDA)")
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([text])
        lda = LatentDirichletAllocation(n_components=1, random_state=42)
        lda.fit(X)

        terms = vectorizer.get_feature_names_out()
        topic_weights = lda.components_[0]
        top_indices = topic_weights.argsort()[-5:][::-1]
        top_words = [terms[i] for i in top_indices]
        top_scores = topic_weights[top_indices]

        st.markdown("**Top Topic Keywords:**")
        for word in top_words:
            st.markdown(f"- {word}")

        # Topic Bar Chart
        st.subheader("Topic Keyword Weights")
        fig, ax = plt.subplots()
        ax.barh(top_words[::-1], top_scores[::-1], color="purple")
        ax.set_xlabel("Weight")
        st.pyplot(fig)
    else:
        st.warning("Please enter some text.")

