# ============================================
# 🏏 Streamlit Web App - Cricket Commentary Sentiment Analyzer
# ============================================

import streamlit as st
import joblib

# ---------- Load Models ----------
try:
    model = joblib.load("sentiment_model_balanced.pkl")
    tfidf = joblib.load("sentiment_vectorizer_balanced.pkl")
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}. Please ensure 'sentiment_model_balanced.pkl' and 'sentiment_vectorizer_balanced.pkl' are in the same directory as the app.")
    st.stop()

# ---------- Cricket-Aware Correction ----------
def correct_sentiment(comment, model_pred):
    text = comment.lower()
    if any(word in text for word in ["four", "sixes", "six", "boundary", "lofted", "driven", "massive hit", "over midwicket"]):
        return "positive"
    elif any(word in text for word in ["out", "bowled", "caught", "lbw", "gone", "edge", "dismissed"]):
        return "negative"
    elif any(word in text for word in ["no run", "defended", "blocked", "dot"]):
        return "neutral"
    return model_pred

# ---------- Prediction Function ----------
def predict_sentiment(comment):
    vec = tfidf.transform([comment])
    model_pred = model.predict(vec)[0]
    final_pred = correct_sentiment(comment, model_pred)
    return final_pred

# ---------- Streamlit UI ----------
st.set_page_config(page_title="🏏 Cricket Commentary Sentiment Analyzer", layout="centered")
st.title("🏏 Cricket Commentary Sentiment Analyzer")
st.markdown("Analyze cricket commentary to detect **Positive**, **Neutral**, or **Negative** sentiment using NLP & domain rules.")

# Text input
user_input = st.text_area("Enter a cricket commentary line:", height=150)

if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() != "":
        result = predict_sentiment(user_input)
        if result == "positive":
            st.success("🟢 Positive — Exciting or successful moment for batting team!")
        elif result == "neutral":
            st.info("⚪ Neutral — Ordinary or balanced commentary.")
        elif result == "negative":
            st.error("🔴 Negative — Dismissal or poor moment.")
    else:
        st.warning("Please enter a commentary line first.")

# Optional: Upload a file
st.markdown("---")
st.subheader("📁 Analyze Full Commentary File")
uploaded_file = st.file_uploader("Upload a commentary .txt file", type=["txt"])

if uploaded_file is not None:
    lines = uploaded_file.read().decode("utf-8").splitlines()
    st.write(f"Total lines found: {len(lines)}")

    results = []
    for line in lines:
        if line.strip():
            sentiment = predict_sentiment(line)
            results.append({"Commentary": line, "Sentiment": sentiment})

    import pandas as pd
    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Downloadable CSV
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Sentiment Results as CSV", data=csv, file_name="commentary_sentiments.csv", mime="text/csv")

st.markdown("---")
st.caption("Built by Subba Rami Reddy Janga | NLP Project on Cricket Commentary | Streamlit App")
