import streamlit as st
import pickle
import re
import string
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- LIGHTWEIGHT CSS ----------------
st.markdown("""
<style>
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 18px;
    margin-bottom: 25px;
}
.result-fake {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 14px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}
.result-real {
    background-color: #dcfce7;
    color: #166534;
    padding: 14px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---------------- UI ----------------
st.title("📰 Fake News Detector")
st.markdown(
    '<div class="subtitle">AI-based system to detect Fake or Real news using NLP</div>',
    unsafe_allow_html=True
)

user_input = st.text_area("📝 Paste the news article here", height=180)

# ---------------- PREDICTION & VISUALS ----------------
if st.button("🔍 Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]

        fake_prob = probability[0]
        real_prob = probability[1]

        # -------- Result --------
        if prediction == 0:
            st.markdown('<div class="result-fake">❌ FAKE NEWS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-real">✅ REAL NEWS</div>', unsafe_allow_html=True)

        # -------- Visualization 1: Bar Chart --------
        st.subheader("📊 Prediction Confidence (Bar Chart)")
        bar_df = pd.DataFrame({
            "News Type": ["Fake", "Real"],
            "Probability": [fake_prob, real_prob]
        }).set_index("News Type")
        st.bar_chart(bar_df)

        # -------- Visualization 2: Pie Chart --------
        st.subheader("🥧 Confidence Distribution (Pie Chart)")
        pie_df = pd.DataFrame({
            "Probability": [fake_prob, real_prob]
        }, index=["Fake", "Real"])
        st.pyplot(pie_df.plot.pie(
            y="Probability",
            autopct="%1.1f%%",
            legend=False,
            figsize=(4, 4)
        ).figure)

        # -------- Visualization 3: Confidence Meter --------
        st.subheader("📈 Model Confidence Meter")
        confidence_score = max(fake_prob, real_prob)
        st.progress(int(confidence_score * 100))
        st.write(f"Model Confidence: **{confidence_score*100:.2f}%**")

        # -------- Visualization 4: Text Statistics --------
        st.subheader("🧾 Input Text Statistics")
        word_count = len(user_input.split())
        char_count = len(user_input)

        stats_df = pd.DataFrame({
            "Metric": ["Word Count", "Character Count"],
            "Value": [word_count, char_count]
        }).set_index("Metric")

        st.bar_chart(stats_df)
