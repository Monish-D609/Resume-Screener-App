import re
import joblib
import streamlit as st
import pdfplumber

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Resume Screener",
    page_icon="📄",
    layout="centered"
)

# ---------------- LOAD ARTIFACTS ----------------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
int_to_label = joblib.load("int_to_label.pkl")

# ---------------- CLEANING FUNCTION ----------------
def cleanResume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'\bRT\b|\bcc\b', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# ---------------- UI ----------------
def main():
    st.title("Resume Screener")
    st.caption("Upload a resume PDF and predict the job role")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        with st.spinner("Extracting text..."):
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if len(resume_text.strip()) < 200:
            st.warning(
                "Low text detected. This may be a scanned PDF. "
                "Prediction may be unreliable."
            )

        with st.expander("View extracted resume text"):
            st.text(resume_text[:2000])

        if st.button("Analyze Resume"):
            with st.spinner("Analyzing resume..."):
                cleaned_text = cleanResume(resume_text)
                vector = tfidf.transform([cleaned_text])
                pred_id = int(model.predict(vector)[0])
                category = int_to_label[pred_id]

            st.success(f"Predicted Job Role: **{category}**")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
