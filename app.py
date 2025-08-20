import streamlit as st
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extra libraries for PDF & DOCX
import PyPDF2
import docx

st.title("AI-Powered Resume Screening")

# --- Step 1: User inputs JD in a text box ---
job_description = st.text_area("Paste Job Description here:")

# --- Step 2: Resume Upload (PDF/DOCX/TXT) ---
uploaded_files = st.file_uploader(
    "Upload Resumes", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text(file):
    """Detect file type and extract text accordingly"""
    if file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return file.read().decode("utf-8")

def fit_label(score):
    """Return label based on similarity score"""
    if score >= 0.7:
        return "High Fit ✅"
    elif score >= 0.4:
        return "Medium Fit ⚠️"
    else:
        return "Low Fit ❌"

if st.button("Rank Candidates") and job_description and uploaded_files:
    resumes = [extract_text(f) for f in uploaded_files]
    resume_names = [f.name for f in uploaded_files]

    # --- Step 3: TF-IDF Vectorization ---
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    # --- Step 4: Cosine Similarity ---
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # --- Step 5: Ranking ---
    ranked = sorted(zip(resume_names, similarities), key=lambda x: x[1], reverse=True)

    st.subheader("Ranked Candidates:")
    for i, (name, score) in enumerate(ranked, 1):
        st.write(f"{i}. {name} → Score: {score:.4f} → {fit_label(score)}")
