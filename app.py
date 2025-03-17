import streamlit as st
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from resumes
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = " ".join([p.text for p in doc.paragraphs])
    else:
        text = ""
    return text

# Function to calculate match score between resume and job description
def calculate_match(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    docs = [job_description, resume_text]
    tfidf_matrix = vectorizer.fit_transform(docs)  
    similarity_score = cosine_similarity(tfidf_matrix)[0][1] * 100
    return round(similarity_score, 2)

# Streamlit UI
st.title("📄 AI Resume Screening System ")

uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description Here")

if st.button("Analyze Resume"):
    if uploaded_file and job_description:
        resume_text = extract_text(uploaded_file)
        match_score = calculate_match(resume_text, job_description)

        st.subheader("🔍 Resume Analysis")
        st.write("**Match Score:**", match_score, "%")
    else:
        st.warning("Please upload a resume and enter a job description.")