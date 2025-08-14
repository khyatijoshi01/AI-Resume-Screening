import re
import streamlit as st
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ----------------------------
# Step 1: Read & clean text
# ----------------------------
def get_text_from_file(file):
    """
    Extract plain text from uploaded resume (PDF/DOCX).
    """
    extracted_text = ""

    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted_text += page.extract_text() or ""

    elif file.name.endswith(".docx"):
        document = docx.Document(file)
        for paragraph in document.paragraphs:
            extracted_text += paragraph.text + " "

    # Remove weird spaces & make lowercase
    extracted_text = re.sub(r"\s+", " ", extracted_text).strip().lower()
    return extracted_text

# ----------------------------
# Step 2: Find similarity score
# ----------------------------
def get_match_percentage(jd_text, resume_text):
    """
    Compare JD and resume using TF-IDF and return percentage match.
    """
    if not jd_text or not resume_text:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf_matrix)[0][1] * 100
    return round(score, 2)

# ----------------------------
# Step 3: Pull keywords from JD
# ----------------------------
def extract_job_keywords(jd_text, count=15):
    """
    Identify most frequent words from the JD (ignoring common fillers).
    """
    words = re.findall(r"\b[a-zA-Z]{3,}\b", jd_text.lower())
    filler_words = {"and", "the", "for", "with", "from", "that", "this", "have", "are", "was", "you", "your", "has", "can", "will", "not"}
    keywords = [w for w in words if w not in filler_words]
    return [word for word, _ in Counter(keywords).most_common(count)]

# ----------------------------
# Step 4: Compare JD keywords with resume
# ----------------------------
def match_keywords(jd_keywords, resume_text):
    """
    Return two lists: matched keywords & missing keywords.
    """
    matched = [kw for kw in jd_keywords if kw in resume_text]
    missing = [kw for kw in jd_keywords if kw not in resume_text]
    return matched, missing

# ----------------------------
# Step 5: Streamlit Interface
# ----------------------------
st.set_page_config(page_title="AI Resume Screening ")
st.title(" AI Resume Screening! ")
st.write("Upload your resume and paste a Job Description to see how well you match!")

resume_file = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste Job Description Here")

if st.button("check"):
    if resume_file and job_description:
        resume_text = get_text_from_file(resume_file)
        jd_text = re.sub(r"\s+", " ", job_description.strip().lower())

        if not resume_text:
            st.error(" No text could be read from your resume. Try saving it as a DOCX or non-scanned PDF.")
        else:
            # Match score
            match_score = get_match_percentage(jd_text, resume_text)

            # JD keywords
            jd_keywords = extract_job_keywords(job_description, count=30)

            # Keyword match
            matched, missing = match_keywords(jd_keywords, resume_text)

            # Show results
            st.subheader(" Match Analysis")
            st.write(f"**Match Score:** {match_score}%")

            st.subheader(" Important Keywords from JD")
            st.write(", ".join(jd_keywords))

            st.subheader(" Found in Your Resume")
            st.write(", ".join(matched) if matched else "_None found_")

            st.subheader(" Missing from Your Resume")
            st.write(", ".join(missing) if missing else "_All covered_")
    else:
        st.warning("Please upload a resume and paste a job description.")
