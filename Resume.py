import streamlit as st
import fitz
import google.generativeai as genai

genai.configure(api_key="AIzaSyDtVqH7hfy0nuFEYvxrGB4G6nWzc3VESro")
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_resume_feedback(text):
    prompt = ("Analyze the following resume and provide feedback on strengths, weaknesses, and areas for improvement:\n\n"
    + text)
    response = model.generate_content(prompt)
    return response.text

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        resume_text = extract_text_from_pdf(uploaded_file)

        with st.spinner("Generating AI feedback..."):
            feedback = get_resume_feedback(resume_text)
            
            st.subheader("AI Feedback and Suggestions:")
            st.markdown(feedback)

            