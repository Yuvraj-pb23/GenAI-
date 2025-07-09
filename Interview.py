import streamlit as st
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="AIzaSyAVdOnE9qyK0gpJwt1KXIg-TIPE9zFn--w")
model = genai.GenerativeModel("gemini-1.5-flash")

# Title
st.title("🧠 AI Interview Simulator")

# Initialize session state
if "questions" not in st.session_state:
    st.session_state.questions = []
if "profile" not in st.session_state:
    st.session_state.profile = ""

# Input section
skills = st.text_input("Enter your skills (comma-separated):")
experience = st.number_input("Years of experience:", min_value=0, max_value=50)
job_position = st.text_input("Desired job position:")

# Generate interview questions and save in session
if st.button("Start Interview"):
    with st.spinner("Generating questions..."):
        profile = f"Skills: {skills}, Experience: {experience} years, Position: {job_position}"
        prompt = f"""You are an expert interviewer. Create 10 interview questions for a candidate with:
        {profile}. Keep them short and relevant."""
        response = model.generate_content(prompt)
        questions = response.text.strip().split("\n")
        questions = [q.strip("-•1234567890. ").strip() for q in questions if "?" in q]
        questions = questions[:10]
        st.session_state.questions = questions
        st.session_state.profile = profile

# If questions already generated, show interview
if st.session_state.questions:
    st.subheader("📝 Interview Questions")
    user_answers = []
    for i, question in enumerate(st.session_state.questions):
        answer = st.text_area(f"Q{i+1}: {question}", key=f"q{i}")
        user_answers.append({"question": question, "answer": answer})

    if st.button("Get My Score"):
        with st.spinner("Evaluating..."):
            all_answers = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in user_answers])
            final_prompt = f"""You are a job interviewer. Read this interview and give a rating (out of 10) and short feedback.

Candidate Profile: {st.session_state.profile}
Interview:
{all_answers}
"""
            result = model.generate_content(final_prompt)
        st.subheader("📊 Your Interview Feedback")
        st.write(result.text)