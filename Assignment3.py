import streamlit as st
import google.generativeai as genai

genai.configure(api_key="AIzaSyAVdOnE9qyK0gpJwt1KXIg-TIPE9zFn--w")

model = genai.GenerativeModel("gemini-2.0-flash")

st.title("AI Math Solver")

st.write("Type a math equation to solve (e.g., `2x + 3 = 7`, `integrate x^2 dx`, `derivative of sin(x)`).")
st.write("Type `exit` to quit the session.")

# Input box for math equation
user_equation = st.text_input("Enter your math equation:")

# Checkbox to ask for step-by-step explanation
explain = st.checkbox("Explain step-by-step?")

# Button to submit
if st.button("Solve"):
    if user_equation.lower().strip() == "exit":
        st.write("Session Ended. You can close this window.")
    elif user_equation.strip() == "":
        st.warning("Please enter a valid equation.")
    else:
        # Construct prompt for Gemini
        if explain:
            prompt = f"Solve this math problem and explain step by step: {user_equation}"
        else:
            prompt = f"Solve this math problem: {user_equation}"

        # Send request to Gemini
        response = model.generate_content(prompt)

        # Display AI-generated answer
        st.subheader("Gemini Answer:")
        st.write(response.text)