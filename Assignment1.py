import google.generativeai as genai
import streamlit as st
genai.configure(api_key= "AIzaSyDtVqH7hfy0nuFEYvxrGB4G6nWzc3VESro")
model = genai.GenerativeModel("gemini-1.5-flash")
st.title("Gemini Note Generator")
prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate Note"):
  with st.spinner("Generating..."):
    response = model.generate_content(prompt)
    st.subheader("Generated Note")
    st.write(response.text)
    