import google.generativeai as genai
import streamlit as st

genai.configure(api_key="AIzaSyDtVqH7hfy0nuFEYvxrGB4G6nWzc3VESro")
model = genai.GenerativeModel("gemini-1.5-flash")

st.title("Poem Generator")
prompt = st.text_input("Enter a topic for your poem: ", "")

# Use session state to store the generated poem and prompt
if 'poem' not in st.session_state:
    st.session_state['poem'] = ''
if 'mod_poem' not in st.session_state:
    st.session_state['mod_poem'] = ''
if 'last_prompt' not in st.session_state:
    st.session_state['last_prompt'] = ''

if st.button("Generate Poem"):
    with st.spinner("Generating..."):
        # Always ask for a poem about the topic
        poem_prompt = f"Write a poem about {prompt}."
        response = model.generate_content(poem_prompt)
        st.session_state['poem'] = response.text
        st.session_state['mod_poem'] = ''  # Reset modified poem
        st.session_state['last_prompt'] = prompt

# Show the generated poem if available
if st.session_state['poem']:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Poem Topic:** {st.session_state['last_prompt']}")
        st.subheader("Generated Poem")
        st.write(st.session_state['poem'])
        # Show the modified poem if available
        if st.session_state['mod_poem']:
            st.subheader("Modified Poem")
            st.write(st.session_state['mod_poem'])
    with col2:
        # Option to edit the prompt and regenerate
        st.markdown("**Edit your topic and regenerate the poem:**")
        edited_prompt = st.text_input("Edit topic:", st.session_state['last_prompt'], key="edit_prompt")
        if st.button("Edit Prompt & Regenerate"):
            with st.spinner("Generating with edited prompt..."):
                poem_prompt = f"Write a poem about {edited_prompt}."
                response = model.generate_content(poem_prompt)
                st.session_state['poem'] = response.text
                st.session_state['mod_poem'] = ''
                st.session_state['last_prompt'] = edited_prompt

        st.subheader("Poem Options")
        option = st.selectbox(
            "",
            ["Shorten", "Expand", "Rewrite in a different style", "Change theme"], index=None, placeholder="Select an option"
        )

        mod_prompt = None
        style = ''
        new_theme = ''
        if option == "Shorten":
            mod_prompt = f"Shorten this poem:\n{st.session_state['poem']}"
        elif option == "Expand":
            mod_prompt = f"Expand this poem:\n{st.session_state['poem']}"
        elif option == "Rewrite in a different style":
            style = st.text_input("Enter the style (e.g., Shakespearean, Haiku, Rap):", key="style")
            if style:
                mod_prompt = f"Rewrite this poem in {style} style:\n{st.session_state['poem']}"
        elif option == "Change theme":
            new_theme = st.text_input("Enter the new theme:", key="theme")
            if new_theme:
                mod_prompt = f"Rewrite this poem with the theme '{new_theme}':\n{st.session_state['poem']}"

        if mod_prompt:
            if st.button("Apply Option"):
                with st.spinner("Processing..."):
                    mod_response = model.generate_content(mod_prompt)
                    st.session_state['mod_poem'] = mod_response.text