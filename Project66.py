import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# For language detection
try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
except ImportError:
    guess_lexer = None
    ClassNotFound = Exception

# Set Streamlit page config for a wider layout
st.set_page_config(
    layout="wide",
    page_title="Universal Code Cleaner",
    page_icon="🧹"
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)

# Supported languages for highlighting
LANGUAGES = [
   "python", "javascript", "java", "c", "cpp", "csharp", "go", "ruby", "php", "rust", "typescript", "kotlin", "swift", "scala", "perl", "r", "bash", "html", "css", "sql", "json", "xml", "yaml", "markdown"
]

# --- Session State for Version History ---
if "history" not in st.session_state:
    st.session_state.history = []

# Add clear flag to session state
if "clear_triggered" not in st.session_state:
    st.session_state.clear_triggered = False

# Add text input state
if "text_input_value" not in st.session_state:
    st.session_state.text_input_value = ""

# --- Professional Header ---
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">🧹 Universal Code Optimizer</h1>
            <p style="color: #e3f2fd; margin: 0; font-size: 1.2rem;">AI-Powered Code Optimization & Documentation</p>
        </div>
        <div style="text-align: right;">
            <h3 style="color: white; margin: 0;">✨ Features</h3>
            <p style="color: #e3f2fd; margin: 0; font-size: 0.9rem;">• Multi-language Support<br>• Smart Optimization<br>• Auto Documentation</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Features Section ---
st.markdown("""
<div class="feature-card">
    <h2 style="color: #007acc; margin-bottom: 1rem;">🚀 What We Do</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div style="background: rgba(0, 122, 204, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc;">
            <h4 style="margin: 0 0 0.5rem 0;">🎯 Smart Optimization</h4>
            <p style="margin: 0; font-size: 0.9rem;">Automatically optimize and structure your code for better readability</p>
        </div>
        <div style="background: rgba(0, 122, 204, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc;">
            <h4 style="margin: 0 0 0.5rem 0;">📝 Auto Documentation</h4>
            <p style="margin: 0; font-size: 0.9rem;">Generate comprehensive comments and documentation</p>
        </div>
        <div style="background: rgba(0, 122, 204, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #007acc;">
            <h4 style="margin: 0 0 0.5rem 0;">🔍 Multi-Language</h4>
            <p style="margin: 0; font-size: 0.9rem;">Support for 20+ programming languages</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Code Input Section ---
col_head, col_upload = st.columns([6, 1], gap="small")
with col_head:
    st.markdown("""
    <div style='margin-top:1.2rem; margin-bottom:0; display:flex; align-items:center;'>
      <h2 style='color:#007acc; font-size:1.6rem; margin:0; display:inline; vertical-align:middle;'>Paste Your Code</h2>
    </div>
    """, unsafe_allow_html=True)
with col_upload:
    uploaded_file = st.file_uploader(" ", type=None, help="Upload any code file to get started", key="main_file_uploader")
    if uploaded_file:
        uploaded_code = uploaded_file.read().decode("utf-8")
    else:
        uploaded_code = ""

# Always use the latest value from the text area (file upload fills it, but typing wins)
def get_initial_code():
    if uploaded_file and uploaded_code:
        return uploaded_code
    return st.session_state.text_input_value

# Initialize messy_code
messy_code = ""

# Always show the text area
messy_code = st.text_area(
    " ",
    value="" if st.session_state.clear_triggered else get_initial_code(),
    height=300,
    placeholder="Paste your messy or unformatted code here...",
    help="Enter the code you want to optimize and document",
    key="main_code_input",
    on_change=lambda: setattr(st.session_state, 'text_input_value', st.session_state.main_code_input)
)

# --- Language Detection ---
detected_language = None
# Extension-based detection mapping
EXTENSION_TO_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".ts": "typescript",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".pl": "perl",
    ".r": "r",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".json": "json",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
}
# If uploaded_file is present, try to detect by extension first
if uploaded_file and uploaded_file.name:
    import os
    _, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()
    if ext in EXTENSION_TO_LANG:
        detected_language = EXTENSION_TO_LANG[ext]
# If not detected by extension, or no file uploaded, use Pygments
if not detected_language and messy_code and guess_lexer:
    try:
        lexer = guess_lexer(messy_code)
        pygments_name = lexer.name.lower()
        # Map Pygments names to LANGUAGES entries (case-insensitive, with aliases)
        pygments_to_lang = {
            "python": "python",
            "py": "python",
            "python3": "python",
            "javascript": "javascript",
            "js": "javascript",
            "java": "java",
            "c": "c",
            "c++": "cpp",
            "cpp": "cpp",
            "c#": "csharp",
            "csharp": "csharp",
            "c-sharp": "csharp",
            "c# (c-sharp)": "csharp",
            "go": "go",
            "golang": "go",
            "ruby": "ruby",
            "rb": "ruby",
            "php": "php",
            "rust": "rust",
            "typescript": "typescript",
            "ts": "typescript",
            "kotlin": "kotlin",
            "swift": "swift",
            "scala": "scala",
            "perl": "perl",
            "r": "r",
            "bash": "bash",
            "sh": "bash",
            "shell": "bash",
            "html": "html",
            "xml+html": "html",
            "css": "css",
            "sql": "sql",
            "json": "json",
            "xml": "xml",
            "yaml": "yaml",
            "yml": "yaml",
            "markdown": "markdown",
            "md": "markdown",
        }
        # Try direct match, then lowercased, then stripped of special chars
        detected_language = pygments_to_lang.get(pygments_name)
        if not detected_language:
            detected_language = pygments_to_lang.get(pygments_name.lower())
        if not detected_language:
            import re
            simple_name = re.sub(r'[^a-z0-9]', '', pygments_name.lower())
            for key, val in pygments_to_lang.items():
                if re.sub(r'[^a-z0-9]', '', key.lower()) == simple_name:
                    detected_language = val
                    break
        # Fallback: show raw lexer name if still not found
        if not detected_language:
            detected_language = pygments_name
    except ClassNotFound:
        detected_language = None

# --- Prompt template for code cleaning and commenting ---
prompt = PromptTemplate.from_template(
    "You are a helpful code formatter and explainer. "
    "Given the following messy or unordered {language} code, return a clean, well-formatted, and readable version with helpful comments explaining the code. "
    "Do not add explanations outside the code, just return the cleaned and commented code.\n\n"
    "Messy code:\n{code}\n\nCleaned and commented code:"
)
chain = LLMChain(llm=llm, prompt=prompt)

# --- Prompt template for code explanation ---
explain_prompt = PromptTemplate.from_template(
    "You are a helpful programming assistant. "
    "Explain what the following {language} code does, step by step, in a visually structured way. "
    "For each important code line or block, start with a callout (➤) and show the code using inline code formatting (single backticks). "
    "Then, use bullet points to explain what that line or block does. "
    "Highlight important terms or concepts in bold. "
    "Use clear, readable markdown, and make the explanation easy to scan, like a professional code review or tutorial.\n\n"
    "Code:\n{code}\n\nExplanation:"
)
explain_chain = LLMChain(llm=llm, prompt=explain_prompt)

# --- Action Buttons ---
st.markdown("### ⚡ Actions")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    optimize_btn = st.button("🧹 Optimize Code", use_container_width=True)
with col2:
    explain_btn = st.button("📖 Explain Code", use_container_width=True)
with col3:
    revert_btn = st.button("↩️ Revert", use_container_width=True)
with col4:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True)

# Show warning immediately below actions if revert is pressed and nothing to revert
if revert_btn and len(st.session_state.history) <= 1:
    st.warning("⚠️ No previous version to revert to.")

# Show success message immediately below actions if revert is successful
if revert_btn and len(st.session_state.history) > 1:
    st.session_state.history.pop()  # Remove current
    prev = st.session_state.history[-1]
    messy_code = prev["messy"]
    optimized = prev["cleaned"]
    detected_language = prev["language"]
    st.success("✅ Reverted to previous version.")

# Clear all data and reset to initial state
if clear_btn:
    # Clear session state
    st.session_state.history = []
    # Clear the text input value
    st.session_state.text_input_value = ""
    # Set clear flag
    st.session_state.clear_triggered = True
    # Rerun to refresh the page
    st.rerun()

# Reset clear flag after widget creation
if st.session_state.clear_triggered:
    st.session_state.clear_triggered = False

# --- Add flag to control comparison display ---
if 'show_explanation_only' not in st.session_state:
    st.session_state['show_explanation_only'] = False

# --- Optimize Code ---
if optimize_btn and messy_code.strip():
    with st.spinner("🧹 Optimizing your code..."):
        result = chain.invoke({"code": messy_code, "language": detected_language})
        optimized = result["text"] if isinstance(result, dict) and "text" in result else result
        # Save to history
        st.session_state.history.append({
            "messy": messy_code,
            "cleaned": optimized,
            "language": detected_language,
        })
    st.session_state['show_explanation_only'] = False  # Reset flag on optimize

# --- Explain Code ---
if explain_btn and messy_code.strip():
    st.session_state['show_explanation_only'] = True
    with st.spinner("📖 Analyzing your code..."):
        explanation = explain_chain.invoke({"code": messy_code, "language": detected_language})
        explanation_text = explanation["text"] if isinstance(explanation, dict) and "text" in explanation else explanation
    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown("### 📖 Code Explanation")
    st.markdown(explanation_text)
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state['show_explanation_only'] = False  # Reset for next run
    st.stop()

# --- Show Side-by-Side Comparison ---
if (
    st.session_state.history
    and st.session_state.history[-1]["cleaned"].strip()
    and st.session_state.history[-1]["messy"].strip()
    and not st.session_state.get('show_explanation_only', False)
):
    last = st.session_state.history[-1]
    optimized = last["cleaned"]
    messy = last["messy"]
    detected_language = last["language"]

    st.markdown('<div class="comparison-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Code Comparison")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📝 Original Code**")
        st.code(messy, language=detected_language)
    with c2:
        st.markdown("**✨ Optimized & Commented Code**")
        st.code(optimized, language=detected_language)
        # Add Explain Code button for optimized code
        explain_optimized_btn = st.button("📖 Explain Optimized Code", key="explain_optimized_btn")

    # --- Download Optimized Code ---
    st.markdown("### 💾 Download")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="📥 Download Optimized Code",
            data=optimized,
            file_name=f"optimized_code.{detected_language}",
            mime="text/plain",
            use_container_width=True
        )
    with col2:
        st.info("💡 Your optimized code is ready for download!")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Optimized Code Explanation (Full Width) ---
    if 'explain_optimized_btn' in locals() and explain_optimized_btn:
        with st.spinner("📖 Analyzing your optimized code..."):
            explanation = explain_chain.invoke({"code": optimized, "language": detected_language})
            explanation_text = explanation["text"] if isinstance(explanation, dict) and "text" in explanation else explanation
        st.markdown("### 📖 Optimized Code Explanation")
        st.markdown(explanation_text)

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; border-top: 1px solid #404040;">
    <p style="color: #666; margin: 0;">🧹 Universal Code Optimizer - Powered by AI</p>
    <p style="color: #666; margin: 0; font-size: 0.9rem;">Optimize, format, and document your code with ease</p>
</div>
""", unsafe_allow_html=True)

# --- Dark Theme CSS (Only) ---
dark_css = """
<style>
.stApp {
    background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
    color: #ffffff;
}
.main-header {
    background: linear-gradient(135deg, #007acc 0%, #005a9e 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 122, 204, 0.3);
}
.feature-card {
    background: rgba(45, 45, 45, 0.8);
    border: 1px solid #404040;
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.stTextArea textarea {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
    border-color: #404040 !important;
    border-radius: 8px !important;
}
.stSelectbox select {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #007acc 0%, #005a9e 100%) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0, 122, 204, 0.4) !important;
}
.stMarkdown {
    color: #ffffff !important;
}
.stCode {
    background-color: #2d2d2d !important;
    border-radius: 8px !important;
}
.stAlert {
    background-color: rgba(45, 45, 45, 0.9) !important;
    border-color: #404040 !important;
    border-radius: 8px !important;
}
.stSuccess {
    background-color: rgba(30, 58, 30, 0.9) !important;
    border-color: #2d5a2d !important;
    border-radius: 8px !important;
}
.stWarning {
    background-color: rgba(58, 42, 30, 0.9) !important;
    border-color: #5a3a2d !important;
    border-radius: 8px !important;
}
.stError {
    background-color: rgba(58, 30, 30, 0.9) !important;
    border-color: #5a2d2d !important;
    border-radius: 8px !important;
}
.upload-section {
    background: rgba(45, 45, 45, 0.6);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid #404040;
}
.comparison-section {
    background: rgba(45, 45, 45, 0.4);
    border-radius: 15px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid #404040;
}
/* Hide drag-and-drop area, show only the Browse files button */
section[data-testid="stFileUploaderDropzone"] > div:first-child {
    display: none !important;
}
/* Hide drag-and-drop text, icon, and file size limit in file uploader */
section[data-testid="stFileUploaderDropzone"] div[role="presentation"],
section[data-testid="stFileUploaderDropzone"] svg,
section[data-testid="stFileUploaderDropzone"] span,
section[data-testid="stFileUploaderDropzone"] p,
section[data-testid="stFileUploaderDropzone"] small {
    display: none !important;
}
section[data-testid="stFileUploaderDropzone"] {
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
    background: none !important;
    height: 2.2rem !important;
    display: flex !important;
    align-items: center !important;
}
/* Hide all file details and filename/size displays in the uploader */
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDetails"],
section[data-testid="stFileUploader"] .uploadedFileName,
section[data-testid="stFileUploader"] .uploadedFileSize,
section[data-testid="stFileUploader"] ul,
section[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
section[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] {
    display: none !important;
}
/* Only show the Browse files button, make it full width and centered */
section[data-testid="stFileUploader"] button {
    width: 100% !important;
    margin: 0 auto !important;
    display: block !important;
}
/* Hide any filename display in or near the file uploader */
section[data-testid="stFileUploader"] .uploadedFileName,
section[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
section[data-testid="stFileUploader"] ul,
section[data-testid="stFileUploader"] li,
section[data-testid="stFileUploader"] div[role="list"],
section[data-testid="stFileUploader"] div[role="listitem"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)
