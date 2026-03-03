import streamlit as st
import time

# Try to import LangChain components. 
# If not installed, the app will run in "Demo Mode" with mock responses.
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Coron AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM STYLES ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F46E5; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #4B5563;}
    .agent-card {padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR & SETTINGS ---
with st.sidebar:
    st.title("🧬 Coron AI")
    st.markdown("### Control Panel")
    
    api_provider = st.selectbox("Select AI Provider", ["Demo Mode (Mock)", "Groq (Free Llama 3)"])
    
    api_key = ""
    if api_provider == "Groq (Free Llama 3)":
        api_key = st.text_input("Enter Groq API Key", type="password", help="Get a free key at console.groq.com")
        if not api_key:
            st.warning("Please enter an API key to use the live agents.")

    st.markdown("---")
    st.markdown("### Available Agents")
    tool_selection = st.radio(
        "Choose your tool:",
        ["Coron Chat (General)", "Code Architect", "Docu-Summarizer"]
    )
    
    st.markdown("---")
    st.info("Coron AI v1.0\nPowered by Open Source Models")

# --- AI LOGIC ENGINE ---
def get_ai_response(prompt_text, system_role, temperature=0.7):
    """
    Handles the logic for fetching AI responses.
    Switches between Mock mode and Real AI mode.
    """
    # 1. Demo Mode Logic
    if api_provider == "Demo Mode (Mock)" or not api_key:
        time.sleep(1) # Simulate latency
        if "code" in system_role.lower():
            return f"```python\n# Mock Code Response for: {prompt_text}\ndef coron_function():\n    print('Hello from Coron AI Demo')\n```"
        return f"**[DEMO MODE]** I received your input: '{prompt_text}'. To get real intelligent responses, please switch to Groq and provide an API Key."

    # 2. Real AI Logic (Groq)
    if HAS_LANGCHAIN and api_key:
        try:
            llm = ChatGroq(temperature=temperature, groq_api_key=api_key, model_name="llama3-8b-8192")
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_role),
                ("human", "{input}")
            ])
            chain = prompt | llm | StrOutputParser()
            return chain.invoke({"input": prompt_text})
        except Exception as e:
            return f"Error connecting to AI Agent: {str(e)}"
    
    return "Configuration Error: Please check your API Key or Libraries."

# --- MAIN INTERFACE ---

st.markdown('<p class="main-header">Welcome to Coron AI</p>', unsafe_allow_html=True)

# 1. GENERAL CHAT AGENT
if tool_selection == "Coron Chat (General)":
    st.markdown('<p class="sub-header">Your General Purpose AI Assistant</p>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask Coron AI anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Coron AI is thinking..."):
                response = get_ai_response(
                    prompt, 
                    "You are Coron AI, a helpful and friendly AI assistant."
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# 2. CODE ARCHITECT AGENT
elif tool_selection == "Code Architect":
    st.markdown('<p class="sub-header">Software Engineering Agent</p>', unsafe_allow_html=True)
    st.markdown("Describe the feature or script you need, and I will generate the code.")
    
    code_prompt = st.text_area("Describe your coding task:", height=150)
    language = st.selectbox("Target Language", ["Python", "JavaScript", "HTML/CSS", "SQL"])
    
    if st.button("Generate Code"):
        if code_prompt:
            with st.spinner("Architecting solution..."):
                system_prompt = f"You are an expert coder. Write clean, commented {language} code. Do not provide conversational filler, just the code and brief explanation."
                response = get_ai_response(code_prompt, system_prompt, temperature=0.2)
                st.markdown(response)
        else:
            st.error("Please enter a description first.")

# 3. SUMMARIZER AGENT
elif tool_selection == "Docu-Summarizer":
    st.markdown('<p class="sub-header">Content Condenser Agent</p>', unsafe_allow_html=True)
    
    txt_input = st.text_area("Paste text to summarize:", height=200)
    summary_style = st.selectbox("Summary Style", ["Bullet Points", "Concise Paragraph", "EL15 (Explain like I'm 5)"])
    
    if st.button("Summarize"):
        if txt_input:
            with st.spinner("Analyzing text..."):
                system_prompt = f"You are a summarization expert. Summarize the text in the following style: {summary_style}."
                response = get_ai_response(txt_input, system_prompt)
                st.markdown("### Summary")
                st.info(response)
        else:
            st.error("Please paste some text first.")

# --- FOOTER ---
st.markdown("---")
st.caption("Coron AI | Built with Streamlit & LangChain")
