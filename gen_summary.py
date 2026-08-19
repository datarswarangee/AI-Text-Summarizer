import os
os.environ["USE_TF"] = "0"
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import io

# Title and Description
st.title("AI Text Summarizer")
st.write("Summarize large chunks of text using advanced Generative AI models (BART or T5).")

# Sidebar Controls
st.sidebar.header("⚙️ Model & Parameters")

# Model selection
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ("facebook/bart-large-cnn", "t5-base")
)

max_len = st.sidebar.slider("Maximum Summary Length", 50, 500, 150, 10)
min_len = st.sidebar.slider("Minimum Summary Length", 10, 100, 40, 5)

# Model Loader
@st.cache_resource
def load_model(model_name):
    return pipeline("summarization", model=model_name)

summarizer = load_model(model_choice)

# Text Input Section
st.subheader("✍️ Input Text or Upload File")

text_input = st.text_area("Enter your text here:", height=250)

uploaded_file = st.file_uploader("Or upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        text_input = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text_input = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

# Summarization Logic
if st.button("🚀 Summarize"):
    if text_input.strip():
        with st.spinner("Generating summary..."):
            summary = summarizer(text_input, max_length=max_len, min_length=min_len, do_sample=False)
            st.subheader("📄 Summary:")
            st.success(summary[0]['summary_text'])
    else:
        st.warning("Please provide text or upload a file first.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Hugging Face Transformers](https://huggingface.co/transformers/) and Streamlit.")
st.markdown("**Deploy free on [Hugging Face Spaces](https://huggingface.co/spaces)** or Streamlit Cloud.")
