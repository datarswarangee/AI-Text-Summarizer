# AI Text Summarizer

An AI-powered text summarization application built using Python, Streamlit, and Hugging Face Transformers.

The application allows users to enter large amounts of text or upload PDF/TXT documents and automatically generate concise summaries using pretrained Generative AI models such as BART and T5.

---

## Features

- AI-powered automatic text summarization
- Supports multiple pretrained Hugging Face models
- BART summarization using:
  - `facebook/bart-large-cnn`
- T5 summarization using:
  - `t5-base`
- Direct text input
- PDF document upload
- TXT document upload
- Adjustable minimum summary length
- Adjustable maximum summary length
- Cached model loading for improved performance
- Custom Streamlit interface
- Pink, black, and white visual theme
- Custom `chats.jpg` background
- Responsive two-column layout
- Dedicated input and summary sections
- File name display after uploading a document
- Error handling during summarization

---

## Application Interface

The application consists of two primary sections:

### 1. Input Text

Users can either:

- Paste text directly into the text area
- Upload a `.txt` file
- Upload a `.pdf` file

The uploaded document is automatically converted into text before summarization.

### 2. Summary

The generated AI summary is displayed in a dedicated summary panel.

The interface uses a pink, black, and white color scheme with the custom `chats.jpg` image as the background.

---

## AI Models

The application currently supports the following models:

### BART

```text
facebook/bart-large-cnn
  
## Installation
```bash
pip install -r requirements.txt
