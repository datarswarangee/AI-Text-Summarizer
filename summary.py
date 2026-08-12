# ============================================================
# AI TEXT SUMMARIZER
# Streamlit + Hugging Face Transformers
# ============================================================

import os

# ------------------------------------------------------------
# Disable TensorFlow BEFORE importing Transformers
# ------------------------------------------------------------

os.environ["USE_TF"] = "0"


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

import base64
import html

import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD BACKGROUND IMAGE
# ============================================================

def get_image_base64(image_path):
    """
    Convert a local image into Base64 so that it can be
    directly embedded into Streamlit CSS.
    """

    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(
                image_file.read()
            ).decode()

    except FileNotFoundError:
        return None


chat_background = get_image_base64("chats.jpg")


# ============================================================
# CUSTOM CSS
# ============================================================

if chat_background:

    st.markdown(
        f"""
        <style>

        /* ==================================================
           GLOBAL PAGE
        ================================================== */

        html,
        body,
        [data-testid="stAppViewContainer"] {{

            margin: 0;
            padding: 0;

            font-family: 'Segoe UI', sans-serif;

            cursor: url('icon.png') 16 16, auto;
        }}


        /* ==================================================
           MAIN APPLICATION BACKGROUND
        ================================================== */

        [data-testid="stAppViewContainer"] {{

            background-image:
                url("data:image/jpeg;base64,{chat_background}");

            background-repeat: no-repeat;

            background-position: center center;

            background-size: cover;

            background-attachment: fixed;

            min-height: 100vh;
        }}


        /* ==================================================
           STREAMLIT HEADER
        ================================================== */

        [data-testid="stHeader"] {{

            background: transparent;
        }}


        /* ==================================================
           MAIN CONTENT
        ================================================== */

        [data-testid="stMain"] {{

            background: transparent;
        }}


        .main .block-container {{

            max-width: none;

            padding-top: 25px;

            padding-left: 80px;

            padding-right: 80px;

            padding-bottom: 90px;
        }}


        /* ==================================================
           TITLE
        ================================================== */

        .page-title {{

            text-align: center;

            color: #222;

            font-size: 34px;

            font-weight: 600;

            margin-top: 5px;

            margin-bottom: 5px;
        }}


        /* ==================================================
           DESCRIPTION
        ================================================== */

        .page-description {{

            text-align: center;

            color: #555;

            font-size: 16px;

            margin-bottom: 20px;
        }}


        /* ==================================================
           WHITE DOCUMENT AREA
        ================================================== */

        .document-area {{

            background: rgba(255, 255, 255, 0.97);

            border-radius: 6px;

            padding: 25px;

            min-height: 500px;

            box-shadow:
                0 2px 10px rgba(0, 0, 0, 0.15);

            margin-bottom: 20px;
        }}


        /* ==================================================
           SECTION HEADINGS
        ================================================== */

        .section-heading {{

            color: #222;

            font-size: 20px;

            font-weight: 600;

            margin-bottom: 10px;
        }}


        /* ==================================================
           TEXT AREA
        ================================================== */

        textarea {{

            background-color: white !important;

            color: #222 !important;

            border: 1px solid #ccc !important;

            border-radius: 5px !important;

            font-family: 'Segoe UI', sans-serif !important;

            font-size: 15px !important;
        }}


        textarea:focus {{

            border: 1px solid #d63384 !important;

            box-shadow:
                0 0 0 1px #d63384 !important;
        }}


        /* ==================================================
           FILE UPLOADER
        ================================================== */

        [data-testid="stFileUploader"] {{

            background: white;

            border: 1px solid #ccc;

            border-radius: 5px;

            padding: 10px;
        }}


        /* ==================================================
           SUMMARIZE BUTTON
        ================================================== */

        .stButton > button {{

            padding: 12px 20px;

            margin-left: 0;

            border: none;

            border-radius: 5px;

            background: #4CAF50;

            color: white;

            cursor: pointer;

            font-size: 15px;

            font-weight: 500;

            width: 100%;
        }}


        .stButton > button:hover {{

            background: #45a049;

            color: white;
        }}


        /* ==================================================
           SUMMARY BOX
        ================================================== */

        .summary-box {{

            margin-top: 20px;

            padding: 20px;

            background: white;

            border-radius: 6px;

            color: #222;

            line-height: 1.7;

            border-left: 4px solid #d63384;

            box-shadow:
                0 1px 5px rgba(0, 0, 0, 0.08);
        }}


        /* ==================================================
           AI LABEL
        ================================================== */

        .bot-label {{

            color: #222;

            font-weight: 600;

            margin-bottom: 8px;
        }}


        /* ==================================================
           SUMMARY TEXT
        ================================================== */

        .summary-text {{

            color: #222;

            font-size: 16px;

            line-height: 1.7;

            text-align: left;

            white-space: pre-wrap;
        }}


        /* ==================================================
           FILE INFORMATION
        ================================================== */

        .file-info {{

            color: #d63384;

            text-align: right;

            margin-top: 8px;

            margin-bottom: 10px;

            font-size: 14px;
        }}


        /* ==================================================
           SIDEBAR
        ================================================== */

        [data-testid="stSidebar"] {{

            background: rgba(255, 255, 255, 0.88);

            backdrop-filter: blur(5px);

            -webkit-backdrop-filter: blur(5px);
        }}


        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{

            color: #222;
        }}


        /* ==================================================
           SIDEBAR LABELS
        ================================================== */

        [data-testid="stSidebar"] label {{

            color: #333 !important;

            font-weight: 500;
        }}


        /* ==================================================
           SELECT BOX
        ================================================== */

        [data-testid="stSidebar"] [data-baseweb="select"] {{

            border-radius: 5px;
        }}


        /* ==================================================
           SLIDER
        ================================================== */

        [data-testid="stSidebar"] [data-testid="stSlider"] {{

            margin-bottom: 15px;
        }}


        /* ==================================================
           WARNING / ALERT
        ================================================== */

        [data-testid="stAlert"] {{

            border-radius: 5px;
        }}


        /* ==================================================
           FOOTER
        ================================================== */

        .custom-footer {{

            position: fixed;

            bottom: 0;

            left: 0;

            width: 100%;

            text-align: center;

            background: #000;

            color: #fff;

            padding: 6px 0;

            font-size: 14px;

            z-index: 9999;
        }}


        /* ==================================================
           RESPONSIVE DESIGN
        ================================================== */

        @media (max-width: 900px) {{

            .main .block-container {{

                padding-left: 25px;

                padding-right: 25px;
            }}

            .page-title {{

                font-size: 28px;
            }}
        }}


        </style>
        """,
        unsafe_allow_html=True
    )

else:

    # --------------------------------------------------------
    # Fallback if chats.jpg is missing
    # --------------------------------------------------------

    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {

            background: #111;
        }

        .main .block-container {

            padding-left: 80px;

            padding-right: 80px;

            padding-bottom: 90px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="page-title">
        AI Text Summarizer
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DESCRIPTION
# ============================================================

st.markdown(
    """
    <div class="page-description">
        Summarize large chunks of text using advanced
        Generative AI models (BART or T5).
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Model & Parameters")


# ------------------------------------------------------------
# Model Selection
# ------------------------------------------------------------

model_choice = st.sidebar.selectbox(
    "Choose Model",
    (
        "facebook/bart-large-cnn",
        "t5-base"
    )
)


# ------------------------------------------------------------
# Maximum Summary Length
# ------------------------------------------------------------

max_len = st.sidebar.slider(
    "Maximum Summary Length",
    50,
    500,
    150,
    10
)


# ------------------------------------------------------------
# Minimum Summary Length
# ------------------------------------------------------------

min_len = st.sidebar.slider(
    "Minimum Summary Length",
    10,
    100,
    40,
    5
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource
def load_model(model_name):

    return pipeline(
        "summarization",
        model=model_name
    )


# ============================================================
# LOAD MODEL
# ============================================================

summarizer = load_model(model_choice)


# ============================================================
# WHITE DOCUMENT AREA
# ============================================================

st.markdown(
    """
    <div class="document-area">
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        ✍️ Input Text
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT INPUT
# ============================================================

text_input = st.text_area(
    "Enter your text here:",
    height=250,
    label_visibility="collapsed",
    placeholder="Paste the text you want to summarize here..."
)


# ============================================================
# FILE UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-heading">
        📎 Upload Document
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a .txt or .pdf file",
    type=["txt", "pdf"],
    label_visibility="collapsed"
)


# ============================================================
# READ UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # TXT FILE
    # --------------------------------------------------------

    if uploaded_file.type == "text/plain":

        text_input = uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

        st.markdown(
            f"""
            <div class="file-info">
                📄 File loaded: {html.escape(uploaded_file.name)}
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PDF FILE
    # --------------------------------------------------------

    elif uploaded_file.type == "application/pdf":

        pdf_reader = PdfReader(uploaded_file)

        extracted_pages = []

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:

                extracted_pages.append(page_text)


        text_input = "\n".join(extracted_pages)


        st.markdown(
            f"""
            <div class="file-info">
                📄 PDF loaded: {html.escape(uploaded_file.name)}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# SUMMARIZE BUTTON
# ============================================================

summarize_button = st.button(
    "🚀 Summarize"
)


# ============================================================
# SUMMARIZATION LOGIC
# ============================================================

if summarize_button:

    # --------------------------------------------------------
    # Check Input
    # --------------------------------------------------------

    if text_input and text_input.strip():

        try:

            # ------------------------------------------------
            # Generate Summary
            # ------------------------------------------------

            with st.spinner("Generating summary..."):

                summary = summarizer(
                    text_input,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False
                )


            # ------------------------------------------------
            # Extract Summary
            # ------------------------------------------------

            summary_text = summary[0]["summary_text"]


            # ------------------------------------------------
            # Escape HTML
            # ------------------------------------------------

            safe_summary = html.escape(
                summary_text
            )


            # ------------------------------------------------
            # Summary Heading
            # ------------------------------------------------

            st.markdown(
                """
                <div class="section-heading">
                    📄 Summary
                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # Summary Result
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="summary-box">

                    <div class="bot-label">
                        AI:
                    </div>

                    <div class="summary-text">
                        {safe_summary}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        except Exception as e:

            st.error(
                f"Unable to generate the summary: {str(e)}"
            )


    else:

        st.warning(
            "Please provide text or upload a file first."
        )


# ============================================================
# CLOSE WHITE DOCUMENT AREA
# ============================================================

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">
        © The copyrights of this work is to datwashere
    </div>
    """,
    unsafe_allow_html=True
)
