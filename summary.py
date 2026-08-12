# ============================================================
# AI TEXT SUMMARIZER
# Streamlit + Hugging Face Transformers
# ============================================================

import os

# Disable TensorFlow before importing Transformers
os.environ["USE_TF"] = "0"

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
# IMAGE LOADER
# ============================================================

def get_image_base64(image_path):

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
           GLOBAL
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
           BACKGROUND IMAGE
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

        .main .block-container {{
            max-width: none;

            padding-top: 10px;

            padding-left: 30px;

            padding-right: 30px;

            padding-bottom: 70px;
        }}


        /* ==================================================
           TITLE
        ================================================== */

        .page-title {{
            text-align: center;

            color: #000000;

            font-size: 34px;

            font-weight: 600;

            margin-top: 5px;

            margin-bottom: 4px;
        }}


        /* ==================================================
           DESCRIPTION
        ================================================== */

        .page-description {{
            text-align: center;

            color: #000000;

            font-size: 15px;

            margin-bottom: 12px;
        }}


        /* ==================================================
           PANEL HEADINGS
        ================================================== */

        .panel-heading {{
            color: #000000;

            font-size: 20px;

            font-weight: 600;

            padding-bottom: 8px;

            margin-bottom: 8px;

            border-bottom: 2px solid #d63384;
        }}


        /* ==================================================
           TEXT AREA
        ================================================== */

        textarea {{
            background-color: #ffffff !important;

            color: #000000 !important;

            border: 2px solid #000000 !important;

            border-radius: 5px !important;

            font-family: 'Segoe UI', sans-serif !important;

            font-size: 15px !important;
        }}


        textarea:focus {{
            border: 2px solid #d63384 !important;

            box-shadow:
                0 0 0 1px #d63384 !important;
        }}


        textarea::placeholder {{
            color: #777777 !important;
        }}


        /* ==================================================
           FILE UPLOADER
        ================================================== */

        [data-testid="stFileUploader"] {{
            background: #ffffff !important;

            border: 2px solid #d63384 !important;

            border-radius: 5px !important;

            padding: 6px !important;

            margin-top: 8px !important;
        }}


        /* ==================================================
           FILE UPLOADER CONTENT
        ================================================== */

        [data-testid="stFileUploader"] section {{
            background: #ffffff !important;

            border: none !important;
        }}


        /* ==================================================
           FILE UPLOADER TEXT
        ================================================== */

        [data-testid="stFileUploader"] * {{
            color: #000000 !important;
        }}


        /* ==================================================
           UPLOAD BUTTON
        ================================================== */

        [data-testid="stFileUploader"] button {{
            background: #f8c8dc !important;

            color: #000000 !important;

            border: 1px solid #d63384 !important;

            border-radius: 5px !important;

            font-weight: 500 !important;
        }}


        /* ==================================================
           UPLOAD BUTTON HOVER
        ================================================== */

        [data-testid="stFileUploader"] button:hover {{
            background: #d63384 !important;

            color: #ffffff !important;

            border-color: #d63384 !important;
        }}


        /* ==================================================
           FILE INFORMATION
        ================================================== */

        .file-info {{
            color: #d63384;

            font-size: 13px;

            font-weight: 600;

            margin-top: 7px;

            margin-bottom: 5px;
        }}


        /* ==================================================
           SUMMARIZE BUTTON
        ================================================== */

        .stButton > button {{
            width: 100%;

            padding: 11px 20px;

            margin-top: 12px;

            border: 2px solid #d63384;

            border-radius: 5px;

            background: #d63384;

            color: #ffffff;

            cursor: pointer;

            font-size: 15px;

            font-weight: 600;
        }}


        /* ==================================================
           SUMMARIZE BUTTON HOVER
        ================================================== */

        .stButton > button:hover {{
            background: #000000;

            border-color: #000000;

            color: #ffffff;
        }}


        /* ==================================================
           SUMMARY BOX
        ================================================== */

        .summary-box {{
            background: #ffffff;

            border: 2px solid #000000;

            border-radius: 5px;

            min-height: 330px;

            padding: 20px;

            box-sizing: border-box;
        }}


        /* ==================================================
           AI LABEL
        ================================================== */

        .ai-label {{
            color: #d63384;

            font-size: 17px;

            font-weight: 600;

            margin-bottom: 10px;
        }}


        /* ==================================================
           SUMMARY TEXT
        ================================================== */

        .summary-content {{
            color: #000000;

            font-size: 16px;

            line-height: 1.8;

            text-align: left;

            white-space: pre-wrap;
        }}


        /* ==================================================
           EMPTY SUMMARY BOX
        ================================================== */

        .empty-summary {{
            background: #ffffff;

            border: 2px solid #000000;

            border-radius: 5px;

            min-height: 330px;

            padding: 20px;

            box-sizing: border-box;

            display: flex;

            flex-direction: column;

            align-items: center;

            justify-content: center;

            text-align: center;

            color: #777777;

            font-size: 15px;

            line-height: 1.7;
        }}


        /* ==================================================
           EMPTY SUMMARY TITLE
        ================================================== */

        .empty-summary-title {{
            color: #000000;

            font-size: 17px;

            font-weight: 600;

            margin-bottom: 12px;
        }}


        /* ==================================================
           EMPTY SUMMARY DESCRIPTION
        ================================================== */

        .empty-summary-text {{
            color: #777777;

            font-size: 15px;

            line-height: 1.7;
        }}


        /* ==================================================
           SIDEBAR
        ================================================== */

        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.90);

            backdrop-filter: blur(5px);

            -webkit-backdrop-filter: blur(5px);

            border-right: 2px solid #d63384;
        }}


        /* ==================================================
           SIDEBAR HEADINGS
        ================================================== */

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #000000;
        }}


        /* ==================================================
           SIDEBAR LABELS
        ================================================== */

        [data-testid="stSidebar"] label {{
            color: #000000 !important;

            font-weight: 500;
        }}


        /* ==================================================
           SIDEBAR SELECTBOX
        ================================================== */

        [data-testid="stSidebar"] [data-baseweb="select"] {{
            border-radius: 5px;

            border: 1px solid #000000;
        }}


        /* ==================================================
           SIDEBAR SLIDER
        ================================================== */

        [data-testid="stSlider"] [role="slider"] {{
            background-color: #d63384;
        }}


        /* ==================================================
           ALERTS
        ================================================== */

        [data-testid="stAlert"] {{
            border: 1px solid #d63384;

            border-radius: 5px;

            color: #000000;
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

            background: #000000;

            color: #ffffff;

            padding: 6px 0;

            font-size: 14px;

            z-index: 9999;
        }}


        /* ==================================================
           RESPONSIVE DESIGN
        ================================================== */

        @media (max-width: 900px) {{

            .main .block-container {{
                padding-left: 20px;

                padding-right: 20px;
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
    # FALLBACK IF chats.jpg IS MISSING
    # --------------------------------------------------------

    st.markdown(
        """
        <style>

        [data-testid="stAppViewContainer"] {
            background: #ffffff;
        }

        .main .block-container {
            padding-left: 30px;
            padding-right: 30px;
            padding-bottom: 70px;
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


# ============================================================
# MODEL SELECTION
# ============================================================

model_choice = st.sidebar.selectbox(
    "Choose Model",
    (
        "facebook/bart-large-cnn",
        "t5-base"
    )
)


# ============================================================
# MAXIMUM SUMMARY LENGTH
# ============================================================

max_len = st.sidebar.slider(
    "Maximum Summary Length",
    50,
    500,
    150,
    10
)


# ============================================================
# MINIMUM SUMMARY LENGTH
# ============================================================

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
# TWO-COLUMN LAYOUT
# ============================================================

input_column, summary_column = st.columns(
    [1, 1],
    gap="medium"
)


# ============================================================
# LEFT COLUMN — INPUT
# ============================================================

with input_column:

    # --------------------------------------------------------
    # Input Heading
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel-heading">
            ✍🏻 Input Text
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Text Input
    # --------------------------------------------------------

    text_input = st.text_area(
        "Input Text",
        height=250,
        label_visibility="collapsed",
        placeholder="Paste the text you want to summarize here..."
    )


    # --------------------------------------------------------
    # Upload Label
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="file-info">
            📎 Upload PDF or TXT
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # File Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=["txt", "pdf"],
        label_visibility="collapsed"
    )


    # ========================================================
    # PROCESS UPLOADED FILE
    # ========================================================

    if uploaded_file is not None:

        # ----------------------------------------------------
        # TXT FILE
        # ----------------------------------------------------

        if uploaded_file.type == "text/plain":

            text_input = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

            st.markdown(
                f"""
                <div class="file-info">
                    🪭 {html.escape(uploaded_file.name)}
                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PDF FILE
        # ----------------------------------------------------

        elif uploaded_file.type == "application/pdf":

            pdf_reader = PdfReader(
                uploaded_file
            )

            extracted_pages = []


            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:

                    extracted_pages.append(
                        page_text
                    )


            text_input = "\n".join(
                extracted_pages
            )


            st.markdown(
                f"""
                <div class="file-info">
                    📄 {html.escape(uploaded_file.name)}
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SUMMARIZE BUTTON
    # ========================================================

    summarize_button = st.button(
        "🚀 Summarize"
    )


# ============================================================
# RIGHT COLUMN — SUMMARY
# ============================================================

with summary_column:

    # --------------------------------------------------------
    # Summary Heading
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="panel-heading">
            📄 Summary
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    if summarize_button:

        # ----------------------------------------------------
        # CHECK INPUT
        # ----------------------------------------------------

        if text_input and text_input.strip():

            try:

                # ------------------------------------------------
                # Generate summary
                # ------------------------------------------------

                with st.spinner(
                    "Generating summary..."
                ):

                    summary = summarizer(
                        text_input,
                        max_length=max_len,
                        min_length=min_len,
                        do_sample=False
                    )


                # ------------------------------------------------
                # Extract summary
                # ------------------------------------------------

                summary_text = summary[0][
                    "summary_text"
                ]


                # ------------------------------------------------
                # Protect HTML characters
                # ------------------------------------------------

                safe_summary = html.escape(
                    summary_text
                )


                # ------------------------------------------------
                # Display summary
                # ------------------------------------------------

                st.markdown(
                    f"""
                    <div class="summary-box">

                        <div class="ai-label">
                            AI:
                        </div>

                        <div class="summary-content">
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


        # ----------------------------------------------------
        # NO INPUT
        # ----------------------------------------------------

        else:

            st.markdown(
                """
                <div class="empty-summary">

                    <div class="empty-summary-title">
                        No text to summarize
                    </div>

                    <div class="empty-summary-text">
                        Enter text or upload a document
                        and click Summarize.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # INITIAL SUMMARY STATE
    # ========================================================

    else:

        st.markdown(
            """
            <div class="empty-summary">

                <div class="empty-summary-title">
                    Your generated summary will appear here.
                </div>

                <div class="empty-summary-text">
                    Enter text or upload a document
                    and click Summarize.
                </div>

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
        © The work is copywrite!
    </div>
    """,
    unsafe_allow_html=True
)
