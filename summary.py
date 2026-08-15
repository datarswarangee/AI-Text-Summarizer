# ============================================================
# AI TEXT SUMMARIZER
# Streamlit + Hugging Face Transformers
#
# Features:
# 1. BART / T5 text summarization
# 2. TXT and PDF upload
# 3. Important word extraction
# 4. Important sentence detection
# 5. Pink highlighting of important words
# 6. Pink highlighting of important sentences
# 7. Custom chats.jpg background
# 8. Pink + black UI
# ============================================================


# ============================================================
# DISABLE TENSORFLOW
# ============================================================

import os

os.environ["USE_TF"] = "0"


# ============================================================
# IMPORTS
# ============================================================

import base64
import html
import math
import re
from collections import Counter

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

        with open(
            image_path,
            "rb"
        ) as image_file:

            return base64.b64encode(
                image_file.read()
            ).decode()

    except FileNotFoundError:

        return None


chat_background = get_image_base64(
    "chats.jpg"
)


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves"
}


# ============================================================
# TEXT TOKENIZATION
# ============================================================

def tokenize_words(text):

    return re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]{2,}\b",
        text.lower()
    )


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# IMPORTANT WORD EXTRACTION
# ============================================================

def extract_important_words(
    text,
    top_n=12
):

    sentences = split_sentences(
        text
    )

    if not sentences:

        return []


    # --------------------------------------------------------
    # All words
    # --------------------------------------------------------

    all_words = tokenize_words(
        text
    )


    # --------------------------------------------------------
    # Remove stopwords
    # --------------------------------------------------------

    meaningful_words = [
        word
        for word in all_words
        if word not in STOPWORDS
    ]


    if not meaningful_words:

        return []


    # --------------------------------------------------------
    # Term Frequency
    # --------------------------------------------------------

    term_frequency = Counter(
        meaningful_words
    )


    # --------------------------------------------------------
    # Document Frequency
    # --------------------------------------------------------

    document_frequency = Counter()


    for sentence in sentences:

        sentence_words = set(
            tokenize_words(
                sentence
            )
        )

        for word in sentence_words:

            if word not in STOPWORDS:

                document_frequency[word] += 1


    total_sentences = max(
        len(sentences),
        1
    )


    # --------------------------------------------------------
    # Importance score
    # --------------------------------------------------------

    scores = {}


    for word, frequency in term_frequency.items():

        df = document_frequency.get(
            word,
            1
        )


        # Term frequency
        tf = (
            frequency
            /
            len(meaningful_words)
        )


        # Inverse document frequency
        idf = (
            math.log(
                (total_sentences + 1)
                /
                (df + 1)
            )
            + 1
        )


        score = tf * idf


        # Small boost for repeated terms
        if frequency > 1:

            score *= (
                1
                +
                0.15 * (frequency - 1)
            )


        scores[word] = score


    # --------------------------------------------------------
    # Rank words
    # --------------------------------------------------------

    ranked_words = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    return ranked_words[:top_n]


# ============================================================
# IMPORTANT SENTENCE RANKING
# ============================================================

def rank_important_sentences(
    text,
    important_words
):

    sentences = split_sentences(
        text
    )

    if not sentences:

        return []


    keyword_scores = dict(
        important_words
    )


    ranked_sentences = []


    for index, sentence in enumerate(
        sentences
    ):

        words = tokenize_words(
            sentence
        )


        meaningful_words = [
            word
            for word in words
            if word not in STOPWORDS
        ]


        if not meaningful_words:

            score = 0.0

        else:

            # ----------------------------------------------
            # Keyword relevance
            # ----------------------------------------------

            keyword_score = sum(
                keyword_scores.get(
                    word,
                    0
                )
                for word in meaningful_words
            )


            # ----------------------------------------------
            # Normalize by sentence length
            # ----------------------------------------------

            keyword_score /= math.sqrt(
                len(meaningful_words)
            )


            # ----------------------------------------------
            # Position score
            # ----------------------------------------------

            position_score = (
                1
                /
                (1 + index * 0.15)
            )


            # ----------------------------------------------
            # Combined score
            # ----------------------------------------------

            score = (
                keyword_score * 0.85
                +
                position_score * 0.15
            )


        ranked_sentences.append(
            {
                "index": index,
                "sentence": sentence,
                "score": score
            }
        )


    # --------------------------------------------------------
    # Highest score first
    # --------------------------------------------------------

    ranked_sentences.sort(
        key=lambda item: item["score"],
        reverse=True
    )


    return ranked_sentences


# ============================================================
# HIGHLIGHT IMPORTANT WORDS
# ============================================================

def highlight_words(
    sentence,
    important_words
):

    # Escape the original text first
    safe_sentence = html.escape(
        sentence
    )


    if not important_words:

        return safe_sentence


    # Longest words first
    keywords = sorted(
        [
            word
            for word, _ in important_words
        ],
        key=len,
        reverse=True
    )


    for keyword in keywords:

        safe_keyword = html.escape(
            keyword
        )


        pattern = (
            r"(?<![A-Za-z])"
            +
            re.escape(keyword)
            +
            r"(?![A-Za-z])"
        )


        replacement = (
            '<span class="important-word">'
            +
            safe_keyword
            +
            '</span>'
        )


        safe_sentence = re.sub(
            pattern,
            replacement,
            safe_sentence,
            flags=re.IGNORECASE
        )


    return safe_sentence


# ============================================================
# BUILD HIGHLIGHTED DOCUMENT
# ============================================================

def build_highlighted_document(
    text,
    important_words,
    ranked_sentences
):

    sentences = split_sentences(
        text
    )

    if not sentences:

        return ""


    total_sentences = len(
        sentences
    )


    # Highlight approximately top 30%
    number_to_highlight = max(
        1,
        min(
            6,
            math.ceil(
                total_sentences * 0.30
            )
        )
    )


    important_indices = {
        item["index"]
        for item in ranked_sentences[
            :number_to_highlight
        ]
    }


    highlighted_parts = []


    for index, sentence in enumerate(
        sentences
    ):

        highlighted_sentence = highlight_words(
            sentence,
            important_words
        )


        if index in important_indices:

            highlighted_parts.append(
                '<div class="important-sentence">'
                + highlighted_sentence
                + '</div>'
            )

        else:

            highlighted_parts.append(
                '<div class="normal-sentence">'
                + highlighted_sentence
                + '</div>'
            )


    return "".join(
        highlighted_parts
    )


# ============================================================
# CUSTOM CSS
# ============================================================

if chat_background:

    st.markdown(
        f"""
<style>

html,
body,
[data-testid="stAppViewContainer"] {{
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', sans-serif;
    cursor: url('icon.png') 16 16, auto;
}}


/* ============================================================
   BACKGROUND
   ============================================================ */

[data-testid="stAppViewContainer"] {{
    background-image:
        url("data:image/jpeg;base64,{chat_background}");

    background-repeat: no-repeat;

    background-position: center center;

    background-size: cover;

    background-attachment: fixed;

    min-height: 100vh;
}}


/* ============================================================
   HEADER
   ============================================================ */

[data-testid="stHeader"] {{
    background: transparent;
}}


/* ============================================================
   MAIN CONTENT
   ============================================================ */

.main .block-container {{
    max-width: none;

    padding-top: 10px;

    padding-left: 30px;

    padding-right: 30px;

    padding-bottom: 70px;
}}


/* ============================================================
   TITLE
   ============================================================ */

.page-title {{
    text-align: center;

    color: #000000;

    font-size: 34px;

    font-weight: 600;

    margin-top: 5px;

    margin-bottom: 4px;
}}


/* ============================================================
   DESCRIPTION
   ============================================================ */

.page-description {{
    text-align: center;

    color: #000000;

    font-size: 15px;

    margin-bottom: 12px;
}}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.panel-heading {{
    color: #000000;

    font-size: 20px;

    font-weight: 600;

    padding-bottom: 8px;

    margin-bottom: 8px;

    border-bottom: 2px solid #d63384;
}}


/* ============================================================
   TEXT AREA
   ============================================================ */

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


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {{
    background: #ffffff !important;

    border: 2px solid #d63384 !important;

    border-radius: 5px !important;

    padding: 6px !important;

    margin-top: 8px !important;
}}


[data-testid="stFileUploader"] section {{
    background: #ffffff !important;

    border: none !important;
}}


[data-testid="stFileUploader"] * {{
    color: #000000 !important;
}}


/* ============================================================
   UPLOAD BUTTON
   ============================================================ */

[data-testid="stFileUploader"] button {{
    background: #f8c8dc !important;

    color: #000000 !important;

    border: 1px solid #d63384 !important;

    border-radius: 5px !important;

    font-weight: 500 !important;
}}


[data-testid="stFileUploader"] button:hover {{
    background: #d63384 !important;

    color: #ffffff !important;

    border-color: #d63384 !important;
}}


/* ============================================================
   FILE INFORMATION
   ============================================================ */

.file-info {{
    color: #d63384;

    font-size: 13px;

    font-weight: 600;

    margin-top: 7px;

    margin-bottom: 5px;
}}


/* ============================================================
   SUMMARIZE BUTTON
   ============================================================ */

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


.stButton > button:hover {{
    background: #000000;

    border-color: #000000;

    color: #ffffff;
}}


/* ============================================================
   SUMMARY BOX
   ============================================================ */

.summary-box {{
    background: #ffffff;

    border: 2px solid #000000;

    border-radius: 5px;

    min-height: 330px;

    padding: 20px;

    box-sizing: border-box;
}}


/* ============================================================
   AI LABEL
   ============================================================ */

.ai-label {{
    color: #d63384;

    font-size: 17px;

    font-weight: 600;

    margin-bottom: 10px;
}}


/* ============================================================
   SUMMARY TEXT
   ============================================================ */

.summary-content {{
    color: #000000;

    font-size: 16px;

    line-height: 1.8;

    text-align: left;

    white-space: pre-wrap;
}}


/* ============================================================
   EMPTY SUMMARY
   ============================================================ */

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


.empty-summary-title {{
    color: #000000;

    font-size: 17px;

    font-weight: 600;

    margin-bottom: 12px;
}}


.empty-summary-text {{
    color: #777777;

    font-size: 15px;

    line-height: 1.7;
}}


/* ============================================================
   IMPORTANT WORDS BOX
   ============================================================ */

.important-words-box {{
    background: #ffffff;

    border: 2px solid #000000;

    border-radius: 5px;

    padding: 18px;

    margin-top: 25px;

    box-sizing: border-box;
}}


.important-words-title {{
    color: #000000;

    font-size: 20px;

    font-weight: 600;

    padding-bottom: 8px;

    margin-bottom: 15px;

    border-bottom: 2px solid #d63384;
}}


/* ============================================================
   KEYWORD CONTAINER
   ============================================================ */

.keyword-container {{
    display: flex;

    flex-wrap: wrap;

    gap: 8px;
}}


/* ============================================================
   KEYWORD TAG
   ============================================================ */

.keyword-tag {{
    display: inline-block;

    background: #f8c8dc;

    color: #000000;

    border: 1px solid #d63384;

    border-radius: 4px;

    padding: 6px 10px;

    font-size: 14px;

    font-weight: 500;
}}


/* ============================================================
   IMPORTANT PARTS TITLE
   ============================================================ */

.highlight-title {{
    color: #000000;

    font-size: 20px;

    font-weight: 600;

    padding-bottom: 8px;

    margin-top: 25px;

    margin-bottom: 10px;

    border-bottom: 2px solid #d63384;
}}


/* ============================================================
   HIGHLIGHTED DOCUMENT
   ============================================================ */

.highlighted-document {{
    background: #ffffff;

    border: 2px solid #000000;

    border-radius: 5px;

    padding: 20px;

    margin-top: 15px;

    color: #000000;

    font-size: 15px;

    line-height: 1.8;
}}


/* ============================================================
   IMPORTANT SENTENCE
   ============================================================ */

.important-sentence {{
    background: #f8c8dc;

    border-left: 5px solid #d63384;

    color: #000000;

    padding: 10px 12px;

    margin: 8px 0;

    border-radius: 4px;
}}


/* ============================================================
   NORMAL SENTENCE
   ============================================================ */

.normal-sentence {{
    color: #000000;

    padding: 5px 0;

    margin: 2px 0;
}}


/* ============================================================
   IMPORTANT WORD
   ============================================================ */

.important-word {{
    background: #d63384;

    color: #ffffff;

    padding: 2px 5px;

    border-radius: 4px;

    font-weight: 600;
}}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {{
    background: rgba(255, 255, 255, 0.90);

    backdrop-filter: blur(5px);

    -webkit-backdrop-filter: blur(5px);

    border-right: 2px solid #d63384;
}}


[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: #000000;
}}


[data-testid="stSidebar"] label {{
    color: #000000 !important;

    font-weight: 500;
}}


[data-testid="stSidebar"] [data-baseweb="select"] {{
    border-radius: 5px;

    border: 1px solid #000000;
}}


[data-testid="stSlider"] [role="slider"] {{
    background-color: #d63384;
}}


/* ============================================================
   ALERTS
   ============================================================ */

[data-testid="stAlert"] {{
    border: 1px solid #d63384;

    border-radius: 5px;

    color: #000000;
}}


/* ============================================================
   FOOTER
   ============================================================ */

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


/* ============================================================
   RESPONSIVE
   ============================================================ */

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

st.sidebar.header(
    "⚙️ Model & Parameters"
)


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
# SUMMARY LENGTH
# ============================================================

max_len = st.sidebar.slider(
    "Maximum Summary Length",
    50,
    500,
    150,
    10
)


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

summarizer = load_model(
    model_choice
)


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

    st.markdown(
        """
<div class="panel-heading">
    ✍🏻 Input Text
</div>
""",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # TEXT INPUT
    # --------------------------------------------------------

    text_input = st.text_area(
        "Input Text",
        height=250,
        label_visibility="collapsed",
        placeholder=(
            "Paste the text you want to "
            "summarize here..."
        )
    )


    # --------------------------------------------------------
    # UPLOAD LABEL
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
    # FILE UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a .txt or .pdf file",
        type=[
            "txt",
            "pdf"
        ],
        label_visibility="collapsed"
    )


    # ========================================================
    # READ UPLOADED FILE
    # ========================================================

    if uploaded_file is not None:

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------

        if uploaded_file.type == "text/plain":

            text_input = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )


            st.markdown(
                f"""
<div class="file-info">
    📄 {html.escape(uploaded_file.name)}
</div>
""",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PDF
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

    st.markdown(
        """
<div class="panel-heading">
    📄 Summary
</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # SUMMARIZATION
    # ========================================================

    if summarize_button:

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
                # Escape summary text
                # ------------------------------------------------

                safe_summary = html.escape(
                    summary_text
                )


                # =================================================
                # DISPLAY SUMMARY
                #
                # st.html() is deliberately used here.
                # No Markdown parser is involved.
                # =================================================

                st.html(
                    f"""
<div class="summary-box">
    <div class="ai-label">AI:</div>

    <div class="summary-content">
        {safe_summary}
    </div>
</div>
"""
                )


                # =================================================
                # IMPORTANT WORD ANALYSIS
                # =================================================

                with st.spinner(
                    "Analyzing important words and sentences..."
                ):

                    important_words = (
                        extract_important_words(
                            text_input,
                            top_n=12
                        )
                    )


                    ranked_sentences = (
                        rank_important_sentences(
                            text_input,
                            important_words
                        )
                    )


                # =================================================
                # IMPORTANT WORDS
                # =================================================

                if important_words:

                    keyword_tags = ""


                    for word, score in important_words:

                        safe_word = html.escape(
                            word
                        )


                        keyword_tags += (
                            '<span class="keyword-tag">'
                            + safe_word
                            + '</span>'
                        )


                    st.html(
                        f"""
<div class="important-words-box">

    <div class="important-words-title">
        🔑 Important Words
    </div>

    <div class="keyword-container">
        {keyword_tags}
    </div>

</div>
"""
                    )


                # =================================================
                # IMPORTANT PARTS
                # =================================================

                if ranked_sentences:

                    highlighted_document = (
                        build_highlighted_document(
                            text_input,
                            important_words,
                            ranked_sentences
                        )
                    )


                    st.html(
                        f"""
<div class="highlight-title">
    📌 Important Parts
</div>

<div class="highlighted-document">
    {highlighted_document}
</div>
"""
                    )


            except Exception as e:

                st.error(
                    "Unable to generate the summary: "
                    +
                    str(e)
                )


        else:

            st.html(
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
"""
            )


    # ========================================================
    # INITIAL STATE
    # ========================================================

    else:

        st.html(
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
"""
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
