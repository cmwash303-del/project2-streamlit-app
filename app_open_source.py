# ===== IMPORT ALL THE TOOLS WE NEED =====
import streamlit as st           # For building the web app
import io                        # For handling file bytes
from pathlib import Path         # For working with file names and extensions
from pypdf import PdfReader      # For reading PDF files
import docx                      # For reading Word .docx files
from bs4 import BeautifulSoup    # For reading HTML files
from transformers import pipeline  # For using an open-source AI model
import re


# ===== SET UP THE OPEN-SOURCE AI MODEL =====
# This creates a "question-answering" model that can read text and answer questions about it.
qa = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)


def get_answer(question, context):
    """
    Uses the open-source model to answer a question based on given context text.
    """
    # Make sure the context is not too long (the model can't handle super long text)
    context = context[:4000]
    result = qa(question=question, context=context)
    return result["answer"]


# ===== FUNCTION TO PULL TEXT OUT OF UPLOADED FILES =====
def extract_text_from_file(uploaded_file):
    """
    Takes a file the user uploaded and returns all the text inside it.
    Works for: .txt, .pdf, .docx, .html / .htm
    """
    # Read the raw bytes from the file
    file_bytes = uploaded_file.read()
    # Get the file extension, like ".pdf" or ".docx"
    suffix = Path(uploaded_file.name).suffix.lower()

    # ----- PLAIN TEXT FILE (.txt) -----
    if suffix == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")

    # ----- PDF FILE (.pdf) -----
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    # ----- WORD FILE (.docx) -----
    if suffix == ".docx":
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)

    # ----- HTML FILE (.html / .htm) -----
    if suffix in [".html", ".htm"]:
        soup = BeautifulSoup(file_bytes, "html.parser")
        # Remove script and style tags (they are not useful text)
        for tag in soup(["script", "style"]):
            tag.extract()
        return soup.get_text(separator="\n")

    # If the type is not recognized, return empty text
    return ""
def extract_abbreviations(text):
    """
    Looks for patterns like 'full term (ABBR)' where ABBR is uppercase.
    Returns a dictionary like: {"WDC": "weighted degree centrality"}
    """
    pattern = r"\b([A-Za-z][A-Za-z\s\-]{2,})\s*\(([A-Z]{2,10})\)"
    matches = re.findall(pattern, text)

    abbr_dict = {}
    for full_term, abbr in matches:
        full_term_clean = " ".join(full_term.split())
        abbr = abbr.strip()
        # Only keep the first meaning we see for each abbreviation
        if abbr not in abbr_dict:
            abbr_dict[abbr] = full_term_clean

    return abbr_dict


# ===== STREAMLIT APP LAYOUT (WHAT THE USER SEES) =====
# ===== STREAMLIT APP LAYOUT (WHAT THE USER SEES) =====

# Big title at the top
st.title("Project 2: Document Question Answering App")

# Smaller subtitle
st.subheader("Programming for Data Science - Streamlit + Open-Source LLM")

# Short instructions for the user
st.write("""
This app lets you:
1. Type a question and ask about documents, **or**
2. Build an abbreviation index from articles.

Use the radio buttons below to choose what you want to do.
""")

# Radio buttons to choose between Q1 and Q2 features
mode = st.radio(
    "What do you want to do?",
    ["Ask questions about documents", "Build abbreviation index"]
)

# ===== MODE 1: QUESTION ANSWERING (Q1) =====
if mode == "Ask questions about documents":
    st.markdown("### Ask a question about your documents")

    # Text box where the user types their question
    user_question = st.text_input("Enter your question:")

    # File uploader where the user can upload multiple documents
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, Word, Text, HTML)",
        type=["pdf", "docx", "txt", "html", "htm"],
        accept_multiple_files=True,
        key="qa_uploader"  # key so it doesn't conflict with the other uploader
    )

    # Button that the user clicks to run the app logic
    if st.button("Get Answer"):
        # First, check if the user typed a question
        if not user_question:
            st.warning("Please type a question first.")
        # Then, check if the user uploaded any files
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            # If both are provided, process the documents
            st.success("Great! Reading your documents now...")

            # Put all the text from all files into one big string
            all_text = ""
            for file in uploaded_files:
                text = extract_text_from_file(file)
                all_text += f"\n\n===== {file.name} =====\n\n"
                all_text += text

            # If no text was read, show an error
            if not all_text.strip():
                st.error("I could not read any text from the uploaded files.")
            else:
                # Show a small preview of the text for you (first 500 characters)
                st.write("### Preview of the document text:")
                st.write(all_text[:500] + "...\n\n(Only showing the beginning.)")

                # Ask the AI model to answer the question using the text
                st.write("### Thinking about your question...")
                answer = get_answer(user_question, all_text)

                # Show the final answer
                st.write("## Answer:")
                st.write(answer)

# ===== MODE 2: ABBREVIATION INDEX (Q2) =====
else:
    st.markdown("### Build an abbreviation index from articles")
    st.write("""
Upload one or more articles. 
I will look for patterns like **'weighted degree centrality (WDC)'**
and create an index like:

`WDC: weighted degree centrality`
    """)

    uploaded_files_abbr = st.file_uploader(
        "Upload article files (PDF, Word, Text, HTML)",
        type=["pdf", "docx", "txt", "html", "htm"],
        accept_multiple_files=True,
        key="abbr_uploader"
    )

    if st.button("Generate Abbreviation Index"):
        if not uploaded_files_abbr:
            st.warning("Please upload at least one article.")
        else:
            # For each uploaded article, make an index
            for file in uploaded_files_abbr:
                st.markdown(f"#### Abbreviation index for **{file.name}**")

                text = extract_text_from_file(file)
                abbr_dict = extract_abbreviations(text)

                if not abbr_dict:
                    st.write("No abbreviations found in this article.")
                else:
                    # Show each abbreviation in alphabetical order
                    for abbr in sorted(abbr_dict.keys()):
                        full_term = abbr_dict[abbr]
                        st.write(f"**{abbr}**: {full_term}")
