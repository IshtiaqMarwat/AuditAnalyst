# -*- coding: utf-8 -*-
"""Audit Assistant with PDF + Excel Support"""

# pip install -U streamlit langchain langchain-openai langchain-community PyPDF2 faiss-cpu sentence-transformers pandas

import streamlit as st
import PyPDF2
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 🔐 API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- 🤖 LLM ---
llm = ChatOpenAI(
    model="gpt-4o",  # new GPT-4 Omni
    temperature=0,
    api_key=openai_api_key
)

# --- 📄 PDF Processing ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# --- 📊 Excel Analysis ---
def analyze_excel_data(df, question):
    data_as_string = df.to_string(index=False)
    prompt = f"""
You are an audit data analyst. Analyze the following Excel data and answer the question.

Data:
{data_as_string}

Question:
{question}
"""
    return llm.invoke(prompt).content

# --- 🧠 Embeddings + QA for PDF ---
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
text_splitter = CharacterTextSplitter(
    separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len
)

def ask_pdf_question(pdf_text, question):
    chunks = text_splitter.split_text(pdf_text)
    db = FAISS.from_texts(chunks, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.invoke(question)["result"]

# --- 🌐 Streamlit UI ---
st.title("🧠 AI Audit Assistant")
st.markdown("Upload an **Audit PDF** or **Excel dataset** and ask questions about it.")

uploaded_file = st.file_uploader("📂 Upload your Audit Document or Dataset", type=["pdf", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF uploaded and processed.")

        user_question = st.text_input("💬 Ask a question about the PDF:")
        if user_question:
            with st.spinner("Analyzing PDF..."):
                try:
                    answer = ask_pdf_question(pdf_text, user_question)
                    st.text_area("📘 Answer", value=answer, height=200)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    elif file_type == "xlsx":
        try:
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file loaded successfully.")
            st.dataframe(df)

            user_question = st.text_input("💬 Ask a question about the Excel data:")
            if user_question:
                with st.spinner("Analyzing Excel data..."):
                    try:
                        answer = analyze_excel_data(df, user_question)
                        st.text_area("📊 Answer", value=answer, height=200)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")
