# -*- coding: utf-8 -*-
"""Excel Audit Analyst"""

# pip install -U streamlit pandas openai langchain langchain-openai

import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI

# --- 🔐 API Key ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OpenAI API key not found. Please set it in Streamlit secrets.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# --- 🤖 LLM ---
llm = ChatOpenAI(
    model="gpt-4o",  # or gpt-3.5-turbo
    temperature=0,
    api_key=openai_api_key
)

# --- 📊 Excel Data Analysis ---
def analyze_excel_data(df, question):
    data_str = df.head(10).to_string(index=False)  # limit to 100 rows
    prompt = f"""
You are a smart internal auditor analyzing tabular Excel data.

Here is a sample of the uploaded data:

{data_str}

Question:
{question}

Answer using only the data provided, focusing on insights and trends.
"""
    return llm.invoke(prompt).content

# --- 🌐 Streamlit UI ---
st.title("📊 AI Excel Audit Analyst")
st.markdown("Upload an Excel file and ask questions about the data inside.")

uploaded_file = st.file_uploader("📂 Upload Excel File", type="xlsx")

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("✅ Excel file loaded successfully.")
        st.dataframe(df)

        question = st.text_input("💬 Ask a question about the Excel data:")
        if question:
            with st.spinner("Analyzing Excel data..."):
                try:
                    answer = analyze_excel_data(df, question)
                    st.text_area("🧠 Answer", value=answer, height=200)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    except Exception as e:
        st.error(f"❌ Failed to read Excel file: {e}")
