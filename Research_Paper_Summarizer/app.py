from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    api_key = st.secrets["GOOGLE_API_KEY"]

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header('AI Research Paper Summarizer')

paper_name=st.text_input("Enter Research paper")

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Select",
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Select",
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)
template=PromptTemplate(
    template="""
You are an expert AI researcher and educator.

Explain the research paper:
**{paper_name}**

Requirements:
- Explanation Style: {style_input}
- Explanation Length: {length_input}

Instructions:
1. Give a brief overview of the paper.
2. Explain the core idea in simple terms.
3. Describe the architecture or methodology used.
4. Explain why this paper is important.
5. Mention its real-world applications.
6. Discuss its limitations, if any.
7. End with a concise summary.

The explanation should strictly follow the selected style and length.
If you didnt have knowledge about asked reseach paper, Kindly write "Insufficient Data" but dont start to guessing on your own

""",
input_variables=['paper_name', 'style_input', 'length_input'],
validation=True
)

prompt=template.invoke({
    'paper_name':paper_name,
    'style_input':style_input,
    'length_input':length_input
})

if st.button('Summarize'):
    with st.spinner("Generating summary..."):
        rs=model.invoke(prompt)
        st.markdown(rs.content)

