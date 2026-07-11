from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
import streamlit as st

st.title("AI Question Difficulty Router")
st.caption("An AI-powered question router built with LangChain that classifies questions into Basic, Intermediate, or Advanced levels and dynamically routes them to specialized response chains using conditional branching.")

load_dotenv()

llm=HuggingFaceEndpoint(
    model="openai/gpt-oss-20b:groq"
)

model=ChatHuggingFace(llm=llm)

parser1=StrOutputParser()

class schema(BaseModel):
    difficulty:Literal["Basic","Intermediate","Advanced"]=Field(description="return the difficulty of the given question")
    question:str=Field(description="return the given question as input ny the user")

parser2=PydanticOutputParser(pydantic_object=schema)

t1=PromptTemplate(
    template="""You are an AI question difficulty classifier.

Your task is to classify the given question into ONLY one of the following levels:

- Basic
- Intermediate
- Advanced

Question:
{ques}

Return ONLY the JSON object.
, {format_instruction}""",
    input_variables=["ques"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

class_chain=t1 | model | parser2

t2=PromptTemplate(
    template="generate simple answer for the basic level question: {ques} also tell the question's level of difficulty in starting line",
    input_variables=["ques"]
)

t3=PromptTemplate(
    template="generate detailed answer for the intermediate level question: {ques} also tell the question's level of difficulty in starting line",
    input_variables=["ques"]
)

t4=PromptTemplate(
    template="generate code/mathematical based answer with content of around 1000 words for the advanced level question: {ques}, also tell the question's level of difficulty in starting line",
    input_variables=["ques"]
)

br_chain=RunnableBranch(
    (lambda x:x.difficulty == "Basic", t2 | model | parser1),
    (lambda x:x.difficulty == "Intermediate", t3 | model | parser1),
    (lambda x:x.difficulty == "Advanced", t4 | model | parser1),
    RunnableLambda(lambda x: "Cannot identify difficulty")
)

chain= class_chain | br_chain

qs=st.text_input("Enter your Question: ")

if st.button("Generate"):
    with st.spinner("Generating..."):
        rs=chain.invoke({"ques": qs})
        st.markdown(rs)


