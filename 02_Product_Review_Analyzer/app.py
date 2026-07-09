from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Optional,Literal
import streamlit as st
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    api_key = st.secrets["GOOGLE_API_KEY"]

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.title("AI Review Analyzer")
st.caption(
    "Paste any product review and let AI extract key points, "
    "sentiment, rating, pros, and cons using structured output.")

prompt=st.text_area("Enter your Review:")

class review(BaseModel):
    summary:str=Field(description="return a one line summary of the review")
    rating:int= Field(gt=0, lt=6, description="Give rating of the review")
    sentiment: Literal['POSITIVE', 'NEGATIVE', "MIXED"]
    pros:Optional[list[str]]=Field(description="It is optinal if pros written explicitly in the review then only return it")
    cons:Optional[list[str]]=Field(description="It is optinal if cons written explicitly in the review then only return it")



str_model=model.with_structured_output(review)

if st.button("Analyze"):
    with st.spinner("Analyzing Review..."):
        rs=str_model.invoke(prompt)
        st.subheader("Summary:")
        st.write(rs.summary)
        st.write("Rating: ",rs.rating)
        st.subheader("Pros:")
        for pro in rs.pros:
            st.write(f"- {pro}")
        st.subheader("Cons:")
        for con in rs.cons:
            st.write(f"- {con}")
