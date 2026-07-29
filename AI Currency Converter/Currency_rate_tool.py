from langchain.tools import tool
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.messages import HumanMessage
import json
import streamlit as st

load_dotenv()

@tool
def get_currency_amount(amount:int,base:str, target:str) -> float:
    """Gets the base currency and target currency and returns the rate factor"""
    url=f'https://v6.exchangerate-api.com/v6/f34ec0c8cd3c728559c8f215/pair/{base}/{target}'
    res=requests.get(url)
    rate=res.json()['conversion_rate']
    total=amount*rate
    return total

# print(get_currency_factor.invoke({'base':'USD','target':'INR'}))
# print(get_currency_rate.invoke({'base_value':150,'rate':95.32}))

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

model2=model.bind_tools([get_currency_amount])

st.markdown("""
<h1 style="
    font-size: 50px;
    font-weight: 800;
    background: linear-gradient(90deg,#6366F1,#A855F7,#EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
AI Currency Converter
</h1>
""", unsafe_allow_html=True)
st.caption("An AI-powered Currency Converter Agent built with LangChain that uses custom tools to fetch live exchange rates and perform currency conversions through LLM tool calling.")

qs=st.text_input("Enter your Prompt",placeholder="How much is 50 EUR in JPY?")

if st.button("Convert"):
    with st.spinner("Exchanging Rates..."):
        human_msg=HumanMessage(qs)
        msg=[human_msg]
        ai_msg=model2.invoke(qs)
        print(ai_msg.tool_calls[0])
        msg.append(ai_msg)
        tool_msg=get_currency_amount.invoke(ai_msg.tool_calls[0])
        msg.append(tool_msg)
        rs=model.invoke(msg)
        st.markdown(rs.text)

