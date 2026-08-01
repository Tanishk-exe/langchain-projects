from langchain_google_community import GooglePlacesTool
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import tool
import requests
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')

gmap_tool=GooglePlacesTool()

@tool
def get_weather(city:str) -> dict:
    """Takes city name and returns the weather of that city"""
    api="9b8c19999e9137115a328d0e56a84562"
    url=f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api}&units=metric'
    res=requests.get(url)
    return res.json()


@tool
def get_currency_rate(base:str, target:str) -> float:
    """Gets the base currency and target currency and returns the rate factor"""
    url=f'https://v6.exchangerate-api.com/v6/f34ec0c8cd3c728559c8f215/pair/{base}/{target}'
    res=requests.get(url)
    return res.json()

prompt="""Answer the following questions as best you can. You have access to the following tools:

tools

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [tool_names]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: input
Thought:agent_scratchpad
"""

agent=create_agent(
    model=model,
    tools=[get_currency_rate,get_weather,gmap_tool],
    system_prompt=prompt
)

rs=agent.invoke({"messages":[{"role":"user","content":"I am planning to visit Japan this year for my vacation, so what's the weather in tokyo and osaka, and also suggest me some plaxes to visit in tokyo ans osaka and also any other places you can suggest."}]})

print(rs,"\n")
print(rs['messages'][-1].content[0]['text'])

