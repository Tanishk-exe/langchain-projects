from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

#VERSION 1

#This flow is good but it cannot store context of previous messages for that we need to use list  
# while True:
#     user_input=input("You: ")
#     if user_input=='exit':
#         break
#     rs=model.invoke(user_input)
#     print("AI: ", rs.content)


# VERSION 2

# hist=[]
# This also has a problem that the chats are saved in hist list will be saved as genric messages so our Model will face difficulty to detect which message is from user and AI and can start to create problems.
# while True:
#     user_input=input("You: ")
#     hist.append(user_input)
#     if user_input=='exit':
#         break
#     rs=model.invoke(hist)
#     hist.append(rs.content)
#     print("AI: ", rs.content)

#OUTPUT
# (venv) PS D:\AI\VS CODE\Langchain> python Prompts\03_chatbot.py
# You: howdy
# AI:  Well howdy! How can I help you today?
# You: which is greater no. 2 or 5
# AI:  Five is greater than two.
# You: now multipy greater no. with 67
# AI:  Okay, the greater number was 5.
# 5 multiplied by 67 is:
# 5 * 67 = **335**
# You: thanks
# AI:  You're welcome! Glad I could help.
# You: exit

#VERSION 3
conv=[
    SystemMessage(content="You are a helpful Assistent")

]
while True:
    user_input=input("You: ")
    conv.append(HumanMessage(content=user_input))
    if user_input=='exit':
        break
    rs=model.invoke(conv)
    conv.append(AIMessage(content=rs.content))
    print("AI: ", rs.content)

print(conv)
