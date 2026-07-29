from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
qs=input("Enter your query: ")

emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

llm=ChatGoogleGenerativeAI(model='gemini-flash-latest')

parser=StrOutputParser()

loader=PyPDFLoader("D:\AI\VS CODE\Langchain\Projects\Pre Semester Registration Form Final (Odd Sem 26-27 ).pdf")

docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n","\n",". "," ",""]
)

docs_sp=splitter.split_documents(docs)

vs=Chroma.from_documents(
    embedding=emb,
    documents=docs_sp
)

t1=PromptTemplate(template=""" You are an expert AI tutor helping students learn concepts from their study materials.

Your task is to answer the student's question ONLY using the provided context.

Instructions:
- Read the context carefully before answering.
- Explain concepts in simple and easy-to-understand language.
- If appropriate, use examples or analogies to make the concept clearer.
- Organize the answer using headings and bullet points when helpful.
- Do NOT invent facts or use knowledge outside the provided context.
- If the answer is not available in the context, respond exactly with:
  "I couldn't find this information in the provided study material."

Context:
{context}

Student's Question:
{question}

Answer:
""", input_variables=['context', 'question'])

ret=vs.as_retriever(search_type='mmr', search_kwargs={'k':4, 'lambda_mult':0.5})

def merge(doc):
    rs="\n\n".join(d.page_content for d in doc)
    return rs

parallel=RunnableParallel(
    context=ret | RunnableLambda(merge),
    question=RunnablePassthrough()

)

seq=t1 | llm | parser

chain= parallel | seq

rs=chain.invoke(qs)

print(rs)