from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# loading PDF
loader=PyMuPDFLoader("D:\AI\VS CODE\Langchain\Projects\HR-Policy.pdf")
docs=loader.load()

#Chunking
splitter=RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=50,
    separators=["\n\n","\n",". "," ",""]
)
split_docs=splitter.split_documents(docs)

emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm=ChatGoogleGenerativeAI(model="gemini-flash-latest")

#Vector store
vs=Chroma.from_documents(
    embedding=emb,
    documents=split_docs
)

#Retriever
ret=vs.as_retriever(search_kwargs={"k":5})
mul=MultiQueryRetriever.from_llm(llm=llm, retriever=ret)

#parser
parser=StrOutputParser()

#Prompt
temp=PromptTemplate(template="""You are an Enterprise HR Policy Assistant.

Your goal is to help employees understand company policies accurately.

Rules:
- Answer ONLY from the provided policy documents.
- Never fabricate information.
- If the answer is partially available, clearly indicate which parts are supported by the documents.
- If the answer cannot be found, respond:

"I couldn't find this information in the available company policy documents. Please contact HR for clarification."

Formatting:
- Start with a short answer.
- Then explain the details.
- Use bullet points for eligibility, conditions, or procedures.
- End with a "Policy Source" section if metadata is available.
- Policy Source section format:Document Name, Page Number (if available)
Question:
{question}
Retrieved Context:
{context}
Answer: """, input_variables=["question", "context"])

question=input("Enter your question: ")


paral=RunnableParallel(question= RunnablePassthrough(),
                       context=ret )

seq=temp | llm | parser

chain=paral | seq

rs=chain.invoke(question)

print(rs)

