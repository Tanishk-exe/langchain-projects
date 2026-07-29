from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

emb=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
llm=ChatGoogleGenerativeAI(model="gemini-flash-latest")

loader=DirectoryLoader(
    path='D:\AI\VS CODE\Langchain\Projects\Medical books for RAG model',
    glob="*.pdf",
    loader_cls=PyMuPDFLoader
)

docs=loader.lazy_load()

splitter=RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500, separators=["\n\n","\n",". "," ",""])

splitted_docs=splitter.split_documents(docs)

vector=FAISS.from_documents(documents=splitted_docs, embedding=emb)

ret=vector.as_retriever(search_kwargs={"k":2})

compressor=LLMChainExtractor.from_llm(llm=llm)

ccr=ContextualCompressionRetriever(base_retriever=ret, base_compressor=compressor)

question=input("Enter your question: ")

parser=StrOutputParser()

temp=PromptTemplate(template="""You are MedRAG, an AI assistant that helps healthcare professionals and researchers understand medical literature.

Your responsibility is to answer questions using ONLY the retrieved medical documents.

Instructions:

- Base every answer strictly on the provided context.
- Never generate information that is not supported by the documents.
- Maintain an objective, scientific tone.
- If multiple studies are retrieved, synthesize the evidence while preserving important differences.
- Mention uncertainties or limitations when they exist.
- Do not provide medical advice, diagnoses, or treatment recommendations beyond what is explicitly stated in the literature.
- If the answer cannot be found, reply:
  "The retrieved medical literature does not contain sufficient information to answer this question."

Response Format:

## Summary

## Key Findings

## Supporting Evidence

## Limitations

## Source
(Document name and page if available)

Question:
{question}
Retrieved Context:
{context}
Response:""")

paral=RunnableParallel(question=RunnablePassthrough(), 
                       context=ccr )

seq=temp | llm | parser

chain=paral | seq

rs=chain.invoke(question)

print(rs)










