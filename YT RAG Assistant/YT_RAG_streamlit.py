from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

parser=StrOutputParser()

emb=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=100,separators=["\n\n","\n",". "," ",""])
st.markdown("""
<style>
.stApp {
    background-color: #0B0F19;
    background-image:
        radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px);
    background-size: 24px 24px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<h1 style="
    font-size: 66px;
    font-weight: 800;
    background: linear-gradient(90deg,#6366F1,#A855F7,#EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
">
YouTube RAG Assistant
</h1>
""", unsafe_allow_html=True)

st.caption("A Retrieval-Augmented Generation (RAG) application that answers questions from YouTube videos using LangChain, FAISS, Hugging Face embeddings, and Gemini.")
url=st.text_input("Enter URL: ")

if url:
    st.video(url)

qs=st.text_input("Enter Your Question: ")


if st.button("Answer"):
    with st.spinner("Generating..."):
        loader=YoutubeLoader.from_youtube_url(url,language=["en-GB", "en", "hi"])

        docs=loader.load()

        docs_splitted=splitter.split_documents(docs)

        vector_store=FAISS.from_documents(embedding=emb,documents=docs_splitted)

        ret=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

        def format(rs):
             context="\n\n".join(doc.page_content for doc in rs)
             return context
        
        template=PromptTemplate(template="""You are a knowledgeable and helpful AI assistant.

Use the retrieved information as your primary source of truth.
If the retrieved information contains the answer, use it accurately and naturally.

If the retrieved information is incomplete but you have reliable general knowledge that helps provide a better answer, you may use it. However, do not contradict the retrieved information.

If neither the retrieved information nor your own knowledge is sufficient to answer confidently, respond with:
"I don't have enough information to answer that."

Do not mention words like "context", "provided information", "retrieved documents", or explain where the information came from. Answer naturally as if you already know it.

Question:
{question}

Retrieved Information:
{context}

Answer:
""",    input_variables=['question', 'context'])

        paral=RunnableParallel(context=ret | RunnableLambda(format),question=RunnablePassthrough())

        seq_chain= template | llm | parser

        main=paral | seq_chain

        rs=main.invoke(qs)
        
        st.markdown(rs)
