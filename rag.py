import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
# Load the document
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("data\Test.pdf")
pages = loader.load()

# Splitting the document 

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=150,
    length_function=len
)
chunks = text_splitter.split_documents(pages)

print(f"Number of chunks: {len(chunks)} and document{len(pages)}")

# Putting embeddings and adding those into database
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory
)
# making a streamlit app
import streamlit as st
# having history
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Question input
question = st.chat_input("Say something")
st.session_state.messages.append({"role": "user", "content": question})

if question:
    # retrival 
    st.chat_message("user").write(question)
    docs = vectordb.similarity_search_with_relevance_scores(question,k=3)
    print(docs[0])
    vectordb.persist()


    template = """Be polite and conversational with the user and if it ask a question then Answer the question based only on the following context: \n\n{context}\n\n Question: {question} \n\n"""

    # Giving the chunk back to gpt-3.5
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA

    from langchain.chat_models import ChatOpenAI


    llm_name = "gpt-3.5-turbo"


    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )


    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        
    )

    result = qa_chain({"query": question})
    # prining the result
    st.chat_message("assistant").write(result["result"])
    st.chat_message("assistant").write(result["source_documents"][0])
    
    st.session_state.messages.append({"role": "assistant", "content": result["result"]})
    