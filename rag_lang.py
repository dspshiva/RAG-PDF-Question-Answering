from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.runnables import RunnableParallel,RunnablePassthrough , RunnableLambda
from langchain_core.output_parsers import StrOutputParser 
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

#Load environment variables (e.g., for API keys)
load_dotenv()

#Streamlit page configuration
st.set_page_config(page_title = "📄 Document Q&A Chatbot" , layout="centered")
st.title("📄 Document Q&A Chatbot")
st.markdown("Upload any PDF document and ask questions to extract relevant information using Google Gemini + LangChain.F")


#Output parser to convert LLM output to string
parser = StrOutputParser()

#File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF document (PDF only)",type=["pdf"])

#Caching heavy resource: PDF -> Embeddings -> FAISS store
@st.cache_resource
def get_store(pdf_bytes):
    with open("temp_res.pdf","wb") as f:
        f.write(pdf_bytes)

    # Load PDF using LangChain's PyPDFLoader    
    loader = PyPDFLoader("temp_res.pdf")
    doc = loader.load()
    n_pages_doc = len(doc)# Total pages in PDF

    # Extract text content from each page
    document_content = [i.page_content for i in doc]

    #Dynamically adjust chunk size based on number of pages (for better performance)
    min_chunks = 800
    max_chunks = 3500
    chunk_size = min_chunks+int((max_chunks-min_chunks)*min(n_pages_doc,100)/100)
    chunk_overlap = int(chunk_size*0.1)

    #Split the document into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size , chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents(document_content)

    embedding = GoogleGenerativeAIEmbeddings(model = 'models/gemini-embedding-001')

    #Create a FAISS vector store from the document chunks
    store = FAISS.from_documents(documents = chunks, embedding = embedding)
    return store
if pdf_file:
    # Read the PDF content as raw bytes
    file_bytes = pdf_file.getvalue()
    with st.spinner("🔄 Processing Please wait..."):
        store = get_store(file_bytes)
    
    #Prompt template for question answering
    Template = PromptTemplate(
        template="""
        You are an assistant helping answer questions based on the provided document content.\n\n
            Context:\n{text}\n\n
            Question: {query}\n\n
            Answer concisely based only on the document. If the answer is not found in the document, say:\n
            I couldn't find the requested information in the document.
        """

    )
    #Create a retriever to fetch the top relevant chunks from the FAISS store
    retriever = store.as_retriever(search_type='similarity', search_kwargs={'k':4})

    #Define a parallel runnable to prepare context + query
    parallel = RunnableParallel({
        'text' : retriever | (lambda X : "\n\n".join(i.page_content for i in X)),
        'query': lambda X:X
    })
    #Final RAG pipeline: Retrieve → Prompt → LLM → Parse Output
    model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')
    chain = parallel | Template | model | parser

    input = st.text_input("Ask a question about the document:")
    
    if input:
        with st.spinner("Analyzing document and generating response..."):
            response = chain.invoke(input)
            st.success("Answer:")
            st.write(response)