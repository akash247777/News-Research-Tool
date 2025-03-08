# Import necessary libraries for environment variables, UI, data processing, and ML models
import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain  # For QA with source tracking
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain.document_loaders import UnstructuredURLLoader  # Load data from URLs
from langchain.vectorstores import FAISS  # Vector store for similarity search
from transformers import pipeline  # Hugging Face pipeline for model inference
from langchain.embeddings import HuggingFaceEmbeddings  # Embedding generation
from langchain.llms import HuggingFacePipeline  # Wrapper for HF pipelines

# Set Hugging Face API token for authentication
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Configure Streamlit UI with a title
st.title("AkashBot: News Research Tool 📈")
st.sidebar.title("News Article URLs 🌐")  # Sidebar for URL input

# Collect up to 3 URLs from the user via the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")  # Input field for each URL
    urls.append(url)

# Button to trigger processing of URLs
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "3_News_Research/vectorindex_huggingface.pkl"  # File to save vector store

# Placeholder for dynamic status messages in the main UI
main_placeholder = st.empty()

# Initialize Hugging Face text generation pipeline with FLAN-T5 model
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",  # Pre-trained model for QA
)
# Wrap pipeline in LangChain's HuggingFacePipeline for integration
llm = HuggingFacePipeline(
    pipeline=hf_pipeline,
    pipeline_kwargs={
        "max_length": 500,  # Maximum output length
        "do_sample": True,  # Enable sampling for diverse outputs
        "temperature": 0.9,  # Control randomness of predictions
    },
)

# Process URLs when the button is clicked
if process_url_clicked:
    # Load data from provided URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()  # Fetch and parse URL content
    
    # Split text into manageable chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],  # Split on paragraphs, lines, sentences
        chunk_size=1000,  # Target chunk size
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)  # Create document chunks
    
    # Generate embeddings and build FAISS vector store for similarity search
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Efficient embedding model
    )
    vectorstore = FAISS.from_documents(docs, embeddings)  # Create vector index
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)  # Simulate processing delay
    
    # Save vector store to disk for later reuse
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)  # Serialize the vector store

# Handle user queries
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load precomputed vector store
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Create QA chain with source tracking
            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),  # Use FAISS as retriever
            )
            result = chain(
                {"question": query},
                return_only_outputs=True,  # Only return answer and sources
            )
            
            # Display results in UI
            st.header("Answer")
            st.write(result["answer"])  # Show generated answer
            
            # List sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split multiple sources
                for source in sources_list:
                    st.write(source)  # Display each source URL
