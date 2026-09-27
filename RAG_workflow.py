from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chromaDB_1"

embedding_model = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

def build_vector_store(file_path : str):
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=45
    )
    chunks = splitter.split_documents(docs)

    vectorStore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vectorStore

def load_vectorStore():
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )

vectorStore = load_vectorStore()
retriever = vectorStore.as_retriever( 
    search_type = "similarity",
    search_kwargs = {"k":5}
)

if __name__ == "__main__":
    build_vector_store(file_path="file path of book")


# D:/Learning_AI/ProjectsFile/AgenticChatbot_RAG_Tools/Agentic_chatbot/books/DeepLearningBook.pdf