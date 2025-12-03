# build_index.py
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader, PyMuPDFLoader

from typing import List
from langchain.schema import Document

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_PATH", os.path.join(os.getcwd(), "documents"))
CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", "/chromadb")  # <- important

DOCUMENTS = [
    "doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf", "doc5.pdf",
    "doc6.pdf", "doc7.pdf", "doc8.pdf", "doc9.pdf", "doc10.pdf",
    "doc11.pdf", "doc12.pdf", "doc13.pdf", "doc14.pdf", "doc15.pdf",
    "doc16.pdf", "doc17.pdf", "doc18.pdf", "doc19.pdf", "doc20.pdf",
    "doc21.pdf", "doc22.pdf", "doc23.pdf", "doc24.pdf", "doc25.pdf",
]

datasource: List[Document] = []
for doc in DOCUMENTS:
    doc_path = os.path.join(DOCUMENTS_DIR, doc)
    if os.path.exists(doc_path):
        try:
            datasource.extend(PDFMinerLoader(doc_path).load())
        except ValueError:
            print(f"PDFMiner failed to load {doc_path}, using PyMuPDFLoader instead.")
            datasource.extend(PyMuPDFLoader(doc_path).load())
    else:
        print(f"WARNING: {doc_path} not found")

print(f"Collected {len(datasource)} documents/pages")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(datasource)
print(f"Split into {len(docs)} chunks")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH,
)

vectorstore.persist()
print(f"Index built and persisted to {CHROMA_DB_PATH}")
