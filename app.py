from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.utils import embedding_functions
import openai
import asyncio
import os
import re
import base64
import json
import csv
import uuid
from gtts import gTTS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from azure.storage.blob import BlobServiceClient
import pandas as pd
from openpyxl import load_workbook
import shutil



azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
print("AZURE_STORAGE_CONNECTION_STRING: ",azure_connection_string)
if not azure_connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

AUDIO_CONTAINER = "audio"
CHAT_CONTAINER = "chathistory"

try:
    blob_service_client.create_container(AUDIO_CONTAINER)
except Exception as e:
    print("Audio container likely exists:", e)

try:
    blob_service_client.create_container(CHAT_CONTAINER)
except Exception as e:
    print("Chat history container likely exists:", e)





CHAT_HISTORY_FILE = "chat_history.xlsx"

def save_to_excel(query, response, feedback=None):
    """Append a query-response pair to an Excel file."""
    try:
        new_entry = pd.DataFrame([{"Query": query, "Response": response, "Feedback": feedback}])

        try:
            # Load existing workbook
            book = load_workbook(CHAT_HISTORY_FILE)
            sheet = book.active

            # Convert DataFrame to list of lists and append rows
            for row in new_entry.itertuples(index=False, name=None):
                sheet.append(row)

            # Save and close the workbook
            book.save(CHAT_HISTORY_FILE)
            book.close()

        except FileNotFoundError:
            # If file does not exist, create a new one with headers
            new_entry.to_excel(CHAT_HISTORY_FILE, index=False, sheet_name="ChatHistory")

    except Exception as e:
        print(f"Error saving to Excel: {e}")

# Load environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
openai.api_key = openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")



# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates for frontend rendering
templates = Jinja2Templates(directory="templates")

# Ensure static directory exists
STATIC_DIR = "static"
CHAT_LOG_DIR = "chat_logs"
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# Serve static files (CSS, JS, images, audio)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize ChromaDB with persistence


CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", "./chromadb")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)

# Load and process documents from the same location as the code file

DOCUMENTS_DIR = os.getenv("DOCUMENTS_PATH", os.path.join(os.getcwd(), "documents"))

DOCUMENTS = ["doc1.pdf", "doc2.pdf", "doc3.pdf", "doc4.pdf", "doc5.pdf", "doc6.pdf", "doc7.pdf", "doc8.pdf", "doc9.pdf"]
datasource = []
for doc in DOCUMENTS:
    doc_path = os.path.join(DOCUMENTS_DIR, doc)
    if os.path.exists(doc_path):
        try:
            datasource.extend(PDFMinerLoader(doc_path).load())
        except ValueError:
            print(f"PDFMiner failed to load {doc_path}, using PyMuPDFLoader instead.")
            datasource.extend(PyMuPDFLoader(doc_path).load())

# Split documents and store in ChromaDB
print("Collected all documents")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(datasource)
print("Documents Split")
vectorstore.add_documents(docs)
print("Documents added to Vector Store")


# Initialize Chat Model
chat_model = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.4)
prompt_template=PromptTemplate(
        input_variables=["context","question"],
        template = """You are a document parser/interpreter for the Human Resources Department at On-Target Supplies and Logictics (or OTSL for short).\n"
        All questions you get come from employees of On-Target Supplies and Logictics.
        You are given the following context information.\n
        ---------------------\n
        {context}\\n
        ---------------------\n
        Given the context information and not prior knowledge,
        answer the query {question}. 
        Please be detailed and provide answers in meaningful, clearly interpretable sentences. No answer should be more than 5 sentences.\n
        If required, provide answers in short bullet points.\n""" )

llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)
retriever = vectorstore.as_retriever()

# Session-based storage
session_data = {}

async def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)

# Function to generate audio from text
async def generate_audio(text: str, filename: str):
    tts = gTTS(text=text, lang='en')
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    tts.save(audio_path)
    return f"/static/audio/{filename}.mp3"

@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)
    
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        if not relevant_docs:
            print("No relevant documents found for:", question)
            context = "No relevant documents found. Answering based on general knowledge."
        else:
            context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Stream response while collecting full text
        async def response_generator():
            full_response = ""
            async for chunk in llm_chain.astream({"context": context, "question": question}):
                text_chunk = chunk.get("text", "")
                full_response += text_chunk
                yield json.dumps({"type": "text", "content": text_chunk}) + "\n\n"
                await asyncio.sleep(0)


            # Generate audio after the response is fully collected
            save_to_excel(question, full_response)
            sanitized_filename = await sanitize_filename(question)
            audio_url = await generate_audio(full_response, sanitized_filename)
            sources = [doc.page_content for doc in relevant_docs]
            yield json.dumps({"type": "metadata", "sources": sources, "audio_url": audio_url}) + "\n\n"
        
        return StreamingResponse(response_generator(), media_type="application/x-ndjson")
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(content={"error": f"Server error: {str(e)}"}, status_code=500)


@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    question = data.get("question")
    feedback = data.get("feedback")
    answer = data.get("full_response")

    if not question or not feedback:
        return JSONResponse(content={"error": "Missing required data"}, status_code=400)

    try:
        # Load existing chat history
        df = pd.read_excel(CHAT_HISTORY_FILE, sheet_name="ChatHistory")

        # Correct filtering using parentheses
        index = df[(df["Query"] == question) & (df["Response"] == answer)].index
        if not index.empty:
            df.loc[index, "Feedback"] = feedback  # Update feedback column
            df.to_excel(CHAT_HISTORY_FILE, index=False, sheet_name="ChatHistory")
            return JSONResponse(content={"message": "Feedback saved successfully"})
        else:
            return JSONResponse(content={"error": "Query not found in chat history"}, status_code=404)

    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save feedback: {str(e)}"}, status_code=500)



@app.get("/static/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse(content={"error": "Audio file not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
