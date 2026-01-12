from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.utils import embedding_functions
import openai
import asyncio
import os
import re, html
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
import datetime
import uuid
from typing import List
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from azure.data.tables import TableServiceClient, UpdateMode
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

session_id = str(uuid.uuid4())

azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# print("AZURE_STORAGE_CONNECTION_STRING: ",azure_connection_string)
if not azure_connection_string:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

#try:
#    blob_service_client.create_container(AUDIO_CONTAINER)
#except Exception as e:
#    print("Audio container likely exists:", e)

#try:
#    blob_service_client.create_container(CHAT_CONTAINER)
#except Exception as e:
#    print("Chat history container likely exists:", e)



# CHAT_LOG_DIR = os.getenv("CHATHISTORY_PATH", os.path.join(os.getcwd(), "chathistory"))
# CHAT_HISTORY_FILE = os.path.join(CHAT_LOG_DIR, f"{session_id}_chat_history.xlsx")

# 1. FIRST create the service client
service_client = TableServiceClient.from_connection_string(
    conn_str=azure_connection_string 
)

# 2. THEN create chat history table
CHAT_TABLE_NAME = os.getenv("CHAT_TABLE_NAME", "chathistory")
table_client = service_client.get_table_client(table_name=CHAT_TABLE_NAME)
try:
    table_client.create_table()
except Exception:
    pass

# 3. THEN create survey table (after service_client exists)
SURVEY_TABLE_NAME = os.getenv("SURVEY_TABLE_NAME", "hrchatbotsurvey")

survey_table_client = service_client.get_table_client(
    table_name=SURVEY_TABLE_NAME
)
try:
    survey_table_client.create_table()
except Exception:
    pass


_RE_FENCE      = re.compile(r"```.*?```", re.S)
_RE_INLINECODE = re.compile(r"`([^`]*)`")
_RE_IMG        = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_RE_LINK       = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RE_BI         = re.compile(r"(\*\*|__|\*|_)([^*_].*?)\1")  # bold/italic markers
_RE_STRIKE     = re.compile(r"~~(.*?)~~")
_RE_HEADERS    = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_RE_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.M)
_RE_UL         = re.compile(r"^\s*([*+\-])\s+", re.M)
_RE_OL         = re.compile(r"^\s*\d+[\.)]\s+", re.M)
_RE_TABLE_RULE = re.compile(r"^\s*\|?\s*[:\-| ]+\s*\|?\s*$", re.M)  # ---|:--- lines
_RE_HTML       = re.compile(r"<[^>]+>")
_RE_TIME_COLON = re.compile(r'(?<=\d):(?=\d)')
_RE_URL = re.compile(r'\b(?:https?|ftp)://\S+')


def _make_row_key() -> str:
    now = datetime.datetime.utcnow()
    return f"{now.strftime('%Y%m%d%H%M%S%f')}_{uuid.uuid4().hex}"


def log_survey_to_table(session_id: str, ratings: dict) -> str:
    row_key = _make_row_key()
    entity = {
        "PartitionKey": session_id,
        "RowKey": row_key,
        "SubmittedAtUTC": datetime.utcnow().isoformat(),

        # store each question rating
        "Q1": int(ratings["q1"]),
        "Q2": int(ratings["q2"]),
        "Q3": int(ratings["q3"]),
        "Q4": int(ratings["q4"]),
        "Q5": int(ratings["q5"]),
    }
    survey_table_client.create_entity(entity=entity)
    return row_key


def log_chat_to_table(session_id: str, question: str, response: str) -> str:
    row_key = _make_row_key()
    now = datetime.datetime.utcnow().isoformat() + "Z"

    entity = {
        "PartitionKey": session_id,     # groups all rows from this session
        "RowKey": row_key,              # unique id per interaction
        "CreatedUtc": now,
        "Question": question,
        "Response": response,
        "Feedback": "",                 # updated later
    }

    # NOTE: Table entity max size is 1 MB. If you expect long responses,
    # store response in Blob and put only a pointer here.
    table_client.create_entity(entity=entity)
    return row_key


def update_feedback_in_table(session_id: str, row_key: str, feedback: str):
    patch = {
        "PartitionKey": session_id,
        "RowKey": row_key,
        "Feedback": feedback,
        "FeedbackUtc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    table_client.update_entity(mode=UpdateMode.MERGE, entity=patch)



def markdown_to_speech_text(md: str, normalize_colons: bool = True, colon_replacement: str = " — ") -> str:
    """Strip Markdown/HTML so TTS won't spell out formatting, and (optionally) turn non-time colons into a pause."""
    if not md:
        return ""

    t = md

    # Remove fenced code blocks entirely
    t = _RE_FENCE.sub("", t)
    # Inline code: keep content
    t = _RE_INLINECODE.sub(r"\1", t)
    # Images: keep alt text
    t = _RE_IMG.sub(lambda m: (m.group(1) or ""), t)
    # Links: keep link text only
    t = _RE_LINK.sub(r"\1", t)
    # Also strip any raw URLs that appear in plain text
    t = _RE_URL.sub("", t)

    # Styling markers
    t = _RE_BI.sub(r"\2", t)
    t = _RE_STRIKE.sub(r"\1", t)

    # Block-level prefixes
    t = _RE_HEADERS.sub("", t)
    t = _RE_BLOCKQUOTE.sub("", t)

    # Lists → readable bullets
    t = _RE_UL.sub("• ", t)
    t = _RE_OL.sub("", t)

    # Tables → readable separators
    t = _RE_TABLE_RULE.sub("", t)
    t = re.sub(r"^\s*\|\s*|\s*\|\s*$", "", t, flags=re.M)
    t = re.sub(r"\s*\|\s*", " — ", t)

    # Strip HTML and unescape entities
    t = _RE_HTML.sub("", t)
    t = html.unescape(t)

    # >>> NEW: normalize non-time, non-URL colons for better TTS rhythm <<<
    if normalize_colons:
        placeholder = "\uFFFF"  # mask time colons temporarily
        masked = _RE_TIME_COLON.sub(placeholder, t)
        # Replace remaining single colons (avoid "::")
        masked = re.sub(r'(?<!:):(?!:)', colon_replacement, masked)
        t = masked.replace(placeholder, ":")

    # Whitespace cleanup
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


'''
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

'''

# Load environment variables
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY environment variable is not set.")
# openai.api_key = openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates for frontend rendering
templates = Jinja2Templates(directory=".")

# Ensure static directory exists

#CHAT_LOG_DIR = "chat_logs"
#AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
#os.makedirs(AUDIO_DIR, exist_ok=True)
#os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# Initialize ChromaDB with persistence

DOCUMENTS_DIR = os.getenv("DOCUMENTS_PATH", os.path.join(os.getcwd(), "documents"))

AUDIO_DIR = os.getenv("AUDIO_PATH", os.path.join(os.getcwd(), "audio"))
os.makedirs(AUDIO_DIR, exist_ok=True)


CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", "./chromadb")
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initialize Chat Model

chat_model = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.2)

prompt_template=PromptTemplate(
        input_variables=["context","question"],
        template = """You are a document parser/interpreter for the Human Resources Department at On-Target Supplies & Logistics (or OTSL for short).\n
        All questions you get come from employees of On-Target Supplies & Logistics.
        You are given the following context information.\n
        ---------------------\n
        {context}\\n
        ---------------------\n
        Given the context information and no prior knowledge, answer the query {question}.\n 
        Provide detailed responses in clear, meaningful sentences that are easy to interpret.\n
        In the response, include which team, company, or department to reach out to regarding this information.
        Don't provide any person's name or contact information (phone number or address) as part of the response.\n
        For example, if any answer/documentation includes the name "Lorene Smith", don't include it in the response.\n
        If you refer them to a website, try to provide the link to the website as well.\n
        If an employee account needs to be created and could be necessary and relevant to the query, provide instructions for the same.\n
        Provide a follow-up question at the end of the response, relevant to the current query, that the user could potentially ask.\n""" )

llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)

def crossencoder_rerank_docs(
    question: str,
    docs: List[Document],
    top_n: int = 6,
    max_chunk_chars: int = 2000,
) -> List[Document]:
    """
    Use a cross-encoder model to score [question, chunk] pairs and
    keep the top_n highest scoring chunks.
    Deterministic and independent of the LLM used for answering.
    """
    if not docs:
        return []

    # Prepare [question, chunk] pairs
    pairs = [(question, d.page_content[:max_chunk_chars]) for d in docs]

    # Predict relevance scores (higher = more relevant)
    scores = cross_encoder.predict(pairs)

    # Zip scores with docs, sort by score descending
    scored_docs = list(zip(scores, docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Take top_n docs
    return [doc for score, doc in scored_docs[:top_n]]


# retriever = vectorstore.as_retriever()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 12,          # how many chunks you ultimately want to consider
        "fetch_k": 40,    # how many to pull before MMR pruning (if supported)
        "lambda_mult": 0.7,  # 0.0 = more diversity, 1.0 = more similarity
    },
)



# Session-based storage
session_data = {}

@app.get("/logo")
async def get_logo():
    # Adjust the file name as needed.
    logo_path = os.path.join(DOCUMENTS_DIR, "Logo.png")
    if not os.path.exists(logo_path):
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(logo_path, media_type="image/png")

async def sanitize_filename(filename):
    sanitized_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', filename)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{session_id}_{timestamp}_{sanitized_filename}"

# Function to generate audio from text
async def generate_audio(text: str, filename: str):
    clean = markdown_to_speech_text(text)
    tts = gTTS(text=clean, lang='en')
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    tts.save(audio_path)
    return f"/audio/{filename}.mp3"

@app.get("/")
async def serve_frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    client_session_id = data.get("session_id") or session_id
    
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)
    
    try:
        # Retrieve relevant documents
        # relevant_docs = retriever.get_relevant_documents(question)
        # if not relevant_docs:
        #    print("No relevant documents found for:", question)
        #    context = "No relevant documents found. Answering based on general knowledge."
        # else:
        #    context = "\n".join([doc.page_content for doc in relevant_docs])

        candidate_docs = retriever.get_relevant_documents(question)
        relevant_docs = crossencoder_rerank_docs(
            question=question,
            docs=candidate_docs,
            top_n=6,            # tune based on how much context you can afford
            max_chunk_chars=2000,
        )
        if not relevant_docs:
            print("No relevant documents found for:", question)
            context = "No relevant documents found. Answering based on general knowledge."
        else:
            context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Stream response while collecting full text
        async def response_generator():
            try:
                full_response = ""
                async for chunk in llm_chain.astream({"context": context, "question": question}):
                    text_chunk = chunk.get("text", "\n")
                    full_response += text_chunk
                    yield json.dumps({"type": "text", "content": text_chunk}) + "\n\n"
                    await asyncio.sleep(0)
                
                # Process audio generation and metadata
                '''
                log_id = log_chat_to_table(session_id, question, full_response)
                sanitized_filename = await sanitize_filename(question)
                audio_url = await generate_audio(full_response, sanitized_filename)
                sources = [doc.page_content for doc in relevant_docs]
                yield json.dumps({"type": "metadata", "sources": sources, "audio_url": audio_url}) + "\n\n"
                '''
                log_id = log_chat_to_table(client_session_id, question, full_response)
                
                sanitized_filename = f"{client_session_id}_{uuid.uuid4().hex}"
                audio_url = await generate_audio(full_response, sanitized_filename)
                
                sources = [doc.page_content for doc in relevant_docs]
                yield json.dumps({
                    "type": "metadata",
                    "sources": sources,
                    "audio_url": audio_url,
                    "log_id": log_id,
                    "session_id": client_session_id
                }) + "\n\n"
                
            except Exception as e:
                print(f"Error in streaming generator: {e}")
                yield json.dumps({"type": "error", "content": str(e)}) + "\n\n"

        return StreamingResponse(response_generator(), media_type="application/x-ndjson")
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return JSONResponse(content={"error": f"Server error: {str(e)}"}, status_code=500)

'''
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
'''

@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()

    feedback = (data.get("feedback") or "").strip()
    row_key = (data.get("log_id") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not feedback or not row_key or not session_id:
        return JSONResponse(
            content={"error": "Missing required data: feedback, log_id, session_id"},
            status_code=400
        )

    try:
        update_feedback_in_table(session_id=session_id, row_key=row_key, feedback=feedback)
        return JSONResponse(content={"message": "Feedback saved successfully"})
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to save feedback: {str(e)}"},
            status_code=500
        )


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    return JSONResponse(content={"error": "Audio file not found"}, status_code=404)

@app.post("/survey")
async def submit_survey(request: Request):
    data = await request.json()
    session_id = (data.get("session_id") or "").strip()
    ratings = data.get("ratings") or {}

    required = ["q1", "q2", "q3", "q4", "q5"]
    if not session_id or any(k not in ratings for k in required):
        return JSONResponse(
            content={"error": "Missing required data: session_id and ratings(q1..q5)"},
            status_code=400
        )

    # validate 1-5
    try:
        for k in required:
            v = int(ratings[k])
            if v < 1 or v > 5:
                raise ValueError(f"{k} out of range")
    except Exception:
        return JSONResponse(
            content={"error": "Ratings must be integers 1-5 for q1..q5"},
            status_code=400
        )

    try:
        row_key = log_survey_to_table(session_id=session_id, ratings=ratings)
        return JSONResponse(content={"message": "Survey saved", "survey_id": row_key})
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save survey: {str(e)}"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)















