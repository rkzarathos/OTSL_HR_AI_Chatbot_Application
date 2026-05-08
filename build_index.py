import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader, PyMuPDFLoader
from langchain.schema import Document

# --- Azure Document Intelligence imports ---
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# ========================
# ENV / CONFIG
# ========================

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_PATH", os.path.join(os.getcwd(), "documents"))
CHROMA_DB_PATH = os.getenv("CHROMADB_PATH", "/chromadb")

# Azure Document Intelligence env vars
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
AZURE_DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")

if not AZURE_DOC_INTELLIGENCE_ENDPOINT or not AZURE_DOC_INTELLIGENCE_KEY:
    raise ValueError(
        "Azure Document Intelligence endpoint/key env vars are not set "
        "(AZURE_DOC_INTELLIGENCE_ENDPOINT, AZURE_DOC_INTELLIGENCE_KEY)."
    )

doc_client = DocumentIntelligenceClient(
    endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_KEY),
)

DOCUMENTS = [
    "2026 Employee Handbook.pdf", "BUILDING ACCESS POLICY.pdf", "Curative Onboarding Steps.pdf", "Curative Pharmacy Need to Know.pdf", "Curative Registration.pdf",
    "Curative Services.pdf", "ExponentHR 401K Enrollment.pdf", "ExponentHR Obtaining Year End Forms - W2 and 1095-C.pdf", "ExponentHR Pay Checks and Direct Deposit.pdf", "FMLA Claim Submission Checklist.pdf",
    "FMLA Policy.pdf", "Fidelity NetBenefits Registration.pdf", "Gallagher Team contact information.pdf", "HR Frequently Asked Questions.pdf", "OTSL 401K Guidlines.pdf",
    "OTSL Employee Referral Form.pdf", "OTSL Out of State Employee Benefits.pdf", "OTSL Performace Management Module.pdf", "OTSL Profit Sharing Plan.pdf",
    "Reporting Time in ExponentHR.pdf", "2026 Benefits Enrollment - old.pdf",
]

# ========================
# HELPERS
# ========================

def has_real_text(docs: List[Document], min_chars: int = 20) -> bool:
    """
    Returns True if the list of Documents has any non-trivial text.
    This is used to detect cases where PDFMiner / PyMuPDF "succeeded"
    but the PDF was image-only and yielded essentially no text.
    """
    for d in docs:
        if d.page_content and len(d.page_content.strip()) >= min_chars:
            return True
    return False


def azure_ocr_to_documents(file_path: str) -> List[Document]:
    """
    Use Azure AI Document Intelligence (prebuilt-read) to extract text
    from a PDF. Returns one Document per page.
    """
    with open(file_path, "rb") as f:
        poller = doc_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=f,
        )
    result = poller.result()

    docs: List[Document] = []

    # Build one LangChain Document per page
    for page in result.pages:
        lines = [line.content for line in page.lines]
        page_text = "\n".join(lines)

        docs.append(
            Document(
                page_content=page_text,
                metadata={
                    "source": file_path,
                    "page": page.page_number,
                    "ocr_provider": "azure_document_intelligence",
                },
            )
        )

    return docs

# ========================
# LOAD & BUILD DATASOURCE
# ========================

datasource: List[Document] = []

for doc_name in DOCUMENTS:
    doc_path = os.path.join(DOCUMENTS_DIR, doc_name)
    if not os.path.exists(doc_path):
        print(f"WARNING: {doc_path} not found")
        continue

    docs_for_file: List[Document] = []

    # 1) Try PDFMiner
    try:
        docs_for_file = PDFMinerLoader(doc_path).load()
    except ValueError as e:
        print(f"PDFMiner failed to load {doc_path}: {e}")

    # 2) If PDFMiner failed or produced no real text, try PyMuPDF
    if not has_real_text(docs_for_file):
        try:
            print(f"Falling back to PyMuPDF for {doc_path}")
            docs_for_file = PyMuPDFLoader(doc_path).load()
        except Exception as e:
            print(f"PyMuPDF failed to load {doc_path}: {e}")

    # 3) If still no real text, use Azure Document Intelligence
    if not has_real_text(docs_for_file):
        try:
            print(f"No extractable text via PDFMiner/PyMuPDF for {doc_path}, using Azure OCR.")
            docs_for_file = azure_ocr_to_documents(doc_path)
        except Exception as e:
            print(f"Azure Document Intelligence failed for {doc_path}: {e}")
            docs_for_file = []

    if not docs_for_file:
        print(f"Skipping {doc_path}: could not extract any text.")
        continue

    datasource.extend(docs_for_file)

print(f"Collected {len(datasource)} documents/pages")

# ========================
# SPLIT, EMBED, INDEX
# ========================

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1200,
    chunk_overlap=400,
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
