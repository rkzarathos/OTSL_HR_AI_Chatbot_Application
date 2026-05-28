import os
import json
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pageindex import PageIndexClient
import pageindex.utils as utils
from pypdf import PdfReader


# ========================
# CONFIG
# ========================

#try:
#    from dotenv import load_dotenv
#    load_dotenv()
#except Exception:
#    pass

PAGEINDEX_API_KEY_1 = os.getenv("PAGEINDEX_API_KEY_1")
PAGEINDEX_API_KEY_2 = os.getenv("PAGEINDEX_API_KEY_2")

if not PAGEINDEX_API_KEY_1:
    raise ValueError("PAGEINDEX_API_KEY_1 environment variable is not set.")

if not PAGEINDEX_API_KEY_2:
    raise ValueError("PAGEINDEX_API_KEY_2 environment variable is not set.")

PAGEINDEX_KEY_1_PAGE_SWITCH_THRESHOLD = 150

DOCUMENTS_DIR = Path(
    os.getenv("DOCUMENTS_PATH", Path.cwd() / "documents")
).resolve()

PAGEINDEX_MANIFEST_DIR = Path(
    os.getenv("PAGEINDEX_MANIFEST_DIR", Path.cwd() / "pageindex-manifest")
).resolve()

PAGEINDEX_MANIFEST_PATH = PAGEINDEX_MANIFEST_DIR / "pageindex_manifest.json"
PAGEINDEX_TREES_DIR = PAGEINDEX_MANIFEST_DIR / "pageindex_trees"

PRINT_TREES_TO_LOG = False

POLL_INTERVAL_SECONDS = 5
MAX_POLLS_PER_DOCUMENT = 120  # 10 minutes per document

# ========================
# DOCUMENT LIST
# ========================

DOCUMENTS = [
    "2026 Employee Handbook.pdf",
    "BUILDING ACCESS POLICY.pdf",
    "Curative Onboarding Steps.pdf",
    "Curative Pharmacy Need to Know.pdf",
    "Curative Registration.pdf",
    "Curative Services.pdf",
    "ExponentHR 401K Enrollment.pdf",
    "ExponentHR Obtaining Year End Forms - W2 and 1095-C.pdf",
    "ExponentHR Pay Checks and Direct Deposit.pdf",
    "FMLA Claim Submission Checklist.pdf",
    "FMLA Policy.pdf",
    "Fidelity NetBenefits Registration.pdf",
    "Gallagher Team contact information.pdf",
    "HR Frequently Asked Questions.pdf",
    "OTSL 401K Guidlines.pdf",
    "OTSL Employee Referral Form.pdf",
    "OTSL Out of State Employee Benefits.pdf",
    "OTSL Performace Management Module.pdf",
    "OTSL Profit Sharing Plan.pdf",
    "Reporting Time in ExponentHR.pdf",
    "2026 Benefits Enrollment - old.pdf",
]


# ========================
# HELPERS
# ========================



def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as e:
        raise RuntimeError(f"Could not count pages for {pdf_path.name}: {e}")

def utc_now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_filename(name: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in name
    )


def extract_doc_id(submit_response: Any) -> str:
    """
    PageIndex cookbook-style response is expected to include:
        {"doc_id": "..."}
    This helper is defensive in case response shape varies.
    """
    if isinstance(submit_response, dict):
        doc_id = (
            submit_response.get("doc_id")
            or submit_response.get("id")
            or submit_response.get("document_id")
        )
    else:
        doc_id = str(submit_response)

    if not doc_id:
        raise RuntimeError(
            f"Could not find doc_id in PageIndex submit response: {submit_response}"
        )

    return doc_id


def wait_until_retrieval_ready(
    client: PageIndexClient,
    doc_id: str,
    doc_name: str,
) -> None:
    """
    Waits until PageIndex says the submitted document is retrieval-ready.
    """
    for attempt in range(1, MAX_POLLS_PER_DOCUMENT + 1):
        try:
            if client.is_retrieval_ready(doc_id):
                print(f"PageIndex ready: {doc_name}")
                return
        except Exception as e:
            print(f"PageIndex readiness check failed for {doc_name}: {e}")

        print(
            f"Waiting for PageIndex: {doc_name} "
            f"({attempt}/{MAX_POLLS_PER_DOCUMENT})"
        )
        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out waiting for PageIndex to process {doc_name}. doc_id={doc_id}"
    )


def get_tree_result(
    client: PageIndexClient,
    doc_id: str,
    doc_name: str,
) -> Any:
    """
    Fetches the actual PageIndex tree object.

    Important:
    app.py expects the saved tree file to contain the tree itself,
    not a wrapper like {"status": "...", "result": {...}}.
    """
    tree_response = client.get_tree(doc_id, node_summary=True)

    if isinstance(tree_response, dict) and "result" in tree_response:
        tree = tree_response["result"]
    else:
        tree = tree_response

    if not tree:
        raise RuntimeError(f"Empty PageIndex tree returned for {doc_name}")

    return tree


def save_tree_file(
    doc_name: str,
    doc_id: str,
    tree: Any,
) -> str:
    PAGEINDEX_TREES_DIR.mkdir(parents=True, exist_ok=True)

    tree_file = PAGEINDEX_TREES_DIR / f"{safe_filename(doc_name)}__{doc_id}.json"

    with tree_file.open("w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)

    return str(tree_file)


def submit_and_save_document(
    client: PageIndexClient,
    doc_path: Path,
) -> Dict[str, Any]:
    doc_name = doc_path.name

    print(f"\nSubmitting to PageIndex: {doc_name}")

    submit_response = client.submit_document(str(doc_path))
    doc_id = extract_doc_id(submit_response)

    print(f"Submitted {doc_name}. doc_id={doc_id}")

    wait_until_retrieval_ready(
        client=client,
        doc_id=doc_id,
        doc_name=doc_name,
    )

    tree = get_tree_result(
        client=client,
        doc_id=doc_id,
        doc_name=doc_name,
    )

    if PRINT_TREES_TO_LOG:
        print(f"\nPageIndex tree for {doc_name}:")
        try:
            utils.print_tree(tree)
        except Exception as e:
            print(f"Could not print PageIndex tree for {doc_name}: {e}")

    tree_file = save_tree_file(
        doc_name=doc_name,
        doc_id=doc_id,
        tree=tree,
    )

    print(f"Saved tree file: {tree_file}")

    return {
        "doc_name": doc_name,
        "doc_path": str(doc_path),
        "pageindex_doc_id": doc_id,
        "tree_file": tree_file,
        "submitted_at_utc": utc_now_iso(),
    }


# ========================
# MAIN
# ========================

def main() -> None:
    print("Starting PageIndex build...")
    print(f"DOCUMENTS_DIR: {DOCUMENTS_DIR}")
    print(f"PAGEINDEX_MANIFEST_DIR: {PAGEINDEX_MANIFEST_DIR}")
    print(f"PAGEINDEX_MANIFEST_PATH: {PAGEINDEX_MANIFEST_PATH}")
    print(f"PAGEINDEX_TREES_DIR: {PAGEINDEX_TREES_DIR}")

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"DOCUMENTS_PATH does not exist: {DOCUMENTS_DIR}")

    PAGEINDEX_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_TREES_DIR.mkdir(parents=True, exist_ok=True)

    client_key_1 = PageIndexClient(api_key=PAGEINDEX_API_KEY_1)
    client_key_2 = PageIndexClient(api_key=PAGEINDEX_API_KEY_2)
    
    key_1_pages_used = 0
    key_2_pages_used = 0

    manifest: Dict[str, Any] = {
        "built_at_utc": utc_now_iso(),
        "documents_dir": str(DOCUMENTS_DIR),
        "pageindex_manifest_dir": str(PAGEINDEX_MANIFEST_DIR),
        "pageindex_trees_dir": str(PAGEINDEX_TREES_DIR),
        "documents_requested": len(DOCUMENTS),
        "documents_submitted": 0,
        "documents_missing": [],
        "documents_failed": [],
        "documents": [],
        "api_key_page_switch_threshold": PAGEINDEX_KEY_1_PAGE_SWITCH_THRESHOLD,
        "key_1_pages_used": 0,
        "key_2_pages_used": 0,
    }

    for doc_name in DOCUMENTS:
        doc_path = DOCUMENTS_DIR / doc_name

        if not doc_path.exists():
            print(f"WARNING: Missing document: {doc_path}")
            manifest["documents_missing"].append(doc_name)
            continue

        if doc_path.suffix.lower() != ".pdf":
            print(f"WARNING: Skipping non-PDF document: {doc_path}")
            manifest["documents_failed"].append(
                {
                    "doc_name": doc_name,
                    "doc_path": str(doc_path),
                    "error": "not_a_pdf",
                }
            )
            continue

        try:
            page_count = get_pdf_page_count(doc_path)
        
            if key_1_pages_used + page_count <= PAGEINDEX_KEY_1_PAGE_SWITCH_THRESHOLD:
                selected_client = client_key_1
                selected_key_label = "PAGEINDEX_API_KEY_1"
                key_1_pages_used += page_count
            else:
                selected_client = client_key_2
                selected_key_label = "PAGEINDEX_API_KEY_2"
                key_2_pages_used += page_count
        
            print(
                f"Using {selected_key_label} for {doc_name} "
                f"({page_count} pages). "
                f"Key1 pages used: {key_1_pages_used}, "
                f"Key2 pages used: {key_2_pages_used}"
            )
        
            doc_record = submit_and_save_document(
                client=selected_client,
                doc_path=doc_path,
            )
        
            doc_record["page_count"] = page_count
            doc_record["api_key_used"] = selected_key_label
        
            manifest["documents"].append(doc_record)
            manifest["documents_submitted"] += 1
        
        except Exception as e:
            print(f"ERROR: Failed PageIndex build for {doc_name}: {e}")
            manifest["documents_failed"].append(
                {
                    "doc_name": doc_name,
                    "doc_path": str(doc_path),
                    "error": str(e),
                }
            )

    manifest["key_1_pages_used"] = key_1_pages_used
    manifest["key_2_pages_used"] = key_2_pages_used
    manifest["total_pages_submitted"] = key_1_pages_used + key_2_pages_used

    with PAGEINDEX_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nPageIndex build complete.")
    print(f"Documents requested: {manifest['documents_requested']}")
    print(f"Documents submitted: {manifest['documents_submitted']}")
    print(f"Documents missing: {len(manifest['documents_missing'])}")
    print(f"Documents failed: {len(manifest['documents_failed'])}")
    print(f"Manifest saved to: {PAGEINDEX_MANIFEST_PATH}")

    if manifest["documents_failed"]:
        raise RuntimeError(
            "One or more documents failed during PageIndex build. "
            "Check the startup logs and pageindex_manifest.json."
        )


if __name__ == "__main__":
    main()
