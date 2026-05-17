import os
import sys
from pathlib import Path

# Add api to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "api"))

from rag_core import CNCExpertRAG

def main():
    rag = CNCExpertRAG(vector_store_path="models/faiss_index")
    
    # Files to ingest
    docs_to_ingest = [
        "docs/information.txt",
        "docs/sumitomo_manual.pdf",
        "docs/Sandvik.pdf",
        "docs/Fundamentals_of_CNC_Machining.pdf"
    ]
    
    print("Starting ingestion process...")
    
    # Clean up existing database first for a fresh start
    if os.path.exists("models/faiss_index"):
        print("Removing old index for a fresh rebuild...")
        import shutil
        shutil.rmtree("models/faiss_index")
        
    for doc in docs_to_ingest:
        doc_path = BASE_DIR / doc
        if doc_path.exists():
            print(f"\n--- Ingesting {doc} ---")
            try:
                rag.ingest_document(str(doc_path))
                print(f"Successfully ingested {doc}")
            except Exception as e:
                print(f"Error ingesting {doc}: {e}")
        else:
            print(f"File not found: {doc}")
            
    print("\nIngestion complete! FAISS Vector DB successfully built with comprehensive cutting tool manuals.")

if __name__ == "__main__":
    main()
