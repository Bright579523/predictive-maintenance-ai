import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
import requests
import tempfile
from dotenv import load_dotenv

# Load environment variables (GROQ_API_KEY must be in .env)
load_dotenv()

class CNCExpertRAG:
    def __init__(self, vector_store_path="models/faiss_index"):
        self.vector_store_path = vector_store_path
        
        # 1. Use a fast, free local embedding model (so we don't pay for embeddings)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initialize Groq LLM (Llama 3 70B - Very Smart & Fast)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("WARNING: GROQ_API_KEY not found in .env file!")
            self.llm = None
        else:
            self.llm = ChatGroq(
                temperature=0.1, 
                model_name="llama-3.3-70b-versatile", # Modern state-of-the-art model
                api_key=groq_api_key
            )
            
        self.vector_store = None

    def ingest_document(self, file_path):
        """Loads a PDF or TXT file, splits it, and saves it to the FAISS Vector Database."""
        print(f"Loading document: {file_path}...")
        
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Unsupported file format. Please use PDF or TXT.")
            
        documents = loader.load()
        
        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        print(f"Creating Embeddings for {len(docs)} chunks... (This runs locally)")
        if self.vector_store is not None:
            self.vector_store.add_documents(docs)
        else:
            # Try to load existing db first, if not found, create new
            if not self.load_vector_db():
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                self.vector_store.add_documents(docs)
        
        # Save the vector database so we don't have to embed again
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"DONE: Vector database saved to {self.vector_store_path}")

    def ingest_url(self, url):
        """Scrapes a web page or downloads a PDF from a URL and adds it to the vector store."""
        print(f"Loading from URL: {url}...")
        
        # Check if URL is a direct PDF link
        if url.lower().endswith('.pdf') or 'pdf' in url.lower():
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                os.unlink(tmp_path) # Clean up temp file
            else:
                raise Exception(f"Failed to download PDF from {url}")
        else:
            # Assume it's a web page
            loader = WebBaseLoader(url)
            documents = loader.load()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        
        print(f"Creating Embeddings for {len(docs)} chunks from web/pdf...")
        if self.vector_store:
            self.vector_store.add_documents(docs)
        else:
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        self.vector_store.save_local(self.vector_store_path)
        print(f"DONE: URL content added to {self.vector_store_path}")

    def load_vector_db(self):
        """Loads the existing FAISS Vector Database from disk."""
        if os.path.exists(self.vector_store_path):
            self.vector_store = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True # Required for FAISS in newer versions
            )
            print("DONE: Vector database loaded successfully.")
            return True
        else:
            print("WARNING: No vector database found. Please ingest a document first.")
            return False

    def ask_question(self, query):
        """Answers a question using RAG (Retrieval-Augmented Generation)."""
        if not self.vector_store:
            if not self.load_vector_db():
                return "System Error: Knowledge base not loaded."
                
        if not self.llm:
            return "System Error: GROQ_API_KEY is missing. Please add it to your .env file."

        # RAG Prompt Template - Helpful expert that uses manuals as reference
        prompt_template = """You are an expert CNC Machining and Cutting Tool Consultant with 15+ years of experience.
You have access to context extracted from official technical manuals (Sandvik, Sumitomo, Haas, and CNC engineering guides).

Instructions:
1. Use the provided context as your primary source to answer the question.
2. If the context contains related information, synthesize a helpful answer from it — even if it doesn't match the question exactly.
3. Use bullet points and bold text for clarity.
4. Include specific values (speeds, feeds, grades) when available in the context.
5. Only say you cannot answer if the context is completely unrelated to the question.

Context from manuals:
{context}

User's Question: {question}

Expert Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create the RAG Chain — retrieve 5 chunks for broader context coverage
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("Groq Llama-3 is thinking...")
        response = qa_chain.invoke(query)
        return response['result']

# --- Quick Test Code (Runs only if you execute this file directly) ---
if __name__ == "__main__":
    rag = CNCExpertRAG()
    
    # Check if DB exists, if not, try to ingest the text file we already have
    if not os.path.exists("models/faiss_index"):
        test_file = "docs/information.txt"
        if os.path.exists(test_file):
            rag.ingest_document(test_file)
        else:
            print("Please provide a PDF or TXT file to test.")
    
    # Try a test question if the Groq key is set
    if os.environ.get("GROQ_API_KEY"):
        rag.load_vector_db()
        ans = rag.ask_question("What is Tool Wear Failure (TWF)?")
        print("\n" + "="*50)
        print("Test Answer:\n", ans)
        print("="*50)
