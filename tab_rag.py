# =============================================
# TAB3 RAG WITH LANGCHAIN, CHROMADB, LANGSMITH
# ============================================

import json
import shutil
import os
import time
from pathlib import Path
import pypdf

import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma

# Import LangSmith for tracing
from langsmith import traceable


def initialize_langsmith():
    """Initialize LangSmith tracing if API key is configured."""
    if os.environ.get("LANGSMITH_API_KEY"):
        return True
    return False


def cleanup_old_chroma_collections(keep_latest: int = 3):
    """Clean up old Chroma collections to prevent disk bloat."""
    try:
        chroma_db_dir = Path(".chroma_db")
        if not chroma_db_dir.exists():
            return
        
        # Get all collection subdirectories (they have UUID names)
        collections = sorted([d for d in chroma_db_dir.glob("*") if d.is_dir()])
        
        # Keep only the latest collections
        if len(collections) > keep_latest:
            for old_collection in collections[:-keep_latest]:
                try:
                    shutil.rmtree(old_collection)
                except Exception as e:
                    # Silently fail for old collections - they might be in use
                    pass
    except Exception:
        pass  # Don't crash if cleanup fails


def load_documents_from_folder(folder_path: str):
    """Load documents from rag_docs folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        st.warning(f"Folder {folder_path} does not exist.")
        return []
    
    documents = []
    
    # Try to load text files
    for text_file in folder.glob("*.txt"):
        try:
            loader = TextLoader(str(text_file), encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error loading {text_file.name}: {e}")
    
    # Try to load PDF files
    for pdf_file in folder.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Error loading {pdf_file.name}: {e}")
    
    return documents


@traceable(name="setup_rag_chain", run_type="tool")
def setup_rag_chain(docs_folder: str = "rag_docs"):
    """Setup RAG chain with ChromaDB vectorstore."""
    
    # Load documents
    documents = load_documents_from_folder(docs_folder)
    
    if not documents:
        raise ValueError(f"No documents found in {docs_folder} folder. Please add .txt or .pdf files.")
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = splitter.split_documents(documents)
    
    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Use persistent Chroma database with unique collection per run
    chroma_db_dir = ".chroma_db"
    Path(chroma_db_dir).mkdir(exist_ok=True)
    
    # Create a unique collection name based on timestamp to avoid file locks
    collection_name = f"rag_docs_{int(time.time() * 1000)}"
    
    try:
        # Create Chroma vectorstore with the unique collection
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=chroma_db_dir,
            collection_name=collection_name
        )
        
        # Persist the vectorstore
        # LangChainDeprecationWarning: Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
        # vectorstore.persist()
        
        return vectorstore
        
    except OSError as e:
        if "already being used" in str(e) or "WinError 32" in str(e):
            # File lock error - wait and retry
            st.warning("⚠️ Database is being released from previous query, retrying in 2 seconds...")
            time.sleep(2)
            # Retry with a fresh collection name
            collection_name = f"rag_docs_{int(time.time() * 1000)}"
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=chroma_db_dir,
                collection_name=collection_name
            )
            # vectorstore.persist()    # No longer needed with Chroma 0.4.x
            return vectorstore
        else:
            raise


@traceable(name="run_rag", run_type="chain")
def run_rag(user_query: str, docs_folder: str = "rag_docs"):
    """Run RAG pipeline to answer questions based on documents."""
    
    try:
        # Clean up old collections before starting new query
        cleanup_old_chroma_collections(keep_latest=3)
        
        # Setup vectorstore and retriever
        vectorstore = setup_rag_chain(docs_folder)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Get LLM
        llm = OllamaLLM(
            # model="gemma4:e2b",
            # model="ExpedientFalcon/Qwen3-4B-UD-Q5_K_XL",
            model= "hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M",
            base_url="http://localhost:11434"
        )
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = retriever.invoke(user_query)
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Helper to call LLM and capture token usage
        def _llm_call(prompt):
            _t0 = time.time()
            gen_result = llm.generate([prompt])
            _elapsed = round(time.time() - _t0, 2)
            generation = gen_result.generations[0][0]
            gen_info = generation.generation_info or {}
            return (
                generation.text,
                {
                    "prompt_tokens": gen_info.get("prompt_eval_count", 0) or 0,
                    "completion_tokens": gen_info.get("eval_count", 0) or 0,
                    "elapsed": _elapsed,
                }
            )

        kickoff_start = time.time()

        # Step 2: Generate answer using LLM
        answer_prompt = f"""Based on the following documents, answer the user's question comprehensively.

Documents:
{context}

User Question: {user_query}

Provide a detailed answer based on the documents. If the answer is not in the documents, state that clearly."""
        
        answer_response, answer_usage = _llm_call(answer_prompt)
        
        # Step 3: Generate summary
        summary_prompt = f"""Create a concise summary of the key findings related to the query: "{user_query}"

Based on:
{answer_response}

Provide 3-5 bullet points summarizing the key findings."""
        
        summary_response, summary_usage = _llm_call(summary_prompt)
        
        # Step 4: Generate source attribution
        sources = [f"- {doc.metadata.get('source', 'Unknown source')}" for doc in retrieved_docs]
        unique_sources = "\n".join(sorted(set(sources)))
        
        # Step 5: Generate structured report
        report_prompt = f"""Generate a structured technical report as JSON about the following:

Query: {user_query}

Answer: {answer_response}

Summary: {summary_response}

Return a JSON object with this exact structure:
{{
  "title": "RAG Analysis Report",
  "sections": [
    {{"heading": "Query", "content": "{user_query}"}},
    {{"heading": "Answer", "content": "..."}},
    {{"heading": "Summary", "content": "..."}},
    {{"heading": "Sources", "content": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""
        
        report_response, report_usage = _llm_call(report_prompt)
        kickoff_elapsed = time.time() - kickoff_start

        try:
            report = json.loads(report_response)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            report = {
                "title": "RAG Analysis Report",
                "sections": [
                    {"heading": "Query", "content": user_query},
                    {"heading": "Answer", "content": answer_response},
                    {"heading": "Summary", "content": summary_response},
                    {"heading": "Sources", "content": unique_sources}
                ]
            }
        
        # Build token summary
        usages = [
            ("Answer", answer_usage),
            ("Summary", summary_usage),
            ("Report", report_usage),
        ]
        total_prompt = sum(u["prompt_tokens"] for _, u in usages)
        total_completion = sum(u["completion_tokens"] for _, u in usages)
        total_tokens = total_prompt + total_completion
        completion_tps = round(total_completion / kickoff_elapsed, 2) if kickoff_elapsed > 0 and total_completion > 0 else 0.0
        total_tps = round(total_tokens / kickoff_elapsed, 2) if kickoff_elapsed > 0 and total_tokens > 0 else 0.0

        token_summary = {
            "per_call": [{"agent": name, "iteration": u["elapsed"]} for name, u in usages],
            "totals": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "cached_prompt_tokens": 0,
                "successful_requests": len(usages),
            },
            "performance": {
                "elapsed_seconds": round(kickoff_elapsed, 2),
                "completion_tokens_per_sec": completion_tps,
                "total_tokens_per_sec": total_tps,
            }
        }

        # Create a simple result object to match the interface
        class RAGResult:
            def __init__(self, data, num_docs):
                self.pydantic = None
                self.json = data
                self.num_docs_retrieved = num_docs

        return RAGResult(report, len(retrieved_docs)), token_summary
        
    except Exception as e:
        raise Exception(f"RAG Pipeline Error: {str(e)}")
