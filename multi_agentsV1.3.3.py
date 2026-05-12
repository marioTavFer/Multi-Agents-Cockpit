# =============================
# Agents Cockpit - Full App
# =============================

import streamlit as st
import os
import datetime
import time
import json
import socket
import urllib.request
import csv
import pandas as pd
from pathlib import Path

import litellm
from crewai import LLM
from langchain_openai import ChatOpenAI


# =============================
# CONFIG
# =============================

# =============================
# Default Ollama endpoint and models
# =============================
OLLAMA_BASE_URL = "http://localhost:11434"

CREW_OLLAMA_MODEL = "ollama/hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:UD-Q4_K_XL"
# CREW_OLLAMA_MODEL = "ollama/qwen35_2b_q4_mario:latest"
# CREW_OLLAMA_MODEL = "ollama/ExpedientFalcon/Qwen3-4B-UD-Q5_K_XL:latest"
# CREW_OLLAMA_MODEL = "ollama/gemma4:e2b"
# CREW_OLLAMA_MODEL = "ollama/Phi4-mini_mario:latest"
# CREW_OLLAMA_MODEL = "ollama/hf.co/unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M"

RAG_OLLAMA_MODEL = "ollama/hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:UD-Q4_K_XL"
OLLAMA_EMBED_MODEL = "ollama/nomic-embed-text:latest"

os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
litellm.num_retries = 0
litellm.request_timeout = 20

# ===========================
# Default LM Studio endpoint and models
# ===========================
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


# LMSTUDIO_MODEL = "openai/llama-3.2-3b-instruct"  
LMSTUDIO_MODEL = "openai/phi-4-mini-instruct"  
# LMSTUDIO_MODEL = "openai/google/gemma-4-e2b"  
# LMSTUDIO_MODEL = "openai/qwen3.5-2b"
# LMSTUDIO_MODEL = "openai/qwen2.5-coder-7b-instruct"
# LMSTUDIO_MODEL = "openai/qwen/qwen3-4b-thinking-2507" 


# =============================
# LLM FACTORY FOR CREWAI (Litellm LLM wrapper for Ollama)
# =============================

def get_llm():
    return LLM(
        model=CREW_OLLAMA_MODEL,
        api_base=f"{OLLAMA_BASE_URL}",
        api_key="ollama",
        timeout=120
    )

# =============================
# LLM FACTORY FOR LM STUDIO (LangChain)
# =============================

def get_lmstudio_llm(base_url: str = LMSTUDIO_BASE_URL, model: str = LMSTUDIO_MODEL, temperature: float = 0.7):
    """Create a LangChain ChatOpenAI instance pointing at LM Studio."""
    return ChatOpenAI(
        base_url=base_url,
        api_key="lm-studio",          # LM Studio accepts any non-empty string
        model=model,
        temperature=temperature,
        timeout=6000,
        max_retries=0,
    )


# ===============================
# Import TAB MODULES
# ==============================

from logging_functions import log_request, read_logs
from tab_crewai import run_crew, ReportModel
from tab_groq import run_groq_direct
from tab_rag import run_rag
from tab_langgraph import run_langgraph_workflow
from tab_crewTokenTest import run_testToken
from tab_LMstudio_Lchain import run_lmstudio_lchain
from tab_LMStudioTokenTest import run_lmstudio_token_test
from pdf_generator import format_text_for_pdf, format_text_for_streamlit, save_report_pdf


# =============================
# API keys  - for testing purposes only - replace with secure input in production
# ********  replace "xxxxxx" with your api-keys   ********
#==============================

#Tavely
tvly_api_key = "xxxxxxxxxxxxxxxxxxx"

#Groq
groq_api_key= "xxxxxxxxxxxxxxxxxxxx"

#LangSmith
langsmith_api_key = "xxxxxxxxxxxxxxxxxxx"  

# =============================
# Streamlit page configuration
# =============================
st.set_page_config(page_title="Multi-Agents Cockpit", layout="wide")
st.title("🧠 Multi-Agents Cockpit / Test Drive")


# Initialize session state with hardcoded API keys
if "groq_api_key" not in st.session_state or not st.session_state["groq_api_key"]:
    st.session_state["groq_api_key"] = groq_api_key
if "tavily_api_key" not in st.session_state or not st.session_state["tavily_api_key"]:
    st.session_state["tavily_api_key"] = tvly_api_key
if "langsmith_api_key" not in st.session_state or not st.session_state["langsmith_api_key"]:
    st.session_state["langsmith_api_key"] = langsmith_api_key

# Configure LangSmith environment if API key is available
if st.session_state.get("langsmith_api_key"):
    os.environ["LANGSMITH_API_KEY"] = st.session_state["langsmith_api_key"]
    
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "multi-agentsv1"
    os.environ["LANGSMITH_TRACING"] = "false" # Set to "true" to enable detailed tracing in LangSmith (may impact performance)
    # os.environ["LANGSMITH_TRACING"] = "true"



# =============================
# SIDEBAR
# =============================

with st.sidebar:
    st.header("🔑 API Keys")

    # Check if groq_api_key already exists in session state
    if "groq_api_key" not in st.session_state or not st.session_state["groq_api_key"]:
        groq_api_key = st.text_input("Groq API Key", type="password")
    else:
        st.success("✓ Groq API Key already configured")
        groq_api_key = st.session_state["groq_api_key"]

    # Check if tavily_api_key already exists in session state
    if "tavily_api_key" not in st.session_state or not st.session_state["tavily_api_key"]:
        tavily_api_key = st.text_input("Tavily API Key", type="password")
    else:
        st.success("✓ Tavily API Key already configured")
        tavily_api_key = st.session_state["tavily_api_key"]
    # Check if langsmith_api_key already exists in session state
    if "langsmith_api_key" not in st.session_state or not st.session_state["langsmith_api_key"]:
        langsmith_api_key = st.text_input("LangSmith API Key", type="password")
    else:
        st.success("✓ LangSmith API Key already configured")
        langsmith_api_key = st.session_state["langsmith_api_key"]

    with st.expander("Advanced Keys"):
        if "x_api_key" not in st.session_state or not st.session_state["x_api_key"]:
            x_api_key = st.text_input("X API Key", type="password")
        else:
            st.success("✓ X API Key already configured")
            x_api_key = st.session_state["x_api_key"]

        if "y_api_key" not in st.session_state or not st.session_state["y_api_key"]:
            y_api_key = st.text_input("Y API Key", type="password")
        else:
            st.success("✓ Y API Key already configured")
            y_api_key = st.session_state["y_api_key"]

        

    st.session_state["groq_api_key"] = groq_api_key
    st.session_state["tavily_api_key"] = tavily_api_key
    st.session_state["x_api_key"] = x_api_key
    st.session_state["y_api_key"] = y_api_key
    st.session_state["langsmith_api_key"] = langsmith_api_key

    st.divider()
    st.header("🤖 Model in Use")
    st.markdown(f"**Crew / LangGraph:**")
    st.code(CREW_OLLAMA_MODEL, language=None)
    st.markdown(f"**RAG:**")
    st.code(RAG_OLLAMA_MODEL, language=None)
    st.markdown(f"**Embeddings:**")
    st.code(OLLAMA_EMBED_MODEL, language=None)


# =============================
# TABS
# =============================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🤖 CrewAI+Ollama", "🚀 Groq+Tavily+LiteLLM", "📚 RAG+LangChain+Ollama", "🔗 LangGraph+LangSmith+LMStudio", "🧪 CrewAI+TokenTest+Ollama", "🖥️ LangChain+LMStudio", "🧪 LMStudio+TokenTest", "📊 Log of Requests"])

# =============================
# TAB 1 - CREWAI
# =============================

with tab1:
    
    st.header(f"CrewAI Multi-Agent - using local ollama with {CREW_OLLAMA_MODEL}")
    
    user_input = st.text_area("We are a Crew of AI experts.  Enter your request")
    output_dir = st.text_input("Report folder", value="reports")

    if st.button("Run Crew"):
        with st.spinner("Running agents..."):
            
            start_time = time.time()
            result, token_usage = run_crew(user_input, get_llm())
            end_time = time.time()
            time_used = end_time - start_time

            st.subheader("Crew (AI Specialist, Technical Educator, Technical Report Writer) Output:")
           
            try:
                if result.pydantic:
                    report = result.pydantic.model_dump()
                else:
                    report = result.json

                st.subheader(report["title"])

                for section in report["sections"]:
                    st.markdown(f"### {section['heading']}")
                    content = section.get("content", "No content available.")
                    st.markdown(format_text_for_streamlit(content))

                st.write(f"Time used: {time_used:.2f} seconds")

                st.subheader("📊 Token Usage (per LLM call)")
                per_call = token_usage.get("per_call", [])
                if per_call:
                    st.table(per_call)
                else:
                    st.warning("No per-call token data captured.")

                st.subheader("🔢 Totals")
                totals = token_usage.get("totals", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                col3.metric("Total Tokens", totals.get("total_tokens", 0))

                perf = token_usage.get("performance", {})
                if perf:
                    st.subheader("⚡ Performance")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                    p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                    p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                token_usage_str = (
                    f"Prompt: {totals.get('prompt_tokens', 0)}, "
                    f"Completion: {totals.get('completion_tokens', 0)}, "
                    f"Total: {totals.get('total_tokens', 0)}"
                )
                pdf_path = save_report_pdf("crew", report, time_used, token_usage_str, output_dir, token_summary=token_usage)
                st.success(f"Report saved to: {pdf_path}")
                
                # Log the request
                log_request("CrewAI+Ollama", CREW_OLLAMA_MODEL, token_usage)

            except Exception as e:
                st.error(f"Error processing report: {e}")

# =============================
# TAB 2 - GROQ
# =============================

with tab2:
    st.header("Groq+Tavily+LiteLLM (Groq for reasoning, Tavily for web search, LiteLLM for token tracking)")

    groq_user_input = st.text_area("Enter your evaluation request (Groq + Tavily)", key="groq_user_input")
    groq_output_dir = st.text_input("Report folder (Groq)", value="reports", key="groq_output_dir")

    if st.button("Run Groq + Tavily"):
        if not st.session_state.get("groq_api_key"):
            st.error("Please add your Groq API Key in the sidebar.")
        elif not st.session_state.get("tavily_api_key"):
            st.error("Please add your Tavily API Key in the sidebar.")
        elif not groq_user_input.strip():
            st.error("Please enter your evaluation request.")
        else:
            with st.spinner("Running Groq with Tavily web search (no CrewAI)..."):
               
                groq_model = "groq/llama-3.3-70b-versatile"
                start_time = time.time()
                result, token_usage = run_groq_direct(
                        groq_user_input,
                        groq_model,
                        st.session_state["groq_api_key"],
                        st.session_state["tavily_api_key"]
                    )
                end_time = time.time()
                time_used = end_time - start_time

                st.subheader("Generated Report")

                try:
                    report = result.json

                    st.subheader(report["title"])

                    for section in report["sections"]:
                        st.markdown(f"### {section['heading']}")
                        content = section.get("content", "No content available.")
                        st.markdown(format_text_for_streamlit(content))

                    st.write(f"Time used: {time_used:.2f} seconds")

                    st.subheader("📊 Token Usage (per LLM call)")
                    per_call = token_usage.get("per_call", [])
                    if per_call:
                        st.table(per_call)
                    else:
                        st.warning("No per-call token data captured.")

                    st.subheader("🔢 Totals")
                    totals = token_usage.get("totals", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                    col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                    col3.metric("Total Tokens", totals.get("total_tokens", 0))

                    perf = token_usage.get("performance", {})
                    if perf:
                        st.subheader("⚡ Performance")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                        p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                        p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                    token_usage_str = (
                        f"Prompt: {totals.get('prompt_tokens', 0)}, "
                        f"Completion: {totals.get('completion_tokens', 0)}, "
                        f"Total: {totals.get('total_tokens', 0)}"
                    )
                    pdf_path = save_report_pdf("groq", report, time_used, token_usage_str, groq_output_dir, token_summary=token_usage)
                    st.success(f"Report saved to: {pdf_path}")
                    
                    # Log the request
                    log_request("Groq+Tavily+LiteLLM", {groq_model}, token_usage)

                except Exception as e:
                    st.error(f"Error processing report: {e}")

# =============================
# TAB 3 - RAG
# =============================

with tab3:
    st.header("RAG Pipeline with LangChain, ChromaDB & Ollama")
    
    st.info("📚 This tab uses RAG to find answers based on documents in the 'rag_docs' folder. Add .txt or .pdf files there.")
    
    # Two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        docs_folder = st.text_input("Documents folder", value="rag_docs", key="rag_docs_folder")
    
    with col2:
        rag_output_dir = st.text_input("Report folder", value="reports", key="rag_output_dir")
    
    # Query input
    rag_query = st.text_area("Ask a question about the documents", key="rag_query", height=100)
    
    # Run button
    if st.button("Run RAG Pipeline"):
        if not rag_query.strip():
            st.error("Please enter your question.")
        else:
            with st.spinner("Setting up RAG pipeline and retrieving documents..."):
                start_time = time.time()
                
                try:
                    result, token_usage = run_rag(rag_query, docs_folder)
                    end_time = time.time()
                    time_used = end_time - start_time
                    
                    st.subheader("📊 RAG Analysis Results")
                    
                    try:
                        report = result.json
                        
                        st.subheader(report["title"])
                        
                        for section in report["sections"]:
                            st.markdown(f"### {section['heading']}")
                            content = section.get("content", "No content available.")
                            st.markdown(format_text_for_streamlit(content))
                        st.write(f"⏱️ Time used: {time_used:.2f} seconds")
                        st.write(f"📄 Documents retrieved: {result.num_docs_retrieved}")

                        st.subheader("📊 Token Usage (per LLM call)")
                        per_call = token_usage.get("per_call", [])
                        if per_call:
                            st.table(per_call)
                        else:
                            st.warning("No per-call token data captured.")

                        st.subheader("🔢 Totals")
                        totals = token_usage.get("totals", {})
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                        col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                        col3.metric("Total Tokens", totals.get("total_tokens", 0))

                        perf = token_usage.get("performance", {})
                        if perf:
                            st.subheader("⚡ Performance")
                            p1, p2, p3 = st.columns(3)
                            p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                            p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                            p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                        token_usage_str = (
                            f"Docs retrieved: {result.num_docs_retrieved} | "
                            f"Prompt: {totals.get('prompt_tokens', 0)}, "
                            f"Completion: {totals.get('completion_tokens', 0)}, "
                            f"Total: {totals.get('total_tokens', 0)}"
                        )
                        # Save PDF
                        pdf_path = save_report_pdf("rag", report, time_used, token_usage_str, rag_output_dir, token_summary=token_usage)
                        st.success(f"✅ Report saved to: {pdf_path}")
                        
                        # Log the request
                        log_request("RAG+LangChain+Ollama", RAG_OLLAMA_MODEL, token_usage)
                    
                    except Exception as e:
                        st.error(f"Error processing report: {e}")
                
                except ValueError as e:
                    st.error(f"⚠️ {str(e)}")
                except Exception as e:
                    st.error(f"❌ Pipeline Error: {str(e)}")
                    st.info("Make sure Ollama is running with models {CREW_OLLAMA_MODEL} and {OLLAMA_EMBED_MODEL}")
                


# =============================
# TAB 4 - LANGGRAPH + LANGSMITH
# =============================

with tab4:
    st.header("🔗 LangGraph Multi-Agent Workflow + LangSmith Monitoring")
    st.info(
        "LangGraph builds stateful, multi-step agentic workflows. "
        "LangSmith provides tracing and debugging for each workflow step."
    )

    langgraph_output_dir = st.text_input("Report folder", value="reports", key="lg_output_dir")
    langgraph_user_input = st.text_area(
        "Enter your evaluation request",
        key="lg_user_input",
        height=120,
        value="Evaluate model performance using key ML/DL metrics"
    )

    if st.button("Run LangGraph + LangSmith", key="run_langgraph"):
        if not langgraph_user_input.strip():
            st.error("Please enter your evaluation request.")
        else:
            with st.spinner("Running LangGraph workflow with LangSmith tracing..."):
                import time as _time
                start_time = _time.time()
                try:
                    report, metrics, explanation, token_summary = run_langgraph_workflow(
                        langgraph_user_input,
                        get_lmstudio_llm()
                    )
                    end_time = _time.time()
                    time_used = end_time - start_time

                    st.subheader(report.get("title", "LangGraph Report"))

                    for section in report.get("sections", []):
                        st.markdown(f"### {section['heading']}")
                        content = section.get("content", "No content available.")
                        st.markdown(format_text_for_streamlit(content))

                    st.write(f"⏱️ Time used: {time_used:.2f} seconds")

                    # ── Token usage per call ──────────────
                    st.subheader("📊 Token Usage (per LLM call)")
                    per_call = token_summary.get("per_call", [])
                    if per_call:
                        st.table(per_call)
                    else:
                        st.warning("No per-call token data captured.")

                    st.subheader("🔢 Totals")
                    totals = token_summary.get("totals", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                    col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                    col3.metric("Total Tokens", totals.get("total_tokens", 0))

                    perf = token_summary.get("performance", {})
                    if perf:
                        st.subheader("⚡ Performance")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                        p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                        p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                    st.markdown(f"**Workflow:** LangGraph Multi-Agent Pipeline")

                    pdf_path = save_report_pdf("langgraph", report, time_used, "LangGraph + LangSmith Workflow", langgraph_output_dir, token_summary=token_summary)
                    st.success(f"✅ Report saved to: {pdf_path}")
                    
                    # Log the request
                    log_request("LangGraph", LMSTUDIO_MODEL, token_summary)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Check that LangGraph and LangSmith are properly configured.")

# =============================
# TAB 5 - CREW TEST TOKEN
# =============================

with tab5:
    st.header("🧪 Crew/Ollama Test Token – Token Usage Tracker")
    st.info("Runs a single-agent CrewAI task and reports token usage captured via the `@after_llm_call` hook.")

    if st.button("Run Token Test", key="run_token_test"):
        with st.spinner("Running crew and collecting token data..."):
            try:
                start_time = time.time()
                result, token_summary = run_testToken(get_llm())
                end_time = time.time()
                time_used = end_time - start_time

                st.success(f"Completed in {time_used:.2f} seconds")

                st.subheader("📝 Agent Output(question:'Summarize the latest AI trends')")
                st.markdown(str(result))

                st.subheader("📊 Token Usage (per LLM call)")
                per_call = token_summary.get("per_call", [])
                if per_call:
                    st.table(per_call)
                else:
                    st.warning("No per-call token data captured.")

                st.subheader("🔢 Totals")
                totals = token_summary.get("totals", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                col3.metric("Total Tokens", totals.get("total_tokens", 0))

                perf = token_summary.get("performance", {})
                if perf:
                    st.subheader("⚡ Performance")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                    p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                    p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                st.markdown(f"**Model:** `{CREW_OLLAMA_MODEL}`")
                
                # Log the request
                log_request("CrewAI+Test+Token+Ollama", CREW_OLLAMA_MODEL, token_summary)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# =============================
# TAB 6 - LANGCHAIN + LM STUDIO
# =============================

with tab6:
    st.header("🖥️ LangChain + LM Studio (Local OpenAI-compatible server)")
    st.info(
        "Make sure **LM Studio** is running with the local server enabled and a model loaded. "
        "Default endpoint: `http://localhost:1234/v1`"
    )

    col1, col2 = st.columns(2)
    with col1:
        lms_base_url = st.text_input("LM Studio base URL", value=LMSTUDIO_BASE_URL, key="lms_base_url")
    with col2:
        lms_model = st.text_input("Model name (as shown in LM Studio)", value=LMSTUDIO_MODEL, key="lms_model")

    lms_output_dir = st.text_input("Report folder", value="reports", key="lms_output_dir")
    lms_user_input = st.text_area(
        "Enter your evaluation request",
        key="lms_user_input",
        height=120,
    )

    if st.button("Run LangChain + LM Studio", key="run_lmstudio"):
        if not lms_user_input.strip():
            st.error("Please enter your evaluation request.")
        else:
            with st.spinner("Running LangChain pipeline against LM Studio..."):
                import time as _time
                start_time = _time.time()
                try:
                    report, metrics, explanation, token_summary = run_lmstudio_lchain(
                        lms_user_input,
                        get_lmstudio_llm()
                    )
                    end_time = _time.time()
                    time_used = end_time - start_time

                    st.subheader(report.get("title", "Report"))

                    for section in report.get("sections", []):
                        st.markdown(f"### {section['heading']}")
                        content = section.get("content", "No content available.")
                        st.markdown(format_text_for_streamlit(content))

                    st.write(f"⏱️ Time used: {time_used:.2f} seconds")

                    # ── Token usage per call (mirrors Tab 5) ──────────────
                    st.subheader("📊 Token Usage (per LLM call)")
                    per_call = token_summary.get("per_call", [])
                    if per_call:
                        st.table(per_call)
                    else:
                        st.warning("No per-call token data captured.")

                    st.subheader("🔢 Totals")
                    totals = token_summary.get("totals", {})
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                    col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                    col3.metric("Total Tokens", totals.get("total_tokens", 0))

                    perf = token_summary.get("performance", {})
                    if perf:
                        st.subheader("⚡ Performance")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                        p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                        p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                    st.markdown(f"**Model:** `{lms_model}`")

                    pdf_path = save_report_pdf("lmstudio", report, time_used, f"LM Studio | model: {lms_model}", lms_output_dir, token_summary=token_summary)
                    st.success(f"✅ Report saved to: {pdf_path}")
                    
                    # Log the request
                    log_request("LangChain+LMStudio", lms_model, token_summary)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Ensure LM Studio is running with the server enabled and a model is loaded.")

# =============================
# TAB 7 - LM STUDIO TOKEN TEST
# =============================

with tab7:
    st.header("🧪 LMStudio + TokenTest – Token Usage Tracker")
    st.info("Runs a single LangChain task against LM Studio and reports token usage.")

    col1, col2 = st.columns(2)
    with col1:
        lms_token_base_url = st.text_input("LM Studio base URL", value=LMSTUDIO_BASE_URL, key="lms_token_base_url")
    with col2:
        lms_token_model = st.text_input("Model name (as shown in LM Studio)", value=LMSTUDIO_MODEL, key="lms_token_model")

    if st.button("Run LM Studio Token Test", key="run_lmstudio_token_test"):
        with st.spinner("Running LangChain task against LM Studio and collecting token data..."):
            try:
                start_time = time.time()
                llm_studio = get_lmstudio_llm(base_url=lms_token_base_url, model=lms_token_model)
                result, token_summary = run_lmstudio_token_test(llm_studio)
                end_time = time.time()
                time_used = end_time - start_time

                st.success(f"Completed in {time_used:.2f} seconds")

                st.subheader("📝 Agent Output(question:'Summarize the latest AI trends in 5 paragraphs')")
                st.markdown(str(result))

                st.subheader("📊 Token Usage (per LLM call)")
                per_call = token_summary.get("per_call", [])
                if per_call:
                    st.table(per_call)
                else:
                    st.warning("No per-call token data captured.")

                st.subheader("🔢 Totals")
                totals = token_summary.get("totals", {})
                col1, col2, col3 = st.columns(3)
                col1.metric("Prompt Tokens", totals.get("prompt_tokens", 0))
                col2.metric("Completion Tokens", totals.get("completion_tokens", 0))
                col3.metric("Total Tokens", totals.get("total_tokens", 0))

                perf = token_summary.get("performance", {})
                if perf:
                    st.subheader("⚡ Performance")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("Elapsed (s)", perf.get("elapsed_seconds", 0))
                    p2.metric("Completion tok/s", perf.get("completion_tokens_per_sec", 0))
                    p3.metric("Total tok/s", perf.get("total_tokens_per_sec", 0))

                st.markdown(f"**Model:** `{lms_token_model}`")
                
                # Log the request
                log_request("LMStudio+TokenTest", lms_token_model, token_summary)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Ensure LM Studio is running with the server enabled and a model is loaded.")

# =============================
# TAB 8 - LOG OF REQUESTS
# =============================

with tab8:
    st.header("📊 Logs of Requests")
    st.info("Display the last 20 requests from all tabs with token usage and performance metrics.")
    
    # Refresh button
    if st.button("🔄 Refresh Logs", key="refresh_logs"):
        st.rerun()
    
    # Read and display logs
    logs_df = read_logs(limit=20)
    
    if logs_df.empty:
        st.warning("No logs found. Run some requests first to see them here.")
    else:
        st.subheader(f"Last {len(logs_df)} Requests")
        
        # Display as table
        st.dataframe(
            logs_df,
            width='stretch',
            height=600,
            hide_index=False
        )
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", len(logs_df))
        with col2:
            st.metric("Avg Total Tokens", int(logs_df["total_tokens"].mean()))
        with col3:
            st.metric("Avg Elapsed (s)", f"{logs_df['elapsed_seconds'].mean():.2f}")
        with col4:
            st.metric("Avg tok/s", f"{logs_df['total_tokens_per_sec'].mean():.2f}")
        
        # Breakdown by tab
        st.subheader("📋 Requests by Tab")
        tab_counts = logs_df["tab_name"].value_counts()
        st.bar_chart(tab_counts)
        
        # Token usage trends
        st.subheader("📊 Token Usage Trends")
        st.line_chart(logs_df[["total_tokens", "prompt_tokens", "completion_tokens"]])
        
        # Performance trends
        st.subheader("⚡ Performance Trends")
        st.line_chart(logs_df[["elapsed_seconds", "total_tokens_per_sec"]])
        
        # Export button
        csv_data = logs_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Logs as CSV",
            data=csv_data,
            file_name=f"request_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

