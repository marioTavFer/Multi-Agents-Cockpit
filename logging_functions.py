"""
Logging utility functions for request tracking and analysis.
"""

import datetime
import csv
from pathlib import Path
import pandas as pd
import streamlit as st


def ensure_logs_folder():
    """Ensure logs folder exists."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def log_request(tab_name: str, model: str, token_summary: dict):
    """Log a request to CSV file."""
    logs_dir = ensure_logs_folder()
    csv_file = logs_dir / "request_logs.csv"
    
    # Extract token data
    totals = token_summary.get("totals", {})
    perf = token_summary.get("performance", {})
    
    log_entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tab_name": tab_name,
        "Model": model,
        "prompt_tokens": totals.get("prompt_tokens", 0),
        "completion_tokens": totals.get("completion_tokens", 0),
        "total_tokens": totals.get("total_tokens", 0),
        "elapsed_seconds": perf.get("elapsed_seconds", 0),
        "completion_tokens_per_sec": perf.get("completion_tokens_per_sec", 0),
        "total_tokens_per_sec": perf.get("total_tokens_per_sec", 0),
    }
    
    # Write to CSV
    file_exists = csv_file.exists()
    try:
        with open(csv_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    except Exception as e:
        st.warning(f"Could not log request: {e}")


def read_logs(limit: int = 20) -> pd.DataFrame:
    """Read the last N log entries."""
    logs_dir = ensure_logs_folder()
    csv_file = logs_dir / "request_logs.csv"
    
    if not csv_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        return df.tail(limit).reset_index(drop=True)
    except Exception as e:
        st.error(f"Could not read logs: {e}")
        return pd.DataFrame()
