"""
Intent-Aware RAG Question Answering System - Main Pipeline Runner.
------------------------------------------------------------------
Usage:
  python rag_system.py "What is virtual memory?"   # Single query mode
  python rag_system.py                             # Interactive terminal mode
  python rag_system.py --rebuild                   # Rebuilds model & FAISS artifacts
"""

import sys
from pathlib import Path

# Ensure root directory is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_system import IntentAwareRAG, main

if __name__ == "__main__":
    main()
