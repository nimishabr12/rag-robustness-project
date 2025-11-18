"""
Verify RAG pipeline setup
"""
import os
from dotenv import load_dotenv

def verify_setup():
    """
    Verify that the RAG pipeline is properly configured
    """
    print("Verifying RAG Pipeline Setup...")
    print("="*80)

    # Check for .env file
    if os.path.exists('.env'):
        print("[OK] .env file found")
        load_dotenv()
    else:
        print("[WARNING] .env file not found")
        print("  Please create a .env file from .env.example:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your OpenAI API key to the OPENAI_API_KEY variable")
        return False

    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"[OK] OpenAI API key found (length: {len(api_key)})")
    else:
        print("[ERROR] OPENAI_API_KEY not set in .env file")
        print("  Please add your OpenAI API key to .env:")
        print("  OPENAI_API_KEY=your_api_key_here")
        return False

    # Check for MS MARCO sample data
    if os.path.exists('data/ms_marco_sample.json'):
        print("[OK] MS MARCO sample data found")
    else:
        print("[ERROR] MS MARCO sample data not found")
        print("  Please run: python src/download_msmarco.py")
        return False

    # Check required packages
    try:
        import openai
        print("[OK] openai package installed")
    except ImportError:
        print("[ERROR] openai package not installed")
        print("  Please run: pip install -r requirements.txt")
        return False

    try:
        import faiss
        print("[OK] faiss-cpu package installed")
    except ImportError:
        print("[ERROR] faiss-cpu package not installed")
        print("  Please run: pip install -r requirements.txt")
        return False

    try:
        from rank_bm25 import BM25Okapi
        print("[OK] rank-bm25 package installed")
    except ImportError:
        print("[ERROR] rank-bm25 package not installed")
        print("  Please run: pip install -r requirements.txt")
        return False

    print("="*80)
    print("[SUCCESS] All checks passed! Ready to run RAG pipeline.")
    print("\nTo test the pipeline, run:")
    print("  python src/test_retrieval_strategies.py")
    return True


if __name__ == "__main__":
    verify_setup()
