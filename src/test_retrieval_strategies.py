"""
Test script for RAG retrieval strategies
"""
import sys
from rag_pipeline import initialize_rag_pipeline

# Function to safely print unicode text
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))


def print_results(query: str, results: list, strategy_name: str):
    """
    Print retrieval results in a formatted way

    Args:
        query: The query used
        results: List of retrieved passages
        strategy_name: Name of the strategy used
    """
    print("\n" + "="*80)
    safe_print(f"Strategy: {strategy_name}")
    safe_print(f"Query: {query}")

    # Check if query was rewritten
    if results and 'rewritten_query' in results[0]:
        safe_print(f"Rewritten Query: {results[0]['rewritten_query']}")

    # Check if multistep
    if results and 'steps' in results[0]:
        print(f"Steps: {results[0]['steps']}")

    print("-"*80)

    for i, result in enumerate(results):
        print(f"\nRank {i+1} | Score: {result['score']:.4f}")

        # Show additional scores for hybrid retrieval
        if 'embedding_score' in result and 'bm25_score' in result:
            print(f"  Embedding Score: {result['embedding_score']:.4f} | BM25 Score: {result['bm25_score']:.4f}")

        safe_print(f"  Passage: {result['passage']['text'][:200]}...")
        print(f"  Query ID: {result['passage']['query_id']}")


def main():
    print("Initializing RAG Pipeline...")
    print("="*80)

    # Initialize pipeline (will build or load indexes)
    pipeline = initialize_rag_pipeline(force_rebuild=False)

    print("\n" + "="*80)
    print("TESTING RETRIEVAL STRATEGIES")
    print("="*80)

    # Test queries - mix of clean and noisy queries
    test_queries = [
        {
            'name': 'Clean Query 1',
            'query': 'what is cost of sales'
        },
        {
            'name': 'Noisy Query 1 (with typos)',
            'query': 'what is ckst of sals'
        },
        {
            'name': 'Clean Query 2',
            'query': 'how much does it cost to park at tampa airport'
        },
        {
            'name': 'Ambiguous Query',
            'query': 'PC'  # Could mean Personal Computer, politically correct, etc.
        },
        {
            'name': 'Context-dependent Query',
            'query': 'How does it work?'
        }
    ]

    top_k = 3  # Retrieve top 3 passages for faster testing

    # Test each query with each strategy
    for test_query in test_queries:
        print("\n\n" + "#"*80)
        safe_print(f"# TEST: {test_query['name']}")
        safe_print(f"# Query: {test_query['query']}")
        print("#"*80)

        query = test_query['query']

        # Strategy 1: Naive Retrieval
        try:
            print("\n[1/4] Running Naive Retrieval...")
            results = pipeline.naive_retrieval(query, top_k=top_k)
            print_results(query, results, "Naive Retrieval")
        except Exception as e:
            print(f"Error in naive retrieval: {e}")

        # Strategy 2: Query Rewrite Retrieval
        try:
            print("\n[2/4] Running Query Rewrite Retrieval...")
            results = pipeline.query_rewrite_retrieval(query, top_k=top_k)
            print_results(query, results, "Query Rewrite Retrieval")
        except Exception as e:
            print(f"Error in query rewrite retrieval: {e}")

        # Strategy 3: Hybrid Retrieval
        try:
            print("\n[3/4] Running Hybrid Retrieval...")
            results = pipeline.hybrid_retrieval(query, top_k=top_k, alpha=0.5)
            print_results(query, results, "Hybrid Retrieval")
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")

        # Strategy 4: Multistep Retrieval
        try:
            print("\n[4/4] Running Multistep Retrieval...")
            results = pipeline.multistep_retrieval(query, top_k=top_k)
            print_results(query, results, "Multistep Retrieval")
        except Exception as e:
            print(f"Error in multistep retrieval: {e}")

        print("\n" + "="*80)
        print("All strategies completed for this query")
        print("="*80)

    print("\n\n" + "#"*80)
    print("# ALL TESTS COMPLETED")
    print("#"*80)
    print("\nSummary:")
    print(f"  Tested {len(test_queries)} queries")
    print("  Strategies tested:")
    print("    1. Naive Retrieval - Simple embedding similarity")
    print("    2. Query Rewrite Retrieval - GPT-based query cleaning")
    print("    3. Hybrid Retrieval - Embedding + BM25 combination")
    print("    4. Multistep Retrieval - Iterative retrieval with analysis")


if __name__ == "__main__":
    main()
