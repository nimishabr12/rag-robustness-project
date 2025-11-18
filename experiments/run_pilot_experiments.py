"""
Pilot Experiment Runner - Small Subset Test

Runs experiments with 20 queries per noise type to verify everything works.
"""
import os
import sys
import json
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline, initialize_rag_pipeline
from src.answer_generator import AnswerGenerator
from src.evaluation import RAGEvaluator


def main():
    """
    Run pilot experiments with small subset
    """
    print("="*80)
    print("RAG ROBUSTNESS PILOT EXPERIMENT")
    print("Testing with 20 queries per noise type")
    print("="*80)
    print(f"\nStarting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize components
    print("Initializing RAG pipeline...")
    try:
        pipeline = initialize_rag_pipeline()
        print("✓ Pipeline initialized")
    except Exception as e:
        print(f"✗ Error initializing pipeline: {e}")
        return

    print("\nInitializing answer generator...")
    try:
        answer_generator = AnswerGenerator()
        print("✓ Answer generator initialized")
    except Exception as e:
        print(f"✗ Error initializing answer generator: {e}")
        return

    print("\nInitializing evaluator...")
    try:
        evaluator = RAGEvaluator()
        evaluator.load_ground_truth()
        print("✓ Evaluator initialized")
    except Exception as e:
        print(f"✗ Error initializing evaluator: {e}")
        return

    # Define test parameters
    noise_types = ['clean', 'noisy', 'ambiguous', 'context_dependent', 'adversarial']
    strategies = ['naive_retrieval', 'query_rewrite_retrieval', 'hybrid_retrieval', 'multistep_retrieval']
    max_queries_per_type = 20

    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'max_queries_per_type': max_queries_per_type,
            'noise_types': noise_types,
            'retrieval_strategies': strategies
        },
        'detailed_results': {},
        'summary': {}
    }

    overall_start = time.time()

    # Run experiments for each noise type
    for noise_type in noise_types:
        print(f"\n{'='*80}")
        print(f"NOISE TYPE: {noise_type.upper()}")
        print(f"{'='*80}\n")

        # Load queries
        filepath = f'data/{noise_type}_queries.json'
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                queries = json.load(f)

            # Limit to first N queries
            queries = queries[:max_queries_per_type]
            print(f"Loaded {len(queries)} {noise_type} queries")
        except Exception as e:
            print(f"Error loading queries: {e}")
            continue

        all_results['detailed_results'][noise_type] = {}

        # Test each retrieval strategy
        for strategy in strategies:
            print(f"\n{'-'*80}")
            print(f"Strategy: {strategy}")
            print(f"{'-'*80}")

            strategy_results = []
            start_time = time.time()

            for i, query_data in enumerate(queries):
                query_id = query_data['query_id']
                query = query_data['query']

                print(f"  [{i+1}/{len(queries)}] Query {query_id}: {query[:60]}...")

                try:
                    # Retrieve passages
                    if strategy == 'naive_retrieval':
                        retrieved_passages = pipeline.naive_retrieval(query, top_k=5)
                    elif strategy == 'query_rewrite_retrieval':
                        retrieved_passages = pipeline.query_rewrite_retrieval(query, top_k=5)
                    elif strategy == 'hybrid_retrieval':
                        retrieved_passages = pipeline.hybrid_retrieval(query, top_k=5)
                    elif strategy == 'multistep_retrieval':
                        retrieved_passages = pipeline.multistep_retrieval(query, top_k=5)
                    else:
                        print(f"    ✗ Unknown strategy: {strategy}")
                        continue

                    print(f"      Retrieved {len(retrieved_passages)} passages")

                    # Generate answer
                    answer_result = answer_generator.generate_answer(
                        query=query,
                        passages=retrieved_passages
                    )
                    print(f"      Generated answer ({answer_result.get('confidence', 'unknown')} confidence)")

                    # Evaluate
                    eval_result = evaluator.evaluate_answer(
                        noise_type=noise_type,
                        retrieval_strategy=strategy,
                        query_id=query_id,
                        generated_answer=answer_result['answer'],
                        retrieved_passages=retrieved_passages
                    )

                    precision = eval_result['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    print(f"      Precision@5: {precision:.3f}")

                    # Store result
                    result = {
                        'query_id': query_id,
                        'query': query,
                        'query_type': query_data.get('query_type', 'unknown'),
                        'original_query': query_data.get('original_query', query),
                        'generated_answer': answer_result['answer'],
                        'answer_confidence': answer_result.get('confidence', 'unknown'),
                        'evaluation': eval_result
                    }
                    strategy_results.append(result)
                    print(f"      ✓ Success")

                except Exception as e:
                    print(f"      ✗ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            elapsed = time.time() - start_time
            print(f"\n  Completed {len(strategy_results)}/{len(queries)} queries in {elapsed:.2f}s")
            print(f"  Average: {elapsed/max(len(strategy_results), 1):.2f}s per query")

            all_results['detailed_results'][noise_type][strategy] = strategy_results

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*80}")
    print(f"PILOT EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {overall_elapsed/60:.2f} minutes")

    # Generate summary
    print("\nGenerating summary...")
    all_results['summary'] = evaluator.generate_summary_statistics()

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    evaluator.print_summary()

    # Save results
    output_path = 'results/pilot_results.json'
    os.makedirs('results', exist_ok=True)

    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✓ Results saved successfully!")

    print("\n" + "="*80)
    print("Pilot experiment completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
