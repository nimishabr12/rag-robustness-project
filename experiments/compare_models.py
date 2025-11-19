"""
Compare Different LLM Models on RAG Robustness

This script tests OpenAI, Google Gemini, and Anthropic Claude models
on the same RAG robustness experiments to compare their performance.
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
from tqdm import tqdm
from src.rag_pipeline import initialize_rag_pipeline
from src.multi_model_generator import MultiModelGenerator
from src.evaluation import RAGEvaluator
from src.noise_generators import add_typos, make_ambiguous, make_context_dependent, make_adversarial

# Configuration
NUM_QUERIES_PER_TYPE = 10  # Small pilot comparison
TOP_K = 5
MODELS_TO_TEST = [
    {"provider": "openai", "model": None},  # gpt-3.5-turbo
    # {"provider": "gemini", "model": None},  # gemini-pro - Skip due to SSL issues in sandbox
    {"provider": "anthropic", "model": None},  # claude-3-haiku
]


def load_test_queries(num_per_type=10):
    """Load test queries for model comparison"""
    with open('data/ms_marco_sample.json', 'r') as f:
        all_data = json.load(f)

    # Extract query info and get subset for quick comparison
    test_queries = []
    for item in all_data[:num_per_type]:
        test_queries.append({
            'query_id': item['query_id'],
            'query': item['query'],
            'query_type': item.get('query_type', 'unknown')
        })

    return test_queries


def apply_noise(query, noise_type):
    """Apply noise transformation"""
    if noise_type == 'clean':
        return query
    elif noise_type == 'noisy':
        return add_typos(query, typo_rate=0.15)
    elif noise_type == 'ambiguous':
        return make_ambiguous(query)
    elif noise_type == 'context_dependent':
        return make_context_dependent(query)
    elif noise_type == 'adversarial':
        return make_adversarial(query)
    return query


def run_model_comparison():
    """Run comparison experiment across all models"""

    print("="*80)
    print("MULTI-MODEL RAG ROBUSTNESS COMPARISON")
    print("="*80)

    # Initialize components
    print("\nInitializing RAG pipeline...")
    pipeline = initialize_rag_pipeline()

    print("Loading evaluator...")
    evaluator = RAGEvaluator()
    evaluator.load_ground_truth()

    # Load test queries
    print(f"\nLoading {NUM_QUERIES_PER_TYPE} test queries...")
    test_queries = load_test_queries(NUM_QUERIES_PER_TYPE)

    # Results storage
    results = {
        'config': {
            'num_queries': NUM_QUERIES_PER_TYPE,
            'top_k': TOP_K,
            'models_tested': MODELS_TO_TEST,
            'noise_types': ['clean', 'noisy', 'ambiguous', 'context_dependent', 'adversarial'],
            'retrieval_strategy': 'hybrid_retrieval',  # Use hybrid for all tests
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'model_results': {}
    }

    noise_types = ['clean', 'noisy', 'ambiguous', 'context_dependent', 'adversarial']

    # Test each model
    for model_config in MODELS_TO_TEST:
        provider = model_config['provider']
        model_name = model_config['model']

        print(f"\n{'='*80}")
        print(f"Testing {provider.upper()}")
        print(f"{'='*80}")

        # Try to initialize model
        try:
            generator = MultiModelGenerator(provider=provider, model=model_name)
            print(f"✓ Model initialized: {generator.model}")
        except Exception as e:
            print(f"✗ Failed to initialize {provider}: {str(e)}")
            results['model_results'][provider] = {
                'status': 'failed',
                'error': str(e)
            }
            continue

        # Test on each noise type
        model_results = {
            'status': 'success',
            'model': f"{provider}/{generator.model}",
            'noise_type_results': {}
        }

        for noise_type in noise_types:
            print(f"\n  Testing noise type: {noise_type}")

            noise_results = {
                'queries_tested': 0,
                'queries_succeeded': 0,
                'queries_failed': 0,
                'total_precision_at_k': 0.0,
                'total_answer_quality': 0.0,
                'errors': []
            }

            # Test each query
            for query_data in tqdm(test_queries, desc=f"  {noise_type}", leave=False):
                query = query_data['query']
                query_id = query_data['query_id']

                try:
                    # Apply noise transformation
                    transformed_query = apply_noise(query, noise_type)

                    # Retrieve passages (using hybrid retrieval)
                    passages = pipeline.hybrid_retrieval(transformed_query, top_k=TOP_K)

                    # Generate answer with this model
                    answer_result = generator.generate_answer(
                        query=transformed_query,
                        passages=passages,
                        temperature=0.3,
                        max_tokens=300
                    )

                    # Check if answer generation succeeded
                    if 'error' in answer_result:
                        noise_results['queries_failed'] += 1
                        noise_results['errors'].append({
                            'query_id': query_id,
                            'error': answer_result['error']
                        })
                        continue

                    # Evaluate
                    precision_metrics = evaluator.calculate_precision_at_k(
                        passages, query_id, k=TOP_K
                    )

                    answer_quality = evaluator.calculate_answer_quality(
                        answer_result['answer'], query_id
                    )

                    # Accumulate metrics
                    noise_results['queries_tested'] += 1
                    noise_results['queries_succeeded'] += 1
                    noise_results['total_precision_at_k'] += precision_metrics['precision_at_k']

                    if answer_quality['similarity_score'] is not None:
                        noise_results['total_answer_quality'] += answer_quality['similarity_score']

                except Exception as e:
                    noise_results['queries_failed'] += 1
                    noise_results['errors'].append({
                        'query_id': query_id,
                        'error': str(e)
                    })

            # Calculate averages
            if noise_results['queries_succeeded'] > 0:
                noise_results['avg_precision_at_k'] = (
                    noise_results['total_precision_at_k'] / noise_results['queries_succeeded']
                )
                noise_results['avg_answer_quality'] = (
                    noise_results['total_answer_quality'] / noise_results['queries_succeeded']
                )
            else:
                noise_results['avg_precision_at_k'] = 0.0
                noise_results['avg_answer_quality'] = 0.0

            # Store results
            model_results['noise_type_results'][noise_type] = noise_results

            # Print summary
            print(f"    Succeeded: {noise_results['queries_succeeded']}/{NUM_QUERIES_PER_TYPE}")
            if noise_results['queries_succeeded'] > 0:
                print(f"    Avg P@{TOP_K}: {noise_results['avg_precision_at_k']:.3f}")
                print(f"    Avg Answer Quality: {noise_results['avg_answer_quality']:.3f}")

        # Store model results
        results['model_results'][provider] = model_results

    # Save results
    output_file = 'results/model_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print(f"\n{'Model':<20} {'Status':<10} {'Clean':<10} {'Noisy':<10} {'Ambiguous':<10} {'Context':<10} {'Adversarial':<10}")
    print("-"*80)

    for provider in MODELS_TO_TEST:
        prov_name = provider['provider']
        if prov_name not in results['model_results']:
            continue

        model_res = results['model_results'][prov_name]

        if model_res['status'] == 'failed':
            print(f"{prov_name:<20} {'FAILED':<10} Error: {model_res.get('error', 'Unknown')}")
            continue

        model_display = model_res['model'][:18]
        status = model_res['status']

        # Get P@K for each noise type
        noise_scores = []
        for noise_type in ['clean', 'noisy', 'ambiguous', 'context_dependent', 'adversarial']:
            noise_res = model_res['noise_type_results'].get(noise_type, {})
            score = noise_res.get('avg_precision_at_k', 0.0)
            noise_scores.append(f"{score:.3f}")

        print(f"{model_display:<20} {status:<10} {noise_scores[0]:<10} {noise_scores[1]:<10} {noise_scores[2]:<10} {noise_scores[3]:<10} {noise_scores[4]:<10}")

    print("\n" + "="*80)
    print("Analysis complete! Check results/model_comparison_results.json for details.")
    print("="*80)

    return results


if __name__ == "__main__":
    start_time = time.time()

    try:
        results = run_model_comparison()

        elapsed_time = time.time() - start_time
        print(f"\nTotal time: {elapsed_time/60:.2f} minutes")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nExperiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
