"""
Experiment Runner for RAG Robustness Project

Runs comprehensive experiments across all noise types and retrieval strategies.
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag_pipeline import RAGPipeline, initialize_rag_pipeline
from src.answer_generator import AnswerGenerator
from src.evaluation import RAGEvaluator


class ExperimentRunner:
    """
    Runs experiments across different noise types and retrieval strategies
    """

    def __init__(self):
        """Initialize the experiment runner"""
        print("="*80)
        print("RAG ROBUSTNESS EXPERIMENT RUNNER")
        print("="*80)
        print(f"\nStarting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Initialize components
        print("Initializing RAG pipeline...")
        self.pipeline = initialize_rag_pipeline()

        print("Initializing answer generator...")
        self.answer_generator = AnswerGenerator()

        print("Initializing evaluator...")
        self.evaluator = RAGEvaluator()
        self.evaluator.load_ground_truth()

        # Define noise types and strategies
        self.noise_types = [
            'clean',
            'noisy',
            'ambiguous',
            'context_dependent',
            'adversarial'
        ]

        self.retrieval_strategies = [
            'naive_retrieval',
            'query_rewrite_retrieval',
            'hybrid_retrieval',
            'multistep_retrieval'
        ]

        # Storage for all results
        self.all_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'noise_types': self.noise_types,
                'retrieval_strategies': self.retrieval_strategies
            },
            'detailed_results': {},
            'examples': {}
        }

    def load_query_variants(self, noise_type: str) -> List[Dict]:
        """
        Load query variants for a specific noise type

        Args:
            noise_type: Type of noise (clean, noisy, ambiguous, etc.)

        Returns:
            List of query dictionaries
        """
        filepath = f'data/{noise_type}_queries.json'

        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            queries = json.load(f)

        print(f"Loaded {len(queries)} {noise_type} queries")
        return queries

    def run_retrieval(self, query: str, strategy: str, top_k: int = 5) -> List[Dict]:
        """
        Run retrieval using specified strategy

        Args:
            query: Query string
            strategy: Retrieval strategy name
            top_k: Number of results to retrieve

        Returns:
            List of retrieved passages
        """
        if strategy == 'naive_retrieval':
            return self.pipeline.naive_retrieval(query, top_k=top_k)
        elif strategy == 'query_rewrite_retrieval':
            return self.pipeline.query_rewrite_retrieval(query, top_k=top_k)
        elif strategy == 'hybrid_retrieval':
            return self.pipeline.hybrid_retrieval(query, top_k=top_k)
        elif strategy == 'multistep_retrieval':
            return self.pipeline.multistep_retrieval(query, top_k=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def run_single_experiment(
        self,
        noise_type: str,
        strategy: str,
        query_data: Dict
    ) -> Dict:
        """
        Run a single experiment for one query

        Args:
            noise_type: Type of noise
            strategy: Retrieval strategy
            query_data: Query dictionary with id, query, type

        Returns:
            Experiment result dictionary
        """
        query_id = query_data['query_id']
        query = query_data['query']

        # Retrieve passages
        retrieved_passages = self.run_retrieval(query, strategy)

        # Generate answer
        answer_result = self.answer_generator.generate_answer(
            query=query,
            passages=retrieved_passages
        )

        # Evaluate
        eval_result = self.evaluator.evaluate_answer(
            noise_type=noise_type,
            retrieval_strategy=strategy,
            query_id=query_id,
            generated_answer=answer_result['answer'],
            retrieved_passages=retrieved_passages
        )

        # Combine all information
        result = {
            'query_id': query_id,
            'query': query,
            'query_type': query_data.get('query_type', 'unknown'),
            'original_query': query_data.get('original_query', query),
            'noise_type': noise_type,
            'retrieval_strategy': strategy,
            'retrieved_passages': [
                {
                    'text': p.get('passage', {}).get('text', '')[:200] + '...',
                    'score': p.get('score', 0.0)
                }
                for p in retrieved_passages[:3]  # Store top 3 for inspection
            ],
            'generated_answer': answer_result['answer'],
            'answer_confidence': answer_result.get('confidence', 'unknown'),
            'evaluation': eval_result
        }

        return result

    def run_experiments_for_noise_type(self, noise_type: str):
        """
        Run experiments for all strategies on a single noise type

        Args:
            noise_type: Type of noise to test
        """
        print(f"\n{'='*80}")
        print(f"NOISE TYPE: {noise_type.upper()}")
        print(f"{'='*80}\n")

        # Load queries
        queries = self.load_query_variants(noise_type)

        if not queries:
            print(f"No queries found for {noise_type}, skipping...\n")
            return

        # Initialize results storage for this noise type
        self.all_results['detailed_results'][noise_type] = {}
        self.all_results['examples'][noise_type] = {}

        # Run experiments for each strategy
        for strategy in self.retrieval_strategies:
            print(f"\n{'-'*80}")
            print(f"Strategy: {strategy}")
            print(f"{'-'*80}")

            strategy_results = []
            start_time = time.time()

            for i, query_data in enumerate(queries):
                # Print progress
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Progress: {i+1}/{len(queries)} queries...")

                try:
                    result = self.run_single_experiment(
                        noise_type=noise_type,
                        strategy=strategy,
                        query_data=query_data
                    )
                    strategy_results.append(result)

                except Exception as e:
                    print(f"  Error processing query {query_data['query_id']}: {str(e)}")
                    continue

            elapsed_time = time.time() - start_time
            print(f"\n  Completed {len(strategy_results)} queries in {elapsed_time:.2f} seconds")
            print(f"  Average time per query: {elapsed_time/len(strategy_results):.2f} seconds")

            # Store results
            self.all_results['detailed_results'][noise_type][strategy] = strategy_results

            # Store example failures (queries with low precision)
            failures = [
                r for r in strategy_results
                if r['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0) < 0.3
            ]

            if failures:
                # Store up to 3 example failures
                self.all_results['examples'][noise_type][strategy] = failures[:3]

    def run_all_experiments(self):
        """
        Run experiments across all noise types and strategies
        """
        overall_start = time.time()

        for noise_type in self.noise_types:
            self.run_experiments_for_noise_type(noise_type)

        overall_elapsed = time.time() - overall_start

        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"{'='*80}")
        print(f"\nTotal time: {overall_elapsed/60:.2f} minutes")
        print(f"Experiments completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def print_summary(self):
        """
        Print summary statistics
        """
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80 + "\n")

        self.evaluator.print_summary()

    def save_results(self, output_path: str = 'results/experiment_results.json'):
        """
        Save all results to JSON file

        Args:
            output_path: Path to save results
        """
        print(f"\nSaving results to {output_path}...")

        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Add evaluation summary
        self.all_results['summary'] = self.evaluator.generate_summary_statistics()

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, indent=2, ensure_ascii=False)

        print(f"Results saved successfully!")

        # Also save evaluator results
        eval_output_path = 'results/evaluation_metrics.json'
        self.evaluator.save_results(eval_output_path)

        # Save a human-readable summary
        summary_path = 'results/summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAG ROBUSTNESS EXPERIMENT SUMMARY\n")
            f.write("="*80 + "\n\n")

            summary = self.all_results['summary']

            f.write(f"Timestamp: {self.all_results['metadata']['timestamp']}\n")
            f.write(f"Total Queries: {summary['overall']['total_queries']}\n")
            f.write(f"Noise Types: {summary['overall']['total_noise_types']}\n")
            f.write(f"Retrieval Strategies: {summary['overall']['total_strategies']}\n\n")

            f.write("Overall Performance:\n")
            f.write(f"  Average Precision@5: {summary['overall']['avg_precision_at_5']:.3f}\n")
            f.write(f"  Std Dev: {summary['overall']['std_precision_at_5']:.3f}\n\n")

            f.write("Performance by Strategy:\n")
            f.write("-"*80 + "\n")
            for strategy, stats in sorted(summary['by_strategy'].items()):
                f.write(f"  {strategy}:\n")
                f.write(f"    Avg P@5: {stats['avg_precision_at_5']:.3f}\n")
                f.write(f"    Std Dev: {stats['std_precision_at_5']:.3f}\n\n")

            f.write("Performance by Noise Type:\n")
            f.write("-"*80 + "\n")
            for noise_type, stats in sorted(summary['by_noise_type'].items()):
                f.write(f"  {noise_type}:\n")
                f.write(f"    Overall Avg P@5: {stats['overall_avg_precision_at_5']:.3f}\n\n")

        print(f"Summary saved to {summary_path}")


def main():
    """
    Main entry point for running experiments
    """
    runner = ExperimentRunner()

    # Run all experiments
    runner.run_all_experiments()

    # Print summary
    runner.print_summary()

    # Save results
    runner.save_results()

    print("\n" + "="*80)
    print("Experiment runner completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
