"""
RAG System Evaluation Framework

Evaluates RAG pipeline performance across different noise types and retrieval strategies.
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv
import statistics

# Load environment variables
load_dotenv()


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems measuring:
    1. Retrieval Precision@k
    2. Answer Quality (semantic similarity)
    3. Failure Analysis
    4. Recovery Rate (for context-dependent queries)
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the evaluator

        Args:
            api_key: OpenAI API key (if None, will load from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"

        # Storage for ground truth data
        self.ground_truth: Dict = {}
        self.results: Dict = defaultdict(lambda: defaultdict(list))

    def load_ground_truth(self, filepath: str = 'data/ms_marco_sample.json'):
        """
        Load ground truth data from MS MARCO sample

        Args:
            filepath: Path to MS MARCO sample JSON file
        """
        print(f"Loading ground truth from {filepath}...")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build ground truth index
        for item in data:
            query_id = item['query_id']
            self.ground_truth[query_id] = {
                'query': item['query'],
                'query_type': item.get('query_type', 'unknown'),
                'answers': item.get('answers', []),
                'relevant_passage_indices': [
                    i for i, is_sel in enumerate(item['passages']['is_selected'])
                    if is_sel == 1
                ],
                'all_passages': item['passages']['passage_text']
            }

        print(f"Loaded ground truth for {len(self.ground_truth)} queries")

    def calculate_precision_at_k(
        self,
        retrieved_passages: List[Dict],
        query_id: int,
        k: int = 5
    ) -> Dict:
        """
        Calculate Precision@k: proportion of retrieved passages that are relevant

        Args:
            retrieved_passages: List of retrieved passage dicts with 'passage' key
            query_id: Query ID to look up ground truth
            k: Number of top results to consider

        Returns:
            Dictionary with precision metrics
        """
        if query_id not in self.ground_truth:
            return {
                'precision_at_k': 0.0,
                'relevant_retrieved': 0,
                'total_retrieved': 0,
                'error': 'query_id not in ground truth'
            }

        gt = self.ground_truth[query_id]
        relevant_indices = set(gt['relevant_passage_indices'])

        # Get the top k retrieved passages
        top_k_passages = retrieved_passages[:k]

        # Match retrieved passages to ground truth by text similarity
        relevant_retrieved = 0

        for passage_dict in top_k_passages:
            # Extract passage text
            if isinstance(passage_dict.get('passage'), dict):
                passage_text = passage_dict['passage'].get('text', '')
            else:
                passage_text = passage_dict.get('text', '')

            # Check if this passage matches any relevant passage
            for idx in relevant_indices:
                if idx < len(gt['all_passages']):
                    gt_passage = gt['all_passages'][idx]
                    # Use exact match or high overlap
                    if self._passages_match(passage_text, gt_passage):
                        relevant_retrieved += 1
                        break

        precision = relevant_retrieved / k if k > 0 else 0.0

        return {
            'precision_at_k': precision,
            'relevant_retrieved': relevant_retrieved,
            'total_retrieved': k,
            'total_relevant': len(relevant_indices)
        }

    def _passages_match(self, passage1: str, passage2: str, threshold: float = 0.9) -> bool:
        """
        Check if two passages match (same content)

        Args:
            passage1: First passage text
            passage2: Second passage text
            threshold: Similarity threshold for matching

        Returns:
            True if passages match
        """
        # Simple exact match or high text overlap
        if passage1 == passage2:
            return True

        # Calculate Jaccard similarity on words
        words1 = set(passage1.lower().split())
        words2 = set(passage2.lower().split())

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard = intersection / union if union > 0 else 0.0

        return jaccard >= threshold

    def calculate_answer_quality(
        self,
        generated_answer: str,
        query_id: int,
        method: str = 'semantic_similarity'
    ) -> Dict:
        """
        Calculate answer quality by comparing to ground truth answers

        Args:
            generated_answer: Generated answer string
            query_id: Query ID to look up ground truth
            method: 'semantic_similarity' or 'exact_match'

        Returns:
            Dictionary with quality metrics
        """
        if query_id not in self.ground_truth:
            return {
                'similarity_score': 0.0,
                'error': 'query_id not in ground truth'
            }

        gt = self.ground_truth[query_id]
        gt_answers = gt['answers']

        if not gt_answers:
            # No ground truth answer available
            return {
                'similarity_score': None,
                'note': 'no ground truth answer available'
            }

        if method == 'exact_match':
            # Simple exact match
            for gt_answer in gt_answers:
                if generated_answer.lower().strip() == gt_answer.lower().strip():
                    return {'similarity_score': 1.0, 'method': 'exact_match'}
            return {'similarity_score': 0.0, 'method': 'exact_match'}

        elif method == 'semantic_similarity':
            # Calculate semantic similarity using embeddings
            try:
                # Get embedding for generated answer
                gen_embedding = self._get_embedding(generated_answer)

                # Get embeddings for all ground truth answers
                max_similarity = 0.0

                for gt_answer in gt_answers:
                    gt_embedding = self._get_embedding(gt_answer)

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(gen_embedding, gt_embedding)
                    max_similarity = max(max_similarity, similarity)

                return {
                    'similarity_score': float(max_similarity),
                    'method': 'semantic_similarity'
                }

            except Exception as e:
                return {
                    'similarity_score': 0.0,
                    'error': str(e),
                    'method': 'semantic_similarity'
                }

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using OpenAI API

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def evaluate_retrieval(
        self,
        noise_type: str,
        retrieval_strategy: str,
        query_id: int,
        retrieved_passages: List[Dict],
        k_values: List[int] = [1, 3, 5]
    ) -> Dict:
        """
        Evaluate retrieval performance for a single query

        Args:
            noise_type: Type of noise applied to query
            retrieval_strategy: Strategy used for retrieval
            query_id: Query ID
            retrieved_passages: List of retrieved passages
            k_values: List of k values to evaluate

        Returns:
            Evaluation metrics
        """
        metrics = {
            'noise_type': noise_type,
            'retrieval_strategy': retrieval_strategy,
            'query_id': query_id,
            'precision': {}
        }

        # Calculate precision at different k values
        for k in k_values:
            precision_metrics = self.calculate_precision_at_k(
                retrieved_passages, query_id, k
            )
            metrics['precision'][f'p@{k}'] = precision_metrics

        # Store for later analysis
        self.results[noise_type][retrieval_strategy].append(metrics)

        return metrics

    def evaluate_answer(
        self,
        noise_type: str,
        retrieval_strategy: str,
        query_id: int,
        generated_answer: str,
        retrieved_passages: List[Dict]
    ) -> Dict:
        """
        Evaluate both retrieval and answer quality

        Args:
            noise_type: Type of noise
            retrieval_strategy: Retrieval strategy used
            query_id: Query ID
            generated_answer: Generated answer
            retrieved_passages: Retrieved passages

        Returns:
            Combined evaluation metrics
        """
        # Retrieval metrics
        retrieval_metrics = self.evaluate_retrieval(
            noise_type, retrieval_strategy, query_id, retrieved_passages
        )

        # Answer quality metrics
        answer_quality = self.calculate_answer_quality(
            generated_answer, query_id
        )

        # Combine metrics
        combined_metrics = {
            **retrieval_metrics,
            'answer_quality': answer_quality,
            'generated_answer': generated_answer
        }

        return combined_metrics

    def analyze_failures(
        self,
        noise_type: str,
        retrieval_strategy: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Analyze which queries failed and why

        Args:
            noise_type: Type of noise to analyze
            retrieval_strategy: Strategy to analyze
            threshold: Threshold for considering a query as failed

        Returns:
            Failure analysis
        """
        if noise_type not in self.results or retrieval_strategy not in self.results[noise_type]:
            return {
                'error': 'No results found for this combination',
                'noise_type': noise_type,
                'retrieval_strategy': retrieval_strategy
            }

        results = self.results[noise_type][retrieval_strategy]

        failures = []
        successes = []

        for result in results:
            query_id = result['query_id']

            # Get precision@5 as main metric
            p_at_5 = result['precision'].get('p@5', {}).get('precision_at_k', 0.0)

            # Get answer quality if available
            answer_quality = result.get('answer_quality', {}).get('similarity_score', None)

            # Determine if failed
            failed = p_at_5 < threshold

            failure_info = {
                'query_id': query_id,
                'query': self.ground_truth.get(query_id, {}).get('query', ''),
                'precision_at_5': p_at_5,
                'answer_quality': answer_quality,
                'query_type': self.ground_truth.get(query_id, {}).get('query_type', 'unknown')
            }

            if failed:
                failures.append(failure_info)
            else:
                successes.append(failure_info)

        # Categorize failures by query type
        failures_by_type = defaultdict(list)
        for failure in failures:
            failures_by_type[failure['query_type']].append(failure)

        return {
            'noise_type': noise_type,
            'retrieval_strategy': retrieval_strategy,
            'total_queries': len(results),
            'num_failures': len(failures),
            'num_successes': len(successes),
            'failure_rate': len(failures) / len(results) if results else 0.0,
            'failures': failures,
            'failures_by_type': dict(failures_by_type),
            'threshold_used': threshold
        }

    def calculate_recovery_rate(
        self,
        context_dependent_results: List[Dict],
        with_clarification_results: List[Dict]
    ) -> Dict:
        """
        Calculate recovery rate for context-dependent queries

        Args:
            context_dependent_results: Results without clarification
            with_clarification_results: Results with clarification/context

        Returns:
            Recovery rate metrics
        """
        if len(context_dependent_results) != len(with_clarification_results):
            return {
                'error': 'Mismatched result counts',
                'context_dependent_count': len(context_dependent_results),
                'with_clarification_count': len(with_clarification_results)
            }

        recovered = 0
        total = len(context_dependent_results)

        recovery_details = []

        for i, (cd_result, clarif_result) in enumerate(
            zip(context_dependent_results, with_clarification_results)
        ):
            # Get precision scores
            cd_precision = cd_result['precision'].get('p@5', {}).get('precision_at_k', 0.0)
            clarif_precision = clarif_result['precision'].get('p@5', {}).get('precision_at_k', 0.0)

            # Check if performance improved with clarification
            improved = clarif_precision > cd_precision

            if improved:
                recovered += 1

            recovery_details.append({
                'query_id': cd_result['query_id'],
                'without_context_precision': cd_precision,
                'with_context_precision': clarif_precision,
                'recovered': improved,
                'improvement': clarif_precision - cd_precision
            })

        recovery_rate = recovered / total if total > 0 else 0.0

        return {
            'total_context_dependent_queries': total,
            'recovered_count': recovered,
            'recovery_rate': recovery_rate,
            'details': recovery_details
        }

    def generate_summary_statistics(self) -> Dict:
        """
        Generate summary statistics across all noise types and strategies

        Returns:
            Summary statistics
        """
        summary = {
            'by_noise_type': {},
            'by_strategy': {},
            'overall': {}
        }

        # Aggregate by noise type
        for noise_type, strategies in self.results.items():
            noise_stats = {
                'strategies': {},
                'avg_precision_at_5': [],
                'total_queries': 0
            }

            for strategy, results in strategies.items():
                precisions = [
                    r['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results
                ]

                noise_stats['strategies'][strategy] = {
                    'num_queries': len(results),
                    'avg_precision_at_5': statistics.mean(precisions) if precisions else 0.0,
                    'std_precision_at_5': statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
                    'min_precision_at_5': min(precisions) if precisions else 0.0,
                    'max_precision_at_5': max(precisions) if precisions else 0.0
                }

                noise_stats['avg_precision_at_5'].extend(precisions)
                noise_stats['total_queries'] += len(results)

            noise_stats['overall_avg_precision_at_5'] = (
                statistics.mean(noise_stats['avg_precision_at_5'])
                if noise_stats['avg_precision_at_5'] else 0.0
            )

            summary['by_noise_type'][noise_type] = noise_stats

        # Aggregate by strategy across noise types
        strategy_stats = defaultdict(lambda: {'precisions': [], 'num_queries': 0})

        for noise_type, strategies in self.results.items():
            for strategy, results in strategies.items():
                precisions = [
                    r['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results
                ]
                strategy_stats[strategy]['precisions'].extend(precisions)
                strategy_stats[strategy]['num_queries'] += len(results)

        for strategy, stats in strategy_stats.items():
            precisions = stats['precisions']
            summary['by_strategy'][strategy] = {
                'num_queries': stats['num_queries'],
                'avg_precision_at_5': statistics.mean(precisions) if precisions else 0.0,
                'std_precision_at_5': statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
                'min_precision_at_5': min(precisions) if precisions else 0.0,
                'max_precision_at_5': max(precisions) if precisions else 0.0
            }

        # Overall statistics
        all_precisions = []
        total_queries = 0

        for noise_type, strategies in self.results.items():
            for strategy, results in strategies.items():
                precisions = [
                    r['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results
                ]
                all_precisions.extend(precisions)
                total_queries += len(results)

        summary['overall'] = {
            'total_queries': total_queries,
            'total_noise_types': len(self.results),
            'total_strategies': len(strategy_stats),
            'avg_precision_at_5': statistics.mean(all_precisions) if all_precisions else 0.0,
            'std_precision_at_5': statistics.stdev(all_precisions) if len(all_precisions) > 1 else 0.0
        }

        return summary

    def save_results(self, output_path: str = 'data/evaluation_results.json'):
        """
        Save evaluation results to JSON file

        Args:
            output_path: Path to save results
        """
        output_data = {
            'results': dict(self.results),
            'summary': self.generate_summary_statistics()
        }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")

    def load_results(self, input_path: str = 'data/evaluation_results.json'):
        """
        Load evaluation results from JSON file

        Args:
            input_path: Path to load results from
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert back to defaultdict structure
        self.results = defaultdict(lambda: defaultdict(list))
        for noise_type, strategies in data['results'].items():
            for strategy, results in strategies.items():
                self.results[noise_type][strategy] = results

        print(f"Results loaded from {input_path}")

    def print_summary(self):
        """
        Print a formatted summary of evaluation results
        """
        summary = self.generate_summary_statistics()

        print("\n" + "="*80)
        print("RAG EVALUATION SUMMARY")
        print("="*80)

        # Overall stats
        print(f"\nOverall Statistics:")
        print(f"  Total Queries: {summary['overall']['total_queries']}")
        print(f"  Noise Types Tested: {summary['overall']['total_noise_types']}")
        print(f"  Retrieval Strategies: {summary['overall']['total_strategies']}")
        print(f"  Average Precision@5: {summary['overall']['avg_precision_at_5']:.3f}")
        print(f"  Std Dev Precision@5: {summary['overall']['std_precision_at_5']:.3f}")

        # By strategy
        print(f"\nPerformance by Retrieval Strategy:")
        print("-" * 80)
        for strategy, stats in sorted(summary['by_strategy'].items()):
            print(f"  {strategy}:")
            print(f"    Queries: {stats['num_queries']}")
            print(f"    Avg P@5: {stats['avg_precision_at_5']:.3f} (±{stats['std_precision_at_5']:.3f})")
            print(f"    Range: [{stats['min_precision_at_5']:.3f}, {stats['max_precision_at_5']:.3f}]")

        # By noise type
        print(f"\nPerformance by Noise Type:")
        print("-" * 80)
        for noise_type, stats in sorted(summary['by_noise_type'].items()):
            print(f"  {noise_type}:")
            print(f"    Queries: {stats['total_queries']}")
            print(f"    Overall Avg P@5: {stats['overall_avg_precision_at_5']:.3f}")

            print(f"    By Strategy:")
            for strategy, strategy_stats in sorted(stats['strategies'].items()):
                print(f"      {strategy}: {strategy_stats['avg_precision_at_5']:.3f}")

        print("="*80 + "\n")
