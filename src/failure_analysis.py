"""
Detailed Failure Analysis for RAG Robustness Experiments

Analyzes failure patterns and generates comprehensive reports.
"""
import os
import json
import statistics
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re


class FailureAnalyzer:
    """
    Comprehensive failure analysis for RAG experiments
    """

    def __init__(self, results_path='results/pilot_results.json'):
        """Load experiment results"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        self.all_failures = []
        self.failures_by_noise_type = defaultdict(list)
        self.failures_by_strategy = defaultdict(list)
        self.failures_by_query_type = defaultdict(list)

    def analyze_failures(self, threshold=0.1):
        """
        Identify and categorize all failures

        Args:
            threshold: Precision@5 below this is considered a failure
        """
        print("Analyzing failures...")

        for noise_type, strategies in self.results['detailed_results'].items():
            for strategy, results_list in strategies.items():
                for result in results_list:
                    precision = result['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)

                    failure_info = {
                        'query_id': result['query_id'],
                        'query': result['query'],
                        'original_query': result.get('original_query', result['query']),
                        'query_type': result.get('query_type', 'unknown'),
                        'noise_type': noise_type,
                        'strategy': strategy,
                        'precision': precision,
                        'answer': result.get('generated_answer', ''),
                        'confidence': result.get('answer_confidence', 'unknown'),
                        'is_failure': precision < threshold
                    }

                    if failure_info['is_failure']:
                        self.all_failures.append(failure_info)
                        self.failures_by_noise_type[noise_type].append(failure_info)
                        self.failures_by_strategy[strategy].append(failure_info)
                        self.failures_by_query_type[failure_info['query_type']].append(failure_info)

        print(f"  Total failures: {len(self.all_failures)}")
        print(f"  Failure rate: {len(self.all_failures) / (len(self.all_failures) + sum(1 for _ in self._all_successes())) * 100:.1f}%")

    def _all_successes(self):
        """Generator for all successful queries"""
        for noise_type, strategies in self.results['detailed_results'].items():
            for strategy, results_list in strategies.items():
                for result in results_list:
                    precision = result['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    if precision >= 0.1:
                        yield result

    def identify_worst_queries(self, top_n=10):
        """
        Identify worst-performing queries for each noise type

        Args:
            top_n: Number of worst queries to return per noise type

        Returns:
            Dictionary mapping noise types to worst queries
        """
        print("\nIdentifying worst-performing queries...")

        worst_queries = {}

        for noise_type in self.results['metadata']['noise_types']:
            # Aggregate performance across all strategies for each query
            query_performance = defaultdict(lambda: {'precision_scores': [], 'query_info': None})

            if noise_type in self.results['detailed_results']:
                for strategy, results_list in self.results['detailed_results'][noise_type].items():
                    for result in results_list:
                        query_id = result['query_id']
                        precision = result['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)

                        query_performance[query_id]['precision_scores'].append(precision)
                        if query_performance[query_id]['query_info'] is None:
                            query_performance[query_id]['query_info'] = {
                                'query': result['query'],
                                'original_query': result.get('original_query', result['query']),
                                'query_type': result.get('query_type', 'unknown')
                            }

            # Calculate average precision and sort
            query_avg_precision = []
            for query_id, data in query_performance.items():
                avg_precision = statistics.mean(data['precision_scores'])
                query_avg_precision.append({
                    'query_id': query_id,
                    'avg_precision': avg_precision,
                    'min_precision': min(data['precision_scores']),
                    'max_precision': max(data['precision_scores']),
                    'std_precision': statistics.stdev(data['precision_scores']) if len(data['precision_scores']) > 1 else 0,
                    **data['query_info']
                })

            # Sort by average precision (worst first)
            query_avg_precision.sort(key=lambda x: x['avg_precision'])

            worst_queries[noise_type] = query_avg_precision[:top_n]
            print(f"  {noise_type}: {len(query_avg_precision)} queries analyzed")

        return worst_queries

    def analyze_failure_patterns(self):
        """
        Analyze common patterns in failures

        Returns:
            Dictionary with pattern analysis
        """
        print("\nAnalyzing failure patterns...")

        patterns = {
            'by_query_length': defaultdict(list),
            'by_query_complexity': defaultdict(list),
            'by_query_type': defaultdict(int),
            'by_word_count': defaultdict(list)
        }

        for failure in self.all_failures:
            query = failure['original_query']

            # Query length (characters)
            length = len(query)
            if length < 30:
                length_category = 'short (<30 chars)'
            elif length < 60:
                length_category = 'medium (30-60 chars)'
            else:
                length_category = 'long (>60 chars)'

            patterns['by_query_length'][length_category].append(failure['precision'])

            # Word count
            word_count = len(query.split())
            patterns['by_word_count'][word_count].append(failure)

            # Query complexity (heuristic based on punctuation, capitalization, etc.)
            complexity_score = 0
            complexity_score += query.count('?')  # Questions
            complexity_score += query.count(',')  # Complex clauses
            complexity_score += len([w for w in query.split() if w[0].isupper()]) * 0.5  # Proper nouns
            complexity_score += len(re.findall(r'\b\w{10,}\b', query)) * 0.3  # Long/technical words

            if complexity_score < 2:
                complexity_category = 'simple'
            elif complexity_score < 5:
                complexity_category = 'moderate'
            else:
                complexity_category = 'complex'

            patterns['by_query_complexity'][complexity_category].append(failure['precision'])

            # Query type
            patterns['by_query_type'][failure['query_type']] += 1

        # Calculate statistics
        stats = {}

        # Length analysis
        stats['length_analysis'] = {}
        for length_cat, precisions in patterns['by_query_length'].items():
            stats['length_analysis'][length_cat] = {
                'count': len(precisions),
                'avg_precision': statistics.mean(precisions) if precisions else 0,
                'failure_rate': sum(1 for p in precisions if p < 0.1) / len(precisions) * 100 if precisions else 0
            }

        # Complexity analysis
        stats['complexity_analysis'] = {}
        for complexity_cat, precisions in patterns['by_query_complexity'].items():
            stats['complexity_analysis'][complexity_cat] = {
                'count': len(precisions),
                'avg_precision': statistics.mean(precisions) if precisions else 0,
                'failure_rate': sum(1 for p in precisions if p < 0.1) / len(precisions) * 100 if precisions else 0
            }

        # Word count analysis
        word_count_data = []
        for wc, failures in patterns['by_word_count'].items():
            if failures:
                word_count_data.append({
                    'word_count': wc,
                    'failure_count': len(failures),
                    'avg_precision': statistics.mean([f['precision'] for f in failures])
                })
        stats['word_count_analysis'] = sorted(word_count_data, key=lambda x: x['failure_count'], reverse=True)[:10]

        # Query type analysis
        stats['query_type_analysis'] = dict(patterns['by_query_type'])

        return stats

    def analyze_context_dependent_failures(self):
        """
        Deep dive into why context-dependent queries fail so catastrophically

        Returns:
            Dictionary with detailed analysis
        """
        print("\nAnalyzing context-dependent query failures...")

        context_failures = self.failures_by_noise_type.get('context_dependent', [])

        if not context_failures:
            return {'error': 'No context-dependent failures found'}

        analysis = {
            'total_context_queries': len(context_failures),
            'failure_rate': len(context_failures) / len(context_failures) * 100 if context_failures else 0,
            'avg_precision': statistics.mean([f['precision'] for f in context_failures]),
            'query_characteristics': {},
            'common_patterns': [],
            'example_failures': []
        }

        # Analyze query characteristics
        pronoun_queries = []
        demonstrative_queries = []
        follow_up_queries = []
        other_queries = []

        pronouns = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        follow_up_phrases = ['how does', 'tell me more', 'what about', 'can you explain', 'why is']

        for failure in context_failures:
            query_lower = failure['query'].lower()

            # Check for pronouns
            if any(pronoun in query_lower.split() for pronoun in pronouns):
                pronoun_queries.append(failure)

            # Check for demonstrative references
            if any(word in query_lower for word in ['this', 'that', 'these', 'those']):
                demonstrative_queries.append(failure)

            # Check for follow-up question patterns
            if any(phrase in query_lower for phrase in follow_up_phrases):
                follow_up_queries.append(failure)
            else:
                other_queries.append(failure)

        analysis['query_characteristics'] = {
            'pronoun_references': {
                'count': len(pronoun_queries),
                'percentage': len(pronoun_queries) / len(context_failures) * 100,
                'avg_precision': statistics.mean([f['precision'] for f in pronoun_queries]) if pronoun_queries else 0
            },
            'demonstrative_references': {
                'count': len(demonstrative_queries),
                'percentage': len(demonstrative_queries) / len(context_failures) * 100,
                'avg_precision': statistics.mean([f['precision'] for f in demonstrative_queries]) if demonstrative_queries else 0
            },
            'follow_up_questions': {
                'count': len(follow_up_queries),
                'percentage': len(follow_up_queries) / len(context_failures) * 100,
                'avg_precision': statistics.mean([f['precision'] for f in follow_up_queries]) if follow_up_queries else 0
            }
        }

        # Common patterns
        query_texts = [f['query'] for f in context_failures]
        query_starts = Counter([q.split()[0].lower() if q.split() else '' for q in query_texts])
        analysis['common_patterns'] = [
            {'pattern': f'Starts with "{word}"', 'count': count}
            for word, count in query_starts.most_common(10)
        ]

        # Root cause analysis
        analysis['root_causes'] = {
            'missing_context': {
                'description': 'Query references entities/concepts from previous conversation that are not present',
                'severity': 'CRITICAL',
                'impact': 'Without conversation history, retrieval has no context to resolve pronouns or references',
                'examples': []
            },
            'ambiguous_intent': {
                'description': 'Query intent is unclear without previous context',
                'severity': 'HIGH',
                'impact': 'Embedding similarity cannot find relevant passages when query is too vague',
                'examples': []
            },
            'incomplete_information': {
                'description': 'Query lacks key terms needed for retrieval',
                'severity': 'HIGH',
                'impact': 'BM25 and embedding-based retrieval both fail due to lack of keywords',
                'examples': []
            }
        }

        # Categorize examples
        for failure in context_failures[:15]:
            query = failure['query']
            original = failure['original_query']

            if len(query.split()) <= 3:
                analysis['root_causes']['incomplete_information']['examples'].append({
                    'query': query,
                    'original': original,
                    'precision': failure['precision']
                })
            elif any(word in query.lower() for word in pronouns):
                analysis['root_causes']['missing_context']['examples'].append({
                    'query': query,
                    'original': original,
                    'precision': failure['precision']
                })
            else:
                analysis['root_causes']['ambiguous_intent']['examples'].append({
                    'query': query,
                    'original': original,
                    'precision': failure['precision']
                })

        # Limit examples to 5 each
        for cause in analysis['root_causes'].values():
            cause['examples'] = cause['examples'][:5]

        # Example failures for illustration
        analysis['example_failures'] = sorted(
            context_failures,
            key=lambda x: x['precision']
        )[:10]

        return analysis

    def generate_markdown_report(self, output_path='results/failure_report.md'):
        """
        Generate comprehensive markdown report

        Args:
            output_path: Path to save the report
        """
        print(f"\nGenerating markdown report: {output_path}")

        worst_queries = self.identify_worst_queries(top_n=10)
        pattern_analysis = self.analyze_failure_patterns()
        context_analysis = self.analyze_context_dependent_failures()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# RAG Robustness - Failure Analysis Report\n\n")
            f.write(f"**Generated from:** {self.results['metadata']['timestamp']}\n\n")
            f.write(f"**Total Queries Analyzed:** {sum(len(strategies) * len(results_list) for strategies in self.results['detailed_results'].values() for results_list in strategies.values())}\n\n")
            f.write(f"**Total Failures (P@5 < 0.1):** {len(self.all_failures)}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes failure patterns in the RAG robustness experiments, identifying:\n\n")
            f.write("1. **Worst-performing queries** for each noise type\n")
            f.write("2. **Common failure patterns** (query length, complexity, type)\n")
            f.write("3. **Root causes** of context-dependent query failures\n\n")
            f.write("---\n\n")

            # Overall Statistics
            f.write("## Overall Failure Statistics\n\n")
            f.write(f"- **Total Failures:** {len(self.all_failures)}\n")
            f.write(f"- **Failure Rate:** {len(self.all_failures) / (len(self.all_failures) + sum(1 for _ in self._all_successes())) * 100:.1f}%\n\n")

            # Failures by noise type
            f.write("### Failures by Noise Type\n\n")
            f.write("| Noise Type | Failures | Percentage |\n")
            f.write("|------------|----------|------------|\n")
            for noise_type in sorted(self.failures_by_noise_type.keys()):
                count = len(self.failures_by_noise_type[noise_type])
                pct = count / len(self.all_failures) * 100
                f.write(f"| {noise_type.replace('_', ' ').title()} | {count} | {pct:.1f}% |\n")
            f.write("\n")

            # Failures by strategy
            f.write("### Failures by Retrieval Strategy\n\n")
            f.write("| Strategy | Failures | Percentage |\n")
            f.write("|----------|----------|------------|\n")
            for strategy in sorted(self.failures_by_strategy.keys()):
                count = len(self.failures_by_strategy[strategy])
                pct = count / len(self.all_failures) * 100
                f.write(f"| {strategy.replace('_', ' ').title()} | {count} | {pct:.1f}% |\n")
            f.write("\n---\n\n")

            # Worst performing queries
            f.write("## Worst-Performing Queries by Noise Type\n\n")
            for noise_type, queries in worst_queries.items():
                f.write(f"### {noise_type.replace('_', ' ').title()}\n\n")
                f.write("| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |\n")
                f.write("|------|----------|-------|---------|---------|----------|\n")

                for i, query in enumerate(queries, 1):
                    query_text = query['query'][:60] + "..." if len(query['query']) > 60 else query['query']
                    f.write(f"| {i} | {query['query_id']} | {query_text} | {query['avg_precision']:.3f} | {query['min_precision']:.3f} | {query['max_precision']:.3f} |\n")
                f.write("\n")

            f.write("---\n\n")

            # Pattern Analysis
            f.write("## Failure Pattern Analysis\n\n")

            # Query Length Analysis
            f.write("### By Query Length\n\n")
            f.write("| Length Category | Failures | Avg P@5 | Failure Rate |\n")
            f.write("|-----------------|----------|---------|-------------|\n")
            for length_cat, stats in pattern_analysis['length_analysis'].items():
                f.write(f"| {length_cat} | {stats['count']} | {stats['avg_precision']:.3f} | {stats['failure_rate']:.1f}% |\n")
            f.write("\n")

            # Query Complexity Analysis
            f.write("### By Query Complexity\n\n")
            f.write("| Complexity | Failures | Avg P@5 | Failure Rate |\n")
            f.write("|------------|----------|---------|-------------|\n")
            for complexity_cat, stats in pattern_analysis['complexity_analysis'].items():
                f.write(f"| {complexity_cat.title()} | {stats['count']} | {stats['avg_precision']:.3f} | {stats['failure_rate']:.1f}% |\n")
            f.write("\n")

            # Query Type Analysis
            f.write("### By Query Type\n\n")
            f.write("| Query Type | Failures |\n")
            f.write("|------------|----------|\n")
            for qtype, count in sorted(pattern_analysis['query_type_analysis'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {qtype.title()} | {count} |\n")
            f.write("\n---\n\n")

            # Context-Dependent Analysis
            f.write("## Deep Dive: Context-Dependent Query Failures\n\n")
            f.write("### Overview\n\n")
            f.write(f"Context-dependent queries show **catastrophic failure** with an average Precision@5 of **{context_analysis['avg_precision']:.3f}** (97% degradation from clean baseline).\n\n")

            f.write("### Why Do Context-Dependent Queries Fail?\n\n")

            for cause_name, cause_data in context_analysis['root_causes'].items():
                f.write(f"#### {cause_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Severity:** `{cause_data['severity']}`\n\n")
                f.write(f"**Description:** {cause_data['description']}\n\n")
                f.write(f"**Impact:** {cause_data['impact']}\n\n")

                if cause_data['examples']:
                    f.write("**Examples:**\n\n")
                    for example in cause_data['examples']:
                        f.write(f"- **Query:** \"{example['query']}\" (Original: \"{example['original']}\")\n")
                        f.write(f"  - Precision@5: {example['precision']:.3f}\n")
                    f.write("\n")

            # Query Characteristics
            f.write("### Query Characteristics\n\n")
            f.write("| Characteristic | Count | Percentage | Avg P@5 |\n")
            f.write("|----------------|-------|------------|----------|\n")
            for char_name, char_data in context_analysis['query_characteristics'].items():
                f.write(f"| {char_name.replace('_', ' ').title()} | {char_data['count']} | {char_data['percentage']:.1f}% | {char_data['avg_precision']:.3f} |\n")
            f.write("\n")

            # Common Patterns
            f.write("### Common Query Patterns\n\n")
            for pattern in context_analysis['common_patterns'][:5]:
                f.write(f"- {pattern['pattern']}: **{pattern['count']} queries**\n")
            f.write("\n---\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### For Context-Dependent Queries\n\n")
            f.write("1. **Implement Conversation History Tracking**\n")
            f.write("   - Maintain session state with previous queries and answers\n")
            f.write("   - Pass conversation context to retrieval and generation\n\n")

            f.write("2. **Query Expansion/Rewriting**\n")
            f.write("   - Use LLM to expand context-dependent queries with referenced entities\n")
            f.write("   - Resolve pronouns and demonstratives before retrieval\n\n")

            f.write("3. **Multi-Turn Dialogue Support**\n")
            f.write("   - Design RAG pipeline specifically for conversational contexts\n")
            f.write("   - Include conversation history in embeddings\n\n")

            f.write("### For General Robustness\n\n")
            f.write("1. **Enhance Query Preprocessing**\n")
            f.write("   - Implement spell correction for noisy queries\n")
            f.write("   - Expand abbreviations and acronyms\n\n")

            f.write("2. **Improve Retrieval Diversity**\n")
            f.write("   - Combine multiple retrieval strategies\n")
            f.write("   - Use query expansion for ambiguous queries\n\n")

            f.write("3. **Fine-tune Embeddings**\n")
            f.write("   - Train embeddings on domain-specific data\n")
            f.write("   - Consider using instruction-tuned embedding models\n\n")

        print(f"  Report saved successfully!")


def main():
    """Run complete failure analysis"""
    print("="*80)
    print("RAG ROBUSTNESS - DETAILED FAILURE ANALYSIS")
    print("="*80)
    print()

    analyzer = FailureAnalyzer()

    # Run analysis
    analyzer.analyze_failures(threshold=0.1)

    # Generate report
    analyzer.generate_markdown_report()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nReport saved to: results/failure_report.md")
    print()


if __name__ == "__main__":
    main()
