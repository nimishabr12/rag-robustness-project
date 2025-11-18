"""
Results Visualization for RAG Robustness Experiments

Generates comprehensive visualizations of experiment results.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(filepath='results/pilot_results.json'):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_heatmap(results, output_dir):
    """
    Create heatmap showing Precision@5 for each (noise_type, strategy) combination
    """
    print("Creating heatmap...")

    noise_types = results['metadata']['noise_types']
    strategies = results['metadata']['retrieval_strategies']

    # Build matrix of average precision@5 scores
    heatmap_data = []

    for noise_type in noise_types:
        row = []
        for strategy in strategies:
            if noise_type in results['detailed_results'] and strategy in results['detailed_results'][noise_type]:
                results_list = results['detailed_results'][noise_type][strategy]
                precisions = [
                    r['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results_list
                ]
                avg_precision = np.mean(precisions) if precisions else 0.0
                row.append(avg_precision)
            else:
                row.append(0.0)
        heatmap_data.append(row)

    # Create DataFrame
    df_heatmap = pd.DataFrame(
        heatmap_data,
        index=[nt.replace('_', ' ').title() for nt in noise_types],
        columns=[s.replace('_retrieval', '').replace('_', ' ').title() for s in strategies]
    )

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=0.25,
        cbar_kws={'label': 'Precision@5'},
        linewidths=1.5,
        linecolor='white'
    )
    plt.title('RAG Performance Heatmap: Precision@5 by Noise Type and Strategy',
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Retrieval Strategy', fontsize=13, fontweight='bold')
    plt.ylabel('Noise Type', fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'heatmap_precision.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_noise_type_comparison(results, output_dir):
    """
    Create bar chart comparing average performance across noise types
    """
    print("Creating noise type comparison...")

    noise_types = results['metadata']['noise_types']

    # Calculate average precision for each noise type
    noise_type_scores = []
    noise_type_names = []
    noise_type_stds = []

    for noise_type in noise_types:
        if noise_type in results['detailed_results']:
            all_precisions = []
            for strategy, results_list in results['detailed_results'][noise_type].items():
                precisions = [
                    r['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results_list
                ]
                all_precisions.extend(precisions)

            if all_precisions:
                noise_type_names.append(noise_type.replace('_', ' ').title())
                noise_type_scores.append(np.mean(all_precisions))
                noise_type_stds.append(np.std(all_precisions))

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(noise_type_names))
    bars = ax.bar(x_pos, noise_type_scores, yerr=noise_type_stds,
                   capsize=7, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Color bars based on performance
    colors = plt.cm.RdYlGn(np.array(noise_type_scores) / max(noise_type_scores))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels on bars
    for i, (score, std) in enumerate(zip(noise_type_scores, noise_type_stds)):
        ax.text(i, score + std + 0.005, f'{score:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Noise Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Precision@5', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison Across Noise Types\n(averaged across all retrieval strategies)',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(noise_type_names, rotation=15, ha='right')
    ax.set_ylim(0, max(noise_type_scores) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'noise_type_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_degradation_plot(results, output_dir):
    """
    Create line plot showing performance degradation from clean baseline
    """
    print("Creating performance degradation plot...")

    strategies = results['metadata']['retrieval_strategies']
    noise_types = ['clean', 'noisy', 'ambiguous', 'context_dependent', 'adversarial']

    # Collect data for each strategy
    fig, ax = plt.subplots(figsize=(14, 8))

    for strategy in strategies:
        strategy_scores = []

        for noise_type in noise_types:
            if (noise_type in results['detailed_results'] and
                strategy in results['detailed_results'][noise_type]):
                results_list = results['detailed_results'][noise_type][strategy]
                precisions = [
                    r['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)
                    for r in results_list
                ]
                avg_precision = np.mean(precisions) if precisions else 0.0
                strategy_scores.append(avg_precision)
            else:
                strategy_scores.append(0.0)

        # Plot line
        strategy_name = strategy.replace('_retrieval', '').replace('_', ' ').title()
        ax.plot(range(len(noise_types)), strategy_scores,
                marker='o', linewidth=2.5, markersize=10,
                label=strategy_name, alpha=0.8)

    # Customize plot
    ax.set_xlabel('Noise Type', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision@5', fontsize=13, fontweight='bold')
    ax.set_title('Performance Degradation Analysis\nHow retrieval strategies handle different noise types',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(range(len(noise_types)))
    ax.set_xticklabels([nt.replace('_', ' ').title() for nt in noise_types],
                        rotation=15, ha='right')
    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max([max(ax.lines[i].get_ydata()) for i in range(len(ax.lines))]) * 1.2)

    # Add shaded region for "acceptable" performance
    ax.axhspan(0.15, 0.25, alpha=0.1, color='green', label='Target Range')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'degradation_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_failure_analysis(results, output_dir):
    """
    Create confusion analysis showing which queries failed most consistently
    """
    print("Creating failure analysis...")

    # Track failure rates per query across all strategies and noise types
    query_failures = defaultdict(lambda: {'total': 0, 'failures': 0, 'query_text': '', 'query_type': ''})

    for noise_type, strategies in results['detailed_results'].items():
        for strategy, results_list in strategies.items():
            for result in results_list:
                query_id = result['query_id']
                precision = result['evaluation']['precision'].get('p@5', {}).get('precision_at_k', 0.0)

                query_failures[query_id]['total'] += 1
                if precision < 0.1:  # Consider < 0.1 as failure
                    query_failures[query_id]['failures'] += 1

                # Store query info
                if not query_failures[query_id]['query_text']:
                    query_failures[query_id]['query_text'] = result.get('original_query', result['query'])
                    query_failures[query_id]['query_type'] = result.get('query_type', 'unknown')

    # Calculate failure rates
    failure_data = []
    for query_id, data in query_failures.items():
        failure_rate = data['failures'] / data['total'] if data['total'] > 0 else 0
        failure_data.append({
            'query_id': query_id,
            'query_text': data['query_text'],
            'query_type': data['query_type'],
            'failure_rate': failure_rate,
            'failures': data['failures'],
            'total': data['total']
        })

    # Sort by failure rate
    failure_data.sort(key=lambda x: x['failure_rate'], reverse=True)

    # Plot top 20 most problematic queries
    top_failures = failure_data[:20]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Top failing queries
    query_labels = [f"Q{f['query_id']}: {f['query_text'][:40]}..." for f in top_failures]
    failure_rates = [f['failure_rate'] * 100 for f in top_failures]

    bars = ax1.barh(range(len(query_labels)), failure_rates, alpha=0.85, edgecolor='black')

    # Color by severity
    colors = plt.cm.Reds([rate/100 for rate in failure_rates])
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax1.set_yticks(range(len(query_labels)))
    ax1.set_yticklabels(query_labels, fontsize=9)
    ax1.set_xlabel('Failure Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Most Problematic Queries\n(failure rate across all strategies and noise types)',
                  fontsize=14, pad=15, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()

    # Add percentage labels
    for i, rate in enumerate(failure_rates):
        ax1.text(rate + 1, i, f'{rate:.1f}%', va='center', fontsize=9)

    # Plot 2: Failure rate by query type
    type_failures = defaultdict(lambda: {'failures': 0, 'total': 0})
    for f in failure_data:
        qtype = f['query_type']
        type_failures[qtype]['failures'] += f['failures']
        type_failures[qtype]['total'] += f['total']

    type_rates = []
    type_names = []
    for qtype, data in type_failures.items():
        type_names.append(qtype.title())
        rate = (data['failures'] / data['total'] * 100) if data['total'] > 0 else 0
        type_rates.append(rate)

    # Sort by rate
    sorted_indices = np.argsort(type_rates)[::-1]
    type_names = [type_names[i] for i in sorted_indices]
    type_rates = [type_rates[i] for i in sorted_indices]

    bars2 = ax2.bar(range(len(type_names)), type_rates, alpha=0.85, edgecolor='black')

    # Color by severity
    colors2 = plt.cm.OrRd([rate/max(type_rates) if max(type_rates) > 0 else 0 for rate in type_rates])
    for bar, color in zip(bars2, colors2):
        bar.set_color(color)

    ax2.set_xticks(range(len(type_names)))
    ax2.set_xticklabels(type_names, rotation=45, ha='right')
    ax2.set_ylabel('Failure Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Failure Rate by Query Type', fontsize=14, pad=15, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels
    for i, rate in enumerate(type_rates):
        ax2.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'failure_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    # Also save detailed failure report as JSON
    report_path = os.path.join(output_dir, 'failure_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'top_failing_queries': failure_data[:50],
            'failure_by_type': dict(type_failures)
        }, f, indent=2)
    print(f"  Saved: {report_path}")


def main():
    """Generate all visualizations"""
    print("="*80)
    print("RAG ROBUSTNESS - RESULTS VISUALIZATION")
    print("="*80)
    print()

    # Create output directory
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load results
    print("Loading results...")
    results = load_results()
    print(f"Loaded {results['metadata']['timestamp']}")
    print(f"Total queries: {len(results['detailed_results'])} noise types × strategies\n")

    # Generate visualizations
    print("Generating visualizations...\n")

    create_heatmap(results, output_dir)
    create_noise_type_comparison(results, output_dir)
    create_degradation_plot(results, output_dir)
    create_failure_analysis(results, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - heatmap_precision.png")
    print("  - noise_type_comparison.png")
    print("  - degradation_analysis.png")
    print("  - failure_analysis.png")
    print("  - failure_report.json")
    print()


if __name__ == "__main__":
    main()
