"""
Download and process MS MARCO dataset sample
"""
import json
import random
from datasets import load_dataset
import numpy as np
import sys

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to safely print unicode text
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

print("Loading MS MARCO dataset from Hugging Face...")
# Load MS MARCO passage ranking dataset
dataset = load_dataset('ms_marco', 'v1.1', split='train')

print(f"Dataset loaded: {len(dataset)} examples")
print(f"\nDataset structure:")
safe_print(f"Features: {dataset.features}")
print(f"\nColumn names: {dataset.column_names}")

# Show a few examples
print("\n" + "="*80)
print("SAMPLE EXAMPLES:")
print("="*80)
for i in range(3):
    example = dataset[i]
    print(f"\nExample {i+1}:")
    safe_print(f"Query: {example['query']}")
    print(f"Query ID: {example.get('query_id', 'N/A')}")
    print(f"Query Type: {example.get('query_type', 'N/A')}")
    safe_print(f"First Answer: {example.get('answers', ['N/A'])[0] if example.get('answers') else 'N/A'}")
    if example['passages'] and 'passage_text' in example['passages']:
        safe_print(f"First passage (truncated): {example['passages']['passage_text'][0][:200]}...")
    print("-" * 80)

# Sample 200 examples
print("\nSampling 200 query-passage pairs...")
sample_size = 200
indices = random.sample(range(len(dataset)), sample_size)
sampled_data = []

for idx in indices:
    example = dataset[idx]
    # Extract relevant information
    sampled_data.append({
        'query_id': example.get('query_id', ''),
        'query': example['query'],
        'passages': example['passages'],
        'query_type': example.get('query_type', ''),
        'answers': example.get('answers', []),
        'wellFormedAnswers': example.get('wellFormedAnswers', [])
    })

# Save to JSON
output_path = 'data/ms_marco_sample.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(sampled_data)} samples to {output_path}")

# Calculate statistics
print("\n" + "="*80)
print("STATISTICS:")
print("="*80)

query_lengths = [len(item['query'].split()) for item in sampled_data]
print(f"\nQuery Length Statistics:")
print(f"  Mean: {np.mean(query_lengths):.2f} words")
print(f"  Median: {np.median(query_lengths):.2f} words")
print(f"  Min: {np.min(query_lengths)} words")
print(f"  Max: {np.max(query_lengths)} words")
print(f"  Std Dev: {np.std(query_lengths):.2f} words")

# Calculate passage statistics
all_passage_lengths = []
for item in sampled_data:
    if item['passages'] and 'passage_text' in item['passages']:
        for passage in item['passages']['passage_text']:
            all_passage_lengths.append(len(passage.split()))

if all_passage_lengths:
    print(f"\nPassage Length Statistics:")
    print(f"  Mean: {np.mean(all_passage_lengths):.2f} words")
    print(f"  Median: {np.median(all_passage_lengths):.2f} words")
    print(f"  Min: {np.min(all_passage_lengths)} words")
    print(f"  Max: {np.max(all_passage_lengths)} words")
    print(f"  Std Dev: {np.std(all_passage_lengths):.2f} words")
    print(f"  Total passages: {len(all_passage_lengths)}")

print(f"\nAverage passages per query: {len(all_passage_lengths) / len(sampled_data):.2f}")

print("\n" + "="*80)
print("SAMPLE QUERY-PASSAGE PAIRS FROM SAVED DATA:")
print("="*80)
for i in range(min(3, len(sampled_data))):
    item = sampled_data[i]
    safe_print(f"\nQuery {i+1}: {item['query']}")
    if item['passages'] and 'passage_text' in item['passages']:
        print(f"Number of passages: {len(item['passages']['passage_text'])}")
        safe_print(f"First passage: {item['passages']['passage_text'][0][:200]}...")
    print("-" * 80)

print("\nDone!")
