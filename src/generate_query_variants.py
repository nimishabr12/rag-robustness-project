"""
Generate query variants using noise generators
"""
import json
import sys
from noise_generators import add_typos, make_ambiguous, make_context_dependent, make_adversarial

# Function to safely print unicode text
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

print("Loading MS MARCO sample...")
with open('data/ms_marco_sample.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} queries\n")

# Create variants
clean_queries = []
noisy_queries = []
ambiguous_queries = []
context_dependent_queries = []
adversarial_queries = []

print("Generating query variants...")
print("="*80)

for item in data:
    original_query = item['query']
    query_id = item['query_id']

    # Clean queries (original)
    clean_queries.append({
        'query_id': query_id,
        'query': original_query,
        'query_type': item['query_type']
    })

    # Noisy queries (with typos)
    noisy_query = add_typos(original_query, typo_rate=0.15)
    noisy_queries.append({
        'query_id': query_id,
        'query': noisy_query,
        'query_type': item['query_type'],
        'original_query': original_query
    })

    # Ambiguous queries
    ambiguous_query = make_ambiguous(original_query)
    ambiguous_queries.append({
        'query_id': query_id,
        'query': ambiguous_query,
        'query_type': item['query_type'],
        'original_query': original_query
    })

    # Context-dependent queries
    context_query = make_context_dependent(original_query)
    context_dependent_queries.append({
        'query_id': query_id,
        'query': context_query,
        'query_type': item['query_type'],
        'original_query': original_query
    })

    # Adversarial queries
    adversarial_query = make_adversarial(original_query)
    adversarial_queries.append({
        'query_id': query_id,
        'query': adversarial_query,
        'query_type': item['query_type'],
        'original_query': original_query
    })

print("[DONE] Generated all variants\n")

# Save all variants
print("Saving query variants...")
with open('data/clean_queries.json', 'w', encoding='utf-8') as f:
    json.dump(clean_queries, f, indent=2, ensure_ascii=False)
print("[DONE] Saved data/clean_queries.json")

with open('data/noisy_queries.json', 'w', encoding='utf-8') as f:
    json.dump(noisy_queries, f, indent=2, ensure_ascii=False)
print("[DONE] Saved data/noisy_queries.json")

with open('data/ambiguous_queries.json', 'w', encoding='utf-8') as f:
    json.dump(ambiguous_queries, f, indent=2, ensure_ascii=False)
print("[DONE] Saved data/ambiguous_queries.json")

with open('data/context_dependent_queries.json', 'w', encoding='utf-8') as f:
    json.dump(context_dependent_queries, f, indent=2, ensure_ascii=False)
print("[DONE] Saved data/context_dependent_queries.json")

with open('data/adversarial_queries.json', 'w', encoding='utf-8') as f:
    json.dump(adversarial_queries, f, indent=2, ensure_ascii=False)
print("[DONE] Saved data/adversarial_queries.json")

print("\n" + "="*80)
print("EXAMPLES OF EACH TRANSFORMATION")
print("="*80)

# Show examples
num_examples = 5
for i in range(num_examples):
    print(f"\n--- Example {i+1} ---")
    safe_print(f"Original:           {clean_queries[i]['query']}")
    safe_print(f"Noisy (typos):      {noisy_queries[i]['query']}")
    safe_print(f"Ambiguous:          {ambiguous_queries[i]['query']}")
    safe_print(f"Context-dependent:  {context_dependent_queries[i]['query']}")
    safe_print(f"Adversarial:        {adversarial_queries[i]['query']}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total queries: {len(data)}")
print(f"Clean queries: {len(clean_queries)}")
print(f"Noisy queries: {len(noisy_queries)}")
print(f"Ambiguous queries: {len(ambiguous_queries)}")
print(f"Context-dependent queries: {len(context_dependent_queries)}")
print(f"Adversarial queries: {len(adversarial_queries)}")
print("\nAll query variants maintain the same query_id structure for tracking.")
print("Done!")
