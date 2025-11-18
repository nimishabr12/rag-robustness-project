# RAG Robustness - Failure Analysis Report

**Generated from:** 2025-11-18T17:26:40.805455

**Total Queries Analyzed:** 1600

**Total Failures (P@5 < 0.1):** 176

## Executive Summary

This report analyzes failure patterns in the RAG robustness experiments, identifying:

1. **Worst-performing queries** for each noise type
2. **Common failure patterns** (query length, complexity, type)
3. **Root causes** of context-dependent query failures

---

## Overall Failure Statistics

- **Total Failures:** 176
- **Failure Rate:** 44.0%

### Failures by Noise Type

| Noise Type | Failures | Percentage |
|------------|----------|------------|
| Adversarial | 20 | 11.4% |
| Ambiguous | 38 | 21.6% |
| Clean | 23 | 13.1% |
| Context Dependent | 68 | 38.6% |
| Noisy | 27 | 15.3% |

### Failures by Retrieval Strategy

| Strategy | Failures | Percentage |
|----------|----------|------------|
| Hybrid Retrieval | 44 | 25.0% |
| Multistep Retrieval | 45 | 25.6% |
| Naive Retrieval | 46 | 26.1% |
| Query Rewrite Retrieval | 41 | 23.3% |

---

## Worst-Performing Queries by Noise Type

### Clean

| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |
|------|----------|-------|---------|---------|----------|
| 1 | 31100 | what is a heart scan | 0.000 | 0.000 | 0.000 |
| 2 | 23604 | good food for underactive thyroid | 0.000 | 0.000 | 0.000 |
| 3 | 31985 | what is Oscillatoria generic name? | 0.000 | 0.000 | 0.000 |
| 4 | 93349 | does polygonum persicaria smell | 0.000 | 0.000 | 0.000 |
| 5 | 98704 | dyad definition synonym | 0.050 | 0.000 | 0.200 |
| 6 | 23177 | when were traffic lights first used | 0.050 | 0.000 | 0.200 |
| 7 | 48383 | what was the first planet to be discovered | 0.150 | 0.000 | 0.200 |
| 8 | 34300 | what is cost of sales | 0.200 | 0.200 | 0.200 |
| 9 | 22977 | did us postal rates change | 0.200 | 0.200 | 0.200 |
| 10 | 55781 | how much does it cost to park at tampa airport | 0.200 | 0.200 | 0.200 |

### Noisy

| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |
|------|----------|-------|---------|---------|----------|
| 1 | 31100 | whta is a heart scan | 0.000 | 0.000 | 0.000 |
| 2 | 31985 | what is Oscillatoria generic name? | 0.000 | 0.000 | 0.000 |
| 3 | 23177 | whne were traffic lights first used | 0.000 | 0.000 | 0.000 |
| 4 | 93349 | does pilygonum persicaria smell | 0.000 | 0.000 | 0.000 |
| 5 | 34300 | what is ckst of sals | 0.050 | 0.000 | 0.200 |
| 6 | 23604 | good food for underactive thyroid | 0.050 | 0.000 | 0.200 |
| 7 | 98704 | dyad definition synonym | 0.050 | 0.000 | 0.200 |
| 8 | 23864 | what is bewn sraw | 0.150 | 0.000 | 0.200 |
| 9 | 86010 | is there pst on serrvice fees | 0.150 | 0.000 | 0.200 |
| 10 | 22977 | did us postal rates change | 0.200 | 0.200 | 0.200 |

### Ambiguous

| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |
|------|----------|-------|---------|---------|----------|
| 1 | 48983 | PC | 0.000 | 0.000 | 0.000 |
| 2 | 33140 | SR | 0.000 | 0.000 | 0.000 |
| 3 | 91264 | WD | 0.000 | 0.000 | 0.000 |
| 4 | 31100 | what is a heart scan | 0.000 | 0.000 | 0.000 |
| 5 | 23604 | good food for underactive thyroid | 0.000 | 0.000 | 0.000 |
| 6 | 31985 | OG | 0.000 | 0.000 | 0.000 |
| 7 | 98704 | DDS | 0.000 | 0.000 | 0.000 |
| 8 | 93349 | does polygonum persicaria smell | 0.000 | 0.000 | 0.000 |
| 9 | 23177 | when were traffic lights first used | 0.050 | 0.000 | 0.200 |
| 10 | 48383 | what was the first planet to be discovered | 0.100 | 0.000 | 0.200 |

### Context Dependent

| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |
|------|----------|-------|---------|---------|----------|
| 1 | 34300 | What about this? | 0.000 | 0.000 | 0.000 |
| 2 | 22977 | How does it work? | 0.000 | 0.000 | 0.000 |
| 3 | 51829 | What about this? | 0.000 | 0.000 | 0.000 |
| 4 | 38001 | what are some criticisms of it | 0.000 | 0.000 | 0.000 |
| 5 | 33140 | salary range of a that | 0.000 | 0.000 | 0.000 |
| 6 | 31100 | what is a that scan | 0.000 | 0.000 | 0.000 |
| 7 | 97192 | Can you explain that? | 0.000 | 0.000 | 0.000 |
| 8 | 75057 | How does it work? | 0.000 | 0.000 | 0.000 |
| 9 | 23864 | Why is that? | 0.000 | 0.000 | 0.000 |
| 10 | 23604 | good food for underactive that | 0.000 | 0.000 | 0.000 |

### Adversarial

| Rank | Query ID | Query | Avg P@5 | Min P@5 | Max P@5 |
|------|----------|-------|---------|---------|----------|
| 1 | 31100 | what is a heart scan in detail but quickly | 0.000 | 0.000 | 0.000 |
| 2 | 31985 | what is Oscillatoria generic name? pros and cons only | 0.000 | 0.000 | 0.000 |
| 3 | 93349 | does polygonum persicaria smell without technical jargon | 0.000 | 0.000 | 0.000 |
| 4 | 98704 | dyad definition synonym in simple terms | 0.050 | 0.000 | 0.200 |
| 5 | 23177 | when were traffic lights first used without using the word '... | 0.050 | 0.000 | 0.200 |
| 6 | 51829 | what is lactobacillus rhamnosus without using the word 'lact... | 0.150 | 0.000 | 0.200 |
| 7 | 48383 | what was the first planet to be discovered in detail but qui... | 0.150 | 0.000 | 0.200 |
| 8 | 34300 | what is cost of sales in one sentence | 0.200 | 0.200 | 0.200 |
| 9 | 22977 | did us postal rates change in detail but quickly | 0.200 | 0.200 | 0.200 |
| 10 | 55781 | how much does it cost to park at tampa airport without techn... | 0.200 | 0.200 | 0.200 |

---

## Failure Pattern Analysis

### By Query Length

| Length Category | Failures | Avg P@5 | Failure Rate |
|-----------------|----------|---------|-------------|
| short (<30 chars) | 63 | 0.000 | 100.0% |
| medium (30-60 chars) | 109 | 0.000 | 100.0% |
| long (>60 chars) | 4 | 0.000 | 100.0% |

### By Query Complexity

| Complexity | Failures | Avg P@5 | Failure Rate |
|------------|----------|---------|-------------|
| Simple | 176 | 0.000 | 100.0% |

### By Query Type

| Query Type | Failures |
|------------|----------|
| Description | 100 |
| Entity | 43 |
| Numeric | 29 |
| Location | 4 |

---

## Deep Dive: Context-Dependent Query Failures

### Overview

Context-dependent queries show **catastrophic failure** with an average Precision@5 of **0.000** (97% degradation from clean baseline).

### Why Do Context-Dependent Queries Fail?

#### Missing Context

**Severity:** `CRITICAL`

**Description:** Query references entities/concepts from previous conversation that are not present

**Impact:** Without conversation history, retrieval has no context to resolve pronouns or references

**Examples:**

- **Query:** "How does it work?" (Original: "did us postal rates change")
  - Precision@5: 0.000
- **Query:** "what are some criticisms of it" (Original: "what are some criticisms of impressionism")
  - Precision@5: 0.000
- **Query:** "salary range of a that" (Original: "salary range of a nutritionist")
  - Precision@5: 0.000
- **Query:** "what is a that scan" (Original: "what is a heart scan")
  - Precision@5: 0.000
- **Query:** "Can you explain that?" (Original: "symmetric vs asymmetric encryption")
  - Precision@5: 0.000

#### Ambiguous Intent

**Severity:** `HIGH`

**Description:** Query intent is unclear without previous context

**Impact:** Embedding similarity cannot find relevant passages when query is too vague

#### Incomplete Information

**Severity:** `HIGH`

**Description:** Query lacks key terms needed for retrieval

**Impact:** BM25 and embedding-based retrieval both fail due to lack of keywords

**Examples:**

- **Query:** "What about this?" (Original: "what is cost of sales")
  - Precision@5: 0.000
- **Query:** "What about this?" (Original: "what is lactobacillus rhamnosus")
  - Precision@5: 0.000
- **Query:** "Why is that?" (Original: "what is bean straw")
  - Precision@5: 0.000
- **Query:** "What happens next?" (Original: "what was the first planet to be discovered")
  - Precision@5: 0.000

### Query Characteristics

| Characteristic | Count | Percentage | Avg P@5 |
|----------------|-------|------------|----------|
| Pronoun References | 32 | 47.1% | 0.000 |
| Demonstrative References | 48 | 70.6% | 0.000 |
| Follow Up Questions | 32 | 47.1% | 0.000 |

### Common Query Patterns

- Starts with "what": **28 queries**
- Starts with "how": **12 queries**
- Starts with "can": **12 queries**
- Starts with "salary": **4 queries**
- Starts with "why": **4 queries**

---

## Recommendations

### For Context-Dependent Queries

1. **Implement Conversation History Tracking**
   - Maintain session state with previous queries and answers
   - Pass conversation context to retrieval and generation

2. **Query Expansion/Rewriting**
   - Use LLM to expand context-dependent queries with referenced entities
   - Resolve pronouns and demonstratives before retrieval

3. **Multi-Turn Dialogue Support**
   - Design RAG pipeline specifically for conversational contexts
   - Include conversation history in embeddings

### For General Robustness

1. **Enhance Query Preprocessing**
   - Implement spell correction for noisy queries
   - Expand abbreviations and acronyms

2. **Improve Retrieval Diversity**
   - Combine multiple retrieval strategies
   - Use query expansion for ambiguous queries

3. **Fine-tune Embeddings**
   - Train embeddings on domain-specific data
   - Consider using instruction-tuned embedding models

