# RAG Robustness Project

Benchmark for testing RAG system robustness against noisy, ambiguous, and adversarial queries

## Overview

This project aims to evaluate and improve the robustness of Retrieval-Augmented Generation (RAG) systems when facing challenging real-world scenarios. We test RAG systems against various types of problematic inputs including:

- **Noisy queries**: Queries with typos, grammatical errors, or unclear phrasing
- **Ambiguous queries**: Questions that could have multiple interpretations
- **Adversarial queries**: Intentionally crafted inputs designed to expose system weaknesses

## Project Structure

```
rag-robustness-project/
├── data/           # Test datasets and query collections
├── src/            # Source code for RAG system and testing framework
├── experiments/    # Experiment configurations and scripts
├── results/        # Experimental results and analysis
└── requirements.txt
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys (OpenAI, etc.)

## Goals

- Develop comprehensive test suites for RAG robustness evaluation
- Identify failure modes and edge cases in RAG systems
- Propose and implement robustness improvements
- Benchmark performance across different RAG architectures
