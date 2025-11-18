"""
RAG Pipeline with Multiple Retrieval Strategies
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
import pickle

# Load environment variables
load_dotenv()

class RAGPipeline:
    """
    RAG Pipeline with multiple retrieval strategies
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the RAG pipeline

        Args:
            api_key: OpenAI API key (if None, will load from environment)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"
        self.gpt_model = "gpt-3.5-turbo"

        # Storage for passages and embeddings
        self.passages: List[Dict] = []
        self.passage_texts: List[str] = []
        self.passage_embeddings: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25: Optional[BM25Okapi] = None

    def load_passages_from_msmarco(self, filepath: str = 'data/ms_marco_sample.json'):
        """
        Load passages from MS MARCO sample file

        Args:
            filepath: Path to MS MARCO sample JSON file
        """
        print(f"Loading passages from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract all passages with metadata
        for item in data:
            query_id = item['query_id']
            query = item['query']
            passages_data = item['passages']

            if 'passage_text' in passages_data:
                for idx, passage_text in enumerate(passages_data['passage_text']):
                    self.passages.append({
                        'query_id': query_id,
                        'query': query,
                        'passage_id': f"{query_id}_{idx}",
                        'text': passage_text,
                        'url': passages_data['url'][idx] if 'url' in passages_data else '',
                        'is_selected': passages_data['is_selected'][idx] if 'is_selected' in passages_data else 0
                    })
                    self.passage_texts.append(passage_text)

        print(f"Loaded {len(self.passages)} passages")

    def build_vector_store(self):
        """
        Build FAISS vector store with OpenAI embeddings
        """
        if not self.passages:
            raise ValueError("No passages loaded. Call load_passages_from_msmarco() first")

        print("Generating embeddings for all passages...")
        embeddings_list = []
        batch_size = 100

        for i in range(0, len(self.passage_texts), batch_size):
            batch = self.passage_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.passage_texts)-1)//batch_size + 1}")

            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings_list.extend(batch_embeddings)

        # Convert to numpy array
        self.passage_embeddings = np.array(embeddings_list, dtype=np.float32)

        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.passage_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.passage_embeddings)

        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def build_bm25_index(self):
        """
        Build BM25 index for hybrid retrieval
        """
        if not self.passage_texts:
            raise ValueError("No passages loaded")

        print("Building BM25 index...")
        # Tokenize passages for BM25
        tokenized_passages = [text.lower().split() for text in self.passage_texts]
        self.bm25 = BM25Okapi(tokenized_passages)
        print("BM25 index built")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text using OpenAI API

        Args:
            text: Input text

        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def naive_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple embedding similarity search

        Args:
            query: Search query
            top_k: Number of passages to retrieve

        Returns:
            List of retrieved passages with relevance scores
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_vector_store() first")

        # Get query embedding
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search in FAISS
        distances, indices = self.faiss_index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            # Convert L2 distance to similarity score (higher is better)
            similarity_score = 1.0 / (1.0 + distance)
            results.append({
                'passage': self.passages[idx],
                'score': float(similarity_score),
                'strategy': 'naive_retrieval'
            })

        return results

    def query_rewrite_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Use GPT to clean/expand the query first, then retrieve

        Args:
            query: Search query (possibly noisy or ambiguous)
            top_k: Number of passages to retrieve

        Returns:
            List of retrieved passages with relevance scores
        """
        # Use GPT to rewrite/expand the query
        rewrite_prompt = f"""Given the following search query, rewrite it to be clearer and more specific.
If the query has typos, fix them. If it's ambiguous, expand it with likely interpretations.
If it's a follow-up question, convert it to a standalone question.
Only output the rewritten query, nothing else.

Original query: {query}

Rewritten query:"""

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites search queries to be clearer and more effective."},
                {"role": "user", "content": rewrite_prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )

        rewritten_query = response.choices[0].message.content.strip()

        # Now perform naive retrieval with the rewritten query
        query_embedding = self.get_embedding(rewritten_query)
        query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            similarity_score = 1.0 / (1.0 + distance)
            results.append({
                'passage': self.passages[idx],
                'score': float(similarity_score),
                'strategy': 'query_rewrite_retrieval',
                'rewritten_query': rewritten_query
            })

        return results

    def hybrid_retrieval(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        """
        Combine embedding similarity with BM25 keyword matching

        Args:
            query: Search query
            top_k: Number of passages to retrieve
            alpha: Weight for embedding similarity (1-alpha for BM25)

        Returns:
            List of retrieved passages with combined scores
        """
        if self.faiss_index is None or self.bm25 is None:
            raise ValueError("Indexes not built. Call build_vector_store() and build_bm25_index() first")

        # Get embedding-based scores
        query_embedding = self.get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search more passages for reranking
        distances, indices = self.faiss_index.search(query_embedding, top_k * 3)

        # Convert L2 distances to similarity scores
        embedding_scores = 1.0 / (1.0 + distances[0])

        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to [0, 1]
        embedding_scores_norm = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-10)
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)

        # Combine scores for the retrieved passages
        combined_results = []
        for i, idx in enumerate(indices[0]):
            combined_score = alpha * embedding_scores_norm[i] + (1 - alpha) * bm25_scores_norm[idx]
            combined_results.append({
                'index': idx,
                'combined_score': combined_score,
                'embedding_score': float(embedding_scores_norm[i]),
                'bm25_score': float(bm25_scores_norm[idx])
            })

        # Sort by combined score and take top_k
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        results = []
        for item in combined_results[:top_k]:
            results.append({
                'passage': self.passages[item['index']],
                'score': float(item['combined_score']),
                'embedding_score': item['embedding_score'],
                'bm25_score': item['bm25_score'],
                'strategy': 'hybrid_retrieval'
            })

        return results

    def multistep_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve → analyze → retrieve again if needed

        Args:
            query: Search query
            top_k: Number of final passages to retrieve

        Returns:
            List of retrieved passages with relevance scores
        """
        # Step 1: Initial retrieval
        initial_results = self.naive_retrieval(query, top_k=top_k)

        # Step 2: Analyze if we need to refine the search
        initial_passages = [r['passage']['text'] for r in initial_results]

        analysis_prompt = f"""Given the search query and initial retrieved passages, determine if the results are relevant.
If they are not relevant, suggest a better reformulated query to find more relevant information.

Query: {query}

Retrieved passages:
{chr(10).join([f"{i+1}. {p[:200]}..." for i, p in enumerate(initial_passages)])}

Are these passages relevant to the query? If not, suggest a reformulated query.
Output format:
RELEVANT: yes/no
REFORMULATED_QUERY: [only if not relevant, otherwise leave empty]"""

        response = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates search relevance."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        analysis = response.choices[0].message.content.strip()

        # Check if we need to retrieve again
        if "RELEVANT: no" in analysis or "RELEVANT:no" in analysis:
            # Extract reformulated query
            lines = analysis.split('\n')
            reformulated_query = query
            for line in lines:
                if "REFORMULATED_QUERY:" in line:
                    reformulated_query = line.split("REFORMULATED_QUERY:")[-1].strip()
                    break

            # Step 3: Retrieve again with reformulated query
            if reformulated_query and reformulated_query != query:
                query_embedding = self.get_embedding(reformulated_query)
                query_embedding = query_embedding.reshape(1, -1)

                distances, indices = self.faiss_index.search(query_embedding, top_k)

                results = []
                for idx, distance in zip(indices[0], distances[0]):
                    similarity_score = 1.0 / (1.0 + distance)
                    results.append({
                        'passage': self.passages[idx],
                        'score': float(similarity_score),
                        'strategy': 'multistep_retrieval',
                        'reformulated_query': reformulated_query,
                        'steps': 2
                    })

                return results

        # If initial results are relevant, return them
        for result in initial_results:
            result['strategy'] = 'multistep_retrieval'
            result['steps'] = 1

        return initial_results

    def save_index(self, index_dir: str = 'data/faiss_index'):
        """
        Save FAISS index and metadata to disk

        Args:
            index_dir: Directory to save index files
        """
        os.makedirs(index_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(index_dir, 'index.faiss'))

        # Save metadata
        with open(os.path.join(index_dir, 'passages.pkl'), 'wb') as f:
            pickle.dump(self.passages, f)

        with open(os.path.join(index_dir, 'passage_texts.pkl'), 'wb') as f:
            pickle.dump(self.passage_texts, f)

        with open(os.path.join(index_dir, 'bm25.pkl'), 'wb') as f:
            pickle.dump(self.bm25, f)

        print(f"Index saved to {index_dir}")

    def load_index(self, index_dir: str = 'data/faiss_index'):
        """
        Load FAISS index and metadata from disk

        Args:
            index_dir: Directory containing index files
        """
        # Load FAISS index
        self.faiss_index = faiss.read_index(os.path.join(index_dir, 'index.faiss'))

        # Load metadata
        with open(os.path.join(index_dir, 'passages.pkl'), 'rb') as f:
            self.passages = pickle.load(f)

        with open(os.path.join(index_dir, 'passage_texts.pkl'), 'rb') as f:
            self.passage_texts = pickle.load(f)

        with open(os.path.join(index_dir, 'bm25.pkl'), 'rb') as f:
            self.bm25 = pickle.load(f)

        print(f"Index loaded from {index_dir}")
        print(f"Loaded {len(self.passages)} passages")


def initialize_rag_pipeline(force_rebuild: bool = False) -> RAGPipeline:
    """
    Initialize RAG pipeline, building or loading indexes as needed

    Args:
        force_rebuild: If True, rebuild indexes even if they exist

    Returns:
        Initialized RAGPipeline instance
    """
    pipeline = RAGPipeline()

    index_dir = 'data/faiss_index'
    index_exists = os.path.exists(os.path.join(index_dir, 'index.faiss'))

    if index_exists and not force_rebuild:
        print("Loading existing index...")
        pipeline.load_index(index_dir)
    else:
        print("Building new index...")
        pipeline.load_passages_from_msmarco()
        pipeline.build_vector_store()
        pipeline.build_bm25_index()
        pipeline.save_index(index_dir)

    return pipeline
