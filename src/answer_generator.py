"""
Answer Generator for RAG System

Generates answers from retrieved passages using OpenAI API.
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AnswerGenerator:
    """
    Generates answers from retrieved passages using OpenAI's GPT models
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the answer generator

        Args:
            api_key: OpenAI API key (if None, will load from environment)
            model: Model to use for generation (default: gpt-3.5-turbo for cost efficiency)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_answer(
        self,
        query: str,
        passages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 300
    ) -> Dict:
        """
        Generate an answer to a query based on retrieved passages

        Args:
            query: User's query
            passages: List of retrieved passage dictionaries with 'passage' and 'score' keys
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in the response

        Returns:
            Dictionary with 'answer', 'model', and 'passages_used' keys
        """
        if not passages:
            return {
                'answer': "I don't have enough information to answer this question.",
                'model': self.model,
                'passages_used': 0,
                'confidence': 'low'
            }

        # Extract passage texts and format them
        passage_texts = []
        for i, p in enumerate(passages):
            # Handle both formats: p['passage'] could be a dict or the passage might be top-level
            if isinstance(p.get('passage'), dict):
                text = p['passage'].get('text', '')
            else:
                text = p.get('text', '')

            if text:
                passage_texts.append(f"[{i+1}] {text}")

        # Construct the prompt
        context = "\n\n".join(passage_texts)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context passages.
Follow these guidelines:
1. Only use information from the provided passages
2. If the passages don't contain enough information to answer the question, say so
3. Be concise and direct
4. Cite passage numbers [1], [2], etc. when referencing specific information
5. If passages contradict each other, acknowledge this"""

        user_prompt = f"""Context passages:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content.strip()

            # Estimate confidence based on whether answer indicates uncertainty
            uncertainty_phrases = [
                "don't have enough information",
                "cannot answer",
                "not enough information",
                "passages don't contain",
                "unclear from",
                "not specified"
            ]

            confidence = 'low' if any(phrase in answer.lower() for phrase in uncertainty_phrases) else 'high'

            return {
                'answer': answer,
                'model': self.model,
                'passages_used': len(passages),
                'confidence': confidence,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': self.model,
                'passages_used': len(passages),
                'confidence': 'error',
                'error': str(e)
            }

    def generate_answer_with_query_context(
        self,
        query: str,
        passages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 300
    ) -> Dict:
        """
        Generate an answer with conversation context for context-dependent queries

        Args:
            query: Current query
            passages: Retrieved passages
            conversation_history: List of previous {query, answer} exchanges
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer and metadata
        """
        if not passages:
            return {
                'answer': "I don't have enough information to answer this question.",
                'model': self.model,
                'passages_used': 0,
                'confidence': 'low'
            }

        # Format passages
        passage_texts = []
        for i, p in enumerate(passages):
            if isinstance(p.get('passage'), dict):
                text = p['passage'].get('text', '')
            else:
                text = p.get('text', '')

            if text:
                passage_texts.append(f"[{i+1}] {text}")

        context = "\n\n".join(passage_texts)

        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n\nPrevious conversation:\n"
            for i, turn in enumerate(conversation_history[-3:]):  # Keep last 3 turns
                conversation_context += f"Q{i+1}: {turn.get('query', '')}\n"
                conversation_context += f"A{i+1}: {turn.get('answer', '')}\n"

        system_prompt = """You are a helpful assistant that answers questions based on provided context passages and conversation history.
Follow these guidelines:
1. Use both the passages and conversation history to understand the current question
2. If the current question refers to "it", "this", "that", use conversation history to resolve the reference
3. Be concise and direct
4. Cite passage numbers when referencing information
5. If you cannot answer with the given context, say so"""

        user_prompt = f"""Context passages:
{context}
{conversation_context}

Current question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content.strip()

            uncertainty_phrases = [
                "don't have enough information",
                "cannot answer",
                "not enough information",
                "passages don't contain",
                "unclear from",
                "not specified"
            ]

            confidence = 'low' if any(phrase in answer.lower() for phrase in uncertainty_phrases) else 'high'

            return {
                'answer': answer,
                'model': self.model,
                'passages_used': len(passages),
                'confidence': confidence,
                'finish_reason': response.choices[0].finish_reason,
                'used_conversation_context': bool(conversation_history)
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': self.model,
                'passages_used': len(passages),
                'confidence': 'error',
                'error': str(e)
            }

    def batch_generate_answers(
        self,
        queries_and_passages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 300
    ) -> List[Dict]:
        """
        Generate answers for multiple query-passage pairs

        Args:
            queries_and_passages: List of dicts with 'query' and 'passages' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response

        Returns:
            List of answer dictionaries
        """
        results = []

        for i, item in enumerate(queries_and_passages):
            query = item.get('query', '')
            passages = item.get('passages', [])

            print(f"Generating answer {i+1}/{len(queries_and_passages)}...")

            answer_result = self.generate_answer(
                query=query,
                passages=passages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Add query info to result
            answer_result['query'] = query
            answer_result['query_id'] = item.get('query_id', None)

            results.append(answer_result)

        return results
