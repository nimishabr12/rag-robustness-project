"""
Multi-Model Answer Generator

Supports OpenAI, Google Gemini, and Anthropic Claude models.
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MultiModelGenerator:
    """
    Answer generator supporting multiple LLM providers
    """

    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize multi-model generator

        Args:
            provider: "openai", "gemini", or "anthropic"
            model: Specific model name (uses defaults if None)
        """
        self.provider = provider.lower()

        # Initialize based on provider
        if self.provider == "openai":
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-3.5-turbo"

        elif self.provider == "gemini":
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
            genai.configure(api_key=api_key)
            self.model = model or "gemini-pro"
            self.client = genai.GenerativeModel(self.model)

        elif self.provider == "anthropic":
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found")
            self.client = Anthropic(api_key=api_key)
            self.model = model or "claude-3-haiku-20240307"  # Fast and cost-effective

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate_answer(
        self,
        query: str,
        passages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 300
    ) -> Dict:
        """
        Generate answer using selected provider

        Args:
            query: User's query
            passages: Retrieved passages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with answer and metadata
        """
        if not passages:
            return {
                'answer': "I don't have enough information to answer this question.",
                'model': f"{self.provider}/{self.model}",
                'provider': self.provider,
                'passages_used': 0,
                'confidence': 'low'
            }

        # Extract passage texts
        passage_texts = []
        for i, p in enumerate(passages):
            if isinstance(p.get('passage'), dict):
                text = p['passage'].get('text', '')
            else:
                text = p.get('text', '')

            if text:
                passage_texts.append(f"[{i+1}] {text}")

        context = "\n\n".join(passage_texts)

        # Generate based on provider
        if self.provider == "openai":
            return self._generate_openai(query, context, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._generate_gemini(query, context, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._generate_anthropic(query, context, temperature, max_tokens)

    def _generate_openai(self, query, context, temperature, max_tokens):
        """Generate answer using OpenAI"""
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
            confidence = self._estimate_confidence(answer)

            return {
                'answer': answer,
                'model': f"openai/{self.model}",
                'provider': 'openai',
                'passages_used': len(context.split('\n\n')),
                'confidence': confidence,
                'finish_reason': response.choices[0].finish_reason
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': f"openai/{self.model}",
                'provider': 'openai',
                'passages_used': 0,
                'confidence': 'error',
                'error': str(e)
            }

    def _generate_gemini(self, query, context, temperature, max_tokens):
        """Generate answer using Google Gemini"""
        prompt = f"""You are a helpful assistant that answers questions based on provided context passages.

Follow these guidelines:
1. Only use information from the provided passages
2. If the passages don't contain enough information, say so
3. Be concise and direct
4. Cite passage numbers [1], [2], etc.

Context passages:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            answer = response.text.strip()
            confidence = self._estimate_confidence(answer)

            return {
                'answer': answer,
                'model': f"gemini/{self.model}",
                'provider': 'gemini',
                'passages_used': len(context.split('\n\n')),
                'confidence': confidence
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': f"gemini/{self.model}",
                'provider': 'gemini',
                'passages_used': 0,
                'confidence': 'error',
                'error': str(e)
            }

    def _generate_anthropic(self, query, context, temperature, max_tokens):
        """Generate answer using Anthropic Claude"""
        system_prompt = """You are a helpful assistant that answers questions based on provided context passages.

Follow these guidelines:
1. Only use information from the provided passages
2. If the passages don't contain enough information, say so
3. Be concise and direct
4. Cite passage numbers [1], [2], etc."""

        user_prompt = f"""Context passages:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            answer = response.content[0].text.strip()
            confidence = self._estimate_confidence(answer)

            return {
                'answer': answer,
                'model': f"anthropic/{self.model}",
                'provider': 'anthropic',
                'passages_used': len(context.split('\n\n')),
                'confidence': confidence,
                'stop_reason': response.stop_reason
            }

        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'model': f"anthropic/{self.model}",
                'provider': 'anthropic',
                'passages_used': 0,
                'confidence': 'error',
                'error': str(e)
            }

    def _estimate_confidence(self, answer: str) -> str:
        """Estimate confidence based on answer content"""
        uncertainty_phrases = [
            "don't have enough information",
            "cannot answer",
            "not enough information",
            "passages don't contain",
            "unclear from",
            "not specified"
        ]

        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            return 'low'
        return 'high'


def test_all_models():
    """Quick test of all three providers"""
    print("Testing all model providers...\n")

    test_query = "what is cost of sales"
    test_passages = [
        {
            'passage': {
                'text': "The cost of sales, also referred to as the cost of goods sold, is a measure of how much it costs a company to sell its products."
            }
        }
    ]

    for provider in ["openai", "gemini", "anthropic"]:
        print(f"\n{'='*60}")
        print(f"Testing {provider.upper()}")
        print('='*60)

        try:
            generator = MultiModelGenerator(provider=provider)
            result = generator.generate_answer(test_query, test_passages)

            print(f"Model: {result['model']}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']}")

        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    test_all_models()
