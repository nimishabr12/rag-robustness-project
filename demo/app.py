"""
Interactive Streamlit Demo for RAG Robustness Testing

Run with: streamlit run demo/app.py
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import json
from src.rag_pipeline import initialize_rag_pipeline
from src.answer_generator import AnswerGenerator
from src.evaluation import RAGEvaluator
from src.noise_generators import add_typos, make_ambiguous, make_context_dependent, make_adversarial

# Page configuration
st.set_page_config(
    page_title="RAG Robustness Demo",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .passage-box {
        background-color: #ffffff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load RAG pipeline (cached)"""
    with st.spinner("Loading RAG pipeline... (this may take a minute on first run)"):
        pipeline = initialize_rag_pipeline()
    return pipeline


@st.cache_resource
def load_answer_generator():
    """Load answer generator (cached)"""
    return AnswerGenerator()


@st.cache_resource
def load_evaluator():
    """Load evaluator (cached)"""
    evaluator = RAGEvaluator()
    evaluator.load_ground_truth()
    return evaluator


def apply_noise(query, noise_type):
    """Apply selected noise type to query"""
    if noise_type == "Clean (No Noise)":
        return query
    elif noise_type == "Noisy (Typos)":
        return add_typos(query, typo_rate=0.15)
    elif noise_type == "Ambiguous (Abbreviations)":
        return make_ambiguous(query)
    elif noise_type == "Context-Dependent":
        return make_context_dependent(query)
    elif noise_type == "Adversarial (Constraints)":
        return make_adversarial(query)
    return query


def get_strategy_description(strategy):
    """Get description of retrieval strategy"""
    descriptions = {
        "naive_retrieval": "Simple embedding similarity using FAISS",
        "query_rewrite_retrieval": "LLM-based query cleaning before retrieval",
        "hybrid_retrieval": "Combined dense (embeddings) + sparse (BM25)",
        "multistep_retrieval": "Iterative retrieval with relevance feedback"
    }
    return descriptions.get(strategy, "")


def main():
    # Header
    st.markdown('<p class="main-header">🔍 RAG Robustness Testing Demo</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Test how different retrieval strategies handle noisy, ambiguous, and adversarial queries</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize components
    try:
        pipeline = load_pipeline()
        answer_generator = load_answer_generator()
        evaluator = load_evaluator()
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.info("Make sure you have set your OPENAI_API_KEY in the .env file")
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Input query
    st.sidebar.subheader("1. Enter Your Query")
    default_query = "what is cost of sales"
    user_query = st.sidebar.text_input(
        "Query:",
        value=default_query,
        help="Enter any question you want to test"
    )

    # Noise type selection
    st.sidebar.subheader("2. Select Noise Type")
    noise_type = st.sidebar.selectbox(
        "Noise Type:",
        [
            "Clean (No Noise)",
            "Noisy (Typos)",
            "Ambiguous (Abbreviations)",
            "Context-Dependent",
            "Adversarial (Constraints)"
        ],
        help="Choose how to transform the query"
    )

    # Strategy selection
    st.sidebar.subheader("3. Choose Retrieval Strategy")
    strategy = st.sidebar.selectbox(
        "Strategy:",
        [
            "naive_retrieval",
            "query_rewrite_retrieval",
            "hybrid_retrieval",
            "multistep_retrieval"
        ],
        format_func=lambda x: x.replace('_', ' ').title(),
        help="Select the retrieval strategy to test"
    )

    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options"):
        top_k = st.slider("Number of passages to retrieve", 1, 10, 5)
        show_full_passages = st.checkbox("Show full passage text", value=False)

    # Run button
    run_query = st.sidebar.button("🚀 Run Query", type="primary", use_container_width=True)

    # Main content area
    if run_query and user_query:
        # Apply noise transformation
        transformed_query = apply_noise(user_query, noise_type)

        # Display query transformation
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Original Query")
            st.code(user_query, language=None)

        with col2:
            st.subheader("🔄 Transformed Query")
            if transformed_query != user_query:
                st.code(transformed_query, language=None)
                st.caption(f"Applied: {noise_type}")
            else:
                st.code(transformed_query, language=None)
                st.caption("No transformation applied")

        st.markdown("---")

        # Run retrieval
        with st.spinner(f"Running {strategy.replace('_', ' ')}..."):
            try:
                # Retrieve passages
                if strategy == "naive_retrieval":
                    results = pipeline.naive_retrieval(transformed_query, top_k=top_k)
                elif strategy == "query_rewrite_retrieval":
                    results = pipeline.query_rewrite_retrieval(transformed_query, top_k=top_k)
                elif strategy == "hybrid_retrieval":
                    results = pipeline.hybrid_retrieval(transformed_query, top_k=top_k)
                elif strategy == "multistep_retrieval":
                    results = pipeline.multistep_retrieval(transformed_query, top_k=top_k)

                # Display strategy info
                st.info(f"**Strategy:** {strategy.replace('_', ' ').title()} - {get_strategy_description(strategy)}")

                # Show rewritten query if applicable
                if strategy == "query_rewrite_retrieval" and results and 'rewritten_query' in results[0]:
                    st.success(f"**Rewritten Query:** {results[0]['rewritten_query']}")

                # Display retrieved passages
                st.subheader("📚 Retrieved Passages")

                if results:
                    for i, result in enumerate(results):
                        passage = result.get('passage', {})
                        score = result.get('score', 0.0)
                        passage_text = passage.get('text', 'N/A')

                        with st.expander(f"**Passage {i+1}** - Score: {score:.4f}", expanded=(i == 0)):
                            # Display passage text
                            if show_full_passages:
                                st.markdown(f"<div class='passage-box'>{passage_text}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='passage-box'>{passage_text[:300]}...</div>", unsafe_allow_html=True)

                            # Additional metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Score", f"{score:.4f}")
                            with col2:
                                if 'embedding_score' in result:
                                    st.metric("Embedding", f"{result['embedding_score']:.4f}")
                            with col3:
                                if 'bm25_score' in result:
                                    st.metric("BM25", f"{result['bm25_score']:.4f}")

                            if passage.get('url'):
                                st.caption(f"Source: {passage['url']}")
                else:
                    st.warning("No passages retrieved")

                st.markdown("---")

                # Generate answer
                st.subheader("💡 Generated Answer")

                with st.spinner("Generating answer..."):
                    answer_result = answer_generator.generate_answer(
                        query=transformed_query,
                        passages=results
                    )

                    # Display answer
                    answer = answer_result.get('answer', 'No answer generated')
                    confidence = answer_result.get('confidence', 'unknown')

                    # Color code by confidence
                    if confidence == 'high':
                        st.success(answer)
                    elif confidence == 'low':
                        st.warning(answer)
                    else:
                        st.info(answer)

                    # Answer metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", confidence.title())
                    with col2:
                        st.metric("Passages Used", answer_result.get('passages_used', 0))
                    with col3:
                        st.metric("Model", answer_result.get('model', 'N/A'))

                st.markdown("---")

                # Evaluation (if ground truth exists)
                st.subheader("📊 Evaluation Metrics")

                # Try to find matching query in ground truth
                query_id = None
                for qid, data in evaluator.ground_truth.items():
                    if data['query'].lower() == user_query.lower():
                        query_id = qid
                        break

                if query_id:
                    # Calculate metrics
                    precision_metrics = evaluator.calculate_precision_at_k(
                        results, query_id, k=top_k
                    )

                    answer_quality = evaluator.calculate_answer_quality(
                        answer, query_id
                    )

                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        p_at_k = precision_metrics.get('precision_at_k', 0.0)
                        st.metric(
                            f"Precision@{top_k}",
                            f"{p_at_k:.3f}",
                            delta=f"{(p_at_k - 0.143) * 100:.1f}% vs baseline" if p_at_k > 0 else None
                        )

                    with col2:
                        st.metric(
                            "Relevant Retrieved",
                            precision_metrics.get('relevant_retrieved', 0)
                        )

                    with col3:
                        st.metric(
                            "Total Relevant",
                            precision_metrics.get('total_relevant', 0)
                        )

                    with col4:
                        if answer_quality.get('similarity_score') is not None:
                            st.metric(
                                "Answer Quality",
                                f"{answer_quality['similarity_score']:.3f}"
                            )

                    # Ground truth info
                    with st.expander("📖 Ground Truth Information"):
                        gt = evaluator.ground_truth[query_id]
                        st.write(f"**Query Type:** {gt['query_type']}")
                        if gt['answers']:
                            st.write(f"**Reference Answer:** {gt['answers'][0]}")
                        st.write(f"**Relevant Passages:** {len(gt['relevant_passage_indices'])} out of {len(gt['all_passages'])}")

                else:
                    st.info("This query is not in the MS MARCO ground truth dataset, so precision metrics cannot be calculated.")
                    st.caption("You can still see the retrieved passages and generated answer above.")

            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # Information section
    else:
        # Welcome message and instructions
        st.info("""
        ### 👋 Welcome to the RAG Robustness Testing Demo!

        This interactive demo lets you test how different retrieval strategies handle various types of query noise.

        **How to use:**
        1. Enter your query in the sidebar (or use the default)
        2. Select a noise type to apply
        3. Choose a retrieval strategy
        4. Click "Run Query" to see results

        **What you'll see:**
        - How the query is transformed by the selected noise type
        - Top retrieved passages with relevance scores
        - AI-generated answer based on the passages
        - Evaluation metrics (if the query exists in our ground truth dataset)
        """)

        # Show example queries
        st.subheader("📝 Example Queries to Try")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Simple Queries:**
            - what is cost of sales
            - where is sahel
            - what is lactobacillus rhamnosus

            **Complex Queries:**
            - why is pure crystalline silicon an electrical insulator
            - what are some criticisms of impressionism
            """)

        with col2:
            st.markdown("""
            **Noisy/Ambiguous:**
            - whta is a heart scan (typo)
            - sr (abbreviation)

            **Context-Dependent:**
            - How does it work?
            - Tell me more
            """)

        # Project statistics
        st.markdown("---")
        st.subheader("📊 Project Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Queries Tested", "400")
        with col2:
            st.metric("Noise Types", "5")
        with col3:
            st.metric("Strategies", "4")
        with col4:
            st.metric("Avg Precision@5", "0.112")

        # Links
        st.markdown("---")
        st.markdown("""
        **📚 Learn More:**
        - [Full Results](../results/RESULTS.md)
        - [Failure Analysis](../results/failure_report.md)
        - [GitHub Repository](https://github.com/nimishabr12/rag-robustness-project)
        """)


if __name__ == "__main__":
    main()
