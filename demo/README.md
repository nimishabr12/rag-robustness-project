# RAG Robustness Interactive Demo

**Test RAG system robustness with an interactive web interface**

![Streamlit Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Overview

This interactive Streamlit demo allows you to:

- ✍️ **Enter custom queries** or use example queries
- 🔄 **Apply noise transformations** (typos, abbreviations, context-dependent, adversarial)
- 🎯 **Test retrieval strategies** (naive, query rewrite, hybrid, multistep)
- 📊 **View results** including retrieved passages, generated answers, and evaluation metrics

## Quick Start

### 1. Install Dependencies

If you haven't already, install the required packages:

```bash
pip install -r ../requirements.txt
```

### 2. Set Up API Key

Make sure you have your OpenAI API key configured in the `.env` file:

```bash
# From the project root directory
cp .env.template .env
# Edit .env and add your API key
```

### 3. Run the Demo

From the project root directory:

```bash
streamlit run demo/app.py
```

Or from the `demo` directory:

```bash
cd demo
streamlit run app.py
```

The app will open automatically in your web browser at `http://localhost:8501`

## Features

### Query Input
- Enter any question you want to test
- Try the provided example queries
- See how different noise types affect your query

### Noise Types

1. **Clean (No Noise)** - Original query unchanged
2. **Noisy (Typos)** - Introduces spelling errors (15% error rate)
3. **Ambiguous (Abbreviations)** - Shortens query to abbreviations
4. **Context-Dependent** - Converts to follow-up questions with pronouns
5. **Adversarial (Constraints)** - Adds conflicting requirements

### Retrieval Strategies

1. **Naive Retrieval** - Simple embedding similarity
2. **Query Rewrite** - LLM-based query cleaning
3. **Hybrid Retrieval** - Dense + sparse (BM25) combination
4. **Multistep Retrieval** - Iterative retrieval with feedback

### Results Display

**Retrieved Passages:**
- Top K passages ranked by relevance
- Relevance scores for each passage
- Optional full text display

**Generated Answer:**
- AI-generated answer based on retrieved passages
- Confidence level indicator
- Model information

**Evaluation Metrics:**
- Precision@K (if query exists in ground truth)
- Answer quality score
- Comparison to baseline performance

## Example Usage

### Example 1: Testing Typo Resilience

1. Enter query: `what is cost of sales`
2. Select noise: `Noisy (Typos)`
3. Choose strategy: `Query Rewrite Retrieval`
4. Click "Run Query"

**Expected Result:** Query rewrite should correct typos and retrieve relevant passages.

### Example 2: Context-Dependent Failure

1. Enter query: `what is symmetric encryption`
2. Select noise: `Context-Dependent`
3. Choose any strategy
4. Click "Run Query"

**Expected Result:** All strategies fail (low precision) because query becomes "How does it work?" without context.

### Example 3: Adversarial Resilience

1. Enter query: `where is sahel`
2. Select noise: `Adversarial (Constraints)`
3. Choose strategy: `Naive Retrieval`
4. Click "Run Query"

**Expected Result:** Surprisingly good performance despite constraints.

## Screenshots

### Main Interface
The app provides a clean, intuitive interface with:
- Left sidebar for configuration
- Main area for results display
- Expandable sections for detailed information

### Query Transformation
See side-by-side comparison of:
- Original query
- Transformed query after noise application

### Retrieved Passages
View top passages with:
- Relevance scores
- Embedding and BM25 scores (for hybrid)
- Source URLs
- Expandable full text

### Generated Answers
AI-generated answers with:
- Confidence indicators (high/low)
- Color-coded display
- Model and passage count information

### Evaluation Metrics
When ground truth exists:
- Precision@K with baseline comparison
- Number of relevant passages retrieved
- Answer quality score

## Advanced Options

Click "Advanced Options" in the sidebar to:
- Adjust number of passages to retrieve (1-10)
- Toggle full passage text display
- View additional metadata

## Troubleshooting

### "Error initializing system"

**Solution:** Make sure you have:
1. Set your `OPENAI_API_KEY` in the `.env` file
2. Installed all dependencies from `requirements.txt`
3. Downloaded the MS MARCO dataset (runs automatically on first use)

### Slow First Run

**Expected:** The first run takes 1-2 minutes to:
- Download and process the MS MARCO dataset
- Build the FAISS index
- Load the BM25 index

**Solution:** Subsequent runs will be much faster (cached).

### "No passages retrieved"

**Possible Causes:**
1. Query is too short or ambiguous
2. No relevant passages in the dataset
3. Noise transformation removed too much information

**Solution:** Try a different query or less extreme noise type.

## Technical Details

### Caching

The app uses Streamlit's `@st.cache_resource` to cache:
- RAG pipeline initialization
- FAISS and BM25 indexes
- Answer generator
- Evaluator

This significantly speeds up subsequent queries.

### Data Flow

```
User Query
    ↓
Apply Noise Transformation
    ↓
Retrieve Passages (selected strategy)
    ↓
Generate Answer (OpenAI GPT-3.5-turbo)
    ↓
Evaluate (if ground truth exists)
    ↓
Display Results
```

### Performance

- **First query:** ~30-60 seconds (loading indexes)
- **Subsequent queries:** ~2-5 seconds per query
- **With query rewrite:** +1-2 seconds (extra LLM call)

## Customization

### Modify Example Queries

Edit the example queries in `app.py`:

```python
default_query = "your custom query"
```

### Add Custom Noise Types

Implement new noise generators in `src/noise_generators.py` and add them to the noise type selection.

### Change Model

Modify the answer generator to use a different model:

```python
answer_generator = AnswerGenerator(model="gpt-4")  # Use GPT-4 instead
```

## Resources

- **Full Results:** [results/RESULTS.md](../results/RESULTS.md)
- **Failure Analysis:** [results/failure_report.md](../results/failure_report.md)
- **Main README:** [README.md](../README.md)
- **GitHub:** [rag-robustness-project](https://github.com/nimishabr12/rag-robustness-project)

## Keyboard Shortcuts

When using the Streamlit app:

- `R` - Rerun the app
- `C` - Clear cache
- `S` - Take a screenshot
- `?` - Show keyboard shortcuts

## Deployment

To deploy this demo publicly:

### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set environment variables (OPENAI_API_KEY)
5. Deploy!

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "demo/app.py"]
```

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages in the app
3. Open an issue on [GitHub](https://github.com/nimishabr12/rag-robustness-project/issues)

---

**Enjoy testing RAG robustness!** 🚀
