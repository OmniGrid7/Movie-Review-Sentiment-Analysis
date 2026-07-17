# Movie Review Sentiment Analysis (BERT) + Agentic Recommendation System

Fine-tunes `bert-base-uncased` on a subset of IMDb reviews to classify sentiment, then plugs that model into a LangChain-based agentic movie recommendation system (structured to match the Coding Blocks research paper intelligence notebook pattern - tools + `create_agent` + `ChatGroq`).

## Running it
Upload `notebook.ipynb` to Google Colab, switch the runtime to GPU (Runtime > Change runtime type > T4 GPU), then run all cells top to bottom. When you reach the agent section you'll need a **Groq API key** (free at console.groq.com) - either add it to Colab's secret manager as `GROQ_API_KEY`, or paste it directly into the `GROQ_API_KEY` variable, or leave it blank and it'll prompt you.

## Dataset
Uses a balanced sample instead of the full 50k reviews so it trains quickly:
- 4000 train
- 1000 validation
- 1000 test

Change `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE` near the top if you want to scale up. The full 25k/25k split will get closer to 92-95% accuracy but takes much longer to train.

## What's in the notebook
1. Setup
2. Load data (sampled, balanced subset of IMDb)
3. Quick EDA - label balance, review lengths, common words, word cloud
4. Minimal cleaning (just HTML tag removal, since BERT tokenizes on its own)
5. Tokenization
6. Fine-tuning with the Hugging Face `Trainer`
7. Evaluation - accuracy, precision, recall, F1, ROC-AUC, confusion matrix
8. Try it on your own review text
9. Save the model
10. Load it back and confirm it still works
11. LIME explainability - see which words drove a prediction
12. **Agentic movie recommendation system** - LangChain tools + `create_agent` + `ChatGroq`, plus a manual tool-calling walkthrough and a critic pass
13. Notes / what to try next

## The agent system
Built with LangChain, matching the tools-and-agent pattern from the reference notebook:
- `search_movies`, `analyze_review_sentiment`, `get_movie_details` are `@tool`-decorated functions
- `analyze_review_sentiment` calls the **BERT model trained earlier in the notebook**, not the LLM
- `create_agent(model=llm, tools=tools)` builds the agent; `llm` is `ChatGroq` (`llama-3.1-8b-instant`)
- a manual walkthrough (`bind_tools`, reading `tool_calls`, building a `ToolMessage`) shows what the agent is doing internally
- a separate critic pass (a plain `llm.invoke` with a `SystemMessage` of rules) checks the recommendations against the original request

The movie database is a small hand-written sample (~10 movies) for demo purposes. Swap `search_movies`/`get_movie_details` for real API calls (TMDB, OMDb) if you want it working with live data - nothing else in the pipeline needs to change.

## Output files
Running the notebook creates:
```
data/imdb_dataset.csv
models/bert_sentiment_model/
models/tokenizer/
outputs/label_distribution.png
outputs/review_length_histogram.png
outputs/wordcloud.png
outputs/training_loss.png
outputs/confusion_matrix.png
outputs/evaluation_report.txt
```

Expect roughly 88-93% test accuracy on the default reduced subset.
