# Movie Review Sentiment Analysis (BERT) + Agentic Recommendation System

Fine-tunes `bert-base-uncased` on a subset of IMDb reviews to classify sentiment, then plugs that model into a LangChain-based agentic movie recommendation system (tools + `create_agent` + `ChatGroq`).

## Requirements
- A Google account (to run Colab)
- A **Groq API key** - free (only needed for the agent section)
- Nothing needs installing locally - the notebook installs everything it needs in its first cells

## Running it
1. Upload `notebook.ipynb` to Google Colab (File > Upload notebook, or drag-and-drop)
2. Runtime > Change runtime type > select **GPU (T4)**
3. Runtime > Run all

When you reach Part 12, you'll need the Groq key. Either:
- add it to Colab's secret manager (key icon in the left sidebar) as `GROQ_API_KEY`, or
- paste it directly into the `GROQ_API_KEY` variable in that cell, or
- leave it blank - the notebook will prompt you for it

## Dataset
Uses `mteb/imdb` - a mirror of the classic Large Movie Review Dataset with the same `text`/`label` columns. (The original `load_dataset("imdb")` uses a loading script Hugging Face has since deprecated, so it now fails for most people - `mteb/imdb` is the fix.)

Uses a balanced sample instead of the full 50k reviews so it trains quickly:
- 4000 train / 1000 validation / 1000 test

Change `TRAIN_SIZE`, `VAL_SIZE`, `TEST_SIZE` near the top of Part 2 to scale up.

## Results from an actual run
On the default settings (4000 train, 3 epochs, T4 GPU):

| Metric | Score |
|---|---|
| Accuracy | 90.3% |
| Precision | 89.1% |
| Recall | 91.8% |
| F1-score | 90.4% |
| ROC-AUC | 96.8% |

Training took about 4 minutes for all 3 epochs on a free-tier T4. Final training loss was 0.24. Numbers will vary slightly run to run since the 4000-review sample is drawn randomly each time.

## What's in the notebook
1. Setup - installs, imports, seed, GPU check
2. Load data - sampled, balanced subset of IMDb
3. Quick EDA - label balance, review lengths, common words, word cloud
4. Minimal cleaning - just HTML tag removal, since BERT tokenizes on its own
5. Tokenization
6. Fine-tuning with the Hugging Face `Trainer` (the slow part - ~4 min on this subset)
7. Evaluation - accuracy, precision, recall, F1, ROC-AUC, confusion matrix
8. Try it on your own review text (interactive input cell)
9. Save the model
10. Load it back and confirm it still works
11. LIME explainability - see which words drove a prediction
12. **Agentic movie recommendation system** - LangChain tools + `create_agent` + `ChatGroq`, a manual tool-calling walkthrough, and a critic pass
13. Notes / what to try next

## The agent system 
- `search_movies`, `analyze_review_sentiment`, `get_movie_details` are `@tool`-decorated functions the agent can call
- `analyze_review_sentiment` runs the **BERT model trained earlier in this same notebook** - not the LLM - so Part 6 needs to have already run in your session
- `create_agent(model=llm, tools=tools)` builds the agent; `llm` is `ChatGroq` (`llama-3.1-8b-instant`, Groq's free tier)
- a manual walkthrough (`bind_tools`, reading `tool_calls`, building a `ToolMessage`) shows what the agent does internally, one step at a time
- a critic pass (a plain `llm.invoke` with a `SystemMessage` of rules) checks the recommendations against what was actually asked for

The movie database is a small hand-written sample (~10 movies) for demo purposes, not a live connection. Swap `search_movies`/`get_movie_details` for real API calls (TMDB, OMDb) if you want live data - nothing else in the agent needs to change.

**Known limitation:** `llama-3.1-8b-instant` is a small, fast model, and it occasionally picks the wrong tool or invents a movie title instead of searching first - in one test run it replied "the movie titles provided do not exist in the database" without actually calling `search_movies`. If you hit this, rerunning the cell usually fixes it, or you can swap in a larger Groq model (e.g. `llama-3.3-70b-versatile`) for more reliable tool selection.

## Output files
Running the full notebook creates:
```
data/imdb_dataset.csv               sampled dataset used for this run
models/bert_sentiment_model/        fine-tuned BERT weights + config
models/tokenizer/                   matching tokenizer files
outputs/label_distribution.png
outputs/review_length_histogram.png
outputs/wordcloud.png
outputs/training_loss.png
outputs/confusion_matrix.png
outputs/evaluation_report.txt
```
Part 9 also zips `models/` and `outputs/` into `models.zip` / `outputs.zip` for download (uncomment the `files.download(...)` lines if you want the browser download popup).
