# Movie Review Sentiment Analysis

Fine-tuning `bert-base-uncased` on a subset of the IMDb dataset to classify movie reviews as positive or negative.

## Running it
Upload `Movie_review_setiment_analysis_using_IMDB.ipynb` to Google Colab. 

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
12. Notes / what to try next

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
