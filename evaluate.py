import numpy as np
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import load_dataset
import summarizer

nltk.download('punkt_tab', quiet=True)

def lead_n(text, num_sentences=3):
    """Lead-N baseline: just take the first N sentences."""
    return ' '.join(sent_tokenize(text)[:num_sentences])


def evaluate(summarize_fn, dataset, label):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge_totals = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    predictions = []
    references = []
    n = 0

    for ex in dataset:
        try:
            predicted = summarize_fn(ex['article'], num_sentences=3)
        except Exception as e:
            print(f"  skipped: {e}")
            continue
        scores = scorer.score(ex['highlights'], predicted)
        for k in rouge_totals:
            rouge_totals[k] += scores[k].fmeasure
        predictions.append(predicted)
        references.append(ex['highlights'])
        n += 1

    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)

    print(f"\n{label} (n={n}):")
    for k, v in rouge_totals.items():
        print(f"  {k}: {v/n:.4f}")
    print(f"  BERTScore F1: {F1.mean():.4f}")



if __name__ == "__main__":
    print("Loading CNN/DailyMail test set...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test[:100]")
    evaluate(summarizer.summarize_text, ds, "PCA summarizer")
    evaluate(lead_n, ds, "Lead-3 baseline")