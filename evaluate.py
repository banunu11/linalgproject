from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import load_dataset
import matplotlib.pyplot as plt
import summarizer
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

def lead_n(text, num_sentences=3):
    """Lead-N baseline: just take the first N sentences."""
    return ' '.join(sent_tokenize(text)[:num_sentences])

def evaluate(summarize_fn, dataset, label, text_field, summary_field):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    rouge_totals = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    predictions = []
    references = []
    n = 0

    for ex in dataset:
        try:
            predicted = summarize_fn(ex[text_field], num_sentences=3)
        except Exception as e:
            print(f"  skipped: {e}")
            continue
        scores = scorer.score(ex[summary_field], predicted)
        for k in rouge_totals:
            rouge_totals[k] += scores[k].fmeasure
        predictions.append(predicted)
        references.append(ex[summary_field])
        n += 1

    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False, model_type='bert-base-uncased')

    print(f"\n{label} (n={n}):")
    for k, v in rouge_totals.items():
        print(f"  {k}: {v/n:.4f}")
    print(f"  BERTScore F1: {F1.mean():.4f}")

    results = {
        "rouge": {k: v/n for k, v in rouge_totals.items()},
        "bert_f1_mean": float(F1.mean()),
        "bert_f1_all": F1.numpy(),
        "predictions": predictions,
        "references": references
    }

    return results

if __name__ == "__main__":
    print("Loading testing datasets...")
    datasets = [load_dataset("cnn_dailymail", "3.0.0", split="test[:100]"), 
                load_dataset("knkarthick/samsum", split="test[:100]"), 
                load_dataset("ccdv/govreport-summarization", split="test[:100]")]
    parameters = [(datasets[0], 'article', 'highlights', 'CNN/DailyMail'),
                  (datasets[1], 'dialogue', 'summary', 'SAMSum'),
                  (datasets[2], 'report', 'summary', 'GovReport')]
    print("Finished loading datasets")
    results_store = {}
    for ds, text_field, summary_field, name in parameters:
        print(f"\n--- {name} ---")
        pca_results = evaluate(summarizer.summarize_text, ds, "PCA summarizer", text_field, summary_field)
        lead_results = evaluate(lead_n, ds, "Lead-3 baseline", text_field, summary_field)

        results_store[name] = {}
        results_store[name]["pca"] = pca_results
        results_store[name]["lead3"] = lead_results

    for dataset_name, methods in results_store.items():
        for method_name, res in methods.items():
            plt.hist(res["bert_f1_all"], bins=20)
            plt.title(f"{dataset_name} - {method_name} BERTScore")

            filename = f"{dataset_name}_{method_name}_bert_hist.png".replace("/", "_").replace(" ", "_")
            plt.savefig(filename)
            plt.clf()

