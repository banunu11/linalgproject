import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import string
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import re
import sys
import text_scraper

nltk.download('punkt_tab', quiet=True)

def preprocess(text):
    sentences = sent_tokenize(text)
    cleaned = []

    for s in sentences:
        tokens = word_tokenize(s)
        tokens = ' '.join([t for t in tokens if t not in string.punctuation])
        cleaned.append(tokens)

    return [sentences, cleaned]

def summarize(inPath, num_sentences):
    with open(inPath, 'r', encoding='utf-8') as file:
        text = file.read()

    # preprocess data
    [sentences, cleaned] = preprocess(text)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = (vectorizer.fit_transform(cleaned)).toarray()

    n_comp = min(num_sentences, len(cleaned)) #we can modify this
    pca = PCA(n_components=n_comp)
    X_pca = (pca.fit_transform(X))

    weights = pca.explained_variance_ratio_
    importance = np.dot(X_pca ** 2, weights)
    importance = importance / importance.sum() # normalize importance

    rank = np.argsort(importance)[::-1]
    
    base = inPath.replace('.txt', '')
    output = '\n'.join([sentences[i] for i in rank[:n_comp]])
    output2 = '\n'.join(sentences)

    with open(f'{base}_output.txt', 'w', encoding='utf-8') as file:
        file.write(output)

    with open(f'{base}_original.txt', 'w', encoding='utf-8') as file:
        file.write(output2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.bar(range(1, n_comp + 1), importance[rank[:n_comp]])
    ax1.set_xlabel('Sentence Rank')
    ax1.set_ylabel('Importance Score')

    top_indices = np.sort(rank[:n_comp])
    ax2.bar(range(1, n_comp + 1), importance[top_indices])
    ax2.set_xticks(range(1, n_comp + 1), top_indices + 1, rotation=90)
    ax2.set_xlabel('Sentence Index (original document order)')
    ax2.set_ylabel('Importance Score')
    fig.suptitle('Ranking of Sentence Importance')

    plt.savefig(f'{base}_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    return output

def summarize_text(text, num_sentences=3):
    """Replicates summarizer.py's ranking logic, returns top-k sentences."""
    sentences, cleaned = preprocess(text)
    if len(sentences) <= 1:
        return ' '.join(sentences)
    
    n_comp = min(num_sentences, len(sentences))

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(cleaned).toarray()

    if n_comp < 1:
        return ' '.join(sentences[:num_sentences])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    weights = pca.explained_variance_ratio_
    importance = np.dot(X_pca ** 2, weights)
    rank = np.argsort(importance)[::-1][:num_sentences]
    rank_sorted = sorted(rank)
    return ' '.join([sentences[i] for i in rank_sorted])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        summarize("articles/article_0001.txt", 50)
        exit(0)
 
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    textPath = text_scraper.scrape(url, out)

    summarize(textPath, 50)