import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import string
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def preprocess(text):
    sentences = sent_tokenize(text)
    cleaned = []

    for s in sentences:
        tokens = word_tokenize(s)
        tokens = ' '.join([t for t in tokens if t not in string.punctuation])
        cleaned.append(tokens)

    return [sentences, cleaned]


if __name__ == "__main__":
    with open('example1.txt', 'r') as file:
        text = file.read()
    
    # preprocess data
    [sentences, cleaned] = preprocess(text)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = (vectorizer.fit_transform(cleaned)).toarray()

    n_comp = min(10, len(cleaned)) #we can modify this 
    pca = PCA(n_components=n_comp)
    X_pca = (pca.fit_transform(X))

    weights = pca.explained_variance_ratio_
    importance = np.dot(X_pca ** 2, weights)
    importance = importance / importance.sum()

    rank = np.argsort(importance)[::-1]
    
    output = '\n'.join([sentences[i] for i in rank])

    with open('output.txt', 'w') as file:
        file.write(output)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.bar(range(1, len(importance) + 1), importance[rank])
    ax1.set_xlabel('Sentence Rank')
    ax1.set_ylabel('Importance Score')

    ax2.bar(range(len(importance)), importance[rank])
    ax2.set_xticks(range(len(importance)), rank + 1)
    ax2.set_xlabel('Sentence Index (original document order)')
    ax1.set_ylabel('Importance Score')
    fig.suptitle('Ranking of Sentence Importance')

    plt.show()