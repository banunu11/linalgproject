import re
import os

def clean_article(text):
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t')
    text = re.sub(r'</?s>', '', text)
    text = re.sub(r'-lrb-', '(', text)
    text = re.sub(r'-rrb-', ')', text)
    text = re.sub(r'\+|\(|\d+\s*$', '', text, flags=re.MULTILINE)
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 3]
    return ' '.join(lines)

def split_bin_to_txt(path, outDir):
    os.makedirs(outDir, exist_ok=True)

    with open(path, 'rb') as file:
        text = file.read().decode("utf-8", errors="ignore")

    sections = re.split(r'\b(abstract|article)\b', text)

    articles = [sections[i+1] for i, s in enumerate(sections) if s.strip() == 'article']

    for i, article in enumerate(articles):
        cleaned = clean_article(article)
        if len(cleaned.strip()) == 0:
            continue
        output_path = os.path.join(outDir, f'article_{i+1:04d}.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)

if __name__ == "__main__":
    split_bin_to_txt('testerfiles/train_000.bin', 'articles')