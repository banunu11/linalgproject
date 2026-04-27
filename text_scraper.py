#imports
import sys
import re
import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

#remove tags from HTML webpage that should be removed
skip_tags = {
    "script", "style", "noscript", "head", "meta", "link",
    "nav", "footer", "header", "aside", "form", "button",
    "iframe", "svg", "img", "figure", "figcaption",
    "input", "select", "textarea", "label",
}

#block-level tags that should be treated as separate paragraphs, or introduce a line break
block_tags = {
    "p", "div", "section", "article", "main", "blockquote",
    "pre", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "dt", "dd", "tr", "br",
    "table", "thead", "tbody", "tfoot",
}

#function to get the html around any bot restrictions
def fetch_html(url:str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text

#function to extract text from html
def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    #remove unwanted tags
    for tag in soup.find_all(skip_tags):
        tag.decompose()

    lines = []
    current_line_parts = []

    def flush_line():
        line = " ".join(current_line_parts).strip()
        #ideally, here, we want to collapse internal whitespace to a single space, but we want to preserve newlines
        line = re.sub(r" {2,}", " ", line)
        if line:
            lines.append(line)
        current_line_parts.clear()

    def walk(node):
        if isinstance(node, str):
            text = node
            # Normalise whitespace but keep single spaces
            text = re.sub(r"[\r\n\t ]+", " ", text)
            if text.strip():
                current_line_parts.append(text.strip())
            return
        tag_name = node.name if node.name else ""

        if tag_name in block_tags:
            flush_line()
        for child in node.children:
            walk(child)
        if tag_name in block_tags:
            flush_line()
    body = soup.find("body") or soup
    walk(body)
    flush_line()  # catch any trailing content that may remain after the last block tag

    # Merge lines and collapse excessive blank lines
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def scrape(url: str, output_dir: str = "scraped_data", output_path: str | None = None) -> str:
    print(f"Fetching: {url}")
    html = fetch_html(url)
    print("Extracting text…")
    text = extract_text(html)

    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        domain = urlparse(url).netloc.replace("www.", "")
        safe = re.sub(r"[^a-z0-9_-]", "_", domain)
        output_path = os.path.join(output_dir, f"{safe}.txt")
    else:
        output_path = os.path.join(output_dir, output_path)
 
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
 
    print(f"Saved {len(text):,} characters → {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <url> [output.txt]")
        sys.exit(1)
 
    url = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    scrape(url, output_path=out)