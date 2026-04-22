from summarizer import summarize
# from text_scraper import read_txt_file

def read_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()
    
text = read_txt_file("example1.txt")

print(summarize(text))

