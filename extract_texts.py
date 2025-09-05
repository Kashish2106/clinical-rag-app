from document_loader import load_all_pdfs
import os

pdf_dir = "data/pdfs"
text_dir = "data/texts"
os.makedirs(text_dir, exist_ok=True)

data = load_all_pdfs(pdf_dir)
for name, content in data.items():
    filename = name.replace(".pdf", ".txt")
    with open(os.path.join(text_dir, filename), "w", encoding="utf-8") as f:
        f.write(content)

print("✅ All PDFs converted to text files in data/texts/")
