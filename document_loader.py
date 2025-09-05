import os
from pypdf import PdfReader
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import io

# Make sure to install pytesseract and PyMuPDF if missing:
# pip install pytesseract pymupdf Pillow

def extract_text_tables_images(pdf_path):
    """
    Extract text, tables, and OCR from images in a PDF.
    Returns a combined string of all extracted content.
    """
    content = []
    
    # Extract text and tables using PyPDF
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            content.append(text)
    
    # Extract images and apply OCR using PyMuPDF
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                content.append(ocr_text)
    
    combined_text = "\n".join(content)
    return combined_text

def load_all_pdfs(pdf_dir):
    """
    Load all PDFs from a folder and return a dict {filename: text}.
    """
    data = {}
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            data[file] = extract_text_tables_images(path)
    return data

if __name__ == "__main__":
    # Test
    texts = load_all_pdfs("data/pdfs")
    for name, text in texts.items():
        print(f"--- {name} ---")
        print(text[:500])
