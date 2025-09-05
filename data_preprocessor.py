import os
import json
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Chunking configuration
CHUNK_SIZE = 1000   # Slightly bigger for medical context
CHUNK_OVERLAP = 150

def clean_text(text: str) -> str:
    """
    Clean text by removing extra spaces and special chars.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("\x0c", " ")  # Remove page break artifacts
    return text.strip()

def split_into_chunks(text: str) -> List[str]:
    """
    Split text into overlapping chunks using a semantic-aware splitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(clean_text(text))

def process_documents(input_dir: str, output_dir: str):
    """
    Process all text files into clean chunks and save as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_chunks = []
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(".txt"):
            path = os.path.join(input_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = split_into_chunks(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "meta": {"source": file, "chunk_id": i}
                })
    
    # Save all chunks
    output_file = os.path.join(output_dir, "chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Processed {len(all_chunks)} chunks. Saved to {output_file}")
    return all_chunks

if __name__ == "__main__":
    process_documents("data/texts", "data/chunks")
