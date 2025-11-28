# src/resume_parser.py
import os
from typing import List

import PyPDF2

from src.utils import clean_text


def extract_text_from_pdf_file(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text


def process_resumes_from_pdf_folder(
    data_dir: str = "data",
    raw_folder: str = "resumes_raw",
    text_folder: str = "resumes_text",
) -> List[str]:
    """
    Convert all PDFs in data/resumes_raw/ to cleaned text files in data/resumes_text/.
    Returns list of created text filenames.
    """
    raw_dir = os.path.join(data_dir, raw_folder)
    text_dir = os.path.join(data_dir, text_folder)

    if not os.path.exists(raw_dir):
        print(f"Resumes folder not found at {raw_dir}")
        return []

    os.makedirs(text_dir, exist_ok=True)

    processed_files: List[str] = []

    for file_name in os.listdir(raw_dir):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(raw_dir, file_name)
            raw_text = extract_text_from_pdf_file(pdf_path)
            cleaned_text = clean_text(raw_text)

            base_name, _ = os.path.splitext(file_name)
            txt_name = f"{base_name}.txt"
            txt_path = os.path.join(text_dir, txt_name)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Processed: {file_name} -> {txt_name}")
            processed_files.append(txt_name)

    if not processed_files:
        print("No PDF resumes found to process.")

    return processed_files


if __name__ == "__main__":
    # CLI usage: python src/resume_parser.py
    process_resumes_from_pdf_folder()
