# src/jd_processor.py
import os
from typing import Optional

from src.utils import clean_text


def process_jd_from_file(
    data_dir: str = "data",
    raw_filename: str = "jd_raw.txt",
    clean_filename: str = "jd_clean.txt",
) -> Optional[str]:
    """
    Read raw JD from file, clean it, save cleaned JD to file,
    and return cleaned JD as string.
    """
    raw_path = os.path.join(data_dir, raw_filename)
    clean_path = os.path.join(data_dir, clean_filename)

    if not os.path.exists(raw_path):
        print(f"Job description file not found at {raw_path}")
        return None

    with open(raw_path, "r", encoding="utf-8") as f:
        jd_raw = f.read()

    jd_clean = clean_text(jd_raw)

    os.makedirs(data_dir, exist_ok=True)
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(jd_clean)

    print(f"Cleaned JD saved to {clean_path}")
    return jd_clean


def process_jd_from_text(
    jd_text: str,
    data_dir: str = "data",
    clean_filename: str = "jd_clean.txt",
) -> str:
    """
    Clean JD text (coming from Streamlit UI), save to file, and return cleaned JD.
    """
    jd_clean = clean_text(jd_text)

    os.makedirs(data_dir, exist_ok=True)
    clean_path = os.path.join(data_dir, clean_filename)
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(jd_clean)

    return jd_clean


if __name__ == "__main__":
    # CLI usage: python src/jd_processor.py
    process_jd_from_file()
