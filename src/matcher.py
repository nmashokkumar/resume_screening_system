# src/matcher.py
import os
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import suggest_keywords_from_jd


def load_jd_and_resumes(
    data_dir: str = "data",
    jd_clean_filename: str = "jd_clean.txt",
    resumes_text_folder: str = "resumes_text",
) -> Tuple[str, list[str], list[str]]:
    jd_path = os.path.join(data_dir, jd_clean_filename)
    resumes_dir = os.path.join(data_dir, resumes_text_folder)

    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Cleaned JD not found at {jd_path}")
    if not os.path.exists(resumes_dir):
        raise FileNotFoundError(f"Resumes text folder not found at {resumes_dir}")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_clean = f.read()

    resume_texts: list[str] = []
    resume_names: list[str] = []

    for file_name in os.listdir(resumes_dir):
        if file_name.lower().endswith(".txt"):
            full_path = os.path.join(resumes_dir, file_name)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            resume_texts.append(text)
            resume_names.append(file_name)

    return jd_clean, resume_names, resume_texts


def compute_match_dataframe(
    jd_clean: str,
    resume_names: list[str],
    resume_texts: list[str],
) -> pd.DataFrame:
    """
    Compute TF-IDF similarity and keyword suggestions.
    Returns a DataFrame with columns:
    - Resume Name
    - Match Score (%)
    - Suggested Keywords
    """
    if not resume_texts:
        raise ValueError("No resume texts provided.")

    corpus = [jd_clean] + resume_texts

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]
    scores = (similarities * 100).round(2).tolist()

    suggestions_list: list[str] = []
    for resume_text in resume_texts:
        sugg = suggest_keywords_from_jd(jd_clean, resume_text, max_suggestions=3)
        suggestions_list.append(", ".join(sugg) if sugg else "")

    df = pd.DataFrame(
        {
            "Resume Name": resume_names,
            "Match Score (%)": scores,
            "Suggested Keywords": suggestions_list,
        }
    ).sort_values(by="Match Score (%)", ascending=False)

    return df


def run_matching_pipeline(
    data_dir: str = "data",
) -> pd.DataFrame:
    """
    Full pipeline using saved jd_clean.txt and resumes_text/.
    """
    jd_clean, resume_names, resume_texts = load_jd_and_resumes(data_dir=data_dir)

    if not resume_texts:
        raise ValueError("No resumes found in resumes_text folder.")

    df = compute_match_dataframe(jd_clean, resume_names, resume_texts)
    output_path = os.path.join(data_dir, "resume_match_scores.csv")
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")
    return df


if __name__ == "__main__":
    # CLI usage: python src/matcher.py
    df_out = run_matching_pipeline()
    print(df_out.to_string(index=False))
