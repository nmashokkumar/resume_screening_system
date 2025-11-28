# src/utils.py

import re
from typing import List

import spacy

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")

# Use spaCy's built-in stopword list
STOP_WORDS = nlp.Defaults.stop_words


def clean_text(text: str) -> str:
    """
    Clean text using spaCy NLP:
    - Lowercase
    - Tokenization
    - Remove stopwords, punctuation, numbers, spaces
    - Lemmatization
    - Keep only meaningful tokens (length > 2)
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    doc = nlp(text)

    tokens: List[str] = []

    for token in doc:
        # Skip punctuation, spaces, numbers
        if token.is_space or token.is_punct or token.like_num:
            continue

        lemma = token.lemma_.strip().lower()

        # Skip empty, stopwords, very short tokens
        if not lemma:
            continue
        if lemma in STOP_WORDS:
            continue
        if len(lemma) <= 2:
            continue

        tokens.append(lemma)

    cleaned_text = " ".join(tokens)
    return cleaned_text


def suggest_keywords_from_jd(
    jd_clean: str,
    resume_clean: str,
    max_suggestions: int = 3
) -> List[str]:
    """
    Suggest missing keywords from JD that are not present in the resume.
    Here we assume jd_clean and resume_clean are already cleaned using clean_text().
    """
    jd_tokens = jd_clean.split()
    resume_tokens = set(resume_clean.split())

    suggestions: List[str] = []
    seen: set[str] = set()

    for token in jd_tokens:
        if (
            token not in resume_tokens
            and token not in STOP_WORDS
            and len(token) > 2
            and token not in seen
        ):
            suggestions.append(token)
            seen.add(token)

        if len(suggestions) >= max_suggestions:
            break

    return suggestions
