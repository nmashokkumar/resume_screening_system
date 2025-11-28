# src/llm_helper.py

import json
import re
from typing import Dict, Any

import streamlit as st
import google.generativeai as genai


def _get_gemini_model():
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in Streamlit secrets.")

    genai.configure(api_key=api_key)
    # Use a model that works for you, e.g. "gemini-pro" or "models/gemini-pro"
    return genai.GenerativeModel("gemini-2.5-flash")  # or whatever you confirmed


def _extract_json_from_text(raw_text: str) -> str:
    """
    Gemini sometimes returns JSON inside ```json ... ``` code fences.
    This helper extracts the inner JSON string.
    """
    text = raw_text.strip()

    # If it starts with ``` remove the fences
    if text.startswith("```"):
        # remove starting ```json or ``` line
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        # remove trailing ```
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()

    # As an extra safety net: take content between first { and last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    return text


def call_llm_for_resume_eval(jd_text: str, resume_text: str) -> Dict[str, Any]:
    """
    Call Gemini to evaluate how well the resume matches the JD.

    Returns a dict:
    {
      "score": int (0-100),
      "summary": str,
      "suggested_keywords": List[str]
    }
    """
    model = _get_gemini_model()

    prompt = f"""
    You are an AI hiring assistant.

    JOB DESCRIPTION:
    {jd_text}

    CANDIDATE RESUME:
    {resume_text}

    Task:
    1. Rate how well this candidate fits the job on a scale of 0 to 100.
       Consider skills, tools, experience, and relevance to the JD.
    2. Write a short 2-3 sentence summary of the candidate's fit.
    3. Suggest 3 to 6 concrete keywords or short phrases the candidate
       could add to their resume to better match this JD.
       Only suggest skills/keywords that are realistic and relevant
       based on the JD (not random buzzwords).

    Very important:
    - Respond ONLY in valid JSON.
    - Do NOT include any extra text outside the JSON.
    - JSON format must be:

    {{
      "score": <number between 0 and 100>,
      "summary": "<short explanation>",
      "suggested_keywords": ["keyword1", "keyword2", "keyword3"]
    }}
    """

    response = model.generate_content(prompt)
    raw_text = (response.text or "").strip()

    # 🔧 New: clean markdown fences and isolate JSON
    json_text = _extract_json_from_text(raw_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Debug fallback so you can see what model actually sent
        return {
            "score": 0,
            "summary": "LLM response could not be parsed as JSON.",
            "suggested_keywords": [],
            "raw_response": raw_text,
            "parsed_attempt": json_text,
        }

    score = int(data.get("score", 0))
    score = max(0, min(100, score))  # clamp to 0–100

    summary = str(data.get("summary", "")).strip()

    suggested = data.get("suggested_keywords", [])
    if not isinstance(suggested, list):
        suggested = []
    suggested = [str(k).strip() for k in suggested if str(k).strip()]

    return {
        "score": score,
        "summary": summary,
        "suggested_keywords": suggested,
    }
