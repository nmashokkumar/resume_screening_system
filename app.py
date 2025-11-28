import os
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

# ✅ Your existing backend pipeline imports
from src.jd_processor import process_jd_from_text
from src.resume_parser import process_resumes_from_pdf_folder
from src.matcher import run_matching_pipeline
from src.llm_helper import call_llm_for_resume_eval


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
RAW_RESUMES_DIR = os.path.join(DATA_DIR, "resumes_raw")
TEXT_RESUMES_DIR = os.path.join(DATA_DIR, "resumes_text")


# -----------------------------
# Helper: Save Uploaded PDFs
# -----------------------------
def save_uploaded_pdfs(uploaded_files: List):
    os.makedirs(RAW_RESUMES_DIR, exist_ok=True)

    # Clear previous PDFs for clean runs
    for f in os.listdir(RAW_RESUMES_DIR):
        if f.lower().endswith(".pdf"):
            os.remove(os.path.join(RAW_RESUMES_DIR, f))

    for file in uploaded_files:
        path = os.path.join(RAW_RESUMES_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Resume Jobscan (ATS + Gemini)",
        layout="wide",
    )

    st.title("🧠 AI Resume Screening System")
    st.caption("TF-IDF + spaCy + Gemini LLM | Built for GenAI & ATS Matching")

    tab_scan, tab_about = st.tabs(["🔍 Resume Scanner", "ℹ️ About"])

    # ============================
    # TAB 1: SCANNER
    # ============================
    with tab_scan:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("📄 Job Description")
            jd_text = st.text_area(
                "Paste the Job Description:",
                height=270,
                placeholder="Paste EY GenAI / AI-ML Internship JD here...",
            )

        with col_right:
            st.subheader("📂 Upload Resume(s)")
            uploaded_files = st.file_uploader(
                "Upload PDF resumes:",
                type=["pdf"],
                accept_multiple_files=True,
            )

            st.info("Upload 1 or multiple resumes to compare against the JD.")

        st.markdown("---")

        # ✅ Run ATS Pipeline Button
        if st.button("🚀 Run ATS Matching", type="primary"):
            if not jd_text.strip():
                st.warning("Please paste the Job Description first.")
                return

            if not uploaded_files:
                st.warning("Please upload at least one resume.")
                return

            with st.spinner("Saving JD and resumes..."):
                process_jd_from_text(jd_text, data_dir=DATA_DIR)
                save_uploaded_pdfs(uploaded_files)

            with st.spinner("Parsing resumes..."):
                processed = process_resumes_from_pdf_folder(data_dir=DATA_DIR)
                if not processed:
                    st.error("No valid resumes were processed.")
                    return

            with st.spinner("Running TF-IDF matching..."):
                df_results = run_matching_pipeline(data_dir=DATA_DIR)

            st.subheader("📊 ATS Match Results (TF-IDF)")

            best_row = df_results.iloc[0]
            st.metric(
                label="Best Resume",
                value=best_row["Resume Name"],
                delta=f"{best_row['Match Score (%)']} %",
            )

            st.dataframe(df_results.reset_index(drop=True), use_container_width=True)

            st.caption("ATS Score = TF-IDF + Cosine Similarity (Keyword Matching)")

            # ✅ Save session state for Gemini
            st.session_state["df_results"] = df_results
            st.session_state["jd_text"] = jd_text

        # ============================
        # LLM SECTION (GEMINI)
        # ============================
        if "df_results" in st.session_state:
            st.markdown("---")
            st.subheader("🤖 LLM Semantic Evaluation (Gemini)")

            df_results = st.session_state["df_results"]
            jd_text_saved = st.session_state["jd_text"]

            selected_resume = st.selectbox(
                "Choose a resume for AI evaluation:",
                options=list(df_results["Resume Name"]),
            )

            if st.button("🔍 Get AI Evaluation (Gemini)", type="secondary"):
                base_name = Path(selected_resume).stem
                txt_path = Path(TEXT_RESUMES_DIR) / f"{base_name}.txt"

                if not txt_path.exists():
                    st.error("Cleaned resume text not found.")
                else:
                    resume_text = txt_path.read_text(encoding="utf-8")

                    with st.spinner("Calling Gemini LLM..."):
                        try:
                            llm_result = call_llm_for_resume_eval(
                                jd_text_saved, resume_text
                            )
                        except Exception as e:
                            st.error(f"LLM call failed: {e}")
                        else:
                            st.success("✅ AI Evaluation Completed")

                            st.markdown(
                                f"### 🧠 LLM Fit Score: **{llm_result['score']} / 100**"
                            )

                            if llm_result.get("summary"):
                                st.markdown("**Summary:**")
                                st.write(llm_result["summary"])

                            suggestions = llm_result.get("suggested_keywords", [])
                            if suggestions:
                                st.markdown("**💡 Suggested Keywords to Add:**")
                                for kw in suggestions:
                                    st.markdown(f"- `{kw}`")
                            else:
                                st.info("No keyword suggestions returned by the LLM.")

                            with st.expander("🔍 Raw LLM Output (Debug)"):
                                st.json(llm_result)

    # ============================
    # TAB 2: ABOUT
    # ============================
    with tab_about:
        st.markdown(
            """
### ✅ About This Project

This is a **Hybrid AI Resume Screening System** built using:

- **ATS Layer:**  
  - TF-IDF Vectorization  
  - Cosine Similarity  
  - spaCy NLP Cleaning  

- **GenAI Layer:**  
  - Google **Gemini LLM (gemini-2.5-flash)**  
  - Semantic Resume Evaluation  
  - AI Fit Score (0–100)  
  - Resume Keyword Improvement Suggestions  

### ✅ What This Demonstrates

- Applied **NLP & Machine Learning**
- **GenAI + LLM Integration**
- Real-world **ATS-style Resume Matching**
- End-to-end **AI System Design**

Built for:
- GenAI / AI-ML Internships  
- EY / Consulting / Data Science Roles  
"""
        )


if __name__ == "__main__":
    main()
