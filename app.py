# app.py
import os
from typing import List

import streamlit as st
import pandas as pd

from src.jd_processor import process_jd_from_text
from src.resume_parser import process_resumes_from_pdf_folder
from src.matcher import run_matching_pipeline


DATA_DIR = "data"
RAW_RESUMES_DIR = os.path.join(DATA_DIR, "resumes_raw")


def save_uploaded_pdfs(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """
    Save uploaded PDF files into data/resumes_raw/ folder.
    """
    os.makedirs(RAW_RESUMES_DIR, exist_ok=True)

    # Clear old resumes so each run is fresh (optional but clean)
    for old_file in os.listdir(RAW_RESUMES_DIR):
        if old_file.lower().endswith(".pdf"):
            os.remove(os.path.join(RAW_RESUMES_DIR, old_file))

    for file in uploaded_files:
        save_path = os.path.join(RAW_RESUMES_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())


def main():
    st.set_page_config(page_title="Resume Scanner Pipeline App", layout="wide")

    st.title("🔎 Resume Screening System – Pipeline + Streamlit")
    st.write(
        "This app uses a **3-file backend pipeline** (JD processor, resume parser, matcher) "
        "and exposes it via Streamlit, similar to a mini Jobscan."
    )

    col_jd, col_resumes = st.columns(2)

    with col_jd:
        st.subheader("📄 Job Description")
        jd_text = st.text_area(
            "Paste the Job Description here:",
            height=250,
            placeholder="Paste the EY / Data Scientist / AI Intern JD here...",
        )

    with col_resumes:
        st.subheader("📂 Upload Resume(s) – PDF")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF resumes:",
            type=["pdf"],
            accept_multiple_files=True,
        )

    st.markdown("---")

    if st.button("🚀 Run Full Pipeline (JD + Resumes + Match)"):
        if not jd_text.strip():
            st.warning("Please paste the Job Description first.")
            return
        if not uploaded_files:
            st.warning("Please upload at least one PDF resume.")
            return

        with st.spinner("Saving JD and resumes, and running pipeline..."):
            # 1️⃣ JD → File 1 logic
            process_jd_from_text(jd_text, data_dir=DATA_DIR)

            # 2️⃣ Save uploaded PDFs to data/resumes_raw/
            save_uploaded_pdfs(uploaded_files)

            # 3️⃣ File 2: parse PDFs into text files
            processed = process_resumes_from_pdf_folder(data_dir=DATA_DIR)
            if not processed:
                st.error("No valid resumes processed. Please check your PDFs.")
                return

            # 4️⃣ File 3: run matcher pipeline
            df_results: pd.DataFrame = run_matching_pipeline(data_dir=DATA_DIR)

        st.subheader("📊 Match Results")
        best_row = df_results.iloc[0]
        st.success(
            f"🎯 Best match: **{best_row['Resume Name']}** "
            f"with **{best_row['Match Score (%)']}%** match."
        )

        st.dataframe(df_results.reset_index(drop=True), use_container_width=True)

        st.caption("💡 ‘Suggested Keywords’ are words from the JD that are missing in the resume, to boost ATS alignment.")

        csv_data = df_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_data,
            file_name="resume_match_scores.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
