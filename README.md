# 🧠 AI Resume Screening System (ATS + Gemini LLM)

A hybrid **AI-powered Resume Screening System** that combines:

- ✅ **ATS-style keyword matching** using TF-IDF + Cosine Similarity  
- ✅ **Advanced NLP cleaning** using spaCy  
- ✅ **Generative AI semantic evaluation** using **Google Gemini LLM**  
- ✅ **Interactive web interface** built with **Streamlit**

This project is designed for:
- GenAI Internships
- Data Science / AI-ML Roles
- Consulting & EY-style Digital Roles
- Portfolio & Resume Showcase

---

## 🚀 Features

### ✅ ATS Matching (Classical ML)
- TF-IDF Vectorization
- Cosine Similarity Scoring
- Resume ranking based on JD

### ✅ NLP Processing
- Tokenization
- Stopword Removal
- Lemmatization (spaCy)
- Cleaned text storage

### ✅ GenAI Semantic Evaluation
- Google Gemini LLM integration
- AI Fit Score (0–100)
- AI-generated summary
- Resume improvement keyword suggestions

### ✅ Modern Web UI
- Built with Streamlit
- Job Description paste area
- Resume PDF upload
- Interactive results & metrics
- Live LLM evaluation button

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frontend:** Streamlit  
- **Machine Learning:** scikit-learn  
- **Natural Language Processing:** spaCy  
- **PDF Processing:** PyPDF2  
- **Generative AI:** Google Gemini LLM  
- **Deployment:** Streamlit Community Cloud  

---

## For Local Use

*Create this file:*

```bash
.streamlit/secrets.toml
```

## Add:
```bash
GEMINI_API_KEY = "YOUR_REAL_API_KEY_HERE"
```

## 🧪 How to Run Locally

```bash
git clone https://github.com/nmashokkumar/resume_screening_system.git
cd resume_screening_system

pip install -r requirements.txt

streamlit run app.py
```

## 🧠 How the System Works

## 1.User pastes a Job Description
## 2.User uploads one or more PDF resumes
## 3.System:
  - Cleans JD
  - Extracts resume text
  - Applies TF-IDF + cosine similarity
  - Ranks resumes

## 4.User selects any resume for:
  ### ✅ Gemini LLM semantic evaluation
  ### ✅ AI score out of 100
  ### ✅ Summary of fit
  ### ✅ Keyword improvement suggestions
  ### ✅ What This Project Demonstrates

## Real-world ATS system logic
 - NLP preprocessing
 - Machine Learning text matching
 - GenAI integration
 - End-to-end AI system design
 - Cloud deployment & MLOps basics

## 🙌 Author

**Ashok Kumar N**
**Aspiring Data Scientist & GenAI Developer**

