import streamlit as st
import requests
import PyPDF2
import json
import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_API_URL = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/v1/chat/completions")
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-30b")

# ── helpers ──────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join(page.extract_text() for page in reader.pages)


def sarvam_chat(user_content: str) -> str:
    """Call Sarvam API (OpenAI-compatible chat completions) and return assistant text."""
    headers = {
        "Authorization": f"Bearer {SARVAM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SARVAM_MODEL,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    resp = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data["choices"][0]["message"]["content"] is None:
        st.error("Sarvam AI returned an empty response.")
        st.write(data) 
        return ""
    return data["choices"][0]["message"]["content"].strip()


def analyse_resume(resume_text, jd_text):
    prompt = f"""
You are an expert ATS resume screener.

Given the resume and job description below, return ONLY a valid JSON object 
with exactly these keys:
- fit_score: integer 0-100
- score_rationale: string (2 sentences explaining the score)
- matched_keywords: list of strings (keywords present in both)
- missing_keywords: list of strings (important JD keywords absent from resume)
- suggested_rewrites: list of exactly 3 strings (improved resume bullets 
  tailored to this JD, each starting with a strong action verb)
- red_flags: list of strings (any concerns a recruiter might have)

Resume:
{resume_text}

Job Description:
{jd_text}

Return only the JSON. No markdown, no explanation.
"""
    raw = sarvam_chat(prompt)
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)

# ── UI ───────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Scanner", page_icon="📄", layout="wide")

st.title("📄 AI Resume Scanner")
st.caption("Paste a job description and upload your resume — get an instant ATS fit analysis powered by Sarvam AI.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"Loaded: {uploaded_file.name}")

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the full JD here", height=300, placeholder="Copy and paste the job description...")

st.divider()

if st.button("🔍 Analyse Fit", type="primary", use_container_width=True):
    if not SARVAM_API_KEY:
        st.error("Missing SARVAM_API_KEY in .env")
    elif not uploaded_file:
        st.error("Please upload your resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analysing with Sarvam AI..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            result = analyse_resume(resume_text, jd_text)

        # ── Score ──
        score = result["fit_score"]
        color = "green" if score >= 75 else "orange" if score >= 55 else "red"
        st.markdown(f"## Fit Score: :{color}[{score}/100]")
        st.info(result["score_rationale"])

        st.divider()

        # ── Keyword columns ──
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched Keywords")
            for kw in result["matched_keywords"]:
                st.markdown(f"- `{kw}`")
        with c2:
            st.subheader("❌ Missing Keywords")
            for kw in result["missing_keywords"]:
                st.markdown(f"- `{kw}`")

        st.divider()

        # ── Rewrites ──
        st.subheader("✏️ Suggested Bullet Rewrites")
        for i, bullet in enumerate(result["suggested_rewrites"], 1):
            st.markdown(f"**{i}.** {bullet}")

        # ── Red flags ──
        if result.get("red_flags"):
            st.divider()
            st.subheader("⚠️ Recruiter Red Flags")
            for flag in result["red_flags"]:
                st.warning(flag)