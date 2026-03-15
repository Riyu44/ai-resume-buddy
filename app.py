from math import log
import streamlit as st
import requests
import PyPDF2
import re
import os
from dotenv import load_dotenv
import logging

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_API_URL = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/v1/chat/completions")
SARVAM_MODEL   = os.getenv("SARVAM_MODEL", "sarvam-30b")

logging.basicConfig(level=logging.INFO,filename='app.log',format='%(asctime)s - %(levelname)s - %(message)s')
# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ── Sarvam API call ───────────────────────────────────────────────────────────
def sarvam_chat(user_content: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {SARVAM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": SARVAM_MODEL,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": 9000,
            "temperature": 0.2,
        }
        resp = requests.post(SARVAM_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        if not content:
            logging.error(f"Sarvam AI returned an empty response,{data}")
            st.error("Sarvam AI returned an empty response.")
            st.write(data)
            st.stop()
        return content.strip()
    except Exception as e:
        logging.error(f"Error calling Sarvam API: {e}")


# ── Delimiter-based parser ────────────────────────────────────────────────────
def extract_section(text: str, tag: str) -> str:
    """Pull content between <TAG> and </TAG>, strip whitespace."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_list_section(text: str, tag: str) -> list[str]:
    """Extract a section and split into a clean list on newlines or • - * bullets."""
    raw = extract_section(text, tag)
    if not raw:
        return []
    items = re.split(r"\n|(?:^|\n)\s*[-•*]\s*", raw)
    return [i.strip().lstrip("-•* ") for i in items if i.strip()]


def analyse_resume(resume_text: str, jd_text: str) -> dict:
    prompt = f"""
You are an expert ATS resume screener. Analyse the resume against the job description.

Respond using EXACTLY this format with these XML-style tags.
Do not add any text outside the tags.

<fit_score>integer between 0 and 100</fit_score>

<score_rationale>Two sentences explaining the score.</score_rationale>

<matched_keywords>
keyword one
keyword two
keyword three
</matched_keywords>

<missing_keywords>
keyword one
keyword two
keyword three
</missing_keywords>

<suggested_rewrites>
First rewrite bullet starting with a strong action verb tailored to the JD.
Second rewrite bullet starting with a strong action verb tailored to the JD.
Third rewrite bullet starting with a strong action verb tailored to the JD.
</suggested_rewrites>

<red_flags>
First concern a recruiter might have.
Second concern a recruiter might have.
</red_flags>

--- RESUME ---
{resume_text}

--- JOB DESCRIPTION ---
{jd_text}
"""
    raw = sarvam_chat(prompt)
    clean_raw = raw.strip()
    clean_raw = clean_raw.removeprefix("```json").removesuffix("```").strip()
    clean_raw = clean_raw.removeprefix("```").strip()

    fit_score_str = extract_section(clean_raw, "fit_score")
    try:
        fit_score = int(re.search(r"\d+", fit_score_str).group())
    except (AttributeError, ValueError):
        fit_score = 0

    return {
        "fit_score":         fit_score,
        "score_rationale":   extract_section(raw, "score_rationale"),
        "matched_keywords":  parse_list_section(raw, "matched_keywords"),
        "missing_keywords":  parse_list_section(raw, "missing_keywords"),
        "suggested_rewrites": parse_list_section(raw, "suggested_rewrites"),
        "red_flags":         parse_list_section(raw, "red_flags"),
        "_raw":              raw,   # kept for debug, hidden in UI
    }


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Scanner", page_icon="📄", layout="wide")

st.title("📄 AI Resume Scanner")
st.caption("Upload your resume and paste a job description — get an instant ATS fit analysis powered by Sarvam AI.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"✓ Loaded: {uploaded_file.name}")

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full JD here",
        height=300,
        placeholder="Copy and paste the job description...",
    )

st.divider()

if st.button("🔍  Analyse Fit", type="primary", use_container_width=True):
    if not SARVAM_API_KEY:
        st.error("Missing SARVAM_API_KEY — add it to your .env or Streamlit secrets.")
        st.stop()
    if not uploaded_file:
        st.error("Please upload your resume PDF.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    with st.spinner("Analysing with Sarvam AI…"):
        resume_text = extract_text_from_pdf(uploaded_file)
        result = analyse_resume(resume_text, jd_text)

    # ── Fit score ─────────────────────────────────────────────────────────────
    score = result["fit_score"]
    color = "green" if score >= 75 else "orange" if score >= 55 else "red"

    st.markdown(f"## Fit Score: :{color}[{score} / 100]")
    if result["score_rationale"]:
        st.info(result["score_rationale"])

    st.divider()

    # ── Keywords ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Matched Keywords")
        if result["matched_keywords"]:
            for kw in result["matched_keywords"]:
                st.markdown(f"- `{kw}`")
        else:
            st.caption("None found.")

    with c2:
        st.subheader("❌ Missing Keywords")
        if result["missing_keywords"]:
            for kw in result["missing_keywords"]:
                st.markdown(f"- `{kw}`")
        else:
            st.caption("None — great coverage!")

    st.divider()

    # ── Suggested rewrites ────────────────────────────────────────────────────
    st.subheader("✏️ Suggested Bullet Rewrites")
    if result["suggested_rewrites"]:
        for i, bullet in enumerate(result["suggested_rewrites"], 1):
            st.markdown(f"**{i}.** {bullet}")
    else:
        st.caption("No rewrites generated.")

    # ── Red flags ─────────────────────────────────────────────────────────────
    if result["red_flags"]:
        st.divider()
        st.subheader("⚠️ Recruiter Red Flags")
        for flag in result["red_flags"]:
            st.warning(flag)

    # ── Debug expander (hidden by default) ───────────────────────────────────
    with st.expander("🔧 Debug — raw model output"):
        st.code(result["_raw"])