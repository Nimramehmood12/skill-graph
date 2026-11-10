"""
End-to-end demo:
  1. Load resume PDF (falls back to inline sample if PDF missing)
  2. Run full engine
  3. Print structured AnalysisResult + save JSON
"""
import json
from pathlib import Path

from pdfminer.pdfparser import PDFSyntaxError

from .graph_builder import build_skill_graph
from .extractor import extract_pdf_text, chunk_text
from .engine import analyze

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESUME_PDF = PROJECT_ROOT / "data" / "sample_resume.pdf"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

JOB_DESCRIPTION = """
We are hiring a Machine Learning Engineer to build and deploy deep learning
models in production. You will work with Python, PyTorch, and AWS. Candidates
should have strong fundamentals in statistics and NumPy, experience building
REST APIs with FastAPI, and a familiarity with Docker and Kubernetes for
deployment. Experience with NLP and distributed systems is a strong plus.
"""

# Fallback resume text (used if sample_resume.pdf is absent)
SAMPLE_RESUME = """
Software Engineer with 4 years building backend services in Python and Flask.
Designed REST APIs serving 50k daily requests. Worked with PostgreSQL and Redis
for caching. Built ETL pipelines using Pandas and NumPy. Some experience with
Docker containers for local development. Familiar with Git and Linux.
Familiar with supervised machine learning using Scikit-Learn.
"""


def main():
    G = build_skill_graph()

    if RESUME_PDF.exists():
        print(f"📄 Extracting: {RESUME_PDF.name}")
        try:
            resume_text = extract_pdf_text(RESUME_PDF)
        except (PDFSyntaxError, OSError) as exc:
            print(f"⚠️  Could not read {RESUME_PDF.name} ({exc}) — using built-in sample.")
            resume_text = SAMPLE_RESUME
    else:
        print("⚠️  No sample_resume.pdf found — using built-in sample.")
        resume_text = SAMPLE_RESUME

    chunks = chunk_text(resume_text)
    print(f"✂️  Resume chunks: {len(chunks)}")

    result = analyze(chunks, JOB_DESCRIPTION, G)

    print("\n" + "=" * 60)
    print(f"FIT SCORE: {result.fit_score}")
    print(f"MATCHED ({len(result.matched_skills)}): {result.matched_skills}")
    print(f"TRANSFERABLE ({len(result.transferable_strengths)}): {result.transferable_strengths}")
    print(f"BLOCKERS ({len(result.critical_blockers)}): {result.critical_blockers}")
    print("\nGAPS (easiest-first):")
    for g in result.gaps:
        chain = " → ".join(g.bridge_path) if g.bridge_path else "NO PATH"
        print(f"  • {g.skill:20s} | {g.estimated_hours:3d}h | {chain}")
    print("\nROADMAP:")
    for r in result.roadmap:
        chain = " → ".join(r.prerequisite_chain)
        print(f"  {r.order}. {r.skill}  ({r.estimated_hours}h)  [{chain}]")
    print("=" * 60)

    out = OUTPUT_DIR / "analysis_result.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n💾 Saved structured result → {out}")


if __name__ == "__main__":
    main()