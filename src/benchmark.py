"""Compare semantic engine vs naive keyword matcher."""
from .graph_builder import build_skill_graph
from .engine import analyze

RESUME = """Software Engineer with 4 years building backend services in Python and Flask.
Designed REST APIs serving 50k daily requests. Built ETL pipelines using Pandas and NumPy.
Familiar with Docker containers and Git."""

JD = """Hiring ML Engineer. Python, PyTorch, AWS required. Strong statistics and NumPy.
FastAPI for REST APIs. Docker and Kubernetes for deployment. NLP and distributed systems a plus."""


def keyword_match(resume, jd, skills):
    r, j = resume.lower(), jd.lower()
    target = {s for s in skills if s.lower() in j}
    matched = {s for s in target if s.lower() in r}
    return target, matched


if __name__ == "__main__":
    G = build_skill_graph()
    skills = list(G.nodes)
    sem = analyze([RESUME], JD, G)
    tgt, kw_matched = keyword_match(RESUME, JD, skills)

    print(f"Semantic engine : fit={sem.fit_score:.0%}, matched={len(sem.matched_skills)}, gaps={len(sem.gaps)}")
    print(f"  Matched: {sem.matched_skills}")
    print(f"  Transferable: {sem.transferable_strengths}")
    print()
    print(f"Keyword baseline: matched={len(kw_matched)}, gaps={len(tgt - kw_matched)}")
    print(f"  Matched: {sorted(kw_matched)}")
    print()
    extra = set(sem.matched_skills) - kw_matched
    print(f"🔥 Engine found {len(extra)} skills the baseline missed: {sorted(extra)}")