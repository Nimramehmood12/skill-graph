"""Streamlit UI for Skill Graph Analyzer."""
import json
import tempfile
from pathlib import Path
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.graph_builder import build_skill_graph
from src.extractor import extract_pdf_text, chunk_text
from src.engine import analyze

st.set_page_config(page_title="Skill Graph Analyzer", page_icon="🧠", layout="wide")

COLORS = {
    "Programming": "#4A90E2", "Tooling": "#7B68EE", "Data/ML": "#2ECC71",
    "Backend": "#E67E22", "DevOps": "#E74C3C", "Cloud": "#16A085",
    "Frontend": "#F39C12",
}

@st.cache_resource
def get_graph():
    return build_skill_graph()

def draw_graph(G, matched, missing):
    fig, ax = plt.subplots(figsize=(12, 8))
    layers = {}
    for n in nx.topological_sort(G):
        preds = list(G.predecessors(n))
        layers[n] = 0 if not preds else max(layers[p] for p in preds) + 1
    pos, buckets = {}, {}
    for n, l in layers.items():
        buckets.setdefault(l, []).append(n)
    for l, nodes in buckets.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (l, i - len(nodes) / 2)

    def color(n):
        if n in matched: return "#2ECC71"     # green = have
        if n in missing: return "#E74C3C"     # red   = missing
        return "#888888"                       # grey  = neutral

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=10,
                           edge_color="#CCC", width=0.8, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=[color(n) for n in G.nodes],
                           node_size=[200 + 80 * G.nodes[n]["difficulty"] for n in G.nodes],
                           edgecolors="white", linewidths=1)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
    handles = [
        mpatches.Patch(color="#2ECC71", label="You have"),
        mpatches.Patch(color="#E74C3C", label="Missing"),
        mpatches.Patch(color="#888888", label="Not relevant"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title("Your Skill Graph", fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig

st.title("🧠 Skill Graph Analyzer")
st.caption("Semantic skill matching + prerequisite-aware roadmap generation")

G = get_graph()

col1, col2 = st.columns(2)
with col1:
    pdf = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
with col2:
    jd = st.text_area("📋 Job Description", height=180, placeholder="Paste the JD here...")

if st.button("🚀 Analyze", type="primary", use_container_width=True):
    if not jd.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Extract resume text
    if pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            resume_text = extract_pdf_text(Path(tmp.name))
    else:
        st.warning("No PDF uploaded — using a sample resume for demo.")
        resume_text = """Software Engineer with 4 years building backend services in Python and Flask.
        Designed REST APIs serving 50k daily requests. Built ETL pipelines using Pandas and NumPy.
        Familiar with Docker containers and Git."""

    chunks = chunk_text(resume_text)
    with st.spinner("Running semantic analysis…"):
        result = analyze(chunks, jd, G)

    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Fit Score", f"{result.fit_score:.0%}")
    k2.metric("Matched Skills", len(result.matched_skills))
    k3.metric("Gaps to Bridge", len(result.gaps))

    # Graph
    missing = {g.skill for g in result.gaps}
    st.subheader("🗺️ Skill Graph")
    st.pyplot(draw_graph(G, set(result.matched_skills), missing))

    # Matched / transferable
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Matched")
        st.write(result.matched_skills or "None")
    with c2:
        st.subheader("🔁 Transferable")
        st.write(result.transferable_strengths or "None")

    # Roadmap
    st.subheader("📚 Actionable Roadmap")
    if result.roadmap:
        for r in result.roadmap:
            chain = " → ".join(r.prerequisite_chain)
            st.markdown(f"**{r.order}. {r.skill}** — *{r.estimated_hours}h*  \n`{chain}`")
    else:
        st.success("You already meet all target skills!")

    # Raw JSON
    with st.expander("🔍 Raw structured output"):
        st.json(json.loads(result.model_dump_json()))