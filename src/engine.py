"""Readiness Engine — semantic match + DAG gap analysis + roadmap."""
from typing import List, Dict
import numpy as np
import networkx as nx

from .schemas import AnalysisResult, SkillGap, RoadmapStep
from .extractor import embed_texts, cosine_sim

MATCH_THRESHOLD = 0.58   # tune this — higher = stricter
TRANSFER_THRESHOLD = 0.50


def _skill_descriptor(skill: str, G: nx.DiGraph) -> str:
    """Build a small text description per skill for embedding."""
    prereqs = list(G.predecessors(skill))
    cat = G.nodes[skill]["category"]
    return f"{skill} in {cat}. Prerequisites: {', '.join(prereqs) or 'none'}."


def semantic_match(
    chunks: List[str],
    G: nx.DiGraph,
    chunk_emb: np.ndarray,
    skill_emb: np.ndarray,
) -> Dict[str, float]:
    """Best similarity of each skill vs all resume chunks."""
    sims = cosine_sim(skill_emb, chunk_emb)          # (n_skills, n_chunks)
    best = sims.max(axis=1) if sims.size else np.zeros(len(G.nodes))
    skills = list(G.nodes)
    return {skills[i]: float(best[i]) for i in range(len(skills))}


def analyze(
    resume_chunks: List[str],
    job_description: str,
    G: nx.DiGraph,
    model_name: str = "all-MiniLM-L6-v2",
) -> AnalysisResult:
    """
    Full pipeline:
      1. Embed resume chunks
      2. Embed skill descriptors + job description
      3. Semantic match → matched vs target skills
      4. DAG traversal → gaps, bridge paths, readiness
      5. Topological roadmap
    """
    skill_names = list(G.nodes)
    skill_desc = [_skill_descriptor(s, G) for s in skill_names]

    # Embed everything
    chunk_emb = embed_texts(resume_chunks)
    skill_emb = embed_texts(skill_desc)
    jd_emb = embed_texts([job_description])          # (1, dim)

    # Score: resume chunk similarity per skill
    resume_sims = cosine_sim(skill_emb, chunk_emb).max(axis=1)  # (n_skills,)

    # Score: job description similarity per skill  → target skills
    jd_sims = cosine_sim(skill_emb, jd_emb).flatten()           # (n_skills,)

    matched, target, transferable = [], [], []
    for i, name in enumerate(skill_names):
        if resume_sims[i] >= MATCH_THRESHOLD:
            matched.append(name)
        if jd_sims[i] >= MATCH_THRESHOLD:
            target.append(name)
        elif jd_sims[i] >= TRANSFER_THRESHOLD:
            transferable.append(name)


    matched_set, target_set = set(matched), set(target)
    missing = [s for s in target if s not in matched_set]

    # --- Fit score ---
    fit = len(matched_set & target_set) / max(len(target_set), 1)

    # --- Gap analysis ---
    gaps: List[SkillGap] = []
    for m in missing:
        best_path, best_cost = None, float("inf")
        for src in matched_set:
            if src in G and m in G and nx.has_path(G, src, m):
                try:
                    path = nx.shortest_path(G, src, m)
                    cost = sum(G.nodes[n]["estimated_hours"] for n in path[1:])
                    if cost < best_cost:
                        best_cost, best_path = cost, path
                except nx.NetworkXNoPath:
                    pass
        if best_path and len(best_path) > 1:
            chain = best_path[1:]
            hours = int(best_cost)
            readiness = min(1.0, G.nodes[best_path[0]]["estimated_hours"] and 1.0)
            gaps.append(SkillGap(
                skill=m,
                prereqs_met=[best_path[0]],
                missing_prereqs=chain[:-1],
                bridge_path=best_path,
                estimated_hours=hours,
                readiness=readiness,
            ))
        else:
            gaps.append(SkillGap(skill=m, readiness=0.0, estimated_hours=999))

    # Sort gaps: easiest first (most "bridgeable" first)
    gaps.sort(key=lambda g: g.estimated_hours)

    # --- Roadmap: topo-sort missing skills by prerequisite order ---
    sub = G.subgraph([n for n in G.nodes if n in target_set or n in matched_set])
    ordered = [n for n in nx.topological_sort(sub) if n in {g.skill for g in gaps}]
    gap_lookup = {g.skill: g for g in gaps}
    roadmap: List[RoadmapStep] = []
    for i, skill in enumerate(ordered, start=1):
        g = gap_lookup.get(skill)
        if g:
            roadmap.append(RoadmapStep(
                order=i,
                skill=skill,
                prerequisite_chain=g.bridge_path,
                estimated_hours=g.estimated_hours,
            ))

    # --- Critical blockers: missing skills that have NO bridge path ---
    blockers = [g.skill for g in gaps if not g.bridge_path]

    return AnalysisResult(
        fit_score=round(fit, 3),
        matched_skills=sorted(matched),
        transferable_strengths=sorted(set(transferable) - matched_set),
        critical_blockers=blockers,
        gaps=gaps,
        roadmap=roadmap,
        metadata={
            "model": model_name,
            "resume_chunks": str(len(resume_chunks)),
            "target_skills": str(len(target_set)),
            "matched_skills": str(len(matched_set)),
        },
    )