"""
Graph Builder — Loads the skill taxonomy JSON into a NetworkX DiGraph.

The graph structure is:
    prerequisite_node → skill_node

Example:
    "Python" → "NumPy"  (means Python is a prerequisite of NumPy)

This direction matters for `nx.shortest_path(G, source, target)`:
- To go FROM what candidate HAS → TO what candidate NEEDS,
  the prerequisite chain flows in the correct direction.
"""

import json
from pathlib import Path
from typing import Dict, Any
import networkx as nx


# Resolve paths relative to the project root, so the code works no matter
# which directory you run it from (VS Code terminal, notebook, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = PROJECT_ROOT / "data" / "skill_taxonomy.json"


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Dict[str, Any]:
    """Load the raw skill_taxonomy.json file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_skill_graph(path: Path = TAXONOMY_PATH) -> nx.DiGraph:
    """
    Build a Directed Acyclic Graph (DAG) of skills.

    Nodes have attributes: category, difficulty, estimated_hours
    Edges point FROM prerequisite TO the dependent skill.

    Returns:
        nx.DiGraph with all skills and prerequisite edges.
    """
    raw = load_taxonomy(path)
    skills = raw["skills"]

    G = nx.DiGraph()

    # First pass: add all nodes with attributes
    for skill_name, meta in skills.items():
        G.add_node(
            skill_name,
            category=meta.get("category", "Unknown"),
            difficulty=meta.get("difficulty", 3),
            estimated_hours=meta.get("estimated_hours", 20),
        )

    # Second pass: add prerequisite edges
    # Edge direction: prerequisite → skill
    for skill_name, meta in skills.items():
        for prereq in meta.get("prerequisites", []):
            if prereq not in G.nodes:
                raise ValueError(
                    f"Prerequisite '{prereq}' for skill '{skill_name}' "
                    f"is not defined in the taxonomy. Fix the JSON."
                )
            G.add_edge(prereq, skill_name)

    # Validate it's a DAG — fails loudly if there's a cycle
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(f"Skill graph contains cycles: {cycles}")

    return G


def print_graph_summary(G: nx.DiGraph) -> None:
    """Quick human-readable summary of the graph."""
    print("=" * 60)
    print(f"Nodes (skills): {G.number_of_nodes()}")
    print(f"Edges (prerequisites): {G.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print("=" * 60)

    # Category breakdown
    categories: Dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        cat = attrs["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("Skills by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat:15s} {count}")
    print("=" * 60)


if __name__ == "__main__":
    # Quick self-test when running `python src/graph_builder.py`
    G = build_skill_graph()
    print_graph_summary(G)

    # Show a sample bridge path
    print("\nSample bridge path: Docker -> Kubernetes")
    print(nx.shortest_path(G, source="Docker", target="Kubernetes"))

    print("\nSample bridge path: Python -> PyTorch")
    print(nx.shortest_path(G, source="Python", target="PyTorch"))