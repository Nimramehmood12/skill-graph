"""
Graph Visualization — Two outputs:
1. Static matplotlib PNG (for README / slides)
2. Interactive pyvis HTML (open in browser, drag/zoom)
"""

from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyvis.network import Network

from .graph_builder import build_skill_graph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color by category — used in both visualizations
CATEGORY_COLORS = {
    "Programming": "#4A90E2",   # blue
    "Tooling":     "#7B68EE",   # purple
    "Data/ML":     "#2ECC71",   # green
    "Backend":     "#E67E22",   # orange
    "DevOps":      "#E74C3C",   # red
    "Cloud":       "#16A085",   # teal
    "Frontend":    "#F39C12",   # gold
}


def visualize_matplotlib(G: nx.DiGraph) -> Path:
    """Static hierarchical plot — prerequisites flow left → right."""
    fig, ax = plt.subplots(figsize=(18, 12))

    # Longest-path layering gives a nice left-to-right reading order
    for node in nx.topological_sort(G):
        pass  # just to validate

    # Manual layering by longest path from a root (node with no prereqs)
    layers = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        layers[node] = 0 if not preds else max(layers[p] for p in preds) + 1

    # Position nodes: x = layer, y = spread within layer
    pos = {}
    layer_buckets = {}
    for node, layer in layers.items():
        layer_buckets.setdefault(layer, []).append(node)
    for layer, nodes in layer_buckets.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (layer, i - len(nodes) / 2)

    colors = [CATEGORY_COLORS.get(G.nodes[n]["category"], "#888888") for n in G.nodes]
    sizes  = [300 + 100 * G.nodes[n]["difficulty"] for n in G.nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=12,
                            edge_color="#CCCCCC", width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                            edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=cat)
                      for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, frameon=True)

    ax.set_title("Skill Dependency Graph (DAG)", fontsize=18, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    out = OUTPUT_DIR / "skill_graph.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def visualize_pyvis(G: nx.DiGraph) -> Path:
    """Interactive HTML — drag, zoom, hover."""
    net = Network(height="900px", width="100%", directed=True,
                  bgcolor="#0f1117", font_color="#ffffff")

    for node, attrs in G.nodes(data=True):
        color = CATEGORY_COLORS.get(attrs["category"], "#888888")
        title = (f"<b>{node}</b><br>"
                 f"Category: {attrs['category']}<br>"
                 f"Difficulty: {attrs['difficulty']}/5<br>"
                 f"Est. Hours: {attrs['estimated_hours']}")
        size = 15 + 5 * attrs["difficulty"]
        net.add_node(node, label=node, color=color, size=size, title=title)

    for src, dst in G.edges:
        net.add_edge(src, dst, color="#555555", arrows="to")

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {"gravitationalConstant": -12000, "springLength": 180}
      },
      "edges": {"smooth": {"type": "cubicBezier"}}
    }
    """)

    out = OUTPUT_DIR / "skill_graph.html"
    net.save_graph(str(out))
    return out


if __name__ == "__main__":
    G = build_skill_graph()
    png = visualize_matplotlib(G)
    html = visualize_pyvis(G)
    print(f"✅ Static graph:      {png}")
    print(f"✅ Interactive graph: {html}")
    print(f"\n👉 Open the HTML file in your browser to interact with it.")