import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations
from collections import Counter
import numpy as np
import logging
import yaml
import os

# 🧠 Co-occurrence graph builder
def build_cooccurrence_graph(df, binary_columns, min_weight=1):
    cooccurrence_counter = Counter()

    for _, row in df.iterrows():
        active_symptoms = [col for col in binary_columns if row[col] == 1]
        for pair in combinations(sorted(active_symptoms), 2):
            cooccurrence_counter[pair] += 1

    G = nx.Graph()
    for (symptom1, symptom2), weight in cooccurrence_counter.items():
        if weight >= min_weight:
            G.add_edge(symptom1, symptom2, weight=weight)

    return G

# 🛠 Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 📥 Load config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 📁 Base directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, config["data_paths"]["processed_data"])
network_base_dir = os.path.join(base_dir, config["data_paths"]["networks"])
network_dirs = {
    "pcos": os.path.join(network_base_dir, "with_pcos"),
    "non_pcos": os.path.join(network_base_dir, "without_pcos")
}
os.makedirs(network_dirs["pcos"], exist_ok=True)
os.makedirs(network_dirs["non_pcos"], exist_ok=True)

# 📄 File & config values
input_file = os.path.join(processed_dir, config["files"]["transformed_csv"])
pcos_label = config["columns"]["pcos_label"]
min_weight = config["network"]["min_edge_weight"]
layout_seed = config["network"]["layout_seed"]

# 🧠 Load dataset
df = pd.read_csv(input_file)
logging.info(f"✅ Loaded dataset: {input_file}")

if pcos_label not in df.columns:
    raise ValueError(f"❌ Column '{pcos_label}' not found in dataset.")

binary_cols = [
    col for col in df.columns
    if col != pcos_label and df[col].nunique() == 2 and sorted(df[col].unique()) == [0, 1]
]
if not binary_cols:
    raise ValueError("❌ No binary symptom columns found.")

# Subsets
df_pcos = df[df[pcos_label] == 1].copy()
df_non_pcos = df[df[pcos_label] == 0].copy()

# 🧱 Build and save graph
def build_and_save_graph(df_subset, label, color, output_dir):
    G = build_cooccurrence_graph(df_subset, binary_cols, min_weight=min_weight)

    # Output paths
    gml_path = os.path.join(output_dir, f"{label}_network.gml")
    image_path = os.path.join(output_dir, f"{label}_network.png")
    csv_path = os.path.join(output_dir, f"{label}_edges.csv")

    # Save GML
    nx.write_gml(G, gml_path)
    logging.info(f"💾 Saved GML: {gml_path}")

    # Save edges
    edge_list = pd.DataFrame([
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ])
    edge_list.to_csv(csv_path, index=False)
    logging.info(f"💾 Saved edge list: {csv_path}")

    # Stats
    if config["network"]["export_stats"]:
        with open(csv_path.replace(".csv", "_stats.txt"), "w") as f:
            f.write(f"Nodes: {G.number_of_nodes()}\n")
            f.write(f"Edges: {G.number_of_edges()}\n")
            f.write(f"Density: {nx.density(G):.4f}\n")

    # Top nodes
    if config["network"]["export_top_nodes"]:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:20]
        pd.DataFrame(top_nodes, columns=["node", "degree"]).to_csv(
            csv_path.replace(".csv", "_top_nodes.csv"), index=False
        )

    # Layout
    try:
        pos = nx.kamada_kawai_layout(G, weight="weight", seed=layout_seed)
    except TypeError:
        np.random.seed(layout_seed)
        pos = nx.kamada_kawai_layout(G)

    edge_widths = [d["weight"] * 0.01 for _, _, d in G.edges(data=True)]
    all_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    top_labels = {n: n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:40]}

    # Plot
    plt.figure(figsize=(20, 22))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.axis("off")

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=all_labels, font_size=7, ax=ax)

    # Draw rectangular nodes
    box_width = 0.15  # Adjust size
    box_height = 0.07
    for node in G.nodes():
        x, y = pos[node]
        rect = Rectangle((x - box_width / 2, y - box_height / 2), box_width, box_height,
                         facecolor=color, edgecolor="black", zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, node, fontsize=9, ha='center', va='center', zorder=3)

    plt.title(f"{label.upper()} Co-occurrence Network", fontsize=16)
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()
    logging.info(f"🖼️ Saved network image: {image_path}")

# 🔴 Build PCOS Network
build_and_save_graph(df_pcos, label="pcos", color="red", output_dir=network_dirs["pcos"])

# 🔵 Build Non-PCOS Network
build_and_save_graph(df_non_pcos, label="non_pcos", color="green", output_dir=network_dirs["non_pcos"])
