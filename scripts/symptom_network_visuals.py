import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities
import yaml
import logging
import os
import sys

from network_utils import build_cooccurrence_graph

# 🛠 Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 📥 Load config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 📁 Paths from config
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, config["data_paths"]["processed_data"])
visuals_base = os.path.join(base_dir, config["data_paths"]["visuals"])
os.makedirs(visuals_base, exist_ok=True)

# 🔧 Parameters
input_file = os.path.join(processed_dir, config["files"]["transformed_csv"])
pcos_label = config["columns"]["pcos_label"]
min_weight = config["network"]["min_edge_weight"]
top_n = config["network"]["top_n_nodes"]
layout_seed = config["network"]["layout_seed"]

# 📄 Visualization Filenames
visual_filenames = {
    "html_network": config["visuals"]["html_network"],
    "heatmap": config["visuals"]["heatmap"],
    "communities": config["visuals"]["communities"],
}


# 🌐 PyVis Interactive Graph
def create_interactive_network(G, filename):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
    net.barnes_hut()

    for node in G.nodes():
        net.add_node(node, label=node)

    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d["weight"], title=f"Co-occurs: {d['weight']} times")

    net.set_options("""
    var options = {
      "edges": {"color": {"inherit": true}, "smooth": false},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)
    net.write_html(filename)
    logging.info(f"🧩 Saved interactive network: {filename}")


# 🔥 Heatmap
def plot_heatmap(G, filename):
    nodes = sorted(G.nodes())
    mat = pd.DataFrame(0, index=nodes, columns=nodes)

    for u, v, d in G.edges(data=True):
        mat.loc[u, v] = d["weight"]
        mat.loc[v, u] = d["weight"]

    plt.figure(figsize=(14, 12))
    sns.heatmap(mat, cmap="Reds", linewidths=0.5, square=True)
    plt.title("Symptom Co-occurrence Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()
    logging.info(f"🧯 Saved heatmap: {filename}")


# 🧠 Communities
def plot_communities(G, filename):
    communities = list(greedy_modularity_communities(G))
    color_map = {}
    palette = sns.color_palette("husl", len(communities)).as_hex()

    for i, group in enumerate(communities):
        for node in group:
            color_map[node] = palette[i]

    pos = nx.spring_layout(G, seed=layout_seed)
    plt.figure(figsize=(18, 14))
    for node in G.nodes():
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node], node_color=color_map[node],
            node_size=500, edgecolors="black"
        )
    nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title("Symptom Clusters via Community Detection", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()
    logging.info(f"🧠 Saved community plot: {filename}")


# 🚀 Group-wise Runner
def generate_for_group(df, label_value, label_name):
    df_group = df[df[pcos_label] == label_value].copy()
    if df_group.empty:
        logging.warning(f"⚠️ No data for group: {label_name}")
        return

    binary_cols = [col for col in df_group.columns if col != pcos_label and df_group[col].nunique() == 2]
    if not binary_cols:
        logging.error(f"❌ No binary symptom columns found for {label_name}")
        return

    G = build_cooccurrence_graph(df_group, binary_cols, min_weight=min_weight)
    if G.number_of_nodes() == 0:
        logging.warning(f"⚠️ Empty graph for {label_name}")
        return

    out_dir = os.path.join(visuals_base, label_name)
    os.makedirs(out_dir, exist_ok=True)

    create_interactive_network(G, filename=os.path.join(out_dir, visual_filenames["html_network"]))
    plot_heatmap(G, filename=os.path.join(out_dir, visual_filenames["heatmap"]))
    plot_communities(G, filename=os.path.join(out_dir, visual_filenames["communities"]))


# ⏯️ Main Entry
def main():
    df = pd.read_csv(input_file)
    logging.info(f"📄 Loaded dataset: {input_file}")

    if pcos_label not in df.columns:
        logging.error(f"❌ PCOS column '{pcos_label}' not found.")
        return

    generate_for_group(df, label_value=1, label_name="with_pcos")
    generate_for_group(df, label_value=0, label_name="without_pcos")


if __name__ == "__main__":
    main()
