import os
import re
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pyvis.network import Network

from network_utils import build_cooccurrence_graph


# logging
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

def log(msg):
    # Adds colorized logging for easier CLI debugging
    if COLOR_ENABLED:
        parts = re.split(r"(\{.*?\})", msg)
        colored = "".join([
            Fore.CYAN + p + Style.RESET_ALL if p.startswith("{") and p.endswith("}") else Fore.RED + p + Style.RESET_ALL
            for p in parts
        ])
        print("•", colored)
    else:
        logging.info(msg)



config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Base project directories
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, config["data_paths"]["processed_data"])
visuals_base = os.path.join(base_dir, config["data_paths"]["visuals"])

# Pull reusable values from config
input_file = os.path.join(processed_dir, config["files"]["transformed_csv"])
pcos_label = config["columns"]["pcos_label"]
default_min_weight = config["network"]["min_edge_weight"]
top_n = config["network"]["top_n_nodes"]
layout_seed = config["network"]["layout_seed"]

# Filenames for output visualizations
visual_filenames = {
    "html_network": config["visuals"]["html_network"],
    "heatmap": config["visuals"]["heatmap"],
    "communities": config["visuals"]["communities"],
}


# Interactive Network Visualization with PyVis

def create_interactive_network(G, filename):
    full_path = Path(filename).resolve()
    full_path.parent.mkdir(parents=True, exist_ok=True)

    # PyVis quirk: needs working directory to match target location when saving HTML
    temp_cwd = os.getcwd()
    try:
        os.chdir(full_path.parent)

        # Configure network appearance and layout
        net = Network(height="800px", width="100%", bgcolor="#fefefe", font_color="#333")
        net.barnes_hut(gravity=-2500, spring_length=90)

        # Add nodes and edges with tooltips
        for node in G.nodes():
            net.add_node(node, label=node, title=f"<b>{node}</b>")
        for u, v, d in G.edges(data=True):
            net.add_edge(u, v, value=d["weight"], title=f"{u} ↔ {v}: {d['weight']} times")

        # Customize the styling via JS config
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "box",
            "font": { "size": 16, "face": "arial", "color": "#111" },
            "color": {
              "background": "#E0F7FA",
              "border": "#0097A7",
              "highlight": {
                "background": "#B2EBF2",
                "border": "#00796B"
              }
            }
          },
          "edges": {
            "color": { "color": "#aaa", "highlight": "#666" },
            "smooth": false
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 90
            },
            "minVelocity": 0.75
          }
        }
        """)
        net.write_html(full_path.name)
        log(f"Saved interactive network: {{{full_path}}}")
    finally:
        os.chdir(temp_cwd)


# Static Heatmap Visualization

def plot_heatmap(G, filename, matrix_out_csv=None):
    # Initialize matrix with all nodes
    nodes = sorted(G.nodes())
    mat = pd.DataFrame(0, index=nodes, columns=nodes)

    # Fill symmetric adjacency matrix
    for u, v, d in G.edges(data=True):
        mat.loc[u, v] = d["weight"]
        mat.loc[v, u] = d["weight"]

    if matrix_out_csv:
        mat.to_csv(matrix_out_csv)
        log(f"Saved co-occurrence matrix: {{{matrix_out_csv}}}")

    # Plot using seaborn
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        mat,
        cmap="coolwarm",
        linewidths=0.7,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Co-occurrence Frequency"},
        annot=False
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title("Symptom Co-occurrence Heatmap", fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=400)
    plt.close()
    log(f"Saved heatmap: {{{filename}}}")


# Louvain Community Detection Plot

def plot_communities(G, filename):
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("Please install `python-louvain` via pip.")

    # Run Louvain clustering algorithm
    partition = community_louvain.best_partition(G)

    # Group nodes by community
    community_dict = defaultdict(list)
    for node, comm_id in partition.items():
        community_dict[comm_id].append(node)
    communities = list(community_dict.values())

    # Assign a color to each community
    color_map = {}
    palette = sns.color_palette("pastel", len(communities)).as_hex()
    for i, group in enumerate(communities):
        for node in group:
            color_map[node] = palette[i]

    # Plot community-labeled network
    pos = nx.spring_layout(G, seed=layout_seed)
    plt.figure(figsize=(18, 14))
    nx.draw(G, pos,
            node_color=[color_map[n] for n in G.nodes()],
            node_size=900,
            edge_color="#AAAAAA",
            linewidths=1.5,
            font_size=10,
            font_color="black",
            edgecolors="black",
            with_labels=True)
    plt.title("Louvain Symptom Clusters", fontsize=18, weight="bold")
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=600)
    plt.close()
    log(f"Saved community plot: {{{filename}}}")


# Run all visualizations for a given group

def generate_for_group(df, label_value, group_folder, min_weight):
    df_group = df[df[pcos_label] == label_value].copy()

    if df_group.empty:
        log(f"⚠️ No data for group: {{{group_folder}}}")
        return

    # Identify binary features (excluding target)
    binary_cols = [col for col in df_group.columns if col != pcos_label and df_group[col].dropna().isin([0, 1]).all()]
    if not binary_cols:
        log(f"❌ No binary columns found for {{{group_folder}}}")
        return

    # Build the symptom co-occurrence network
    G = build_cooccurrence_graph(df_group, binary_cols, min_weight=min_weight)
    log(f"Graph has {{{G.number_of_nodes()}}} nodes and {{{G.number_of_edges()}}} edges")

    if G.number_of_nodes() == 0:
        log(f"⚠️ Empty graph for {{{group_folder}}}")
        return

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(visuals_base, group_folder, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # Generate all visualizations
    create_interactive_network(G, filename=os.path.join(out_dir, visual_filenames["html_network"]))
    plot_heatmap(G,
                 filename=os.path.join(out_dir, visual_filenames["heatmap"]),
                 matrix_out_csv=os.path.join(out_dir, "cooccurrence_matrix.csv"))
    plot_communities(G, filename=os.path.join(out_dir, visual_filenames["communities"]))


def main(group: str, min_weight_override: int = None):
    df = pd.read_csv(input_file)
    log(f"Loaded dataset: {{{input_file}}}")

    if pcos_label not in df.columns:
        log(f"❌ Missing label column: {{{pcos_label}}}")
        return

    min_weight = min_weight_override if min_weight_override is not None else default_min_weight

    # Run visual generation based on selected group(s)
    if group in ["pcos", "both"]:
        generate_for_group(df, label_value=1, group_folder="with_pcos", min_weight=min_weight)
    if group in ["non_pcos", "both"]:
        generate_for_group(df, label_value=0, group_folder="without_pcos", min_weight=min_weight)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🎨 Generate styled visualizations for PCOS symptom co-occurrence networks.")
    parser.add_argument("--group", choices=["pcos", "non_pcos", "both"], default="both", help="Which cohort to visualize")
    parser.add_argument("--min-weight", type=int, help="Override minimum edge weight threshold")
    args = parser.parse_args()
    main(group=args.group, min_weight_override=args.min_weight)
