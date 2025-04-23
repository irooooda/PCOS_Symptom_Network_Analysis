import os
import re
import yaml
import argparse
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime

# Core utilities for building and saving the network graph
from network_utils import (
    build_cooccurrence_graph,
    get_kamada_layout,
    save_graph,
    export_edges_csv,
    export_network_stats,
    export_top_nodes
)

# Optional colored terminal logging
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

# Custom logger with optional color for better visibility
def log(msg):
    if COLOR_ENABLED:
        parts = re.split(r"(\{.*?\})", msg)
        colored = "".join([
            Fore.CYAN + p + Style.RESET_ALL if p.startswith("{") and p.endswith("}") else Fore.RED + p + Style.RESET_ALL
            for p in parts
        ])
        print("•", colored)
    else:
        logging.info(msg)

# Constructs, analyzes, and optionally plots a symptom co-occurrence network
def build_and_save_graph(df_subset, label, color, base_output_dir, binary_cols, config, layout_seed, top_n_nodes, min_weight):
    # Build co-occurrence graph from binary symptom data
    G = build_cooccurrence_graph(df_subset, binary_cols, min_weight=min_weight)
    log(f"Building {{{label.upper()}}} graph with {{{G.number_of_nodes()}}} nodes")

    # Use timestamped subdirectory for reproducibility and historical tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    base_name = f"{label}_network"

    # Paths for all output files related to this graph
    paths = {
        "gml": os.path.join(output_dir, f"{base_name}.gml"),
        "image": os.path.join(output_dir, f"{base_name}.png"),
        "csv": os.path.join(output_dir, f"{base_name}_edges.csv"),
        "stats": os.path.join(output_dir, f"{base_name}_stats.txt"),
        "top_nodes": os.path.join(output_dir, f"{base_name}_top_nodes.csv")
    }

    # Save GML and export edge list
    save_graph(G, label, output_dir)
    log(f"Saved GML → {{{paths['gml']}}}")

    export_edges_csv(G, label, output_dir)
    log(f"Saved edge list → {{{paths['csv']}}}")

    # Export network statistics
    if config["network"].get("export_stats", False):
        export_network_stats(G, paths["stats"])
        log(f"Saved stats → {{{paths['stats']}}}")

    # Export top-n nodes by degree or centrality
    if config["network"].get("export_top_nodes", False):
        export_top_nodes(G, paths["top_nodes"], n=top_n_nodes)
        log(f"Saved top nodes → {{{paths['top_nodes']}}}")

    # Large networks are skipped to avoid long rendering times
    if G.number_of_nodes() > 1000:
        log(f"⚠️Image is skipped. Reason: Too large to plot ({{{G.number_of_nodes()}}} nodes).")
        return

    # Generate layout coordinates using Kamada-Kawai
    pos = get_kamada_layout(G, layout_seed=layout_seed)

    # Set edge weights for drawing
    edge_widths = [d["weight"] * 0.01 for _, _, d in G.edges(data=True)]
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}

    # Node appearance settings
    node_fill = "#009999" if color == "red" else "#B0B0B0"
    node_edge = "#666666"
    node_text_color = "#FFFFFF" if color == "red" else "#222222"

    # Start drawing
    plt.figure(figsize=(24, 26))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.axis("off")

    nx.draw_networkx_edges(
        G, pos, width=edge_widths, alpha=0.25, edge_color="#AAAAAA", ax=ax
    )

    # Only show edge labels if the graph is small enough
    if len(G.edges) < 200: # change number if needed
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6, font_color="#666666", ax=ax
        )

    # Draw nodes as rectangles
    box_width, box_height = 0.20, 0.07
    for node in G.nodes():
        x, y = pos[node]
        rect = Rectangle((x - box_width / 2, y - box_height / 2),
                         box_width, box_height,
                         facecolor=node_fill, edgecolor=node_edge,
                         linewidth=1.2, zorder=2,
                         joinstyle='round', capstyle='round')
        ax.add_patch(rect)
        ax.text(x, y, node, fontsize=10,
                ha='center', va='center', zorder=3,
                color=node_text_color, fontname="DejaVu Sans")

    # Title and save
    plt.title(
        f"{label.upper()} Symptom Co-occurrence Network",
        fontsize=20,
        pad=20,
        loc='center',
        fontweight='bold',
        color="#444444"
    )
    plt.tight_layout()
    plt.savefig(paths["image"], dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Saved image → {{{paths['image']}}}")

# Main entry: handles dataset I/O and decides which graphs to generate
def main(group: str, min_weight_override: int = None):
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve data paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, config["data_paths"]["processed_data"])
    network_base_dir = os.path.join(base_dir, config["data_paths"]["networks"])
    input_file = os.path.join(processed_dir, config["files"]["transformed_csv"])

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"❌Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    log(f"Loaded dataset: {{{input_file}}}")

    pcos_label = config["columns"]["pcos_label"]
    if pcos_label not in df.columns:
        raise ValueError(f"❌Missing target column '{pcos_label}'.")

    # Find all binary columns (excluding the target)
    binary_cols = [
        col for col in df.columns
        if col != pcos_label and df[col].dropna().isin([0, 1]).all()
    ]
    if not binary_cols:
        raise ValueError("❌No binary features found.")

    # Split dataset based on PCOS label
    df_pcos = df[df[pcos_label] == 1].copy()
    df_non_pcos = df[df[pcos_label] == 0].copy()

    layout_seed = config["network"]["layout_seed"]
    top_n_nodes = config["network"].get("top_n_nodes", 40)
    default_min_weight = config["network"]["min_edge_weight"]
    min_weight = min_weight_override if min_weight_override is not None else default_min_weight

    # Generate graphs as per user-specified group
    if group in ["pcos", "both"]:
        build_and_save_graph(df_pcos, "pcos", "red", os.path.join(network_base_dir, "with_pcos"),
                             binary_cols, config, layout_seed, top_n_nodes, min_weight)

    if group in ["non_pcos", "both"]:
        build_and_save_graph(df_non_pcos, "non_pcos", "green", os.path.join(network_base_dir, "without_pcos"),
                             binary_cols, config, layout_seed, top_n_nodes, min_weight)

# CLI entry point (allows script to be run with arguments)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PCOS symptom co-occurrence networks.")
    parser.add_argument("--group", choices=["pcos", "non_pcos", "both"], default="both", help="Which cohort to process")
    parser.add_argument("--min-weight", type=int, help="Minimum edge weight to include")

    args = parser.parse_args()
    main(group=args.group, min_weight_override=args.min_weight)
