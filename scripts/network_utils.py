import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from itertools import combinations
from collections import Counter
import numpy as np
import logging  # Add this import for logging functionality

def build_cooccurrence_graph(df, binary_columns, min_weight=1):
    """
    Builds a symptom co-occurrence graph from a binary dataframe.

    Parameters:
        df (pd.DataFrame): Patient data with binary symptom columns.
        binary_columns (list): Columns to include in co-occurrence calculation.
        min_weight (int): Minimum number of co-occurrences to retain an edge.

    Returns:
        G (networkx.Graph): Weighted, undirected co-occurrence network.
    """
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

def build_and_save_graph(df_subset, binary_columns, label, color, output_dir, config, min_weight=1, layout_seed=42):
    """
    Builds and saves a co-occurrence graph, including its visualization and statistics.

    Parameters:
        df_subset (pd.DataFrame): Subset of the dataframe (PCOS or non-PCOS).
        binary_columns (list): List of binary symptom columns.
        label (str): Label for the network ("pcos" or "non_pcos").
        color (str): Color for the node rectangles in the plot.
        output_dir (str): Directory where the output files will be saved.
        config (dict): Configuration dictionary for settings (e.g., export options).
        min_weight (int): Minimum co-occurrence weight to include an edge.
        layout_seed (int): Seed for graph layout.
    """
    # Create the graph using co-occurrence
    G = build_cooccurrence_graph(df_subset, binary_columns, min_weight=min_weight)

    # Output paths
    gml_path = os.path.join(output_dir, f"{label}_network.gml")
    image_path = os.path.join(output_dir, f"{label}_network.png")
    csv_path = os.path.join(output_dir, f"{label}_edges.csv")

    # Save GML
    nx.write_gml(G, gml_path)

    # Save edges
    edge_list = pd.DataFrame([
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ])
    edge_list.to_csv(csv_path, index=False)

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
