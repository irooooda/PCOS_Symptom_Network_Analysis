import os
import networkx as nx
import pandas as pd
from itertools import combinations
from collections import Counter
import numpy as np


# Build a symptom co-occurrence graph from a binary-encoded DataFrame
def build_cooccurrence_graph(df, binary_columns, min_weight=1):
    cooccurrence_counter = Counter()

    # Count pairwise co-occurrences for active symptoms per row
    for _, row in df.iterrows():
        active_symptoms = [col for col in binary_columns if row[col] == 1]
        for pair in combinations(sorted(active_symptoms), 2):
            cooccurrence_counter[pair] += 1

    # Build the graph using only edges that meet the weight threshold
    G = nx.Graph()
    for (symptom1, symptom2), weight in cooccurrence_counter.items():
        if weight >= min_weight:
            G.add_edge(symptom1, symptom2, weight=weight)

    return G


# Generate a layout using Kamada-Kawai algorithm with optional seeding
def get_kamada_layout(G, layout_seed=42):
    try:
        pos = nx.kamada_kawai_layout(G, weight="weight", seed=layout_seed)
    except TypeError:
        # Fallback for older NetworkX versions that don't accept 'seed'
        np.random.seed(layout_seed)
        pos = nx.kamada_kawai_layout(G)
    return pos


# Save the graph to GML format for reuse or external tools
def save_graph(G, label, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gml_path = os.path.join(output_dir, f"{label}_network.gml")
    nx.write_gml(G, gml_path)


# Export the graph's edge list to CSV
def export_edges_csv(G, label, output_dir):
    edge_list = pd.DataFrame([
        {"source": u, "target": v, "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ])
    csv_path = os.path.join(output_dir, f"{label}_edges.csv")
    edge_list.to_csv(csv_path, index=False)
    return csv_path


# Output basic network statistics to a text file
def export_network_stats(G, stats_path):
    with open(stats_path, "w") as f:
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Density: {nx.density(G):.4f}\n")


# Save the top-N nodes by degree to CSV
def export_top_nodes(G, top_n_path, n=20):
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n]
    pd.DataFrame(top_nodes, columns=["node", "degree"]).to_csv(top_n_path, index=False)
