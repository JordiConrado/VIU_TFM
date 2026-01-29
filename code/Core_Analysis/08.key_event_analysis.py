import networkx as nx
import community as community_louvain  # python-louvain
import numpy as np
import os
import time

def key_event_analysis(editor_file, page_file, bipartite_file):
    """
    Extracts structural statistics from Wiki graph files.
    Focuses on Modularity, Density, and Nestedness.
    """
    results = {}

    # 1. Load Graphs
    try:
        G_editor = nx.read_graphml(editor_file)
        G_page = nx.read_graphml(page_file)
        G_bip = nx.read_graphml(bipartite_file)
    except Exception as e:
        return f"Error loading files: {e}"

    # 2. Basic Topology & Modularity (Unimodal Projections)
    for name, G in [("Editor", G_editor), ("Page", G_page)]:
        # Basic Metrics
        nodes = G.number_of_nodes()
        edges = G.number_of_edges()
        density = nx.density(G)

        # Modularity (Louvain)
        # Low modularity suggests a breakdown of distinct sub-communities
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)

        # Clustering
        avg_clustering = nx.average_clustering(G)

        results[name] = {
            "Nodes": nodes,
            "Edges": edges,
            "Density": f"{density:.5f}",
            "Modularity": f"{modularity:.4f}",
            "Avg Clustering": f"{avg_clustering:.4f}"
        }

    # 3. Bipartite Specifics (Nestedness)
    # Nestedness indicates if specialists edit a subset of what generalists edit.
    # A drop in nestedness suggests fragmented, disorganized editing patterns.

    # Identify partitions
    top_nodes = {n for n, d in G_bip.nodes(data=True) if d.get('bipartite') == 0}
    if not top_nodes:  # Fallback if attribute is missing
        top_nodes = set(n for n, d in G_bip.degree() if n in G_editor)

    bottom_nodes = set(G_bip) - top_nodes

    # Simple Nestedness Proxy: NODF-like measure (Overlap and Decreasing Fill)
    # For a full NODF, we'd need the adjacency matrix
    adj_matrix = nx.bipartite.biadjacency_matrix(G_bip, list(top_nodes), list(bottom_nodes)).toarray()

    def calculate_nodf(matrix):
        """Simple NODF implementation for nestedness."""
        rows, cols = matrix.shape
        if rows < 2 or cols < 2: return 0

        # Sort matrix by row and column sums
        matrix = matrix[np.argsort(matrix.sum(axis=1))[::-1], :]
        matrix = matrix[:, np.argsort(matrix.sum(axis=0))[::-1]]

        def pair_overlap(v1, v2):
            if v1.sum() == 0 or v2.sum() == 0 or v1.sum() <= v2.sum():
                return 0
            return (np.logical_and(v1, v2).sum() / v2.sum())

        row_overlaps = []
        for i in range(rows):
            for j in range(i + 1, rows):
                row_overlaps.append(pair_overlap(matrix[i], matrix[j]))

        return np.mean(row_overlaps) if row_overlaps else 0

    results["Bipartite"] = {
        "Editors": len(top_nodes),
        "Pages": len(bottom_nodes),
        "Nestedness_Proxy": f"{calculate_nodf(adj_matrix):.4f}"
    }

    return results


if __name__ == "__main__":

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'

    #files = {
    #    "editor": f"{GRAPH_DIR}run_14_wiki_2013_month_10_clean_c_e_e_20251216_115748_7.graphml",
    #    "page": f"{GRAPH_DIR}run_14_wiki_2013_month_10_clean_c_p_p_20251216_115748_7.graphml",
    #    "bipartite": f"{GRAPH_DIR}run_14_wiki_2013_month_10_clean_bipartite_20251216_115748_7.graphml"
    #}

    files = {
        "editor": f"{GRAPH_DIR}e_e_co-occurence_lcc_2013_10_20260118_171535.graphml",
        "page": f"{GRAPH_DIR}p_p_co-occurence_lcc_2013_10_20260118_171835.graphml",
        "bipartite": f"{GRAPH_DIR}bipartite_lcc_2013_10_20260118_172114.graphml"
    }

    # Check if files exist before running
    if all(os.path.exists(f) for f in files.values()):
        stats = key_event_analysis(files["editor"], files["page"], files["bipartite"])

        print(f"{'Metric':<20} | {'Editor Proj.':<15} | {'Page Proj.':<15}")
        print("-" * 55)
        for key in ["Nodes", "Edges", "Density", "Modularity", "Avg Clustering"]:
            print(f"{key:<20} | {stats['Editor'][key]:<15} | {stats['Page'][key]:<15}")

        print(f"\nBipartite Nestedness Proxy: {stats['Bipartite']['Nestedness_Proxy']}")
    else:
        print("Please ensure the .graphml files are in the same directory.")