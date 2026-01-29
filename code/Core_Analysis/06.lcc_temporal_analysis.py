import networkx as nx
import pandas as pd
import os
import re
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, List


def calculate_nodf(G):
    """
    Calculates the Nestedness based on Overlap and Decreasing Fill (NODF).
    Applicable to bipartite graphs.
    """
    if not nx.is_bipartite(G):
        return None

    # Get the bipartite matrix
    top_nodes = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
    if not top_nodes:  # Fallback if bipartite attribute is missing
        top_nodes = set(nx.bipartite.sets(G)[0])

    adj_matrix = nx.bipartite.biadjacency_matrix(G, list(top_nodes)).toarray()

    # Simple implementation of nestedness concept:
    # Percentage of overlap in neighborhoods
    rows, cols = adj_matrix.shape
    if rows < 2 or cols < 2:
        return 0

    # Sort matrix by fill (standard for nestedness)
    row_sums = np.sum(adj_matrix, axis=1)
    col_sums = np.sum(adj_matrix, axis=0)
    matrix = adj_matrix[np.argsort(-row_sums)][:, np.argsort(-col_sums)]

    def get_overlap(vec1, vec2):
        sum1 = np.sum(vec1)
        sum2 = np.sum(vec2)

        # If the first vector is empty (sum is 0), overlap is undefined or 0.
        # Also, the existing logic returns 0 if vec1 is larger/equal to vec2 or vec2 is 0.
        if sum1 == 0 or sum2 == 0 or sum1 >= sum2:
            return 0

        # Perform bitwise AND and divide by the sum of the first vector
        # sum1 is guaranteed to be > 0 at this point
        return np.sum(vec1 & vec2) / sum1

    # Simplified NODF-like metric for temporal comparison
    # (Checking if smaller degree nodes are subsets of larger ones)
    total_nodf = 0
    count = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            overlap = get_overlap(matrix[j], matrix[i])
            total_nodf += overlap
            count += 1

    return (total_nodf / count) if count > 0 else 0


def extract_metadata(filename):
    match = re.search(r'(\d{4})_(\d{2})_\d+', filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def calculate_editor_metrics(G):
    """
    Calculates specific metrics used in co-authorship/editor network papers.
    Assumes G is the Largest Connected Component (LCC).
    """
    metrics = {}

    # 1. Clustering Coefficient (Global)
    metrics['avg_clustering'] = nx.average_clustering(G)

    # 2. Mean Degree (k = 2E / N)
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    metrics['mean_degree'] = (2 * edges) / nodes if nodes > 0 else 0

    # 3. Diameter and Mean Distance
    # Note: These can be slow on very large networks.
    try:
        metrics['diameter'] = nx.diameter(G)
        metrics['mean_distance'] = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        metrics['diameter'] = None
        metrics['mean_distance'] = None

    # 4. Degree Correlation (Assortativity)
    # Measures if high-degree nodes connect to other high-degree nodes
    metrics['degree_correlation'] = nx.degree_pearson_correlation_coefficient(G)

    return metrics


def analyze_graph_type(directory, prefix, target_years, graph_type):
    results = []
    search_pattern = os.path.join(directory, f"{prefix}*.graphml")
    files = glob.glob(search_pattern)

    print(f"\nProcessing Group: {prefix}")

    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        year, month = extract_metadata(filename)

        if year and year in target_years:
            print(f"  Analysing {year}-{month:02d}...")
            try:
                G = nx.read_graphml(filepath)
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()

                metrics = {
                    "year": year,
                    "month": month,
                    "nodes": n_nodes,
                    "edges": n_edges,
                    "density": nx.density(G)
                }

                # Conditional Metrics
                if graph_type == "Bipartite":
                    metrics["avg_clustering"] = nx.bipartite.average_clustering(G)
                    # ADDING NESTEDNESS HERE
                    metrics["nestedness_nodf"] = calculate_nodf(G)
                else:
                    metrics["avg_clustering"] = nx.average_clustering(G)
                    metrics["nestedness_nodf"] = None  # Not applicable

                if nx.is_connected(G):
                    metrics["avg_path_length"] = nx.average_shortest_path_length(G)
                else:
                    metrics["avg_path_length"] = None

                if graph_type == "Editor":
                    # Mean degree: k = 2E / N
                    metrics['mean_degree'] = (2 * n_edges) / n_nodes if n_nodes > 0 else 0

                    # Clustering coefficient (Global)
                    metrics['clustering_coeff'] = nx.average_clustering(G)

                    # Degree correlation (Assortativity)
                    metrics['degree_correlation'] = nx.degree_pearson_correlation_coefficient(G)

                    # Diameter and Mean Distance (Average Shortest Path)
                    # Warning: These are O(V*E) and can be slow for very large graphs
                    try:
                        metrics['diameter'] = nx.diameter(G)
                        metrics['mean_distance'] = nx.average_shortest_path_length(G)
                    except Exception as e:
                        metrics['diameter'] = None
                        metrics['mean_distance'] = None

                results.append(metrics)
            except Exception as e:
                print(f"    Error: {e}")

    df = pd.DataFrame(results)
    return df.sort_values(['year', 'month']).reset_index(drop=True) if not df.empty else df


def plot_lcc_metrics(
        df: pd.DataFrame,
        plot_filepath: str,
        foot_note: str = "",
        event_line_dates: Optional[List[str]] = None,
        plot_type: str = 'Editor'
) -> object:
    """
    Plots structural metrics over time using a 3-panel matplotlib layout.
    """
    if event_line_dates is None:
        event_line_dates = ['2013-01-01', '2014-01-01']
    if df.empty:
        print(f"DataFrame for {plot_type} is empty. Skipping plot.")
        return

    # Ensure index is datetime for plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.set_index('date').sort_index()

    start_year = df.index.min().year
    end_year = df.index.max().year
    title = f"Monthly Core Structure Metrics LCC ({plot_type})"

    # Column mapping (adjusting to what we collect in analyze_graph_type)
    comp_nodes = 'nodes'
    comp_edges = 'edges'

    # For bipartite, we might have specific node types; otherwise we use total
    comp_secondary = 'avg_clustering'
    q_col = 'avg_clustering'  # Placeholder for modularity if not calculated
    nest_col = 'nestedness_nodf' if 'nestedness_nodf' in df.columns else None

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    scatter_size = 25

    # --- Panel 1: Nodes / Size ---
    axes[0].scatter(df.index, df[comp_nodes], s=scatter_size, label='Total Nodes', color='#1f77b4', alpha=0.7)
    if 'nodes_type_2' in df.columns:  # If we split bipartite nodes later
        axes[0].scatter(df.index, df['nodes_type_2'], s=scatter_size, label='Secondary Nodes', color='#FFA07A',
                        alpha=0.7)

    axes[0].set_title(f'{title} ({start_year} - {end_year})', loc='left', fontsize=14)
    axes[0].set_ylabel('Node Count', color='#1f77b4', fontsize=9)
    axes[0].grid(axis='y', linestyle='--')
    axes[0].legend(loc='best', fontsize=8)

    # --- Panel 2: Edges ---
    axes[1].scatter(df.index, df[comp_edges], s=scatter_size, label='Edges', color='#5D2E8C', alpha=0.7)
    axes[1].set_ylabel('Edge Count', color='#5D2E8C', fontsize=9)
    axes[1].grid(axis='y', linestyle='--')
    axes[1].legend(loc='best', fontsize=8)

    # --- Panel 3: Complex Metrics (Clustering / Nestedness) ---
    if plot_type.lower() == 'editor':
        # Primary axis for Clustering (0-1)
        ax3_left = axes[2]
        c_col = 'clustering_coeff' if 'clustering_coeff' in df.columns else 'avg_clustering'

        if c_col in df.columns:
            ax3_left.scatter(df.index, df[c_col], s=scatter_size, label='Avg Clustering', color='red', alpha=0.7)
            ax3_left.set_ylabel('Clustering Coeff', color='red', fontsize=10)
            ax3_left.set_ylim(0, max(1.0, df[c_col].max() * 1.1))

        # Secondary axis for Distance/Diameter (integers/floats > 1)
        ax3_right = axes[2].twinx()
        if 'mean_distance' in df.columns:
            ax3_right.plot(df.index, df['mean_distance'], label='Mean Distance', color='green', marker='o',
                           markersize=4, alpha=0.6)
        if 'diameter' in df.columns:
            ax3_right.plot(df.index, df['diameter'], label='Diameter', color='black', linestyle=':', alpha=0.5)

        ax3_right.set_ylabel('Distance (Steps)', color='black', fontsize=10)

        # Combine legends
        lines, labels = ax3_left.get_legend_handles_labels()
        lines2, labels2 = ax3_right.get_legend_handles_labels()
        ax3_left.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)

    elif plot_type.lower() == 'bipartite':
        q_col = 'avg_clustering' if 'avg_clustering' in df.columns else 'modularity'
        nest_col = 'nestedness_nodf'

        if q_col in df.columns:
            axes[2].scatter(df.index, df[q_col], s=scatter_size, label='Avg Clustering', color='red', alpha=0.7)
        if nest_col in df.columns:
            axes[2].scatter(df.index, df[nest_col], s=scatter_size, label='Nestedness (NODF)', color='#2ca02c',
                            alpha=0.7)

        axes[2].set_ylabel('Score (0-1)', fontsize=10)
        axes[2].set_ylim(0, 1.1)
        axes[2].legend(loc='upper left', fontsize=8)

    axes[2].grid(axis='y', linestyle='--', alpha=0.6)

    # 4. Global Adjustments (Event Lines & Formatting)
    for ax in [axes[0], axes[1], axes[2]]:
        for date_str in event_line_dates:
            ax.axvline(pd.to_datetime(date_str), color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.8)

    axes[2].set_xlabel('Year')
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[2].xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha='right')

    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    plt.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    print(f"  Plot saved to: {plot_filepath}")


if __name__ == "__main__":
    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'

    event_line_dates = ['2013-10-01', '2012-04-01']

    target_years = [2012, 2013, 2014]

    # Section 1: Editor-Editor (e_e) Co-occurrence
    ee_df = analyze_graph_type(GRAPH_DIR, "e_e_co-occurence_lcc", target_years, graph_type="Editor")
    ee_df.to_csv(f"{CSV_DIR}metrics_lcc_editor_cooccurrence_{execution_timestamp}.csv", index=False)
    print(f"Saved: {CSV_DIR}metrics_lcc_editor_cooccurrence_{execution_timestamp}.csv")
    plot_lcc_metrics(
        ee_df,
        f"{PLOT_DIR}lcc_editor_cooccurrence_metrics_{execution_timestamp}.png",
        foot_note="",
        event_line_dates=event_line_dates,
        plot_type='Editor'
    )

    # Section 2: Page-Page (p_p) Co-occurrence
    pp_df = analyze_graph_type(GRAPH_DIR, "p_p_co-occurence_lcc", target_years, graph_type="Page")
    pp_df.to_csv(f"{CSV_DIR}metrics_lcc_page_cooccurrence_{execution_timestamp}.csv", index=False)
    print(f"Saved: {CSV_DIR}metrics_lcc_page_cooccurrence_{execution_timestamp}.csv")
    plot_lcc_metrics(
        pp_df,
        f"{PLOT_DIR}lcc_page_cooccurrence_metrics_{execution_timestamp}.png",
        foot_note="",
        event_line_dates=event_line_dates,
        plot_type='Page'
    )

    # Section 3: Bipartite Graphs
    bipartite_df = analyze_graph_type(GRAPH_DIR, "bipartite_lcc", target_years, graph_type="Bipartite")
    bipartite_df.to_csv(f"{CSV_DIR}metrics_lcc_bipartite_{execution_timestamp}.csv", index=False)
    print(f"Saved: {CSV_DIR}metrics_lcc_bipartite_{execution_timestamp}.csv")
    plot_lcc_metrics(
        bipartite_df,
        f"{PLOT_DIR}lcc_bipartite_metrics_{execution_timestamp}.png",
        foot_note="",
        event_line_dates=event_line_dates,
        plot_type='Bipartite'
    )
