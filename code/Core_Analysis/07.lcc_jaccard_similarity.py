import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
import time
import matplotlib.dates as mdates


def calculate_jaccard(set1, set2):
    """Calculates Jaccard similarity: intersection / union."""
    if not set1 or not set2:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def extract_metadata(filename):
    match = re.search(r'(\d{4})_(\d{2})_\d+', filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def analyze_and_plot_with_stability(directory, prefix, plot_filename="network_evolution.png",
                                    csv_filename="network_metrics.csv", target_years=None):

    files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.graphml")))

    results = []
    prev_nodes = set()       # Nodes from M-1
    prev_prev_nodes = set()  # Nodes from M-2

    for filepath in files:
        # Extract date/month identifier from filename
        filename = os.path.basename(filepath)
        year, month = extract_metadata(filename)

        if year and year in target_years:
            print(f"  Analysing {year}-{month:02d}...")

            G = nx.read_graphml(filepath)
            current_nodes = set(G.nodes())

            # Basic Metrics
            nodes_count = G.number_of_nodes()
            density = nx.density(G)
            avg_clustering = nx.average_clustering(G)

            # 1. Two-Month Stability (M vs M-1)
            # Default to 1.0 for the very first month, then calculate Jaccard
            if not prev_nodes:
                stability_2 = 1.0  # Baseline for Month 1
            else:
                stability_2 = calculate_jaccard(prev_nodes, current_nodes)

            # 2. Three-Month Stability (M vs M-1 vs M-2)
            if not prev_nodes:
                stability_3 = 1.0  # Baseline for Month 1
            elif not prev_prev_nodes:
                stability_3 = None  # Month 2: Impossible to have a 3-month window
            else:
                # Month 3 onwards: Standard calculation
                intersection = current_nodes.intersection(prev_nodes).intersection(prev_prev_nodes)
                union = current_nodes.union(prev_nodes).union(prev_prev_nodes)
                stability_3 = len(intersection) / len(union) if union else 0.0

            results.append({
                "Year": year,
                "Month": month,
                "Nodes": nodes_count,
                "Density": density,
                "Clustering": avg_clustering,
                "2_Stability": stability_2,
                "3_Stability": stability_3
            })

            # Shift the buffer: M-2 becomes M-1, M-1 becomes Current
            prev_prev_nodes = prev_nodes
            prev_nodes = current_nodes

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV for external use
    df.to_csv(csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    # Visualization
    # Ensure index is datetime for plotting
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1))
        df = df.set_index('date').sort_index()

    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    scatter_size = 25

    # 1. Node Count
    axes[0].scatter(df.index, df['Nodes'], s=scatter_size, color='tab:blue', alpha=0.7)
    axes[0].set_title('Network Size (Total Nodes)', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Density
    axes[1].scatter(df.index, df['Density'], s=scatter_size, color='tab:orange', alpha=0.7)
    axes[1].set_title('Network Density', fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Clustering Coefficient
    axes[2].scatter(df.index, df['Clustering'], s=scatter_size, color='tab:green', alpha=0.7)
    axes[2].set_title('Average Clustering Coefficient', fontsize=14, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # 4. Stability Comparison (Jaccard Score)
    # Plot 2-month stability
    axes[3].plot(df.index, df['2_Stability'], color='tab:red', alpha=0.6, label='2-Month Stability', marker='o',
                 markersize=4)
    # Plot 3-month stability
    axes[3].plot(df.index, df['3_Stability'], color='tab:purple', alpha=0.8, label='3-Month Stability', marker='s',
                 markersize=4)

    axes[3].set_title('Temporal Stability (Jaccard Similarity Index)', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Similarity Score')
    axes[3].set_ylim(0, 1.1)
    axes[3].grid(True, linestyle='--', alpha=0.7)
    axes[3].axhline(y=0.5, color='gray', linestyle=':', label='Continuity Threshold (0.5)')
    axes[3].legend()

    # Formatting X-Axis
    axes[3].set_xlabel('Year', fontsize=12)
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[3].xaxis.set_minor_locator(mdates.MonthLocator())

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

    return df


if __name__ == "__main__":
    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'

    event_line_dates = ['2013-01-01', '2014-01-01']

    target_years = [2012, 2013, 2014]

    JACKARD_SIM_EDITOR_CSV_FILE_PATH = f"{CSV_DIR}/jaccard_sim_editor_lcc_{execution_timestamp}.csv"
    JACKARD_SIM_PAGE_CSV_FILE_PATH = f"{CSV_DIR}/jaccard_sim_page_lcc_{execution_timestamp}.csv"
    JACKARD_SIM_BIPARTITE_CSV_FILE_PATH = f"{CSV_DIR}/jaccard_sim_bipartite_lcc_{execution_timestamp}.csv"

    JACKARD_SIM_EDITOR_PLOT_FILE_PATH = f"{PLOT_DIR}/jaccard_sim_editor_lcc_{execution_timestamp}.png"
    JACKARD_SIM_PAGE_PLOT_FILE_PATH = f"{PLOT_DIR}/jaccard_sim_page_lcc_{execution_timestamp}.png"
    JACKARD_SIM_BIPARTITE_PLOT_FILE_PATH = f"{PLOT_DIR}/jaccard_sim_bipartite_lcc_{execution_timestamp}.png"

    df = analyze_and_plot_with_stability(
        directory=GRAPH_DIR,
        prefix="e_e_co-occurence_lcc",
        plot_filename=JACKARD_SIM_EDITOR_PLOT_FILE_PATH,
        csv_filename=JACKARD_SIM_EDITOR_CSV_FILE_PATH,
        target_years=target_years
    )

    df = analyze_and_plot_with_stability(
        directory=GRAPH_DIR,
        prefix="p_p_co-occurence_lcc",
        plot_filename=JACKARD_SIM_PAGE_PLOT_FILE_PATH,
        csv_filename=JACKARD_SIM_PAGE_CSV_FILE_PATH,
        target_years=target_years
    )

    df = analyze_and_plot_with_stability(
        directory=GRAPH_DIR,
        prefix="bipartite_lcc",
        plot_filename=JACKARD_SIM_BIPARTITE_PLOT_FILE_PATH,
        csv_filename=JACKARD_SIM_EDITOR_CSV_FILE_PATH,
        target_years=target_years
    )

    print("\nEnd of processing.")
