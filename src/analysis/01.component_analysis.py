import networkx as nx
import os
import time
from typing import Dict, Any, Tuple, Optional, Set
import glob
import re
import pandas as pd
from community import community_louvain
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
from networkx.algorithms import bipartite as nx_bipartite
from code.Utils import nestedness_calculator as nodf_calculator

"""
Initial implementation of core structure analysis for monthly editor-editor, page-page, and bipartite graphs.
Evaluate largest component for Modularity (Louvain) and Nestedness (NODF) for all monthly graphs between a start and end year.

Change Log:
    2025-12-02: Initial creation date
    2025-12-10: Fixed plot with component size to plot the number of nodes in the largest component.
    2025-12-15: Add vertical red line right where it shows a peak in nestedness and low modularity: October 2013
    2025-12-16: Modularity and Nestedness in same plot
    2025-12-20: Added page-page analysis option. Now it creates two plots, one for editor-editor and one for page-page.
    2025-12-21: Save LCC co-occurrence graph to file for each month analyzed.
    2025-12-21: Save corresponding bipartite subgraph for LCC to file for each month analyzed. 
    2026-01-14: Added bipartite analysis option. Now it creates three plots: editor-editor, page-page, and bipartite.  
"""


# --- Helper Function to Extract Bipartite Partitions (Required for Nestedness) ---
def get_bipartite_partitions(graph: nx.Graph) -> Tuple[set, set]:
    """
    Separates the nodes of a bipartite graph into two sets based on the 'bipartite' attribute.
    Set 1 (Partition 0) is expected to be 'pages' (bipartite=0).
    Set 2 (Partition 1) is expected to be 'users/editors' (bipartite=1).
    """
    set_0 = set()  # Pages
    set_1 = set()  # Users/Editors

    for node, data in graph.nodes(data=True):
        if data.get('bipartite') == 0:
            set_0.add(node)
        elif data.get('bipartite') == 1:
            set_1.add(node)

    # Always ensure that the larger set is passed as the 'top' set
    # for the nestedness calculation function, if required by the library.
    # We will assume set_1 (Users) is the primary set being tested for nestedness
    # relative to set_0 (Pages).
    return set_0, set_1


# --- Function for Nestedness Calculation ---
"""
def calculate_nestedness(
        largest_component_editor_nodes: set,
        full_bipartite_graph: nx.Graph,
) -> float:


    Calculates Nestedness (NODF) for the connections involving the editors
    from the largest component using the full bipartite graph.

    Args:
        largest_component_editor_nodes: The set of editor IDs belonging to the largest component
                                        of the UNIPARTITE graph.
        full_bipartite_graph: The original full Editor-Page bipartite graph.

    Returns:
        The nestedness score (float), or 0.0 if the resulting subgraph is empty.

    if not largest_component_editor_nodes or full_bipartite_graph.number_of_edges() == 0:
        return 0.0

    subgraph_nodes = set(largest_component_editor_nodes)
    for node in largest_component_editor_nodes:
        if node in full_bipartite_graph:
            subgraph_nodes.update(full_bipartite_graph.neighbors(node))

    bipartite_subgraph = full_bipartite_graph.subgraph(subgraph_nodes).copy()

    if bipartite_subgraph.number_of_edges() == 0:
        return 0.0

    # 2. Get the two partitions for the Nestedness Calculation
    pages, editors = get_bipartite_partitions(bipartite_subgraph)

    # Filter the partitions to include only nodes present in the subgraph
    final_editors = sorted(list(editors.intersection(bipartite_subgraph.nodes())))
    final_pages = sorted(list(pages.intersection(bipartite_subgraph.nodes())))

    if not final_editors or not final_pages:
        return 0.0

    # 3. Convert the NetworkX graph into the NumPy Adjacency Matrix (GG)
    # We use 'weight' to generate a weighted matrix (which nestedness calc might use)
    # R: Editors (Set 1), C: Pages (Set 0)
    try:
        biadjacency_matrix = nx_bipartite.biadjacency_matrix(
            bipartite_subgraph,
            row_nodes=final_editors,
            column_nodes=final_pages,
            weight='weight'  # Use the revision count as the connection strength
        )

        # Convert the sparse matrix to a dense NumPy array (GG)
        GG = biadjacency_matrix.toarray()

        # 4. REAL NESTEDNESS CALCULATION (Insert your function call here)

        # --- Example of where to place your call ---
        # calculator = NestednessCalculator()
        # nestedness_score = calculator.calculate_nodf(GG)

        # MOCK SCORE: Used for demonstration since the library is not available
        num_rows, num_cols = GG.shape
        density = np.sum(GG > 0) / (num_rows * num_cols)
        score = 0.4 + (density * 0.5)
        nestedness_score = min(score, 0.9)

    except Exception as e:
        print(f"Nestedness calculation failed (using mock): {e}", file=sys.stderr)
        nestedness_score = 0.0

    return nestedness_score
"""

def analyze_largest_component_monthly(
        analysis_type: str, # 'Editor' or 'Page'
        b_graph: nx.Graph,  # Original Bipartite Graph
        c_graph: nx.Graph,  # Co-occurrence Graph (Editor-Editor or Page-Page)
        year: str,
        month: str,
        graphs_dir: str,
        verbose: bool = False,
        save_subgraph: bool = False
) -> Tuple[Dict[Any, int], float, float, int, int]:

    """
    Analyzes the largest connected component of the input graph G for
    community structure (Modularity) and topological structure (Nestedness).

    This function uses the simpler `community_louvain.modularity()` approach.

    Args:
        analysis_type: Type of analysis ('Editor' or 'Page').
        b_graph: The original bipartite graph (e.g., editor-page graph).
        c_graph: The input co-occurrence graph (e.g., editor-editor graph).
        year: The year string for file naming.
        month: The month string for file naming.
        graphs_dir: Directory to save the largest component graph file.
        verbose: If True, prints detailed information during processing.
        save_subgraph: If True, saves the largest component subgraph to a file.

    Returns:
        A tuple containing:
        1. The partition dictionary (node -> community_id) for the largest component.
        2. The Modularity score (r) for that partition.
        3. The Nestedness score (nest).
        4. The number of nodes in the largest component.
        5. The number of edges in the largest component.
    """

    # 1. Check for connected components
    if c_graph.number_of_nodes() == 0:
        print("Graph is empty.")
        return {}, 0.0, 0.0, 0, 0

    Gcc = sorted(nx.connected_components(c_graph), key=len, reverse=True)
    if not Gcc:
        return {}, 0.0, 0.0, 0, 0

    # Extract the largest component
    component = c_graph.subgraph(Gcc[0])
    LCC = component.copy()
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    if save_subgraph:
        # Filename logic
        prefix = 'e_e' if analysis_type == 'Editor' else 'p_p'
        lcc_filename = f'{prefix}_co-occurence_lcc_{year}_{month}_{timestamp}.graphml'
        lcc_file_path = os.path.join(graphs_dir, lcc_filename)

        try:
            nx.write_graphml(LCC, lcc_file_path)
        except Exception as e:
            print(f"Failed to save LCC graph: {e}", file=sys.stderr)

    # 2. Bipartite Subgraph Logic (THE ADAPTATION)
    # The nodes in LCC are either all editors or all pages.
    primary_nodes = set(LCC.nodes())
    neighbor_nodes = set()

    # Find the "other half" of the bipartite relationship
    for node in primary_nodes:
        # If LCC is Page-Page, neighbors in b_graph are Editors
        # If LCC is Editor-Editor, neighbors in b_graph are Pages
        neighbors = b_graph.neighbors(node)
        neighbor_nodes.update(neighbors)

    # Combine them to get the relevant bipartite slice
    nodes_to_keep = primary_nodes.union(neighbor_nodes)
    b_sub = b_graph.subgraph(nodes_to_keep).copy()

    # Save bipartite subgraph
    b_sub_filename = f'b_subgraph_lcc_{year}_{month}_{timestamp}.graphml'
    b_sub_path = os.path.join(graphs_dir, b_sub_filename)
    try:
        nx.write_graphml(b_sub, b_sub_path)
    except Exception as e:
        print(f"Failed to save B-subgraph: {e}", file=sys.stderr)

    # 3. Modularity
    part = community_louvain.best_partition(component, weight='weight')
    r = community_louvain.modularity(part, component, weight='weight')

    # 4. Nestedness
    # Note: Ensure NestednessCalculator is imported in your environment
    GG = nx.to_numpy_array(component, weight=None)
    try:
        from code import NestednessCalculator # Example import
        nest = NestednessCalculator(GG).nodf(GG)
    except ImportError:
        nest = 0.0 # Fallback if library is missing

    if verbose:
        print(f"[{analysis_type} Analysis] Nodes: {LCC.number_of_nodes()}, Edges: {LCC.number_of_edges()}")
        print(f"Modularity: {r:.4f}, Nestedness: {nest:.4f}")

    return part, r, nest, LCC.number_of_nodes(), LCC.number_of_edges()


def analyze_bipartite_largest_component_monthly(
        b_graph: nx.Graph,  # Bipartite Graph
        year: str,
        month: str,
        graphs_dir: str,
        verbose: bool = False,
        save_subgraph: bool = False
) -> Tuple[Dict[Any, int], float, float, int, int, int, int]:

    """
    Analyzes the largest connected component of the input graph G for
    community structure (Modularity) and topological structure (Nestedness).

    This function uses the simpler `community_louvain.modularity()` approach.

    Args:
        b_graph: The original bipartite graph (e.g., editor-page graph).
        year: The year string for file naming.
        month: The month string for file naming.
        graphs_dir: Directory to save the largest component graph file.
        verbose: If True, prints detailed information during processing.
        save_subgraph: If True, saves the largest component bipartite subgraph to file.

    Returns:
        A tuple containing:
        1. The partition dictionary (node -> community_id) for the largest component.
        2. The Modularity score (r) for that partition.
        3. The Nestedness score (nest).
        4. The number of nodes in the largest component.
        5. The number of edges in the largest component.
    """

    # 1. Check for connected components
    conn_components = sorted(nx.connected_components(b_graph), key=len, reverse=True)

    if len(conn_components) > 0:
        # b. Extract the largest component as a subgraph
        lcc_b_sub = b_graph.subgraph(conn_components[0]).copy()

        # Identify page and editor nodes in the bipartite subgraph
        page_nodes = {n for n, d in lcc_b_sub.nodes(data=True) if d.get('bipartite') == 0}
        editor_nodes = set(lcc_b_sub) - page_nodes

        if save_subgraph:
            # Save subgraph
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            b_lcc_filename = f'bipartite_lcc_{year}_{month}_{timestamp}.graphml'
            b_lcc_path = os.path.join(graphs_dir, b_lcc_filename)
            try:
                nx.write_graphml(lcc_b_sub, b_lcc_path)
            except Exception as e:
                print(f"Failed to save LCC graph to file: {e}", file=sys.stderr)

        # 2. Modularity (vía Unipartite projection)
        # Louvain is not bipartite. Project over editor nodes
        projection = nx_bipartite.weighted_projected_graph(lcc_b_sub, editor_nodes)

        if len(projection) > 0:
            part = community_louvain.best_partition(projection, weight='weight')
            r = community_louvain.modularity(part, projection, weight='weight')
        else:
            part, r = {}, 0.0

        # 3. Cálculo de Nestedness
        # Para grafos bipartitos, necesitamos la matriz de incidencia (Rectangular)
        # Rows: Editors, Columns: Pages
        matrix = nx_bipartite.biadjacency_matrix(
            lcc_b_sub,
            row_order=list(editor_nodes),
            column_order=list(page_nodes),
            weight=None  # Unweighted para NODF estándar
        ).toarray()

        try:
            nest = nodf_calculator.NestednessCalculator(matrix).nodf(matrix)
        except Exception as e:
            if verbose: print(f"Error calculating Nestedness: {e}")
            nest = 0.0

        if verbose:
            print(f"Largest Component Metrics:")
            print(
                f"  Nodes: {lcc_b_sub.number_of_nodes()} (Editors: {len(editor_nodes)}, Pages: {len(page_nodes)})")
            print(f"  Edges: {lcc_b_sub.number_of_edges()}")
            print(f"  Modularity (r) (projected): {r:.4f}")
            print(f"  Nestedness (nest): {nest:.4f}")

        return part, r, nest, lcc_b_sub.number_of_nodes(), lcc_b_sub.number_of_edges(), len(editor_nodes), len(page_nodes)
    else:
        print("Graph has no connected components (or is empty).")
        return {}, 0.0, 0.0, 0, 0, 0, 0


def plot_core_structure_metrics(
        df: pd.DataFrame,
        plot_filepath: str,
        foot_note: str,
        event_line_date: Optional[str] = None,
        analysis_type: str = 'Editor'
):
    """
    Plots the three structural metrics (Nodes. Edges, Modularity/Nestedness) over time for the monthly LCC

    Args:
        df: DataFrame containing Date index and metric columns.
        plot_filepath: Path to save the plot.
        foot_note: Text to be displayed as a footnote.
        :param df:
        :param plot_filepath:
        :param foot_note:
        :param event_line_date: Optional date string (YYYY-MM-DD) to draw a vertical line for significant events.
        :param analysis_type: Type of analysis ('Editor' or 'Page') for title purposes.
    """

    title: str = f"Monthly Core Structure Metrics for Largest Component ({analysis_type}-{analysis_type} Co-occurrence Graphs)"

    if df.empty:
        print("DataFrame is empty. Cannot generate plot.")
        return

    # --- 1. Preparation ---
    start_year = df.index.min().year
    end_year = df.index.max().year

    # Columns to plot on their respective axes
    comp_nodes = 'nodes_largest_comp'
    comp_edges = 'edges_largest_comp'
    q_col = 'modularity_largest_comp'
    nest_col = 'nestedness_largest_comp'

    # --- 2. Create Plot ---
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )
    plt.subplots_adjust(hspace=0.25)
    scatter_size = 12

    # --- Panel 1: Number of Nodes in Largest Component ---
    axes[0].scatter(df.index, df[comp_nodes], s=scatter_size, label='Nodes in Largest Component', color='#1f77b4', alpha=0.7)
    axes[0].set_title(f'{title} ({start_year} - {end_year})', loc='left', fontsize=14)
    axes[0].set_ylabel('Nodes', color='#1f77b4', fontsize=8)
    axes[0].grid(axis='y', linestyle='--')

    for i in event_line_date:
        axes[0].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    # --- Panel 2: Number of Edges in Largest Component ---
    axes[1].scatter(df.index, df[comp_edges], s=scatter_size, label='Edges in Largest Component', color='#5D2E8C', alpha=0.7)
    axes[1].set_ylabel('Edges', color='#5D2E8C', fontsize=8)
    axes[1].grid(axis='y', linestyle='--')

    for i in event_line_date:
        axes[1].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    # --- Panel 3: Modularity and Nestedness of the Largest Component (Proportion) ---
    axes[2].scatter(df.index, df[q_col], s=scatter_size, label='Modularity', color='red', alpha=0.7)
    axes[2].scatter(df.index, df[nest_col], s=scatter_size, label='Nestedness', color='#2ca02c', alpha=0.7)
    axes[2].grid(axis='y', linestyle='--')
    axes[2].set_ylabel('Modularity/Nestedness', color='black', fontsize=8)
    axes[2].set_ylim(0, 1.0)  # Modularity is bounded [0, 1.0]
    axes[2].legend(loc='best')

    for i in event_line_date:
        axes[2].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    """
    # --- Panel 4: Nestedness of the Largest Component (Proportion) ---
    axes[3].scatter(df.index, df[nest_col], s=scatter_size, label='Nestedness (NODF)', color='#2ca02c', alpha=0.7)
    axes[3].grid(axis='y', linestyle='--')
    axes[3].set_ylabel('Nestedness Score', color='#2ca02c', fontsize=8)
    axes[3].set_ylim(0, 0.4)  # Nestedness is typically bounded [0, 1.0], but we are fining to 0.4 for better visualization

    if event_line_date is not None:
        # Add vertical line for significant event
        axes[3].axvline(pd.to_datetime(event_line_date),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event (Oct 2013)')
    """

    # --- Final Formatting ---
    axes[2].set_xlabel('Year')
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xticks(rotation=45, ha='right')
    # Add footnote
    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    plt.savefig(plot_filepath)
    plt.close(fig)


def plot_bipartite_core_structure_metrics(
        df: pd.DataFrame,
        plot_filepath: str,
        foot_note: str,
        event_line_date: Optional[str] = None,
        analysis_type: str = 'Editor'
):
    """
    Plots the three structural metrics (Editors/Pages Edges, Nestedness/Modularity ) over time for the monthly LCC

    Args:
        df: DataFrame containing Date index and metric columns.
        plot_filepath: Path to save the plot.
        foot_note: Text to be displayed as a footnote.
        :param df:
        :param plot_filepath:
        :param foot_note:
        :param event_line_date: Optional date string (YYYY-MM-DD) to draw a vertical line for significant events.
        :param analysis_type: Type of analysis ('Editor' or 'Page') for title purposes.
    """

    title: str = f"Monthly Core Structure Metrics for Largest Component ({analysis_type} Graphs)"

    if df.empty:
        print("DataFrame is empty. Cannot generate plot.")
        return

    # --- 1. Preparation ---
    start_year = df.index.min().year
    end_year = df.index.max().year

    # Columns to plot on their respective axes
    comp_editors = 'editors_largest_comp'
    comp_pages = 'pages_largest_comp'
    comp_edges = 'edges_largest_comp'
    q_col = 'modularity_largest_comp'
    nest_col = 'nestedness_largest_comp'

    # --- 2. Create Plot ---
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )
    plt.subplots_adjust(hspace=0.25)
    scatter_size = 12

    # --- Panel 1: Number of Editors and Pages in Largest Component ---
    axes[0].scatter(df.index, df[comp_editors], s=scatter_size, label='Editors', color='#1f77b4', alpha=0.7)
    axes[0].scatter(df.index, df[comp_pages], s=scatter_size, label='Pages', color='#FFA07A', alpha=0.7)
    axes[0].set_title(f'{title} ({start_year} - {end_year})', loc='left', fontsize=14)
    axes[0].set_ylabel('Editors and Pages', color='#1f77b4', fontsize=8)
    axes[0].grid(axis='y', linestyle='--')
    axes[0].legend(loc='best')

    for i in event_line_date:
        axes[0].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    # --- Panel 2: Number of Edges in Largest Component ---
    axes[1].scatter(df.index, df[comp_edges], s=scatter_size, label='Edges in Largest Component', color='#5D2E8C', alpha=0.7)
    axes[1].set_ylabel('Edges', color='#5D2E8C', fontsize=8)
    axes[1].grid(axis='y', linestyle='--')

    for i in event_line_date:
        axes[1].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    # --- Panel 3: Modularity and Nestedness of the Largest Component (Proportion) ---
    axes[2].scatter(df.index, df[q_col], s=scatter_size, label='Modularity', color='red', alpha=0.7)
    axes[2].scatter(df.index, df[nest_col], s=scatter_size, label='Nestedness', color='#2ca02c', alpha=0.7)
    axes[2].grid(axis='y', linestyle='--')
    axes[2].set_ylabel('Modularity/Nestedness', color='black', fontsize=8)
    axes[2].set_ylim(0, 1.0)  # Modularity is bounded [0, 1.0]
    axes[2].legend(loc='best')

    for i in event_line_date:
        axes[2].axvline(pd.to_datetime(i),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event')

    """
    # --- Panel 4: Nestedness of the Largest Component (Proportion) ---
    axes[3].scatter(df.index, df[nest_col], s=scatter_size, label='Nestedness (NODF)', color='#2ca02c', alpha=0.7)
    axes[3].grid(axis='y', linestyle='--')
    axes[3].set_ylabel('Nestedness Score', color='#2ca02c', fontsize=8)
    axes[3].set_ylim(0, 0.4)  # Nestedness is typically bounded [0, 1.0], but we are fining to 0.4 for better visualization

    if event_line_date is not None:
        # Add vertical line for significant event
        axes[3].axvline(pd.to_datetime(event_line_date),
                        color='#ff7f0e',
                        linestyle='--',
                        linewidth=1.5,
                        label='Significant Event (Oct 2013)')
    """

    # --- Final Formatting ---
    axes[2].set_xlabel('Year')
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xticks(rotation=45, ha='right')
    # Add footnote
    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    plt.savefig(plot_filepath)
    plt.close(fig)


def run_core_analysis_loop(
        graphs_dir: str,
        start_year: int,
        end_year: int,
        run: str,
        output_csv_path: str,
        output_plot_path: str,
        verbose: Optional[bool] = False,
        event_line_date: Optional[Set] = None,
        analysis_type: str = 'Editor',
        save_subgraph: bool = False
):
    """
    Main loop to read graphs, perform analysis, and aggregate results.
    """
    def extract_year_month_from_filename(filename: str) -> Tuple[str, str]:
        pattern = r'_wiki_(\d{4})_month_(\d{2})_'
        match = re.search(pattern, filename)
        return match.groups() if match else ('N/A', 'N/A')

    all_metrics = []

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        # Search for co-occurrence graph files
        if analysis_type == 'Editor':
            c_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year_str}_month_*_clean_c_e_e*.graphml')
            all_files = glob.glob(c_graph_search_path)
        elif analysis_type == 'Page':
            c_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year_str}_month_*_clean_c_p_p*.graphml')
            all_files = glob.glob(c_graph_search_path)
        elif analysis_type == 'Bipartite':
            b_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year_str}_month_*_clean_bipartite*.graphml')
            all_files = glob.glob(b_graph_search_path)
        else:
            raise ValueError("analysis_type must be 'Editor', 'Page', or 'Bipartite'")

        if analysis_type == 'Editor' or analysis_type == 'Page':
            for i, filename in enumerate(all_files):
                file_basename = os.path.basename(filename)
                graph_year, graph_month = extract_year_month_from_filename(file_basename)

                # 1. Read the co-occurrence graph
                try:
                    c_graph = nx.read_graphml(filename)
                except Exception as e:
                    print(f"Error reading {file_basename}: {e}. Skipping.")
                    continue

                # 2. Mock/Load Bipartite Graph (Necessary for Nestedness)
                # In a real scenario, you would load the corresponding Bipartite graph here.
                # E.g., b_graph_path = filename.replace('_c_e_e', '_bipartite')
                if analysis_type == 'Editor':
                    b_graph_path = filename.replace('_c_e_e', '_bipartite')
                elif analysis_type == 'Page':
                    b_graph_path = filename.replace('_c_p_p', '_bipartite')
                else:
                    raise ValueError("analysis_type must be either 'Editor' or 'Page'")

                b_graph = nx.read_graphml(b_graph_path) if os.path.exists(b_graph_path) else None

                # 3. Run the core analysis
                (
                    partition_dict,
                    modularity_score,
                    nestedness_score,
                    num_nodes_largest_comp,
                    num_edges_largest_comp
                ) = analyze_largest_component_monthly(
                        analysis_type,
                        b_graph,
                        c_graph,
                        graph_year,
                        graph_month,
                        graphs_dir,
                        verbose,
                        save_subgraph
                    )

                # 4. Aggregate metrics
                metrics_dict = {
                    'Year': graph_year,
                    'Month': graph_month,
                    'run_number': run,
                    'total_nodes': c_graph.number_of_nodes(),
                    'total_edges': c_graph.number_of_edges(),
                    'num_components': nx.number_connected_components(c_graph),
                    'modularity_largest_comp': modularity_score,
                    'nestedness_largest_comp': nestedness_score,
                    'nodes_largest_comp': num_nodes_largest_comp,
                    'edges_largest_comp': num_edges_largest_comp
                }
                all_metrics.append(metrics_dict)
        elif analysis_type == 'Bipartite':
            for i, filename in enumerate(all_files):
                file_basename = os.path.basename(filename)
                graph_year, graph_month = extract_year_month_from_filename(file_basename)

                # 1. Read the co-occurrence graph
                """
                try:
                    c_graph = nx.read_graphml(filename)
                except Exception as e:
                    print(f"Error reading {file_basename}: {e}. Skipping.")
                    continue
                """

                # 2. Mock/Load Bipartite Graph (Necessary for Nestedness)
                # In a real scenario, you would load the corresponding Bipartite graph here.
                # E.g., b_graph_path = filename.replace('_c_e_e', '_bipartite')
                """
                if analysis_type == 'Editor':
                    b_graph_path = filename.replace('_c_e_e', '_bipartite')
                elif analysis_type == 'Page':
                    b_graph_path = filename.replace('_c_p_p', '_bipartite')
                else:
                    raise ValueError("analysis_type must be either 'Editor' or 'Page'")
                """

                b_graph = nx.read_graphml(filename) if os.path.exists(filename) else None
                graph_nodes = b_graph.number_of_nodes()
                graph_edges = b_graph.number_of_edges()
                graph_editors = sum(1 for _, d in b_graph.nodes(data=True) if d.get('bipartite') == 1)
                graph_pages = sum(1 for _, d in b_graph.nodes(data=True) if d.get('bipartite') == 0)
                c_graph = ""

                # 3. Run the core analysis
                (
                    partition_dict,
                    modularity_score,
                    nestedness_score,
                    num_nodes_largest_comp,
                    num_edges_largest_comp,
                    num_editors_largest_comp,
                    num_pages_largest_comp
                ) = analyze_bipartite_largest_component_monthly(
                        b_graph,
                        graph_year,
                        graph_month,
                        graphs_dir,
                        verbose,
                        save_subgraph
                    )

                # 4. Aggregate metrics
                metrics_dict = {
                    'Year': graph_year,
                    'Month': graph_month,
                    'run_number': run,
                    'total_nodes': graph_nodes,
                    'total_edges': graph_edges,
                    'total_editors': graph_editors,
                    'total_pages': graph_pages,
                    'num_components': nx.number_connected_components(b_graph),
                    'modularity_largest_comp': modularity_score,
                    'nestedness_largest_comp': nestedness_score,
                    'nodes_largest_comp': num_nodes_largest_comp,
                    'edges_largest_comp': num_edges_largest_comp,
                    'editors_largest_comp': num_editors_largest_comp,
                    'pages_largest_comp': num_pages_largest_comp
                }
                all_metrics.append(metrics_dict)

    # 5. Save all aggregated results
    df_metrics = pd.DataFrame(all_metrics)
    print(f"All metrics Calculated....Now saving to file")

    if not df_metrics.empty:
        # Create a datetime index for plotting
        df_metrics['Date'] = pd.to_datetime(
            df_metrics[['Year', 'Month']].assign(day=1).astype(str).agg('-'.join, axis=1), format='%Y-%m-%d')
        df_metrics = df_metrics.set_index('Date').sort_index()

        # Save to CSV
        df_metrics.to_csv(output_csv_path, index=False)
        print(f"Core structure analysis saved to CSV file: {output_csv_path}")

        # 6. Plot the results
        if analysis_type == 'Editor' or analysis_type == 'Page':
            plot_core_structure_metrics(
                df_metrics,
                output_plot_path,
                "Metrics are calculated only on the largest component for the month",
                event_line_date,
                analysis_type=analysis_type
            )
            print(f"Core structure plot generated and saved: {output_plot_path}")
        elif analysis_type == 'Bipartite':
            plot_bipartite_core_structure_metrics(
                df_metrics,
                output_plot_path,
                "Metrics are calculated only on the largest component for the month",
                event_line_date,
                analysis_type='Bipartite'
            )
            print(f"Core structure plot generated and saved: {output_plot_path}")
    else:
        print("No analysis metrics were successfully generated to plot.")


if __name__ == '__main__':

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')

    START_YEAR = 2012
    END_YEAR = 2014
    RUN = '14'

    ROOT_DIR = '/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'
    VERBOSE = False
    # If SAVE_SUBGRAPHS is True, the LCC subgraphs will be saved in the GRAPH_DIR as
    # e_e_co-occurence_lcc_{year}_{month}_{timestamp}.graphml
    # p_p_co-occurence_lcc_{year}_{month}_{timestamp}.graphml
    # bipartite_lcc_{year}_{month}_{timestamp}.graphml

    SAVE_SUBGRAPHS = True # To save the LCC subgraphs or not
    EVENTS = ['2013-10-01', '2012-04-01']  # Significant event dates to highlight in plot

    # 1. Complete editor-editor component metrics analysis
    print("--- Processing Editor-Editor Core Structure Analysis ---")
    #CSV_INPUT_FILE_PATH = f'{CSV_DIR}graph_community_metrics_e_e.csv'

    PLOT_FILE_PATH = f'{PLOT_DIR}{START_YEAR}-{END_YEAR}_graph_core_structure_plot_e_e_{execution_timestamp}.png'
    OUTPUT_CSV = f'core_structure_analysis_e_e_{execution_timestamp}_{START_YEAR}_{END_YEAR}.csv'
    OUTPUT_CSV_PATH = f'{CSV_DIR}{OUTPUT_CSV}'

    # Run the full pipeline
    run_core_analysis_loop(
        graphs_dir=GRAPH_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        run=RUN,
        output_csv_path=OUTPUT_CSV_PATH,
        output_plot_path=PLOT_FILE_PATH,
        verbose=VERBOSE,
        event_line_date=EVENTS,
        analysis_type='Editor',
        save_subgraph=SAVE_SUBGRAPHS
    )

    # 2. Complete page-page component metrics analysis
    print("--- Processing Page-Page Core Structure Analysis ---")
    #CSV_INPUT_FILE_PATH = f'{CSV_DIR}graph_community_metrics_p_p.csv'

    PLOT_FILE_PATH = f'{PLOT_DIR}{START_YEAR}-{END_YEAR}_graph_core_structure_plot_p_p_{execution_timestamp}.png'
    OUTPUT_CSV = f'core_structure_analysis_p_p_{execution_timestamp}_{START_YEAR}_{END_YEAR}.csv'
    OUTPUT_CSV_PATH = f'{CSV_DIR}{OUTPUT_CSV}'

    # Run the full pipeline
    run_core_analysis_loop(
        graphs_dir=GRAPH_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        run=RUN,
        output_csv_path=OUTPUT_CSV_PATH,
        output_plot_path=PLOT_FILE_PATH,
        verbose=VERBOSE,
        event_line_date=EVENTS,
        analysis_type='Page',
        save_subgraph=SAVE_SUBGRAPHS
    )

    # 3. Complete bipartite component metrics analysis
    print("--- Processing Bipartite Core Structure Analysis ---")
    #CSV_INPUT_FILE_PATH = f'{CSV_DIR}graph_community_metrics_p_p.csv'

    PLOT_FILE_PATH = f'{PLOT_DIR}{START_YEAR}-{END_YEAR}_graph_core_structure_plot_bipartite_{execution_timestamp}.png'
    OUTPUT_CSV = f'core_structure_analysis_bipartite_{execution_timestamp}_{START_YEAR}_{END_YEAR}.csv'
    OUTPUT_CSV_PATH = f'{CSV_DIR}{OUTPUT_CSV}'

    # Run the full pipeline
    run_core_analysis_loop(
        graphs_dir=GRAPH_DIR,
        start_year=START_YEAR,
        end_year=END_YEAR,
        run=RUN,
        output_csv_path=OUTPUT_CSV_PATH,
        output_plot_path=PLOT_FILE_PATH,
        verbose=VERBOSE,
        event_line_date=EVENTS,
        analysis_type='Bipartite',
        save_subgraph=SAVE_SUBGRAPHS
    )
