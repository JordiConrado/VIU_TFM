import networkx as nx
import os
import time
import community as community_louvain
from typing import Dict, Any, Tuple, List
from collections import defaultdict
import glob
import re
import pandas as pd

import nestedness_calculator as nodf_calculator
from networkx.algorithms import bipartite

"""
Change Log
    2025-10-28: Initial creation date
    2025-11-05: Removed resolution parameter in the Louvain algorithm call for simplicity
    2025-11-10: Save metrics to a CSV file
    2025-11-26: Added Components calculation
    2025-12-16: Add page-page co-occurrence graph analysis. It creates two csv files, one for editor-editor, and
                one for page-page co-occurrence graphs.
    2025-12-22: Refactored bipartite graph analysis into its own function.
"""

"""
def calculate_nodf(G, user_nodes, page_nodes):
    # Create the bi-adjacency matrix (Users as rows, Pages as columns)
    # Rows/Columns sorted by degree (descending) as required by NODF definition
    user_nodes_sorted = sorted(user_nodes, key=lambda n: G.degree(n), reverse=True)
    page_nodes_sorted = sorted(page_nodes, key=lambda n: G.degree(n), reverse=True)

    # Map nodes to indices
    user_idx = {node: i for i, node in enumerate(user_nodes_sorted)}
    page_idx = {node: i for i, node in enumerate(page_nodes_sorted)}

    matrix = np.zeros((len(user_nodes), len(page_nodes)))
    for u, p in G.edges():
        if u in user_idx and p in page_idx:
            matrix[user_idx[u], page_idx[p]] = 1
        elif p in user_idx and u in page_idx:
            matrix[user_idx[p], page_idx[u]] = 1

    def calculate_n_paired(m):
        rows, cols = m.shape
        if rows < 2: return 0
        total_paired = 0
        for i in range(rows):
            for j in range(i + 1, rows):
                # Degree must be decreasing (N_paired = 0 if degree_i <= degree_j)
                deg_i = np.sum(m[i, :])
                deg_j = np.sum(m[j, :])
                if deg_i > deg_j and deg_j > 0:
                    overlap = np.sum(np.logical_and(m[i, :], m[j, :]))
                    total_paired += (overlap / deg_j) * 100
        return total_paired / (rows * (rows - 1) / 2)

    nodf_rows = calculate_n_paired(matrix)
    nodf_cols = calculate_n_paired(matrix.T)

    return (nodf_rows + nodf_cols) / 2
"""

"""
def get_nodes_and_edges(b_graph=None, c_graph=None):
    if bipartite_graph:
        print(f'Working on Bipartite graph...')

        # Print basic graph information
        num_user_nodes = sum(1 for _, d in b_graph.nodes(data=True) if d.get('type') == 'user')
        num_page_nodes = sum(1 for _, d in b_graph.nodes(data=True) if d.get('type') == 'page')

        print(f"Number of User nodes: {num_user_nodes}")
        print(f"Number of Page nodes: {num_page_nodes}")
        print(f"Number of edges: {b_graph.number_of_edges()}")

    if co_occurrence_graph:
        print(f'\nWorking on Co-occurrence graph...')

        # Print basic graph information
        print(f"Number of nodes: {c_graph.number_of_nodes()}")
        print(f"Number of edges: {c_graph.number_of_edges()}")

        # Print the top 5 edges with the highest weights
        top_edges = sorted(c_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
        print("Top 5 edges by weight:")
        for edge in top_edges:
            print(edge)
"""

"""
def print_component_nodes(G):

    # nx.connected_components(G) returns an iterator of sets, where each set is a component.
    component_iterator = nx.connected_components(G)

    print(f"--- Component Node Listing for Graph with {G.number_of_nodes()} Nodes ---")

    for i, component in enumerate(component_iterator):
        component_size = len(component)

        # Convert the set of nodes to a list for cleaner printing
        nodes_list = sorted(list(component))

        print(f"Component {i + 1}:")
        print(f"  Size: {component_size} nodes")
        print(f"  Nodes: {nodes_list}")
        print("-" * 20)
"""


def find_c_graph_stats(
        G_co_occurrence: nx.Graph, verbose) -> Tuple[Dict[Any, int], float, int, int, int, int, int]:
    """
    1- Detection of commponents using nx.connected_components. Filtering for size >= 2, and tracking the largest component
    2- Detection of communities using community_louvain.best_partition. Calculates modularity (Q), and it counts
       sub-groups (communities) within the graph.

    Args:
        G_co_occurrence: The one-mode weighted co-occurrence graph (c_graph).

    Returns:
        A tuple containing:
        1. The partition dictionary (node -> community_id).
        2. The modularity score (Q).
        3. The total number of nodes in the graph.
        4. The total number of edges in the graph.
        5. The number of components with size >= 2.
        6. The total number of nodes in those components.
        7. The size of the largest component.
    """
    # ---. PART 1. Basic Graph Metrics
    num_nodes = G_co_occurrence.number_of_nodes()
    num_edges = G_co_occurrence.number_of_edges()
    # components = nx.number_connected_components(G_co_occurrence)

    # --- PART 2: COMPONENT ANALYSIS ---
    # Components are "islands" - no edges exist between different components.

    # We are selecting the components to count only those with 2 or more nodes
    # on those components, we will get the number of nodes
    num_components = 0
    nodes_in_meaningful_components = 0
    largest_component_size = 0

    for component in nx.connected_components(G_co_occurrence):
        component_size = len(component)
        if component_size >= 2:
            num_components += 1
            nodes_in_meaningful_components += component_size
            if component_size > largest_component_size:
                largest_component_size = component_size

    if not num_nodes or not G_co_occurrence:
        print("Graph is empty. Cannot perform analysis.")
        return {}, 0.0, 0, 0, 0, 0, 0

    if verbose:
        print(f"\n--- Global Graph Metrics ---")
        print(f"Nodes: {num_nodes} | Edges: {num_edges}")
        print(f"Meaningful Components (Size >= 2): {num_components}")
        print(f"Largest Component Size: {largest_component_size}")

        # Top edges by weight
        top_edges = sorted(G_co_occurrence.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
        print("\nTop 5 edges by weight:")
        for u, v, d in top_edges:
            print(f"  {u} <--> {v} | Weight: {d.get('weight', 1):.2f}")

    # --- PART 3: COMMUNITY DETECTION (LOUVAIN) ---
    # Communities are clusters of nodes that are more densely connected
    # internally than with the rest of the graph.
    if verbose:
        print(f'\n--- Running Louvain Algorithm ---')

    # Detect communities using the Louvain method
    # partition is a dictionary where keys are node IDs and values are community IDs
    partition = community_louvain.best_partition(
        G_co_occurrence,
        weight='weight'
    )

    # Group nodes into sets by community ID for modularity calculation
    communities_as_sets = defaultdict(set)
    for node, community_id in partition.items():
        communities_as_sets[community_id].add(node)

    community_list = list(communities_as_sets.values())

    # Calculate Modularity Score (Q)
    # Q > 0.3 usually indicates a strong community structure
    if num_edges >0:
        Q = nx.community.quality.modularity(
            G_co_occurrence,
            community_list,
            weight='weight'
        )
    else:
        Q = 0.0

    if verbose:
        print(f"Modularity Score (Q): {Q:.4f}")
        print(f"Number of Communities Found: {len(community_list)}")

    return (
        partition,      # partition dictionary, with communities numbered from 0 to number of communities
        Q,              # modularity score
        num_nodes,      # nodes in the graph
        num_edges,      # edges in the graph
        num_components, # number of components with size >= 2
        nodes_in_meaningful_components, # total nodes in those components
        largest_component_size  # size of the largest component
    )


def find_b_graph_stats(G, verbose=False):
    """
    Analyzes a bipartite graph using existing node attributes:
    bipartite=1 for Users, bipartite=0 for Pages.
    """
    # 1. Extract sets based on your specific attribute values<
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    user_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    page_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]

    # 2. Component Analysis
    components = list(nx.connected_components(G))

    # Filter for components with size >= 2
    meaningful_components = [c for c in components if len(c) >= 2]
    num_components_min2 = len(meaningful_components)

    # Total nodes in meaningful components
    meaningful_nodes_list = [node for c in meaningful_components for node in c]
    nodes_in_meaningful_components = len(meaningful_nodes_list)

    # Largest component size
    largest_component_size = len(max(components, key=len)) if components else 0

    # 3. Obtain Nestedness (NODF) score from nestedness_calculator
    nodf_score = 0.0
    try:
        # Verify that we have enough nodes to calculate NODF
        if len(user_nodes) > 1 and len(page_nodes) > 1 and num_edges > 0:
            # Build adjacency matrix from the bipartite graph
            matrix_sparse = bipartite.biadjacency_matrix(G, user_nodes, page_nodes)
            adj_matrix = matrix_sparse.toarray()

            # Instanciate the calculator
            calculator = nodf_calculator.NestednessCalculator(adj_matrix)
            nodf_score = calculator.nodf(adj_matrix)
        else:
            if verbose: print("Graph is too small o is not a bipartite graph to calculate NODF")

    except Exception as e:
        nodf_score = 0.0
        if verbose:
            print(f"Error calculating NODF: {e}")

    counts_dict = {
        "users": len(user_nodes),
        "pages": len(page_nodes)
    }

    # Console Output
    if verbose:
        print(f"--- Bipartite Analysis (NODF) ---")
        print(f"Users: {len(user_nodes)} | Pages: {len(page_nodes)}")
        print(f"Edges: {num_edges}")
        print(f"Nestedness (NODF): {nodf_score:.4f}")
        print(f"Meaningful Components (>= 2): {num_components_min2}")
        print(f"Largest Component Size: {largest_component_size}")

    return (
        nodf_score,      # NODF score
        num_nodes,
        num_edges,
        num_components_min2,
        nodes_in_meaningful_components,
        largest_component_size,
        counts_dict
    )


"""
def plot_network_communities(G_co_occurrence: nx.Graph, partition: Dict[Any, int],
                             title: str = "Network Communities (Louvain)"):

    plt.figure(figsize=(10, 8))
    plt.title(title, fontsize=16)

    # 1. Calculate Layout (Fruchterman-Reingold is good for cluster visualization)

    # --- CALCULATE EXPANDED K VALUE ---
    N = len(G_co_occurrence.nodes)
    # Base optimal distance is usually sqrt(Area / N). We multiply by a factor (e.g., 5.0)
    # to significantly increase node repulsion and spread out the dense core.
    repulsion_factor = 8.0
    k_value = np.sqrt(1.0 / N) * repulsion_factor if N > 0 else 0.5

    pos = nx.spring_layout(
        G_co_occurrence,
        iterations=50,
        seed=42,
        k=k_value  # Use the calculated, expanded k value
    )
    # --- END K VALUE CALCULATION ---

    # 2. Get only meaningful community IDs (>= 2 nodes)
    community_sizes = defaultdict(int)
    for comm_id in partition.values():
        community_sizes[comm_id] += 1

    # FIX: Define num_total_communities here for use in the legend logic
    num_total_communities = len(community_sizes)

    # Get the set of IDs that represent groups of 2 or more nodes
    meaningful_community_ids = {
        comm_id for comm_id, size in community_sizes.items() if size >= 2
    }

    # Calculate the number of unique, meaningful communities
    num_meaningful_communities = len(meaningful_community_ids)

    # 3. Use ListedColormap to discretize the continuous map
    # We only need colors for the meaningful communities + 1 gray color for singletons

    # Create a mapping from Louvain ID to a standardized color index (0 to N-1)
    # This ensures color consistency, ignoring singletons
    id_to_color_index = {
        comm_id: i for i, comm_id in enumerate(sorted(list(meaningful_community_ids)))
    }

    # Get the continuous 'hsv' colormap
    base_cmap = mpl.colormaps.get_cmap('hsv')

    # Create the color list: All meaningful colors + one reserved gray for singletons
    meaningful_colors = base_cmap(np.linspace(0, 1, num_meaningful_communities))
    singleton_color = np.array([0.7, 0.7, 0.7, 1.0])  # Light gray for singletons

    # Combine the meaningful colors with the gray color
    color_list = np.vstack([meaningful_colors, singleton_color])
    cmap = mpl.colors.ListedColormap(color_list)

    # 4. Map partition IDs to the new discrete color index (plus the gray index)
    node_color_indices = []
    gray_index = num_meaningful_communities  # The index for the gray color

    for comm_id in partition.values():
        if comm_id in meaningful_community_ids:
            # Assign the standardized color index
            node_color_indices.append(id_to_color_index[comm_id])
        else:
            # Assign the gray index for single-node communities
            node_color_indices.append(gray_index)

    # 5. Draw Nodes (Colored by Community)
    nx.draw_networkx_nodes(
        G_co_occurrence,
        pos,
        node_color=node_color_indices,  # Use the new index list
        cmap=cmap,
        node_size=80,
        linewidths=0.5,
        edgecolors='black'
    )

    # 6. Draw Edges (Weighting thickness by co-occurrence count)
    # edge_weights = [d['weight'] * 0.1 for u, v, d in G_co_occurrence.edges(data=True)]

    nx.draw_networkx_edges(
        G_co_occurrence,
        pos,
        alpha=0.4,
        # width=edge_weights,
        width=1.0,
        edge_color='gray'
    )

    # 7. Draw Labels (Only for top 10 nodes to reduce clutter)
    node_degrees = dict(G_co_occurrence.degree())
    sorted_degrees = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)
    top_nodes = dict(sorted_degrees[:min(10, len(G_co_occurrence.nodes))])

    nx.draw_networkx_labels(
        G_co_occurrence,
        pos,
        labels=top_nodes,
        font_size=10,
        font_color='black'
    )

    # 8. Create a Legend (Only for meaningful communities)
    legend_elements = [
        plt.scatter([], [], color=cmap(id_to_color_index[comm_id]), label=f'Community {comm_id}', s=100)
        for comm_id in sorted(list(meaningful_community_ids))
    ]

    # Add a legend entry for the filtered singletons
    if num_total_communities != num_meaningful_communities:
        legend_elements.append(
            plt.scatter([], [], color=cmap(gray_index), label='Single-Node (Filtered)', s=100)
        )

    # Reduce clutter if too many communities (including the singleton entry)
    if len(legend_elements) <= 10:
        plt.legend(handles=legend_elements, title="Communities", loc='best')

    plt.box(False)
    plt.tight_layout()
    plt.show()
"""
"""
def plot_top_n_communities(G_co_occurrence: nx.Graph, partition: Dict[Any, int], n_top: int = 3):


    # 1. Group nodes and calculate sizes
    communities_as_sets = defaultdict(set)
    for node, community_id in partition.items():
        communities_as_sets[community_id].add(node)

    # 2. Filter for meaningful communities (size >= 2) and sort by size
    meaningful_communities = {
        comm_id: nodes for comm_id, nodes in communities_as_sets.items() if len(nodes) >= 2
    }

    # Sort the meaningful communities by size (descending)
    sorted_community_ids = sorted(
        meaningful_communities.keys(),
        key=lambda comm_id: len(meaningful_communities[comm_id]),
        reverse=True
    )

    # Get the top N IDs
    top_n_community_ids = sorted_community_ids[:n_top]

    if not top_n_community_ids:
        print(f"No communities with 2 or more nodes found to plot individually.")
        return

    # 3. Prepare a distinct colormap for the individual plots
    # Using a high-contrast colormap like 'Set1' for clarity
    base_cmap = mpl.colormaps.get_cmap('Set1')

    print(f"\n--- Generating individual plots for Top {len(top_n_community_ids)} Communities ---")

    # 4. Loop and plot each top community
    for i, comm_id in enumerate(top_n_community_ids):
        node_set = meaningful_communities[comm_id]

        # Create a subgraph containing only the nodes of this community and their internal edges
        subgraph = G_co_occurrence.subgraph(node_set)

        # Get edges weights and node degrees for the subgraph
        edge_weights = [d['weight'] * 0.1 for u, v, d in subgraph.edges(data=True)]
        node_degrees = dict(subgraph.degree())
        # Node size scaled by degree within the community
        node_sizes = [node_degrees[n] * 50 for n in subgraph.nodes]

        # Determine a color for this specific community
        community_color = base_cmap(i % base_cmap.N)

        # Set figure size for the individual plot
        plt.figure(figsize=(10, 8))

        # Calculate layout for the SUBGRAPH
        N_sub = len(subgraph.nodes)
        repulsion_factor = 2.0  # Adjust repulsion for smaller, focused subgraphs
        k_value = np.sqrt(1.0 / N_sub) * repulsion_factor if N_sub > 0 else 0.5

        pos = nx.spring_layout(
            subgraph,
            iterations=50,
            seed=42,
            k=k_value
        )

        title = f"Community {comm_id} (Size: {N_sub} nodes) - Rank {i + 1}"
        plt.title(title, fontsize=14)

        # Draw Edges
        nx.draw_networkx_edges(
            subgraph,
            pos,
            alpha=0.6,
            width=edge_weights,
            edge_color='black'
        )

        # Draw Nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=[community_color] * N_sub,  # Uniform color for the whole community
            node_size=node_sizes,
            linewidths=0.5,
            edgecolors='black'
        )

        # Draw Labels (for ALL nodes in the smaller subgraph)
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels={n: n for n in subgraph.nodes},
            font_size=8,
            font_color='black'
        )

        plt.box(False)
        plt.tight_layout()
        plt.show()

    print("--- Finished generating individual community plots. ---")

"""

"""   
def analyze_community_results(partition: Dict[Any, int], modularity_score: float):

    # Group nodes by community ID
    communities_as_sets = defaultdict(set)
    for node, community_id in partition.items():
        communities_as_sets[community_id].add(node)

    # Filter: Keep only communities with 2 or more nodes
    meaningful_communities = {
        comm_id: nodes for comm_id, nodes in communities_as_sets.items() if len(nodes) >= 2
    }

    num_total_communities = len(communities_as_sets)
    num_filtered_communities = len(meaningful_communities)

    print("\n--- COMMUNITY ANALYSIS SUMMARY (Filtered) ---")
    print(f"Total Modularity Score (Q): {modularity_score:.4f}")
    print(f"Total Communities Found (Initial): {num_total_communities}")
    print(f"Meaningful Communities (>= 2 nodes): {num_filtered_communities}")
    print(f"Single-Node Communities Filtered: {num_total_communities - num_filtered_communities}")
    print("-" * 45)

    # Sort the meaningful communities by size (descending) for reporting
    sorted_communities = sorted(
        [(comm_id, len(nodes)) for comm_id, nodes in meaningful_communities.items()],
        key=lambda item: item[1],
        reverse=True
    )

    # Print size distribution
    print("Community Size Distribution (Top 10 Largest):")
    if not sorted_communities:
        print("  No meaningful communities found (all had 1 node).")
    else:
        for i, (comm_id, size) in enumerate(sorted_communities[:10]):
            print(f"  Community {comm_id}: {size} nodes")

    print("-" * 45)
    print("\n--- COMMUNITY MEMBER LISTS (Top 3 Largest) ---")

    # Print the nodes for the top 3 largest communities
    for i, (comm_id, size) in enumerate(sorted_communities[:3]):
        nodes = meaningful_communities[comm_id]
        print(f"Community {comm_id} ({size} nodes):")
        # Use simple string representation for printing
        print(f"  Members: {', '.join(map(str, nodes))}")
        print("")

    if len(sorted_communities) > 3:
        print(f"... and {len(sorted_communities) - 3} more communities (use the 'partition' dictionary for all).")
"""

""""""
def extract_year_month_from_filename(filename: str) -> Tuple[str, str]:
    """
    Extracts the year and month from a filename formatted as:
    'run_<run>_wiki_<year>_month_<month>_clean_c_e_e_*.graphml'

    Args:
        filename: The filename string.

    Returns:
        A tuple containing the year and month as strings.
    """
    pattern = r'run_\d+_wiki_(\d{4})_month_(\d{2})_'
    match = re.search(pattern, filename)
    if match:
        year, month = match.groups()
        return year, month
    else:
        return 'N/A', 'N/A'


def save_metrics_to_csv(all_metrics: List[Dict[str, Any]], output_path: str):
    """
    Converts the collected list of dictionaries into a DataFrame and saves it to CSV.
    """
    if not all_metrics:
        print("\n[WARNING] No metrics collected. Skipping CSV write.")
        return

    try:
        df = pd.DataFrame(all_metrics)

        # Check if the file already exists
        file_exists = os.path.exists(output_path)

        # Write to CSV in append mode ('a')
        df.to_csv(
            output_path,
            mode='w',    #'a',
            index=False,
            # Only write the header if the file does NOT exist
            header=not file_exists
        )

        # Confirmation message change based on action
        action = "appended to" if file_exists else "saved to"
        print(f"\n[SUCCESS] All metrics {action}: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to save/append metrics to CSV: {e}")


if __name__ == "__main__":

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    verbose = False
    run = '14'
    start_year = 2001
    end_year = 2025
    PLOT_NETWORKS = False  # Flag to plot graphs

    # File names and paths
    #b_graph_filename = 'run_14_wiki_2002_month_02_clean_bipartite_20251108_104216_3.graphml'
    #c_graph_filename = 'run_14_wiki_2002_month_02_clean_c_e_e_20251108_104216_3.graphml'
    ROOT_DIR = '//'

    OUTPUT_E_E_CSV_FILE = os.path.join(ROOT_DIR, f'data/02_csv/graph_community_metrics_e_e.csv')
    OUTPUT_P_P_CSV_FILE = os.path.join(ROOT_DIR, f'data/02_csv/graph_community_metrics_p_p.csv')
    OUTPUT_BIPARTITE_CSV_FILE = os.path.join(ROOT_DIR, f'data/02_csv/graph_community_metrics_bipartite.csv')

    BIPARTITE_GRAPH = True  # Flag to build bipartite graph
    CO_OCCURRENCE_GRAPH = True  # Flag to build co-occurrence graph
    graph_execution_date = '20251216'

    graphs_dir = os.path.join(ROOT_DIR, 'data/03_graph/')

    # SECTION 1: Analyze editor-editor co-occurrence graphs
    if CO_OCCURRENCE_GRAPH:
        all_metrics = []  # List with all metrics to save

        for year in range(start_year, end_year + 1):
            year = str(year)
            c_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year}_month_*_clean_c_e_e_{graph_execution_date}_*.graphml')
            all_files = glob.glob(c_graph_search_path)

            for i, filename in enumerate(all_files):
                print(f"\n--- Starting processing file {i + 1}/{len(all_files)}: {os.path.basename(filename)} ---")
                file_basename = os.path.basename(filename)
                graph_year, graph_month = extract_year_month_from_filename(file_basename)
                c_graph_path = filename
                # Read c0-occurrence graph file
                c_graph = nx.read_graphml(c_graph_path)

                (partition,
                 modularity_score,
                 num_nodes,
                 num_edges,
                 components,
                 nodes_in_components,
                 larges_component_size) = find_c_graph_stats(c_graph, verbose)

                print(f'Modularity Score: {modularity_score:.5f}')
                print('Num Nodes:', num_nodes)
                print('Num Edges:', num_edges)
                print('Num Components:', components)
                print('Nodes in Components (size >=2):', nodes_in_components)
                print('Largest Component Size:', larges_component_size)

                # Store metrics in dictionary
                metrics_dict = {
                    'timestamp': execution_timestamp,
                    'run': run,
                    'year': graph_year,
                    'month': graph_month,
                    'modularity_score': modularity_score,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'num_components': components,
                    'nodes_in_components': nodes_in_components,
                    'largest_component_size': larges_component_size
                }

                # add the dictionary to the collector list
                all_metrics.append(metrics_dict)

        save_metrics_to_csv(all_metrics, OUTPUT_E_E_CSV_FILE)

    # SECTION 2: Analyze page-page co-occurrence graphs
    if CO_OCCURRENCE_GRAPH:
        all_metrics = []  # List with all metrics to save

        for year in range(start_year, end_year + 1):
            year = str(year)
            c_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year}_month_*_clean_c_p_p_{graph_execution_date}_*.graphml')
            all_files = glob.glob(c_graph_search_path)

            for i, filename in enumerate(all_files):
                print(f"\n--- Starting processing file {i + 1}/{len(all_files)}: {os.path.basename(filename)} ---")
                file_basename = os.path.basename(filename)
                graph_year, graph_month = extract_year_month_from_filename(file_basename)
                c_graph_path = filename
                # Read c0-occurrence graph file
                c_graph = nx.read_graphml(c_graph_path)

                (partition,
                 modularity_score,
                 num_nodes,
                 num_edges,
                 components,
                 nodes_in_components,
                 larges_component_size) = find_c_graph_stats(c_graph, verbose)

                print(f'Modularity Score: {modularity_score:.5f}')
                print('Num Nodes:', num_nodes)
                print('Num Edges:', num_edges)
                print('Num Components:', components)
                print('Nodes in Components (size >=2):', nodes_in_components)
                print('Largest Component Size:', larges_component_size)

                # Store metrics in dictionary
                metrics_dict = {
                    'timestamp': execution_timestamp,
                    'run': run,
                    'year': graph_year,
                    'month': graph_month,
                    'modularity_score': modularity_score,
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'num_components': components,
                    'nodes_in_components': nodes_in_components,
                    'largest_component_size': larges_component_size
                }

                # add the dictionary to the collector list
                all_metrics.append(metrics_dict)

        save_metrics_to_csv(all_metrics, OUTPUT_P_P_CSV_FILE)

    # SECTION 3: Analyze bipartite graphs
    if BIPARTITE_GRAPH:
        all_metrics = []  # List with all metrics to save

        for year in range(start_year, end_year + 1):
            year = str(year)
            b_graph_search_path = os.path.join(graphs_dir,
                                               f'run_{run}_wiki_{year}_month_*_clean_bipartite_{graph_execution_date}_*.graphml')
            all_files = glob.glob(b_graph_search_path)

            for i, filename in enumerate(all_files):
                print(f"\n--- Starting processing file {i + 1}/{len(all_files)}: {os.path.basename(filename)} ---")
                file_basename = os.path.basename(filename)
                graph_year, graph_month = extract_year_month_from_filename(file_basename)
                b_graph_path = filename

                # Read bipartite graph file
                b_graph = nx.read_graphml(b_graph_path)

                (nestedness_score,
                 num_nodes,
                 num_edges,
                 components,
                 nodes_in_components,
                 largest_component_size,
                 node_counts) = find_b_graph_stats(b_graph, verbose)

                # Store metrics in dictionary
                metrics_dict = {
                    'timestamp': execution_timestamp,
                    'run': run,
                    'year': graph_year,
                    'month': graph_month,
                    'nestedness_score': nestedness_score,
                    'num_users': node_counts["users"],
                    'num_pages': node_counts["pages"],
                    'num_nodes': num_nodes,
                    'num_edges': num_edges,
                    'num_components': components,
                    'nodes_in_components': nodes_in_components,
                    'largest_component_size': largest_component_size
                }

                # add the dictionary to the collector list
                all_metrics.append(metrics_dict)

        save_metrics_to_csv(all_metrics, OUTPUT_BIPARTITE_CSV_FILE)

    print(f"\nEnd time [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
