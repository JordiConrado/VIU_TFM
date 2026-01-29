import networkx as nx
import os
import time
from typing import Dict, Any, List, Tuple
import glob
import re
import pandas as pd
import datetime

"""
Reads monthly co-occurrence nad bipartite graph files, iterates through all connected components,
and extracts the size of each component along with the month and year, saving 
the result to a detailed CSV file.

The output CSV will serve as the input for the plot_component_node_counts function.
"""

"""
Change Log:
    2025-11-25: Initial creation of the script to extract component details from co-occurrence graphs.
    2025-12-16: Add page-page co-occurrence graph component extraction. It creates two files, one for e_e
                    and another for p_p.
    2025-12-22: Add bipartite graph component extraction.    
"""


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


def get_component_details(
        G_co_occurrence: nx.Graph,
        run: str,
        year: str,
        month: str,
        min_component_size_to_save: int = 2
) -> List[Dict[str, Any]]:
    """
    Iterates through all connected components in a graph and extracts details
    (size, year, month, unique ID) for each component larger than a minimum size.

    Args:
        G_co_occurrence: The one-mode weighted co-occurrence graph.
        run (str): The run identifier.
        year (str): The year of the graph data.
        month (str): The month of the graph data.
        min_component_size_to_save (int): Minimum size of a component to save.

    Returns:
        A list of dictionaries, where each dictionary represents one component.
    """

    component_details = []

    # nx.connected_components returns an iterator of sets, one set per component
    for i, component_nodes in enumerate(nx.connected_components(G_co_occurrence)):
        component_size = len(component_nodes)

        if component_size >= min_component_size_to_save:
            # We create a unique ID for this instance of the component for potential tracking
            # Note: This ID is only unique for the component within the month, not across time.
            # We will use 'year', 'month', and 'component_size' for plotting later.
            component_details.append({
                'run': run,
                'year': year,
                'month': month,
                # Simple sequential ID for this component within the current month/graph
                'component_instance_id': i + 1,
                'component_size': component_size,
            })

    return component_details


def save_details_to_csv(details: List[Dict[str, Any]], output_path: str):
    """
    Converts the collected list of component detail dictionaries into a DataFrame and saves it to CSV.
    """
    if not details:
        print("\n[WARNING] No component details collected. Skipping CSV write.")
        return

    try:
        df = pd.DataFrame(details)
        # Convert year/month to integers for proper sorting if needed later
        df['year'] = pd.to_numeric(df['year'])
        df['month'] = pd.to_numeric(df['month'])

        # Check if the file already exists
        file_exists = os.path.exists(output_path)

        # Write to CSV in append mode ('a')
        df.to_csv(
            output_path,
            mode='a',
            index=False,
            # Only write the header if the file does NOT exist
            header=not file_exists
        )

        action = "appended to" if file_exists else "saved to"
        print(f"\n[SUCCESS] Component details {action}: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to save/append component details to CSV: {e}")


def get_bipartite_component_details(
        G: nx.Graph,
        run: str,
        year: str,
        month: str,
        min_component_size_to_save: int = 2
) -> List[Dict[str, Any]]:
    """
    Extracts details for connected components in a bipartite graph by
    inspecting the 'bipartite' node attribute.

    Args:
        G: The bipartite networkx Graph where nodes have a 'bipartite' attribute.
        run (str): The run identifier.
        year (str): The year of the graph data.
        month (str): The month of the graph data.
        min_component_size_to_save: Minimum total nodes in a component to save.

    Returns:
        A list of dictionaries with metadata and partition counts.
    """
    component_details = []

    # Extract the bipartite attributes as a dictionary: {node_id: 0 or 1}
    # This assumes all nodes have the attribute set.
    node_types = nx.get_node_attributes(G, 'bipartite')

    for i, component_nodes in enumerate(nx.connected_components(G)):
        component_size = len(component_nodes)

        if component_size >= min_component_size_to_save:
            # Count users (bipartite=1) vs pages (bipartite=0)
            # We look up each node in the node_types dictionary we extracted above
            num_users = sum(1 for n in component_nodes if node_types.get(n) == 1)
            num_pages = component_size - num_users

            component_details.append({
                'run': run,
                'year': year,
                'month': month,
                'component_instance_id': i + 1,
                'component_size': component_size,
                'user_count': num_users,
                'page_count': num_pages
            })

    return component_details


if __name__ == "__main__":

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    # --- Configuration ---
    run = '14'
    start_year = 2012  # 2002
    end_year = 2014  # 2025

    # Set the minimum size of a component to be saved in the detail CSV.
    # Set this to 1 if you want to include singletons.
    MIN_COMPONENT_SIZE_TO_SAVE = 2

    # File paths
    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/'
    graphs_dir = os.path.join(ROOT_DIR, 'data/03_graph/')
    # This is the new output file that will feed the plotting function
    OUTPUT_E_E_CSV_FILE = os.path.join(ROOT_DIR, 'data/02_csv/component_details_e_e.csv')
    OUTPUT_P_P_CSV_FILE = os.path.join(ROOT_DIR, 'data/02_csv/component_details_p_p.csv')
    OUTPUT_BIPARTITE_CSV_FILE = os.path.join(ROOT_DIR, 'data/02_csv/component_details_bipartite.csv')

    graph_execution_date = '20251216'

    # Ensure the directory for the output CSV exists
    os.makedirs(os.path.dirname(OUTPUT_E_E_CSV_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_P_P_CSV_FILE), exist_ok=True)

    # Delete the output file if it exists to ensure a clean run
    for file in [OUTPUT_E_E_CSV_FILE, OUTPUT_P_P_CSV_FILE, OUTPUT_BIPARTITE_CSV_FILE]:
        if os.path.exists(file):
            os.remove(file)
        print(f"Existing file deleted: {file}")

    # 1. editor-editor co-occurrence graph component details
    print(f"\n--- Processing editor-editor co-occurrence: ---")
    all_component_details = []

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        # Search pattern for co-occurrence graph files
        c_graph_search_path = os.path.join(graphs_dir,
                                           f'run_{run}_wiki_{year_str}_month_*_clean_c_e_e_{graph_execution_date}_*.graphml')
        all_files = glob.glob(c_graph_search_path)

        if not all_files:
            print(f"No graph files found for year {year_str}.")
            continue

        print(f"\n--- Processing {len(all_files)} files for year {year_str} ---")

        for i, filename in enumerate(all_files):
            file_basename = os.path.basename(filename)
            graph_year, graph_month = extract_year_month_from_filename(file_basename)

            # Read co-occurrence graph file
            try:
                c_graph = nx.read_graphml(filename)
            except Exception as e:
                print(f"Error reading {file_basename}: {e}. Skipping.")
                continue

            print(f"  -> Analyzing {file_basename} (Nodes: {c_graph.number_of_nodes()})")

            # Extract details for all components
            details = get_component_details(
                c_graph,
                run,
                graph_year,
                graph_month,
                MIN_COMPONENT_SIZE_TO_SAVE
            )

            # Add to the master list
            all_component_details.extend(details)

    # --- Final Save ---
    save_details_to_csv(all_component_details, OUTPUT_E_E_CSV_FILE)

    # 2. page-page co-occurrence graph component details
    print(f"\n--- Processing page-page co-occurrence ---")
    all_component_details = []

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        # Search pattern for co-occurrence graph files
        c_graph_search_path = os.path.join(graphs_dir,
                                           f'run_{run}_wiki_{year_str}_month_*_clean_c_p_p_{graph_execution_date}_*.graphml')
        all_files = glob.glob(c_graph_search_path)

        if not all_files:
            print(f"No graph files found for year {year_str}.")
            continue

        print(f"\n--- Processing {len(all_files)} files for year {year_str} ---")

        for i, filename in enumerate(all_files):
            file_basename = os.path.basename(filename)
            graph_year, graph_month = extract_year_month_from_filename(file_basename)

            # Read co-occurrence graph file
            try:
                c_graph = nx.read_graphml(filename)
            except Exception as e:
                print(f"Error reading {file_basename}: {e}. Skipping.")
                continue

            print(f"  -> Analyzing {file_basename} (Nodes: {c_graph.number_of_nodes()})")

            # Extract details for all components
            details = get_component_details(
                c_graph,
                run,
                graph_year,
                graph_month,
                MIN_COMPONENT_SIZE_TO_SAVE
            )

            # Add to the master list
            all_component_details.extend(details)

    # --- Final Save ---
    save_details_to_csv(all_component_details, OUTPUT_P_P_CSV_FILE)

    # 3. bipartite graph component details
    print(f"\n--- Processing bipartite ---")
    all_component_details = []

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        # Search pattern for co-occurrence graph files
        b_graph_search_path = os.path.join(graphs_dir,
                                           f'run_{run}_wiki_{year_str}_month_*_clean_bipartite_{graph_execution_date}_*.graphml')
        all_files = glob.glob(b_graph_search_path)

        if not all_files:
            print(f"No graph files found for year {year_str}.")
            continue

        print(f"\n--- Processing {len(all_files)} files for year {year_str} ---")

        for i, filename in enumerate(all_files):
            file_basename = os.path.basename(filename)
            graph_year, graph_month = extract_year_month_from_filename(file_basename)

            # Read co-occurrence graph file
            try:
                b_graph = nx.read_graphml(filename)
            except Exception as e:
                print(f"Error reading {file_basename}: {e}. Skipping.")
                continue

            print(f"  -> Analyzing {file_basename} (Nodes: {b_graph.number_of_nodes()})")

            # Extract details for all components
            details = get_bipartite_component_details(
                b_graph,
                run,
                graph_year,
                graph_month,
                MIN_COMPONENT_SIZE_TO_SAVE
            )

            # Add to the master list
            all_component_details.extend(details)

    # --- Final Save ---
    save_details_to_csv(all_component_details, OUTPUT_BIPARTITE_CSV_FILE)

    print(f"\nEnd time [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
