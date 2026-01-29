import networkx as nx
import pandas as pd
import time
import os
import re
from typing import Dict, Any, Tuple, List, Optional
import glob


"""
NOT DONE YET.

Initial implementation of component analysis to obtain most active editors and most active pages being edited for the 
largest component in the monthly graphs.

Change Log:
    2025-12-02: Initial creation date
"""


def obtain_revision_data(b_graph: Optional[nx.Graph]) -> pd.DataFrame:
    """Obtains revision data from the bipartite graph.

    Args:
        b_graph (nx.Graph): Bipartite graph of users and pages.

    Returns:
        pd.DataFrame: DataFrame containing revision data.
    """
    if b_graph is None:
        return pd.DataFrame()

    user_nodes = {n for n, d in b_graph.nodes(data=True) if d.get('bipartite') == 1}
    page_nodes = set(b_graph) - user_nodes
    page_nodes_2 = {n for n, d in b_graph.nodes(data=True) if d.get('bipartite') == 0}

    revisions = []
    for u, v, data in b_graph.edges(data=True):
        weight = data.get('weight', 1)
        # Lógica para gestionar el órden en que nos devuelve u y v

        # Caso 1: u is user and v is page
        if u in user_nodes and v in page_nodes:
            editor_id = u
            page_id = v

        # Caso 2: v is user and u is page
        elif v in user_nodes and u in page_nodes:
            editor_id = v
            page_id = u

        revisions.append({
            'editor_id': editor_id,
            'page_id': page_id,
            'revisions': weight
        })

    df_revisions = pd.DataFrame(revisions)
    return df_revisions


def evaluate_largest_component(
        graphs_dir: str,
        start_year: int,
        end_year: int,
        run: str,
        verbose: bool = False
):
    """Evaluates the largest connected component of graphs in the specified directory over multiple years.

    Args:
        graphs_dir (str): Directory containing graph files.
        start_year (int): Starting year for analysis.
        end_year (int): Ending year for analysis.
        run (str): Run identifier for file naming.
        verbose (bool): If True, prints detailed logs.
    """

    def extract_year_month_from_filename(filename: str) -> Tuple[str, str]:
        pattern = r'_wiki_(\d{4})_month_(\d{2})_'
        match = re.search(pattern, filename)
        return match.groups() if match else ('N/A', 'N/A')

    for year in range(start_year, end_year + 1):
        year_str = str(year)

        c_graph_search_path = os.path.join(graphs_dir, f'run_{run}_wiki_{year_str}_month_*_clean_c_e_e*.graphml')
        all_files = glob.glob(c_graph_search_path)

        for i, filename in enumerate(all_files):
            file_basename = os.path.basename(filename)
            graph_year, graph_month = extract_year_month_from_filename(file_basename)

            # 1. Read co-occurerence graph
            try:
                c_graph = nx.read_graphml(filename)
                if verbose:
                    print(f"[{i+1}/{len(all_files)}] Loaded graph for {graph_year}-{graph_month} with {c_graph.number_of_nodes()} nodes and {c_graph.number_of_edges()} edges.")
            except Exception as e:
                print(f"Error reading graph file {filename}: {e}")
                continue

            #2. Read bipartite graph
            b_graph_path = filename.replace('_c_e_e', '_bipartite')
            b_graph = nx.read_graphml(b_graph_path) if os.path.exists(b_graph_path) else None

            df_revisions = obtain_revision_data(b_graph)


if __name__ == '__main__':

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")


    START_YEAR = 2012
    END_YEAR = 2014
    RUN = '14'

    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'
    CSV_INPUT_FILE_PATH = f'{CSV_DIR}graph_community_metrics.csv'

    #PLOT_FILE_PATH = f'{PLOT_DIR}{START_YEAR}-{END_YEAR}_graph_core_structure_plot_{execution_timestamp}.png'
    #OUTPUT_CSV = f'core_structure_analysis_{execution_timestamp}_{START_YEAR}_{END_YEAR}.csv'
    #OUTPUT_CSV_PATH = f'{CSV_DIR}{OUTPUT_CSV}'

    VERBOSE = False

    # Run the full pipeline
    evaluate_largest_component(
        graphs_dir=GRAPH_DIR,
        start_year= START_YEAR,
        end_year= END_YEAR,
        run= RUN,
        #output_csv_path=OUTPUT_CSV_PATH,
        #output_plot_path=PLOT_FILE_PATH,
        verbose = VERBOSE
    )

    print(f"\nEnd time [{time.strftime('%Y-%m-%d %H:%M:%S')}]")