import pandas as pd
import networkx as nx
import os
import time
from datetime import datetime
import tools
import glob

"""
Create bipartite graph creation and analysis from a CSV file.
The bipartite graph connects users and pages based on revision history.
The edges are weighted by the number of revisions a user has made to a page.

Create Co-occurence graphs as projected graphs from bipartite graphs. 

Change Log
    2025-10-28: Initial creation date
    2025-11-07: Added functionality to build and save both bipartite and co-occurrence graphs.
    2025-11-08: Added multi-year processing capability.
    2025-12-16> Build co-occurence page-page graph function added.

"""


def create_b_graph(df, min_revisions=2):
    """
    Creates a bipartite graph from a CSV file containing wiki revision history.

    :param min_revisions: number of revisions to be included in the graph
    :param df: dataframe
    :return: A NetworkX graph object
    """
    # Initialize a bipartite graph
    graph = nx.Graph()

    # Aggregate the number of revisions (weights) between users and pages
    edge_weights = df.groupby(['page_id', 'title', 'user_id', 'user_name']).size().reset_index(name='weight')

    # Add nodes and edges to the graph
    for _, row in edge_weights.iterrows():
        page_id = row['page_id']
        if row['user_id'] == 0:
            user_id = row['user_name']
        else:
            user_id = row['user_id']

        weight = row['weight']

        # ignoring those with only 1 revision
        if weight >= min_revisions:
            # Add User Node
            if not graph.has_node(user_id):
                graph.add_node(user_id, type='user', bipartite=1)

            # Add Page Node
            if not graph.has_node(page_id):
                graph.add_node(page_id, type='page', bipartite=0)
            # Add Edge with weight
            graph.add_edge(user_id, page_id, weight=weight)

    #enrich_graph = tools.enrich_bipartite_graph(graph, page_category_map)
    return graph


def create_c_e_e_graph(b_graph):
    """
    Creates a co-occurrence network of authors based on shared pages.

    :param b_graph: Bipartite graph
    :return: A NetworkX co-ocurrence graph built as weighted projected graph from bipartite graph
    """

    # 2. Editor-to-Editor Projection: Builds a NEW graph (G_E_E)
    editor_nodes = {n for n, d in b_graph.nodes(data=True) if d["bipartite"] == 1 }
    g_e_e = nx.bipartite.weighted_projected_graph(b_graph, editor_nodes)

    return g_e_e


def create_c_p_p_graph(b_graph):
    """
    Creates a co-occurrence network of authors based on shared pages.

    :param b_graph: Bipartite graph
    :return: A NetworkX co-ocurrence graph built as weighted projected graph from bipartite graph
    """

    # 2. Editor-to-Editor Projection: Builds a NEW graph (G_E_E)
    page_nodes = {n for n, d in b_graph.nodes(data=True) if d["bipartite"] == 0}
    g_p_p = nx.bipartite.weighted_projected_graph(b_graph, page_nodes)

    return g_p_p

def build_graph(
        df,
        file_name,
        min_revisions,
        output_dir,
        bipartite_graph=True,
        co_occurrence=True,
        save_graphs=True,
        plot_graphs=True
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if bipartite_graph:
        print(f'Working on Bipartite graph...')
        # Build Bi-partite network (editor-article)
        b_graph = create_b_graph(df, min_revisions)

        # Print basic graph information
        num_user_nodes = sum(1 for _, d in b_graph.nodes(data=True) if d.get('type') == 'user')
        num_page_nodes = sum(1 for _, d in b_graph.nodes(data=True) if d.get('type') == 'page')

        print(f"Number of User nodes: {num_user_nodes}")
        print(f"Number of Page nodes: {num_page_nodes}")
        print(f"Number of edges: {b_graph.number_of_edges()}")

        # Plot the graph
        if plot_graphs:
            tools.plot_bipartite_graph(b_graph, "Title: Bipartite Graph of Editors and Articles")

        # Save the graph
        if save_graphs:
            save_graphs_path = os.path.join(output_dir, f'{file_name}_bipartite_{ts}_{min_revisions}.graphml')
            tools.save_graphs_to_file(b_graph, save_graphs_path)

    if co_occurrence:
        print(f'\nWorking on Co-occurrence editor-editor graph...')
        # Build Co-occurrence network (editor-editor)
        c_e_e_graph = create_c_e_e_graph(b_graph)

        # Print basic graph information
        print(f"co-ocurrence editor-editor Number of Nodes: {c_e_e_graph.number_of_nodes()}")
        print(f"co-ocurrence editor-editor Number of Edges: {c_e_e_graph.number_of_edges()}")

        # Build Co-occurrence page-page graph
        c_p_p_graph = create_c_p_p_graph(b_graph)

        # Print basic graph information
        print(f"co-ocurrence page-page Number of Nodes: {c_p_p_graph.number_of_nodes()}")
        print(f"co-ocurrence page-page Number of Edges: {c_p_p_graph.number_of_edges()}")

        # Print the top 5 edges with the highest weights
        #top_edges = sorted(c_e_e_graph.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:5]
        #print("Top 5 edges by weight:")
        #for edge in top_edges:
        #    print(edge)

        # Plot the co-occurrence network
        if plot_graphs:
            print("No plot functionality at the moment for co-occurrence graphs.")
            #tools.plot_co_occurrence_graph(c_e_e_graph, "Title: Co-occurrence Graph of Editors")

        # Save the graph
        if save_graphs:
            # Save co-occurence page-page graph
            save_graphs_path = os.path.join(output_dir, f'{file_name}_c_e_e_{ts}_{min_revisions}.graphml')
            tools.save_graphs_to_file(c_e_e_graph, save_graphs_path)
            # Save co-occurence page-page graph
            save_graphs_path = os.path.join(output_dir, f'{file_name}_c_p_p_{ts}_{min_revisions}.graphml')
            tools.save_graphs_to_file(c_p_p_graph, save_graphs_path)

    return


if __name__ == "__main__":
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    # If multi_year = True, we are expecting multiple files per year, hence the multi_file will be ignored.
    multi_year = True  # Set to True if processing multiple years at once

    multi_file = False  # Set to True if processing multiple files at once
    start_year = 2006  # Starting year if multi_year is True
    end_year = 2025  # Ending year if multi_year is True
    run_number = 14  # Run number for file naming
    ROOT_DIR = '//'
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'data/03_graph/')
    INPUT_DIR = 'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/Monthly_Splits/'

    min_revisions = 7  # Minimum number of revisions in an article to consider an edge
    bipartite_graph = True  # Flag to build bipartite graph
    co_occurrence = True  # Flag to build co-occurrence editor-editor graph
    plot_graphs = False  # Flag to plot graphs
    save_graphs = True  # Flag to save graphs

    if not multi_file and not multi_year:
        # Single file processing
        year = '2002'
        month = '02'
        file_name = f'run_{run_number}_wiki_{year}_month_{month}_clean'
        csv_file_path = os.path.join(INPUT_DIR, f'{year}', f'{file_name}.csv')
        df = pd.read_csv(csv_file_path)
        print(f'Processing file: {file_name}')

         # Build Graphs
        if bipartite_graph:
            build_graph(
                df,
                file_name,
                min_revisions,
                OUTPUT_DIR,
                bipartite_graph,
                co_occurrence,
                save_graphs,
                plot_graphs
            )

    if multi_year:
        # Multi-year processing
        search_path = os.path.join(INPUT_DIR, '*.csv')
        for year in range(start_year, end_year + 1):
            year_search_path = os.path.join(INPUT_DIR, f'{year}', f'run_{run_number}_wiki_{year}_month_*_clean.csv')
            all_files = glob.glob(year_search_path)
            for file_path in all_files:
                csv_file_path = os.path.join(ROOT_DIR, file_path)

                file_name = os.path.splitext(os.path.basename(file_path))[0]
                print(f'Processing file: {file_name}')

                df = pd.read_csv(csv_file_path)
                build_graph(
                    df,
                    file_name,
                    min_revisions,
                    OUTPUT_DIR,
                    bipartite_graph,
                    co_occurrence,
                    save_graphs,
                    plot_graphs
                )

    print(f"End time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
