import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


def save_graphs_to_file(graph, graph_path):
    """Saves the graph using the GraphML format."""

    print("\n--- Saving Graphs ---")
    nx.write_graphml(graph, graph_path)
    print(f"Graph {graph} saved to: {graph_path}")


def plot_bipartite_graph(G_bipartite: nx.Graph, title: str):
    """
    Plots the bipartite network with distinct colors and a layout
    that clearly separates the two partitions (Articles and Editors).

    Args:
        G_bipartite: The weighted bipartite graph (must have 'bipartite' node attributes).
        title: The title for the plot.
    """
    # 1. Identify the two sets of nodes (partition 0: Articles, partition 1: Editors)
    articles = {n for n, d in G_bipartite.nodes(data=True) if d.get("bipartite") == 0}
    editors = {n for n, d in G_bipartite.nodes(data=True) if d.get("bipartite") == 1}

    if not articles or not editors:
        print("Error: Bipartite graph is empty or node attributes are missing.")
        return

    # 2. Get the layout that separates the two partitions cleanly
    pos = nx.bipartite_layout(G_bipartite, articles)

    # 3. Define node colors and sizes
    node_colors = []
    for node in G_bipartite.nodes():
        if node in articles:
            node_colors.append('skyblue')  # Articles
        else:
            node_colors.append('salmon')  # Editors

    # Scale node size by degree for general visibility
    node_sizes = [G_bipartite.degree(n) * 100 for n in G_bipartite.nodes()]

    # 4. Extract edge weights and map them to line widths
    edge_weights = [d['weight'] for u, v, d in G_bipartite.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 3 + 0.5 for w in edge_weights]  # Scale thickness

    # 5. Plot the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G_bipartite,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=False,  # Labels clutter small graphs, better to skip
        edge_color='gray',
        alpha=0.8
    )

    # 6. Add title and legend (using two dummy points for the legend)
    plt.title(title, fontsize=16)
    plt.scatter([], [], color='skyblue', label='Articles (Partition 0)')
    plt.scatter([], [], color='salmon', label='Editors (Partition 1)')
    plt.legend(scatterpoints=1, frameon=False)
    plt.show()


def plot_co_occurrence_graph(G_co_occurrence: nx.Graph, title: str):
    """
    Plots the co-occurrence network, sizing nodes by degree and
    weighting edges by co-occurrence strength.

    Args:
        G_co_occurrence: The weighted one-mode graph (e.g., Editor-Editor).
        title: The title for the plot.
    """
    if not G_co_occurrence:
        print("Error: Co-occurrence graph is empty.")
        return

    # 1. Use the Spring Layout (standard for visualizing communities)
    pos = nx.spring_layout(G_co_occurrence, k=0.3, iterations=50)

    # 2. Scale Node Size by Degree (how active/connected the node is)
    degrees = dict(G_co_occurrence.degree(weight='weight'))
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [degrees.get(n, 0) / max_degree * 800 + 100 for n in G_co_occurrence.nodes()]

    # 3. Scale Edge Width by Weight (how strong the co-occurrence is)
    edge_weights = [d['weight'] for u, v, d in G_co_occurrence.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [w / max_weight * 5 + 0.5 for w in edge_weights]  # Scale thickness

    # 4. Plot the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        G_co_occurrence,
        pos,
        node_size=node_sizes,
        width=edge_widths,
        with_labels=True,
        font_size=8,
        node_color='lightgreen',
        edge_color='darkgray',
        alpha=0.7
    )
    plt.title(title, fontsize=16)
    plt.show()


def plot_wiki_metrics_from_dict(metrics_dict):
    """
    Generates a multi-panel plot of monthly Wikipedia metrics from a dictionary.

    The dictionary must contain lists/arrays for:
    'Year', 'month', 'total_users', 'IP_users', 'BOT_users',
    'revised_pages', and 'monthly_revisions'.

    Args:
        metrics_dict (dict): A dictionary containing the monthly metrics data.
    """
    # --- 1. Convert Dictionary to DataFrame and Prepare Data ---
    try:
        df = pd.DataFrame(metrics_dict)
    except Exception as e:
        print(f"An error occurred while converting the dictionary to a DataFrame: {e}")
        return

    # Create a proper datetime index for plotting
    # We assign day=1 to combine Year and month into a monthly date
    df['Date'] = pd.to_datetime(df[['Year', 'month']].assign(day=1))
    df = df.set_index('Date').sort_index()

    # --- 2. Create the Multi-Panel Plot ---
    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(12, 10),  # Adjust size to your preference
        sharex=True
    )
    plt.subplots_adjust(hspace=0.1)  # Reduce space between subplots

    # Define colors for better contrast
    colors = {
        'total_users': '#5D2E8C',  # Dark Purple
        'IP_users': '#FF6961',  # Light Red/Coral
        'BOT_users': '#77DD77',  # Pastel Green
        'revised_pages': '#4682B4',  # Steel Blue
        'monthly_revisions': '#FFA07A'  # Light Salmon
    }

    # --- 3. Plotting Each Panel ---

    # Panel 1: Total Users (Similar to N. Users in the example image)
    axes[0].plot(df.index, df['total_users'], label='Total Users', color=colors['total_users'])
    axes[0].set_title('Monthly Wikipedia Metrics Over Time', loc='left', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', linestyle='--')

    # Panel 2: Total Revisions (Similar to N. Hashtags in the example image)
    axes[1].plot(df.index, df['monthly_revisions'], label='Monthly Revisions', color=colors['monthly_revisions'])
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', linestyle='--')

    # Panel 3: IP vs. BOT Users (Similar to N. Unique users/hashtags)
    axes[2].plot(df.index, df['IP_users'], label='IP Users', color=colors['IP_users'])
    axes[2].plot(df.index, df['BOT_users'], label='BOT Users', color=colors['BOT_users'])
    axes[2].legend(loc='upper right')
    axes[2].grid(axis='y', linestyle='--')

    # Panel 4: Revised Pages (One metric as a standalone plot)
    axes[3].plot(df.index, df['revised_pages'], label='Revised Pages', color=colors['revised_pages'])
    axes[3].legend(loc='upper right')
    axes[3].grid(axis='y', linestyle='--')

    # Panel 5: Derived Metrics (As a proxy for the 'Entropy' plots)
    # Revisions_per_User and Revisions_per_Page show activity intensity
    df['Revisions_per_User'] = df['monthly_revisions'] / df['total_users']
    df['Revisions_per_Page'] = df['monthly_revisions'] / df['revised_pages']

    axes[4].plot(df.index, df['Revisions_per_User'], label='Revisions per User', color='red')
    axes[4].plot(df.index, df['Revisions_per_Page'], label='Revisions per Page', color='purple')
    axes[4].legend(loc='upper right')
    axes[4].set_ylabel('Derived Activity Metrics')
    axes[4].grid(axis='y', linestyle='--')

    # --- 4. Final Formatting ---

    # Configure the X-axis for all panels
    axes[-1].set_xlabel('Date (Year-Month)')

    # Format the x-axis to show monthly labels clearly
    date_form = mdates.DateFormatter("%Y-%m")
    axes[-1].xaxis.set_major_formatter(date_form)
    # Rotate date labels for better fit
    plt.xticks(rotation=45, ha='right')

    # Save the figure instead of showing it directly in a plotting file context
    plt.savefig('wiki_metrics_plot.png')
    plt.close(fig)  # Close the figure to free up memory

    print("Plot generated and saved as 'wiki_metrics_plot.png'")