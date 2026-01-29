import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from matplotlib.ticker import FuncFormatter
import numpy as np
import os

"""
CHANGE LOG
    2025-11-16: Initial creation of the script to plot Wikipedia graph community metrics from CSV.
    2025-11-25: Add footnotes to clarify counts of users and components.
    2025-11-25: Added function to plot component node counts over time.
"""


def plot_graph_metrics_from_csv(csv_filepath, plot_filepath, foot_note, component_type: str):
    """
    Generates a 2-panel plot of monthly Wikipedia graph community metrics from a CSV file,
    showing yearly major ticks and the total time range in the title.
    Graphs include number of nodes, edges, components, nodes in components, and largest component size.

    The CSV must contain the columns:
    year, month, num_nodes, num_edges, num_components, nodes_in_components, largest_component_size.

    Args:
        csv_filepath (str): The file path to the generated metrics CSV file.
        plot_filepath (str): The file path to save the generated plot image.
        foot_note (str): A footnote to include in the plot.
        component_type (str): Type of component ('Editor' for editor-editor, 'Page' for page-page, 'Bipartite' for Bipartite).
    """

    # --- 1. Load and Prepare Data ---
    try:
        # Load the CSV file
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        return

    # Create a proper datetime index from 'year' and 'month'
    df['Date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    df = df.set_index('Date').sort_index()

    # Determine start and end years for the title
    start_year = df.index.min().year
    end_year = df.index.max().year

    # --- 2. Create the Multi-Panel Plot (4 Rows) ---
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )
    plt.subplots_adjust(hspace=0.1)


    # Define colors
    colors = {
        'num_nodes': '#5D2E8C',
        'num_edges': '#4682B4',
        'num_components': '#FF6961',
        #'nodes_in_components': '#77DD77',
        'nodes_in_components': '#56B4E9',
        #'largest_component_size': '#FFA07A',
        'largest_component_size': '#77DD77',
        'modularity_score': '#F7B731',
        'nestedness_score': '#F7B731'
    }

    # --- NEW: Define Y-axis formatter function ---
    def number_formatter(x, pos):
        """Formats numbers for Y-axis ticks (e.g., 100000 -> 100K)"""
        if x >= 1e6:
            return f'{x * 1e-6:.1f}M'
        elif x >= 1e3:
            return f'{x * 1e-3:.1f}K'
        return f'{x:.1f}'

    # Create the formatter instance
    formatter = FuncFormatter(number_formatter)

    # --- 3. Plotting Each Panel ---
    scatter_size = 10
    # Panel 1: Total Nodes and Edges

    axes[0].scatter(df.index, df['num_nodes'], s=scatter_size, label=f'Nodes ({component_type}s)', color=colors['num_nodes'])
    axes[0].set_title(f'Monthly Spanish Wikipedia Graph Metrics {component_type}-to-{component_type} ({start_year} - {end_year})', loc='left',
                      fontsize=14)
    axes[0].scatter(df.index, df['num_edges'], s=scatter_size, label=f'Edges (Between {component_type}s)', color=colors['num_edges'])
    axes[0].legend(loc='upper left')
    axes[0].grid(axis='y', linestyle='--')
    axes[0].yaxis.set_major_formatter(formatter)

    # Panel 2: Components
    axes[1].scatter(df.index, df['num_components'], s=scatter_size, label='Components', color=colors['num_components'])
    axes[1].scatter(df.index, df['nodes_in_components'], s=scatter_size, label='Nodes in Components (size >=2)',
                    color=colors['nodes_in_components'])
    axes[1].scatter(df.index, df['largest_component_size'], s=scatter_size, label='Largest Component',
                    color=colors['largest_component_size'])
    axes[1].legend(loc='upper left')
    axes[1].grid(axis='y', linestyle='--')
    axes[1].yaxis.set_major_formatter(formatter)

    # Panel 3: Components
    if component_type == 'Bipartite':
        axes[2].scatter(df.index, df['nestedness_score'], s=scatter_size, label='Nestedness Score', color=colors['nestedness_score'])
    else:
        axes[2].scatter(df.index, df['modularity_score'], s=scatter_size, label='Modularity Score', color=colors['modularity_score'])
    axes[2].legend(loc='upper left')
    axes[2].grid(axis='y', linestyle='--')
    #axes[2].yaxis.set_major_formatter(formatter)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_yticks(np.arange(0, 1.1, 0.2))
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))

    # --- 4. Final Formatting and Saving ---

    # Configure the X-axis for all panels
    axes[-1].set_xlabel('Year')

    # Set X-axis to show every year
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xticks(rotation=45, ha='right')

    fig.suptitle('Monthly Wiki Graph Analysis', fontsize=16, y=1.02)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    # Add footnote
    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)

    # Save the figure to a file

    plt.savefig(plot_filepath)
    plt.close(fig)

    print(f"Plot generated and saved to {plot_filepath}")


def plot_component_node_counts(
        component_csv_filepath: str,
        plot_filepath: str,
        foot_note: str,
        min_component_size: int = 10,
        component_type: str = 'Editor'
):
    """
    Generates a scatter plot showing the node count of every component over time.

    The component CSV file is assumed to have the following columns for each component:
    year, month, component_size, component_id (optional, for labeling).

    Args:
        component_csv_filepath (str): Path to the CSV file containing component details.
        plot_filepath (str): The file path to save the generated plot image.
        foot_note (str): A footnote to include in the plot.
        min_component_size (int): Minimum size of components to include in the plot.
        max_components_to_label (int): Maximum number of largest components to label.
    """
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(component_csv_filepath)
    except FileNotFoundError:
        print(f"Error: Component CSV file not found at {component_csv_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while loading the component CSV: {e}")
        return

    # Filter for components larger than the minimum size
    df_filtered = df[df['component_size'] >= min_component_size].copy()

    if df_filtered.empty:
        print(f"No components found with size >= {min_component_size}. Aborting plot.")
        return

    # Create a proper datetime index
    df_filtered['Date'] = pd.to_datetime(df_filtered[['year', 'month']].assign(day=1))
    df_filtered = df_filtered.set_index('Date').sort_index()

    # Determine start and end years for the title
    start_year = df_filtered.index.min().year
    end_year = df_filtered.index.max().year

    # --- 2. Create the Scatter Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Scatter plot: x=Date, y=component_size. Node size based on component_size.
    scatter = ax.scatter(
        df_filtered.index,
        df_filtered['component_size'],
        #s=df_filtered['component_size'] * 0.5,  # Size the marker based on component size
        alpha=0.6,
        color='#1E90FF'  # Dodger Blue for a distinct look
    )

    # --- 4. Formatting ---
    if component_type == 'Bipartite':
        ax.set_title(
            f'Monthly Distribution of Component Node Counts (Editors and Pages) ({start_year} - {end_year})',
            loc='left',
            fontsize=14
        )
    else:
        ax.set_title(
            f'Monthly Distribution of Component Node Counts ({component_type}s) ({start_year} - {end_year})',
            loc='left',
            fontsize=14
        )
    ax.set_xlabel('Time (Year)')
    ax.set_ylabel(f'Component Size (Number of Nodes) [min >= {min_component_size}]')
    ax.grid(axis='y', linestyle='--')

    # Configure the X-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha='right')

    # Add footnote
    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    # Save the figure
    plt.savefig(plot_filepath)
    plt.close(fig)

    print(f"Component count plot generated and saved: {plot_filepath}")


def plot_component_bin_sizes(
        component_csv_filepath: str,
        plot_filepath: str,
        foot_note: str,
        min_component_size: int = 10,
        component_type: str = 'Editor'
):
    """
    Generates a scatter plot showing the node count of every component over time,
    colored by component size bins (Small, Medium, Large).

    The component CSV file is assumed to have the following columns for each component:
    year, month, component_size, component_instance_id (optional, for labeling).

    Args:
        component_csv_filepath (str): Path to the CSV file containing component details.
        plot_filepath (str): The file path to save the generated plot image.
        foot_note (str): A footnote to include in the plot.
        min_component_size (int): Minimum size of components to include in the plot.
        max_components_to_label (int): Maximum number of largest components to label.
    """
    # --- 1. Load and Prepare Data ---
    try:
        df = pd.read_csv(component_csv_filepath)
    except FileNotFoundError:
        print(f"Error: Component CSV file not found at {component_csv_filepath}")
        return
    except Exception as e:
        print(f"An error occurred while loading the component CSV: {e}")
        return

    # Filter for components larger than the minimum size
    df_filtered = df[df['component_size'] >= min_component_size].copy()

    if df_filtered.empty:
        print(f"No components found with size >= {min_component_size}. Aborting plot.")
        return

    # Create a proper datetime index
    df_filtered['Date'] = pd.to_datetime(df_filtered[['year', 'month']].assign(day=1))
    df_filtered = df_filtered.set_index('Date').sort_index()

    # Determine start and end years for the title
    start_year = df_filtered.index.min().year
    end_year = df_filtered.index.max().year

    # --- NEW: Define Size Bins and Colors ---

    bins = [min_component_size, 100, 500, np.inf]  # Bins: [min, 100), [100, 500), [500, inf)
    labels = ['Small (2-100 Nodes)', 'Medium (101-500 Nodes)', 'Large (>500 Nodes)']
    bin_colors = {'Small (2-100 Nodes)': '#4682B4',  # Blue
                  'Medium (101-500 Nodes)': '#E69F00',  # Orange
                  'Large (>500 Nodes)': '#009E73'}  # Green


    # Apply the binning
    df_filtered['Size_Category'] = pd.cut(
        df_filtered['component_size'],
        bins=bins,
        labels=labels,
        right=False,  # Interval is [a, b)
        include_lowest=True
    )

    # --- 2. Create the Scatter Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))

    scatter_handles = []

    # Plot each category separately to generate a legend handle
    for category, color in bin_colors.items():
        subset = df_filtered[df_filtered['Size_Category'] == category]

        # We only want to plot categories that actually exist in the data
        if not subset.empty:
            handle = ax.scatter(
                subset.index,
                subset['component_size'],
                #s=subset['component_size'] * 0.5,  # Size the marker based on component size
                alpha=0.6,
                color=color,
                label=category
            )
            scatter_handles.append(handle)
    """
    # --- 3. Labeling Largest Components (Unchanged) ---
    # Find the overall largest components across all months for labeling
    if 'component_instance_id' in df_filtered.columns:
        # Get the top N largest components based on their size (across all time points)
        top_components = df_filtered.nlargest(max_components_to_label, 'component_size')

        for _, row in top_components.iterrows():
            ax.annotate(
                f"ID {row['component_instance_id']} (Size {row['component_size']})",
                (row.name, row['component_size']),  # row.name is the Date index
                textcoords="offset points",
                xytext=(5, 5),
                ha='left',
                fontsize=8,
                color='#00008B'
            )
    """
    # --- 4. Formatting ---

    ax.set_title(
        f'Monthly Distribution of Component Node Counts ({component_type}s) by Size Category ({start_year} - {end_year})',
        loc='left',
        fontsize=14
    )
    ax.set_xlabel('Time (Year)')
    ax.set_ylabel(f'Component Size (Number of Nodes) [min >= {min_component_size}]')
    ax.grid(axis='y', linestyle='--')
    ax.grid(axis='x', linestyle='--')

    # Add the size category legend
    ax.legend(handles=scatter_handles, title="Component Size", loc='upper left')

    # Configure the X-axis
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha='right')

    # Add footnote
    if foot_note:
        plt.figtext(0.5, 0.01, foot_note, wrap=True, horizontalalignment='center', fontsize=8)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    # Save the figure
    plt.savefig(plot_filepath)
    plt.close(fig)

    print(f"Component count plot generated and saved: {plot_filepath}")


if __name__ == "__main__":

    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    GRAPH_METRICS_PLOTS = True  # Set to True to plot graph metrics from CSV
    COMPONENT_NODE_PLOTS = True  # Set to True to plot component node counts from CSV

    ROOT_DIR = '//'
    csv_dir = os.path.join(ROOT_DIR, 'data/02_csv/')
    graph_dir = os.path.join(ROOT_DIR, 'data/03_graph/')
    plot_dir = os.path.join(ROOT_DIR, 'data/04_plots/')

    # Community metrics CSV paths
    CSV_E_E_INPUT_FILE_PATH = f'{csv_dir}graph_community_metrics_e_e.csv'
    CSV_P_P_INPUT_FILE_PATH = f'{csv_dir}graph_community_metrics_p_p.csv'
    CSV_BIPARTITE_INPUT_FILE_PATH = f'{csv_dir}graph_community_metrics_bipartite.csv'

    PLOT_E_E_FILE_PATH = f'{plot_dir}graph_community_plot_e_e_{execution_timestamp}.png'
    PLOT_P_P_FILE_PATH = f'{plot_dir}graph_community_plot_p_p_{execution_timestamp}.png'
    PLOT_BIPARTITE_FILE_PATH = f'{plot_dir}graph_community_plot_bipartite_{execution_timestamp}.png'

    MY_E_E_FOOTNOTE = 'Nodes (Editors) with a minimum of 7 revisions in the month, and components of size >=2.'
    MY_P_P_FOOTNOTE = 'Nodes (Pages) with a minimum of 7 revisions in the month, and components of size >=2.'
    MY_BIPARTITE_FOOTNOTE = 'Bipartite Graph: Nodes (Editors and Pages) with a minimum of 7 revisions in the month, and components of size >=2.'

    if GRAPH_METRICS_PLOTS:
        plot_graph_metrics_from_csv(CSV_E_E_INPUT_FILE_PATH, PLOT_E_E_FILE_PATH, MY_E_E_FOOTNOTE, 'Editor')
        plot_graph_metrics_from_csv(CSV_P_P_INPUT_FILE_PATH, PLOT_P_P_FILE_PATH, MY_P_P_FOOTNOTE, 'Page')
        plot_graph_metrics_from_csv(CSV_BIPARTITE_INPUT_FILE_PATH, PLOT_BIPARTITE_FILE_PATH, MY_BIPARTITE_FOOTNOTE, 'Bipartite')

    # --- Example Usage for plot_component_node_counts (New) ---
    if COMPONENT_NODE_PLOTS:
        # editor-editor component details CSV path
        COMPONENT_CSV_PATH = f'{csv_dir}component_details_e_e.csv'
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_e_e_{execution_timestamp}.png'
        for min_component_size in [10, 200]:
            FOOTNOTE_ORIGINAL = "Scatter plot editor-editor co-occurrence graph showing the size of every component >= 10 nodes over time."
            COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_e_e_{execution_timestamp}_{min_component_size}.png'
            # Plot several graphs based on different minimum component sizes
            plot_component_node_counts(
                COMPONENT_CSV_PATH,
                COMPONENT_PLOT_PATH,
                FOOTNOTE_ORIGINAL,
                min_component_size,
                'Editor'
            )
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_e_e_{execution_timestamp}_binned_{min_component_size}.png'
        FOOTNOTE_BINNED = "Scatter plot editor-editor co-occurrence graph showing the size of every component >= 10 nodes over time, colored by size bin."
        plot_component_bin_sizes(
            COMPONENT_CSV_PATH,
            COMPONENT_PLOT_PATH,
            FOOTNOTE_BINNED,
            4,
            'Editor'
        )

        # page-page component details CSV path
        COMPONENT_CSV_PATH = f'{csv_dir}component_details_p_p.csv'
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_p_p_{execution_timestamp}.png'
        for min_component_size in [10, 200]:
            FOOTNOTE_ORIGINAL = "Scatter plot showing the size of every component >= 10 nodes over time."
            COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_p_p_{execution_timestamp}_{min_component_size}.png'
            # Plot several graphs based on different minimum component sizes
            plot_component_node_counts(
                COMPONENT_CSV_PATH,
                COMPONENT_PLOT_PATH,
                FOOTNOTE_ORIGINAL,
                min_component_size,
                component_type='Page'
            )
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_p_p_{execution_timestamp}_binned_{min_component_size}.png'
        FOOTNOTE_BINNED = "Scatter plot showing the size of every component >= 10 nodes over time, colored by size bin."
        plot_component_bin_sizes(
            COMPONENT_CSV_PATH,
            COMPONENT_PLOT_PATH,
            FOOTNOTE_BINNED,
            min_component_size=4,
            component_type='Page'
        )

        # bipartite component details CSV path
        COMPONENT_CSV_PATH = f'{csv_dir}component_details_bipartite.csv'
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_bipartite_{execution_timestamp}.png'
        for min_component_size in [10, 200]:
            FOOTNOTE_ORIGINAL = "Scatter plot showing the size of every component >= 10 nodes over time."
            COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_bipartite_{execution_timestamp}_{min_component_size}.png'
            # Plot several graphs based on different minimum component sizes
            plot_component_node_counts(
                COMPONENT_CSV_PATH,
                COMPONENT_PLOT_PATH,
                FOOTNOTE_ORIGINAL,
                min_component_size,
                component_type='Bipartite'
            )
        COMPONENT_PLOT_PATH = f'{plot_dir}component_scatter_plot_bipartite_{execution_timestamp}_binned_{min_component_size}.png'
        FOOTNOTE_BINNED = "Scatter plot showing the size of every component >= 10 nodes over time, colored by size bin."
        plot_component_bin_sizes(
            COMPONENT_CSV_PATH,
            COMPONENT_PLOT_PATH,
            FOOTNOTE_BINNED,
            min_component_size=4,
            component_type='Bipartite'
        )
