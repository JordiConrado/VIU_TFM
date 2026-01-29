import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from matplotlib.ticker import FuncFormatter


def plot_wiki_metrics_from_csv(csv_filepath, plot_filepath, full_plot):
    """
    Generates a 4-panel plot of monthly Wikipedia metrics from a CSV file,
    showing yearly major ticks and the total time range in the title.

    The CSV must contain the columns:
    year, month, total_users, ip_users, bot_users, regular_users,
    revised_pages, revisions.

    Args:
        csv_filepath (str): The file path to the generated metrics CSV file.
        plot_filepath (str): The file path to save the generated plot image.
        full_plot (bool): If True, generates a 5-panel plot including total revised pages. If False, generates a 3-panel plot.
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
    n_rows = 5 if full_plot else 3
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(12, 8),
        sharex=True
    )
    plt.subplots_adjust(hspace=0.1)

    std_colors = {
        'blue': '#4682B4',
        'orange': '#E69F00',
        'green': '#009E73',
        'red': '#FF0000',
        'purple': '#5D2E8C',
        'sky_blue': '#56B4E9',
        'yellow': '#F0E442',
        'black': '#000000'
        }
    # Good combinations:
    # 2 plots: blue, orange
    # 3 plots: blue, orange, green
    # 4 plots: blue, orange, green, red
    # 5+ plots: blue, orange, green, red, purple, sky_blue

    # Define colors
    colors = {
        'total_users': std_colors['black'],
        'regular_users': std_colors['blue'],
        'ip_users': std_colors['red'],
        'bot_users': std_colors['green'],
        'revised_pages': std_colors['purple'],
        'revisions':  std_colors['sky_blue']
    }

    # --- NEW: Define Y-axis formatter function ---
    def number_formatter(x, pos):
        """Formats numbers for Y-axis ticks (e.g., 100000 -> 100K)"""
        if x >= 1e6:
            return f'{x * 1e-6:.0f}M'
        elif x >= 1e3:
            return f'{x * 1e-3:.0f}K'
        return f'{x:.0f}'

    # Create the formatter instance
    formatter = FuncFormatter(number_formatter)
    panel = 0

    # --- 3. Plotting Each Panel ---

    # Panel 1: Total Users
    axes[panel].plot(df.index, df['total_users'], label='Total Users', color=colors['total_users'])
    #axes[panel].set_title(f'Monthly Spanish Wikipedia Revisions Basic Metrics ({start_year} - {end_year})',
    #                  loc='left',
    #                  fontsize=14)
    axes[panel].legend(loc='upper right')
    axes[panel].grid(axis='y', linestyle='--')
    axes[panel].yaxis.set_major_formatter(formatter)

    # Panel 2: User Breakdown: Regular vs. IP vs. Bot
    panel += 1
    axes[panel].plot(df.index, df['regular_users'], label='Regular (Registered) Users', color=colors['regular_users'])
    axes[panel].plot(df.index, df['ip_users'], label='IP Users', color=colors['ip_users'])
    axes[panel].plot(df.index, df['bot_users'], label='BOT Users', color=colors['bot_users'])
    axes[panel].legend(loc='upper right')
    axes[panel].grid(axis='y', linestyle='--')
    axes[panel].yaxis.set_major_formatter(formatter)


    # Panel Optional : Regular vs. IP Users
    if full_plot:
        panel += 1
        axes[panel].plot(df.index, df['regular_users'], label='Regular Users', color=colors['regular_users'], linestyle='-')
        axes[panel].plot(df.index, df['ip_users'], label='IP Users', color=colors['ip_users'], linestyle=':')
        axes[panel].legend(loc='upper right')
        axes[panel].grid(axis='y', linestyle='--')
        axes[panel].yaxis.set_major_formatter(formatter)

    # Panel : Total Revisions and Revised Pages
    panel += 1
    axes[panel].plot(df.index, df['revisions'], label='Total Revisions', color=colors['revisions'])
    axes[panel].plot(df.index, df['revised_pages'], label='Revised Pages', color=colors['revised_pages'])
    axes[panel].legend(loc='upper right')
    axes[panel].grid(axis='y', linestyle='--')
    axes[panel].yaxis.set_major_formatter(formatter)

    # Panel Optional: Total Revised Pages
    if full_plot:
        panel += 1
        axes[4].plot(df.index, df['revised_pages'], label='Revised Pages', color=colors['revised_pages'])
        axes[4].legend(loc='upper right')
        axes[4].grid(axis='y', linestyle='--')
        axes[4].yaxis.set_major_formatter(formatter)

    # ---  Final Formatting and Saving ---

    # Configure the X-axis for all panels
    last_ax = axes[n_rows - 1]
    last_ax.set_xlabel('Year')
    last_ax.xaxis.set_major_locator(mdates.YearLocator())
    last_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.xticks(rotation=45, ha='right')

    fig.suptitle('Monthly Wiki Metrics Analysis', fontsize=16, y=1.02)

    plt.tight_layout(rect=(0, 0.03, 1, 1.0))

    # Save the figure to a file

    plt.savefig(plot_filepath)
    plt.close(fig)

    print(f"Plot generated and saved")


if __name__ == "__main__":

    full_plot = False # Set to True to generate the full 5-panel plot. If False. it generates only three panels.
    execution_timestamp = time.strftime('%Y%m%d_%H%M%S')
    ROOT_DIR = 'G:/My Drive/Masters/VIU/09MIAR-TFM/Pycharm/VIU_TFM/data'
    GRAPH_DIR = f'{ROOT_DIR}/03_graph/'
    CSV_DIR = f'{ROOT_DIR}/02_csv/'
    PLOT_DIR = f'{ROOT_DIR}/04_plots/'

    CSV_INPUT_FILE_PATH = f'{CSV_DIR}monthly_wiki_metrics.csv'
    PLOT_FILE_PATH = f'{PLOT_DIR}wiki_metrics_plot_{execution_timestamp}.png'

    plot_wiki_metrics_from_csv(CSV_INPUT_FILE_PATH, PLOT_FILE_PATH, full_plot)
