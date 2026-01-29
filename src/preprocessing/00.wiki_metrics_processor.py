import gzip
import csv
from xml.etree import ElementTree as ET
import os
import time
import tools

"""
CHANGE LOG
    2025-11-08: Initial creation of the script to process Wikipedia XML dumps and extract monthly user metrics.
    2025-11-12: Plot data generation from separate function in tools.py for modularity.
"""


# --- CONFIGURATION ---
DUMP_FILENAME = 'C:/Users/jordi/Documents/09miar/Wiki_original_files/20251021_version/eswiki-latest-stub-meta-history.xml.gz'
OUTPUT_CSV_FILE = '/data/02_csv/monthly_wiki_metrics.csv'
BOT_LIST_FILE = '/data/01_wiki/wiki_bot_users.csv'
# XML namespace required for Wikipedia dumps
WIKI_NAMESPACE = '{http://www.mediawiki.org/xml/export-0.11/}'

# ---------------------


def load_bot_list(bot_list_path: str) -> set:
    """
    Reads the list of known bot usernames from the CSV file and returns them as a set
    for fast lookup (O(1) complexity).
    """
    if not os.path.exists(bot_list_path):
        print(f"\n[ERROR] Bot list file not found: {bot_list_path}. Bot classification will be skipped.")
        return set()

    bot_usernames = set()
    print(f"Loading bot usernames from {bot_list_path}...")
    try:
        with open(bot_list_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip the header row
            try:
                header = next(reader)
                if header[0] != 'BOT_User':
                    print(f"[WARNING] Expected header 'BOT_User', found '{header[0]}'. Check your CSV file.")
            except StopIteration:
                print("[WARNING] Bot list file is empty.")
                return set()

            for row in reader:
                if row and row[0].strip():
                    bot_usernames.add(row[0].strip())  # Add only valid usernames

        print(f"[INFO] Successfully loaded {len(bot_usernames):,} bot usernames.")
        return bot_usernames

    except Exception as e:
        print(f"[ERROR] Failed to load bot list: {e}. Bot classification will be skipped.")
        return set()


def write_summary_csv(metrics_data: dict, output_path: str):
    """
    Calculates final metrics (like non-bot registered users) and writes the aggregated data to a CSV file.
    """
    if not metrics_data:
        print("\n[WARNING] No data collected. Summary CSV file not created.")
        return

    final_data = []
    for date_key, data in metrics_data.items():
        year, month = date_key.split('-')

        # Calculate Registered Users: (Total - IP - Bot)
        # Using set difference ensures we only count users who are neither IP nor Bot.
        registered_users = data['total_users'] - data['ip_users'] - data['bot_users']

        final_data.append({
            'year': int(year),
            'month': int(month),
            'total_users': len(data['total_users']),
            'ip_users': len(data['ip_users']),
            'bot_users': len(data['bot_users']),
            'regular_users': len(registered_users),
            'revised_pages': len(data['revised_pages']),
            'revisions': data['revisions']
        })

    # Sort data by year and month
    final_data.sort(key=lambda x: (x['Year'], x['Month']))

    fieldnames = list(final_data[0].keys())

    try:
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_data)

        print(f"\n[SUCCESS] Analysis complete. Monthly metrics saved to '{output_path}'")
        print(f"Total months analyzed: {len(final_data)}")

    except Exception as e:
        print(f"\n[ERROR] Could not write summary CSV file: {e}")


# --- MAIN EXECUTION BLOCK ---
def main_processor(file_path, bot_list_file, output_path):
    start_time = time.time()
    total_revisions_processed = 0
    page_counter = 0

    # 1. Load Bot List and Initialize Aggregation Structure
    bot_set = load_bot_list(bot_list_file)

    # Aggregation dictionary: { 'YYYY-MM': { 'total_users': set(), 'ip_users': set(), ... } }
    metrics = {}

    current_page_data = {}
    PROGRESS_THRESHOLD = 5000

    print(f"\nStarting analysis of '{file_path}'...")

    try:
        # 'rt' reads as text, 'errors="ignore"' handles common Unicode errors.
        with gzip.open(file_path, 'rt', encoding='utf-8-sig', errors='ignore') as compressed_input_file:
            context = ET.iterparse(compressed_input_file, events=('start', 'end'))

            for event, elem in context:
                tag = elem.tag

                # --- 1. CAPTURE PAGE START (Clear state) ---
                if event == 'start' and tag == f"{WIKI_NAMESPACE}page":
                    current_page_data = {}
                    page_counter += 1
                    if page_counter % PROGRESS_THRESHOLD == 0:
                        print(f"Processed {page_counter:12,} pages...", end='\r', flush=True)

                # --- 2. CAPTURE PAGE METADATA (Title, NS, ID) ---
                # We capture the page ID to track unique pages revised.
                elif event == 'end' and tag == f"{WIKI_NAMESPACE}id" and 'page_id' not in current_page_data:
                    current_page_data['page_id'] = elem.text if elem.text else 'N/A'

                # --- 3. PROCESS REVISION DATA ---
                elif event == 'end' and tag == f"{WIKI_NAMESPACE}revision":
                    try:
                        total_revisions_processed += 1

                        # a. Extract timestamp
                        timestamp_element = elem.find(f"{WIKI_NAMESPACE}timestamp")
                        record_timestamp = timestamp_element.text if timestamp_element is not None else None

                        if record_timestamp is None or len(record_timestamp) < 10:
                            elem.clear()
                            continue

                        # Extract the required monthly key (YYYY-MM)
                        date_key = record_timestamp[:7]
                        page_id = current_page_data.get('page_id')

                        # Initialize metric entry for the month if it doesn't exist
                        if date_key not in metrics:
                            metrics[date_key] = {
                                'total_users': set(), 'ip_users': set(), 'bot_users': set(),
                                'revised_pages': set(), 'revisions': 0
                            }
                        month_metrics = metrics[date_key]

                        # AGGREGATION STEP 1: Revision and Page Counts
                        month_metrics['revisions'] += 1
                        if page_id:
                            month_metrics['revised_pages'].add(page_id)

                        # b. Extract Contributor Data
                        contributor_elem = elem.find(f"{WIKI_NAMESPACE}contributor")
                        user_name = None
                        is_ip = False

                        if contributor_elem is not None:
                            ip_elem = contributor_elem.find(f"{WIKI_NAMESPACE}ip")
                            username_elem = contributor_elem.find(f"{WIKI_NAMESPACE}username")

                            if ip_elem is not None:
                                user_name = ip_elem.text
                                is_ip = True
                            elif username_elem is not None:
                                user_name = username_elem.text

                        if user_name:
                            # AGGREGATION STEP 2: User Classification

                            # All users are added to the 'total_users' set
                            month_metrics['total_users'].add(user_name)

                            if is_ip:
                                # IP User Classification
                                month_metrics['ip_users'].add(user_name)
                            # Registered User Classification
                            elif user_name in bot_set:
                                # Bot Classification using Set Lookup
                                month_metrics['bot_users'].add(user_name)

                    except Exception as inner_e:
                        print(
                            f"\nWarning: Skipping revision {total_revisions_processed:,} due to inner data error: {inner_e}")

                    # Mandatory: Clear the revision element immediately after processing
                    elem.clear()

                # --- 4. CLEAR PAGE MEMORY ---
                elif event == 'end' and tag == f"{WIKI_NAMESPACE}page":
                    # Clear the memory tree of the entire page element
                    elem.clear()
                    current_page_data = {}

    except ET.ParseError as e:
        print(f"\n🚨 FATAL PARSE ERROR (File Corruption/Structure Error) 🚨")
        print(f"Error: {e}")
        print(f"Exiting gracefully after processing {total_revisions_processed:,} revisions.")
    except Exception as e:
        print(f"\n🚨 UNEXPECTED ERROR: {e}")

    finally:
        # --- FINAL WRITE ---
        print(f"\nProcessing complete. Total pages analyzed: {page_counter:,}")
        print(f"Total revisions processed: {total_revisions_processed:,}")
        write_summary_csv(metrics, output_path)
        plot_wiki_metrics_from_dict = tools.plot_wiki_metrics_from_dict(metrics)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    # Ensure the script runs with the configured files
    main_processor(DUMP_FILENAME, BOT_LIST_FILE, OUTPUT_CSV_FILE)