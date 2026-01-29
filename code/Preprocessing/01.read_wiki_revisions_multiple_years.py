import xml.etree.ElementTree as ET
import gzip
import csv
import os
import time

"""
Change Log:
    2025-10-18: Initial version created.
    2025-10-19: Adapted to use a single pass over the XML file to process and split revisions across a 
                defined range of years, greatly improving efficiency.
"""


def process_wiki_dump_once(
        file_path,
        output_dir,
        start_year,
        end_year,
        revisions_per_split=50000,
        sequence_out='01',
        name_space=0
):
    """
    Reads a compressed Wikipedia XML dump file in a single pass and extracts revision data
    for all years within the specified range (inclusive). It splits the year into multiple CSV files with a limit
    of 'revisions_per_split' revisions each.

    :param file_path: Path to the compressed Wikipedia dump file (XML format).
    :param output_dir: Directory where the output CSV files will be saved.
    :param start_year: The first year (integer) in the range to filter revisions by.
    :param end_year: The last year (integer) in the range to filter revisions by.
    :param revisions_per_split: Number of revisions per output CSV file split (per year).
    :param sequence_out: Suffix for output files to distinguish different runs.
    :param name_space: Wikipedia namespace to process (0 = main/article).
    """
    # Namespaces (required for finding XML tags)
    namespace = '{http://www.mediawiki.org/xml/export-0.11/}'

    # CSV Field Names
    field_names = [
        'page_id',
        'title',
        'revision_id',
        'timestamp',
        'user_id',
        'user_name'
    ]

    # --- INITIALIZATION ---
    start_time = time.time()
    current_page_data = {}
    total_revisions_processed = 0
    filtered_count_total = 0

    # Dictionary to manage file handles, writers, and counters for EACH year.
    file_managers = {str(y): {'handle': None, 'writer': None, 'count': 0, 'split': 0, 'filtered': 0} for y in
                     range(start_year, end_year + 1)}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting single-pass parser for years {start_year} to {end_year}...")
    print(f"Target file: {file_path}")
    print(f"Output directory: {output_dir}\n")

    # Helper function to open and initialize a new split file for a specific year
    def open_new_split_file(year_manager, year_str):
        year_manager['split'] += 1

        # Close the previous file if it was open
        if year_manager['handle']:
            year_manager['handle'].close()

        file_split_count = year_manager['split']
        output_path = os.path.join(output_dir, f'run_{sequence_out}_wiki_{year_str}_part_{file_split_count:03d}.csv')

        # Open new file with UTF-8 BOM encoding for Excel
        new_handle = open(output_path, 'w', newline='', encoding='utf-8-sig')
        new_writer = csv.DictWriter(new_handle, fieldnames=field_names)
        new_writer.writeheader()

        # Update manager state
        year_manager['handle'] = new_handle
        year_manager['writer'] = new_writer
        year_manager['count'] = 0

        print(f"--- Opened new split file for {year_str}: {output_path} ---")

    # --- MAIN PARSING LOGIC ---
    try:
        # 'rt' reads as text, 'errors="ignore"' handles common Unicode errors.
        # Open compressed XML file and processed as continuous stream
        with gzip.open(file_path, 'rt', encoding='utf-8-sig', errors='ignore') as compressed_input_file:
            context = ET.iterparse(compressed_input_file, events=('start', 'end'))

            for event, elem in context:
                tag = elem.tag

                # --- 1. CAPTURE PAGE START (Clear state) ---
                if event == 'start' and tag == f"{namespace}page":
                    current_page_data = {}

                # --- 2. CAPTURE PAGE METADATA (Title) ---
                elif event == 'end' and tag == f"{namespace}title":
                    current_page_data['title'] = elem.text if elem.text else 'N/A'

                # --- 2.1 CAPTURE PAGE METADATA (Namespace) ---
                elif event == 'end' and tag == f"{namespace}ns":
                    current_page_data['namespace'] = int(elem.text) if elem.text else -1

                # --- 3. CAPTURE PAGE METADATA (Page ID) ---
                elif event == 'end' and tag == f"{namespace}id" and 'page_id' not in current_page_data:
                    current_page_data['page_id'] = elem.text if elem.text else 'N/A'

                # --- 4. PROCESS REVISION DATA ---
                elif event == 'end' and tag == f"{namespace}revision":
                    try:
                        total_revisions_processed += 1

                        # Inherit page_id and title
                        revision_record = current_page_data.copy()

                        # a. Extract revision ID and timestamp
                        revision_record['revision_id'] = elem.find(f"{namespace}id").text if elem.find(
                            f"{namespace}id") is not None else 'N/A'

                        timestamp_element = elem.find(f"{namespace}timestamp")
                        record_timestamp = timestamp_element.text if timestamp_element is not None else None
                        revision_record['timestamp'] = record_timestamp

                        # CRITICAL: EXTRACT YEAR FOR FILTERING
                        if record_timestamp is None or len(record_timestamp) < 4:
                            # Skip revisions with bad timestamps
                            elem.clear()
                            continue

                        # Extract the year string (e.g., '2001')
                        revision_year_str = record_timestamp[:4]

                        # b. FILTERING: Check if the timestamp year is in the target range
                        if revision_year_str in file_managers:

                            # c. Extract Contributor Data (only if we know we'll keep the record)
                            contributor_elem = elem.find(f"{namespace}contributor")

                            if contributor_elem is not None:
                                user_id_elem = contributor_elem.find(f"{namespace}id")
                                username_elem = contributor_elem.find(f"{namespace}username")
                                ip_elem = contributor_elem.find(f"{namespace}ip")

                                revision_record[
                                    'user_id'] = user_id_elem.text if user_id_elem is not None else 'Anonymous_ID'

                                if username_elem is not None:
                                    revision_record['user_name'] = username_elem.text
                                elif ip_elem is not None:
                                    revision_record['user_name'] = ip_elem.text
                                else:
                                    revision_record['user_name'] = 'Anonymous_Name'
                            else:
                                revision_record['user_id'] = 'Unknown'
                                revision_record['user_name'] = 'Unknown'

                            # --- SPLITTING AND WRITING LOGIC (Specific to this year) ---
                            year_manager = file_managers[revision_year_str]

                            if year_manager['count'] >= revisions_per_split or year_manager['writer'] is None:
                                # Open new file or the first file for this year
                                open_new_split_file(year_manager, revision_year_str)

                            # Filter the revision_record to include only the allowed field names
                            filtered_record = {key: revision_record[key] for key in field_names if
                                               key in revision_record}

                            # Write the filtered record to the current split file for that year
                            year_manager['writer'].writerow(filtered_record)
                            year_manager['count'] += 1
                            year_manager['filtered'] += 1
                            filtered_count_total += 1

                    except Exception as inner_e:
                        print(
                            f"Warning: Skipping revision {total_revisions_processed} due to inner data error: {inner_e}")

                    # Mandatory: Clear the revision element immediately after processing
                    elem.clear()

                # --- 5. CLEAR PAGE MEMORY ---
                elif event == 'end' and tag == f"{namespace}page":
                    # Only process pages with namespace = name_space
                    if current_page_data.get('namespace') == name_space:
                        # Clear the memory tree of the entire page element
                        elem.clear()
                    else:
                        # Skip pages with other namespaces
                        current_page_data = {}

    except ET.ParseError as e:
        print(f"\n🚨 FATAL PARSE ERROR (File Corruption/Structure Error) 🚨")
        print(f"Error: {e}")
        print(f"Exiting gracefully after processing {filtered_count_total} valid revisions.")
    except Exception as e:
        print(f"\n🚨 UNEXPECTED ERROR: {e}")
    finally:
        # --- CLOSING ALL OPEN FILES ---
        print("\n--- Closing all open CSV files ---")
        for year_str, manager in file_managers.items():
            if manager['handle']:
                manager['handle'].close()
                print(f"Closed final file for {year_str}. Total revisions: {manager['filtered']}")

        end_time = time.time()
        print(f"\n--- Processing Complete ---")
        print(f"Total revisions streamed (filtered for all years): {filtered_count_total}")
        print(f"Total revisions processed in XML: {total_revisions_processed}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")


def main():
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")

    # --- CONFIGURATION ---
    sequence_in = ''
    sequence_out = '14'
    name_space = 0  # Wikipedia namespace to process (0 = main/article)

    file_path = f'C:/Users/jordi/Documents/09miar/Wiki_original_files/20251021_version/eswiki-latest-stub-meta-history{sequence_in}.xml.gz'
    output_path = f'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version'

    # Define the range of years to process in a single pass
    START_YEAR = 2006
    END_YEAR = 2025
    revisions_per_split = 50000

    # ONE SINGLE CALL to process all years
    process_wiki_dump_once(
        file_path,
        output_path,
        START_YEAR,
        END_YEAR,
        revisions_per_split,
        sequence_out,
        name_space
    )

    print(f"End time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n\n")


if __name__ == "__main__":
    main()
