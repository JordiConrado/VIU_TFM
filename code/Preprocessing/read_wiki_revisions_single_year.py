import xml.etree.ElementTree as ET
import gzip
import csv
import os
import time

"""
Change Log
    2025-10-18: Initial version created to read Wikipedia XML dumps and extract revisions by year.
    2025-10-18: Define function and main.
    2025-10-19: Include only pages from the main namespace (namespace = 0).

"""


def read_wiki_file_by_year(file_path, output_dir, target_year, revisions_per_split=500000, sequence_out='01', name_space=0):
    """
    Reads a compressed Wikipedia XML dump file and extracts revision data for a specified year.
    The extracted data is saved into multiple CSV files, each containing a specified number of revisions.

    :param file_path: Path to the compressed Wikipedia dump file (XML format).
    :param output_dir: Directory where the output CSV files will be saved.
    :param target_year: Year (as a string, e.g., '2001') to filter revisions by.
    :param revisions_per_split: Number of revisions per output CSV file.
    :param sequence_out: Suffix for output files to distinguish different runs.
    """
    # Namespaces (required for finding XML tags)
    # This namespace is standard for MediaWiki dumps
    namespace = '{http://www.mediawiki.org/xml/export-0.11/}'

    # CSV Field Names (ensure these match the keys in revision_record)
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
    csv_writer = None
    current_csv_file = None
    file_split_count = 0
    revisions_in_current_file = 0
    filtered_count = 0
    total_revisions_processed = 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- MAIN PARSING LOGIC ---
    print(f"Starting parser for year {target_year}...")
    print(f"Target file: {file_path}")
    print(f"Output directory: {output_dir}\n")

    try:
        # 'rt' reads as text, 'errors="ignore"' handles common Unicode errors.
        # 'utf-8-sig' ensures Excel reads accents correctly later.
        with gzip.open(file_path, 'rt', encoding='utf-8-sig', errors='ignore') as compressed_input_file:
            # Use 'start' and 'end' events for better memory control
            context = ET.iterparse(compressed_input_file, events=('start', 'end'))

            for event, elem in context:
                tag = elem.tag

                # --- 1. CAPTURE PAGE START (Clear state) ---
                if event == 'start' and tag == f"{namespace}page":
                    current_page_data = {}

                # --- 2. CAPTURE PAGE METADATA (Title) ---
                elif event == 'end' and tag == f"{namespace}title":
                    # Only capture title when the element ends
                    current_page_data['title'] = elem.text if elem.text else 'N/A'

                # --- 2.1 CAPTURE PAGE METADATA (Namespace) ---
                elif event == 'end' and tag == f"{namespace}ns":
                    current_page_data['namespace'] = int(elem.text) if elem.text else -1

                # --- 3. CAPTURE PAGE METADATA (Page ID) ---
                # This complex check ensures we only grab the page ID (the first <id> tag)
                elif event == 'end' and tag == f"{namespace}id" and 'page_id' not in current_page_data:
                    current_page_data['page_id'] = elem.text if elem.text else 'N/A'

                # --- 4. PROCESS REVISION DATA ---
                elif event == 'end' and tag == f"{namespace}revision":

                    # --- START OF GRANULAR ERROR HANDLING ---
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

                        # b. Extract Contributor Data (User ID and Name)
                        contributor_elem = elem.find(f"{namespace}contributor")

                        if contributor_elem is not None:
                            user_id_elem = contributor_elem.find(f"{namespace}id")
                            username_elem = contributor_elem.find(f"{namespace}username")
                            ip_elem = contributor_elem.find(f"{namespace}ip")

                            # User ID
                            revision_record[
                                'user_id'] = user_id_elem.text if user_id_elem is not None else 'Anonymous_ID'

                            # UserName (Prioritize username, then IP, then fallback)
                            if username_elem is not None:
                                revision_record['user_name'] = username_elem.text
                            elif ip_elem is not None:
                                revision_record['user_name'] = ip_elem.text
                            else:
                                revision_record['user_name'] = 'Anonymous_Name'
                        else:
                            # Fallback for old or missing contributor records
                            revision_record['user_id'] = 'Unknown'
                            revision_record['user_name'] = 'Unknown'

                        # c. FILTERING: Check if the timestamp matches the target year
                        if record_timestamp and record_timestamp.startswith(target_year):

                            # --- SPLITTING LOGIC ---
                            if revisions_in_current_file >= revisions_per_split or csv_writer is None:
                                if current_csv_file:
                                    current_csv_file.close()  # Close the old file

                                file_split_count += 1
                                output_path = os.path.join(output_dir,
                                                           f'run_{sequence_out}_wiki_{target_year}_part_{file_split_count:03d}.csv')

                                # Open new file with UTF-8 BOM encoding for Excel
                                current_csv_file = open(output_path, 'w', newline='', encoding='utf-8-sig')
                                csv_writer = csv.DictWriter(current_csv_file, fieldnames=field_names)
                                csv_writer.writeheader()
                                revisions_in_current_file = 0
                                print(f"\n--- Opened new split file: {output_path} ---")

                            # Write the revision record to the current split file
                            filtered_record = {key: revision_record[key] for key in field_names if
                                               key in revision_record}
                            csv_writer.writerow(filtered_record)
                            revisions_in_current_file += 1
                            filtered_count += 1

                    except Exception as inner_e:
                        # Log data extraction failure (e.g., tag missing unexpectedly) but keep parsing
                        print(
                            f"Warning: Skipping revision {total_revisions_processed} due to inner data error: {inner_e}")
                    # --- END OF GRANULAR ERROR HANDLING ---

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

                # --- FATAL ERROR CATCH ---
    except ET.ParseError as e:
        # This catches the critical 'mismatched tag' error.
        print(f"\n🚨 FATAL PARSE ERROR (File Corruption/Structure Error) 🚨")
        print(f"Error: {e}")
        print(f"Exiting gracefully after processing {filtered_count} valid revisions.")
    except Exception as e:
        print(f"\n🚨 UNEXPECTED ERROR: {e}")
    finally:
        # Ensure the last open file is closed
        if current_csv_file:
            current_csv_file.close()

        end_time = time.time()
        print(f"\n--- Processing Complete ---")
        print(f"Total revisions streamed (filtered for {target_year}): {filtered_count}")
        print(f"Total split files created: {file_split_count}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")


def main():
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    name_space = 0  # Wikipedia namespace to process (0 = main/article)

    sequence_in = ''  # Sequence identifier for input files
    sequence_out = '14'  # Sequence identifier for output files

    file_path = f'C:/Users/jordi/Documents/09miar/Wiki_original_files/20251021_version/eswiki-latest-stub-meta-history{sequence_in}.xml.gz'

    output_path = 'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/'
    start_year = 2001
    end_year = 2001
    revisions_per_split = 500000  # Number of revisions per output CSV file

    for i in range(start_year, end_year+1):
        print(f'Processing year: {i}')
        target_year = str(i)
        read_wiki_file_by_year(file_path, output_path, target_year, revisions_per_split, sequence_out, name_space)

    print(f"End time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n\n")


if __name__ == "__main__":
    main()
