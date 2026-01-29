import pandas as pd
import os
import glob
import time



"""
CHANGE LOG
    2025-10-28: Initial version created to iteratively read large CSV files in chunks and split them into monthly files.
    2025-11-06: Added multi-year processing capability.
"""
# --- CONFIGURATION ---
RUN_SEQUENCE = '14'
YEAR = '2005'

INPUT_DIRECTORY = 'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/'
# Pattern to match your individual CSV files (e.g., "daily_dump_2024-01-01.csv")
INPUT_FILE_PATTERN = f'run_{RUN_SEQUENCE}_wiki_{YEAR}_part*.csv'
# The column name that holds the revision timestamp
TIMESTAMP_COLUMN = 'timestamp'
# The directory where the 12 monthly CSV files will be saved
OUTPUT_DIRECTORY = f'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/Monthly_Splits/{YEAR}/'
# Size of the chunks to read from the input files. 100,000 is a good starting point.
CHUNK_SIZE = 100000 # Numer of rows Pandas reads into memory at once


def get_monthly_filepath(output_dir, year, month_num):
    """Generates the standardized file path for a given month."""
    month_str = str(month_num).zfill(2)
    output_filename = f'run_{RUN_SEQUENCE}_wiki_{year}_month_{month_str}.csv'
    return os.path.join(output_dir, output_filename)


def process_files_iteratively_and_split(input_dir, file_pattern, timestamp_col, output_dir, chunk_size):
    """
    Iterates through input files, reads them in chunks, and writes data to
    the correct monthly output file using append mode ('a').
    This is the memory-safe way to process very large datasets.
    """
    all_files_path = os.path.join(input_dir, file_pattern)
    all_files = glob.glob(all_files_path)

    if not all_files:
        print(f"Error: No files found matching '{all_files_path}'")
        return

    print(f"Found {len(all_files)} files to process in chunks of {chunk_size} rows.")

    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to track which monthly files have already had their header written
    headers_written = {i: False for i in range(1, 13)}

    for i, filename in enumerate(all_files):
        print(f" Starting processing file {i + 1}/{len(all_files)}: {os.path.basename(filename)} ---")

        # Read the input file in chunks
        chunk_iterator = pd.read_csv(filename, chunksize=chunk_size, encoding='utf-8-sig')

        for chunk_num, chunk in enumerate(chunk_iterator):

            # Convert the timestamp column to datetime
            try:
                chunk[timestamp_col] = pd.to_datetime(chunk[timestamp_col])
            except Exception as e:
                print(f"Error converting timestamps in chunk {chunk_num}: {e}. Skipping chunk.")
                continue

            # Get the year (used for naming the output file)
            # Assuming all data is from the same single year, take the first valid year
            year = chunk[timestamp_col].dt.year.dropna().iloc[0] if not chunk[
                timestamp_col].dt.year.dropna().empty else 'YYYY'

            # Group the current chunk by month
            monthly_groups = chunk.groupby(chunk[timestamp_col].dt.month)

            # Iterate through the monthly groups in this chunk
            for month_num, group_df in monthly_groups:
                output_path = get_monthly_filepath(output_dir, year, month_num)

                # Determine if the header should be written
                write_header = not headers_written[month_num]

                # Write/Append the data to the correct monthly file
                group_df.to_csv(
                    output_path,
                    index=False,
                    mode='a',  # CRUCIAL: Use 'a' for append mode
                    header=write_header,
                    encoding='utf-8-sig'
                )

                # Mark the header as written for this month
                if write_header:
                    headers_written[month_num] = True

            # print(f"  Processed Chunk {chunk_num + 1}...")

    print("\n--- ITERATIVE SPLIT PROCESS COMPLETE ---")


if __name__ == '__main__':
    print(f"Start time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    multi_year = True  # Set to True to process multiple years
    start_year = 2006
    end_year = 2025

    if not multi_year:
        # Single year processing
        process_files_iteratively_and_split(
            INPUT_DIRECTORY,
            INPUT_FILE_PATTERN,
            TIMESTAMP_COLUMN,
            OUTPUT_DIRECTORY,
            CHUNK_SIZE
        )
    else:
        # Multiple years processing
        for year in range(start_year, end_year + 1):
            print(f"--- Processing year: {year}")

            input_file_pattern_yearly = f'run_{RUN_SEQUENCE}_wiki_{year}_part*.csv'
            output_directory_yearly = f'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/Monthly_Splits/{year}/'
            process_files_iteratively_and_split(
                INPUT_DIRECTORY,
                input_file_pattern_yearly,
                TIMESTAMP_COLUMN,
                output_directory_yearly,
                CHUNK_SIZE
            )
    print(f"End time [{time.strftime('%Y-%m-%d %H:%M:%S')}]\n\n")