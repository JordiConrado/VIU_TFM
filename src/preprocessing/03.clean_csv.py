import pandas as pd
import os
import glob
from typing import Set

"""
    1) Removes bot activity from CSV files containing wiki revision history.
    Uses a dedicated CSV file with bot usernames to filter out bot edits
    from all other CSV files in a specified directory.
    
    2) Removes anonymous user edits based on a predefined list of anonymous IDs.
    
    Select the year to process by changing the YEAR variable below.

    Returns new CSV files with '_clean' appended to the original filenames.
    These files will exclude any revisions made by bot users.
    Are located in the same directory as the input files.
    
Change Log:
    2025-10-29: Initial creation date
    2024-06-23: Added summary statistics after processing all files.
    2025-11-03: Add Anonymous user ID removal feature.
    2025-11-06: Added multi-year processing capability.
"""

USER_NAME_COLUMN = 'user_name'  # Ensure this matches the column name in your edit history files
USER_ID_COLUMN = 'user_id'  # Column name for user IDs
ANONYMOUS_ID_TO_REMOVE = {'Anonymous_ID'}  # Set of anonymous user IDs to remove, if any
BOT_LIST_FILENAME = 'C:/Users/jordi/Documents/09miar/additional_data/ewiki_bot_users.csv'

def load_bot_usernames(bot_file_path: str) -> Set[str]:
    """Loads bot usernames from the dedicated CSV file and cleans the names."""
    print(f"Loading bot usernames from: {bot_file_path}")
    try:
        # Read the CSV. The 'Bot_Username' column is the first one.
        df_bots = pd.read_csv(bot_file_path, header=0, usecols=['Bot_Username'], keep_default_na=False)

        # Convert the column to a set for fast lookup
        bot_names = set(df_bots['Bot_Username'].str.strip().tolist())

        print(f"Loaded {len(bot_names)} unique bot user names.")
        return bot_names

    except FileNotFoundError:
        print(f"Error: Bot list file not found at {bot_file_path}")
        return set()
    except Exception as e:
        print(f"An error occurred while loading the bot list: {e}")
        return set()


def process_directory(directory: str, bot_usernames: Set[str]):
    """Processes all CSV files in a directory, filters out bot edits, and saves new files."""
    if not bot_usernames:
        print("Bot list is empty. Aborting file processing.")
        return

    # Global counters
    total_revisions_before = 0
    total_revisions_after = 0
    bot_revisions_deleted = 0
    anonymous_revisions_deleted = 0

    # Find all CSV files in the directory that are not the bot list itself,
    # and do not already contain '_NoBOT'.
    search_path = os.path.join(directory, '*.csv')
    all_files = glob.glob(search_path)

    files_to_process = [
        f for f in all_files
        if os.path.basename(f) != BOT_LIST_FILENAME and '_NoBOT' not in os.path.basename(f)
    ]

    if not files_to_process:
        print(f"No new CSV files found in '{directory}' to process.")
        return

    print(f"\nFound {len(files_to_process)} files to process.")

    for input_file_path in files_to_process:
        try:
            input_filename = os.path.basename(input_file_path)

            # 1. Read the input data file
            df_input = pd.read_csv(input_file_path, keep_default_na=False, encoding='utf-8-sig')

            # 2. Check if the required column exists
            if USER_NAME_COLUMN not in df_input.columns:
                print(f"  Skipping {input_filename}: Column '{USER_NAME_COLUMN}' not found.")
                continue

            # 3. Filter the data: Keep rows where 'user_name' is NOT in the bot_usernames set
            initial_count = len(df_input)
            df_no_bots = df_input[~df_input[USER_NAME_COLUMN].isin(bot_usernames)]
            bots_removed = initial_count - len(df_no_bots)

            # Additionally remove anonymous user edits if specified
            df_filtered = df_no_bots[~df_no_bots[USER_ID_COLUMN].isin(ANONYMOUS_ID_TO_REMOVE)]
            anonymous_removed = len(df_no_bots) - len(df_filtered)

            #records_removed = initial_count - len(df_filtered)
            records_removed = bots_removed + anonymous_removed

            # 4. Construct the output filename (inserting '_NoBOT' before the extension)
            base, ext = os.path.splitext(input_filename)
            output_filename = f"{base}_clean{ext}"
            output_file_path = os.path.join(directory, output_filename)

            # 5. Save the filtered DataFrame
            df_filtered.to_csv(output_file_path, index=False, encoding='utf-8-sig')

            print(f"  Processed: {input_filename}")
            print(f"    - Original records: {initial_count:,}")
            print(f"    - Bot records removed: {bots_removed:,}")
            print(f"    - Anonymous records removed: {anonymous_removed:,}")
            print(f"    - Saved to: {output_filename}")
            total_revisions_before += initial_count
            total_revisions_after += len(df_filtered)
            bot_revisions_deleted += bots_removed
            anonymous_revisions_deleted += anonymous_removed

        except Exception as e:
            print(f"  An error occurred while processing {input_filename}: {e}")
            continue

    print("\nProcessing complete.")
    print('Number of files processed:', len(files_to_process))
    print(f"Total revisions before removals: {total_revisions_before:,}")
    print(f"Total revisions after removals: {total_revisions_after:,}")
    print(f"Total bot activity records removed across all files: {bot_revisions_deleted:,}")
    print(f"Total anonymous activity records removed across all files: {anonymous_revisions_deleted:,}")


if __name__ == '__main__':
    multi_year = True  # Set to True if processing multiple years at once
    start_year = 2006  # Starting year if multi_year is True
    end_year = 2025    # Ending year if multi_year is True

    YEAR = '2005'   # only if single year processing

    # Ensure the script starts execution from the directory where the files are located
    # This assumes the script and all your CSVs are in the same directory.
    if multi_year:
        for year in range(start_year, end_year + 1):
            DATA_DIRECTORY = f'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/Monthly_Splits/{year}'
            print(f"\n--- Processing Year: {year} ---")
            os.chdir(DATA_DIRECTORY)

            # 1. Load the list of bot user names
            bot_usernames = load_bot_usernames(BOT_LIST_FILENAME)

            # 2. Process all data files in the directory
            process_directory(DATA_DIRECTORY, bot_usernames)
    else:
        DATA_DIRECTORY = f'C:/Users/jordi/Documents/09miar/Split_CSV_Revisions/20251021_version/Monthly_Splits/{YEAR}'
        os.chdir(DATA_DIRECTORY)

        # 1. Load the list of bot user names
        bot_usernames = load_bot_usernames(BOT_LIST_FILENAME)

        # 2. Process all data files in the directory
        process_directory(DATA_DIRECTORY, bot_usernames)
