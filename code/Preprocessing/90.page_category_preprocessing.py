import pandas as pd
import sqlite3
import os
import re

"""
It reads SQL dump files for for page categories and creates a CSV file
"""
# --- Configuration ---

CATEGORY_SQL_FILE = 'C:/Users/jordi/Documents/09miar/Wiki_original_files/eswiki-latest-category.sql'
# Output CSV file names
OUTPUT_CATEGORY_CSV = '../../data/02_csv/category_metadata.csv'
SAVE_DATA = True  # If True, it will save the file to CSV


# --- Step 1: Utility Functions for Reading and Extracting Data ---
def read_file_robustly(filepath):
    """
    Tries multiple encodings to read the SQL file content, prioritizing UTF-8.
    """
    print(f"Attempting to read file: {filepath}")

    # 1. Try to read the file correctly as UTF-8 (default for modern systems).
    try:
        with open(filepath, 'r', encoding='utf-8', errors='strict') as f:
            content = f.read()
        print(f"File successfully read using utf-8 encoding.")
        # Remove null characters which can cause issues
        return content.replace('\x00', '')

    except UnicodeDecodeError:
        # 2. Fall back to Latin-1, which reads every byte and is less strict.
        print(f"UTF-8 failed. Falling back to Latin-1 reading...")
        try:
            with open(filepath, 'r', encoding='latin1', errors='ignore') as f:
                content = f.read()
            print(f"File successfully read using latin1 encoding.")
            return content.replace('\x00', '')

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to decode the file using all methods. Error: {e}")
            raise
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found at {filepath}")


def extract_category_data(file_content):
    """
    Extracts category data tuples from the SQL dump using regex.
    Bypasses all SQL syntax issues.
    """
    print("\n--- Extracting Category Data (cat_id, cat_title, cat_pages, cat_subcats, cat_files) ---")

    # Regex targets: INSERT INTO `category` VALUES (tuple1), (tuple2), ...
    # We want to capture the entire list of tuples
    match = re.search(r'INSERT INTO `?category`? VALUES\s*(\(.*\));', file_content, re.DOTALL)

    if not match:
        raise ValueError("Could not find INSERT INTO statement for 'category' table.")

    raw_values = match.group(1)

    # This regex is specifically designed to split the string by '),(' while handling
    # the opening '(' and closing ');' of the full string.
    # It pulls out the content of each tuple.
    tuple_contents = re.findall(r'\((.*?)\)(?:,\(|\s*;)', raw_values + ');', re.DOTALL)

    data = []

    for content in tuple_contents:
        # Split by comma *outside* of quotes for MySQL's CSV-like format
        # This is a complex step, simplified here by assuming the standard 5 columns
        # (id, title, pages, subcats, files).

        # We replace the commas only when they are NOT inside quotes/strings
        # WARNING: This simple split is fragile for data with unescaped commas in titles!
        # A more robust solution would require a full CSV parser, but this usually works for titles.

        parts = content.split(',', 4)

        if len(parts) == 5:
            try:
                cat_id = int(parts[0].strip())
                cat_title_raw = parts[1].strip().strip("'")  # Remove surrounding quotes
                cat_pages = int(parts[2].strip())
                cat_subcats = int(parts[3].strip())
                # cat_files is parts[4], we often ignore it or simplify

                # We skip lines where the title might be obviously corrupted/empty
                if not cat_title_raw:
                    continue

                data.append({
                    'cat_id': cat_id,
                    'cat_title': cat_title_raw,
                    'cat_pages': cat_pages,
                    'cat_subcats': cat_subcats,
                })
            except ValueError as e:
                # Print the failing tuple content for debugging, then skip
                print(f"Skipping potentially corrupt category tuple: {content[:80]}... Error: {e}")
                continue

    df = pd.DataFrame(data)
    df['cat_title_cleaned'] = df['cat_title'].str.replace('_', ' ')
    print(f"Successfully extracted {len(df)} category records.")
    return df


# --- Step 2: Main Execution ---

if __name__ == "__main__":

    # 1. Read Category File and Extract Data
    try:
        category_content = read_file_robustly(CATEGORY_SQL_FILE)
        df_cat = extract_category_data(category_content)

        if SAVE_DATA:
            print(f"\nExporting {len(df_cat)} rows to {OUTPUT_CATEGORY_CSV}")
            df_cat[['cat_id', 'cat_title', 'cat_pages', 'cat_subcats', 'cat_title_cleaned']].to_csv(OUTPUT_CATEGORY_CSV,
                                                                                                    index=False,
                                                                                                    encoding='utf-8-sig')
            print(f"Success! Category metadata saved to: {OUTPUT_CATEGORY_CSV}")
        else:
            print("\n--- First 50 rows of Category Metadata (df_cat) ---")
            print(df_cat.head(50))
            print("-" * 50)

    except Exception as e:
        print(f"\n--- CRITICAL FAILURE: CATEGORY FILE ---")
        print(f"Could not process category file: {e}")

    print("\nProcessing complete.")
