import re
import sys
import os
import time

# --- Configuration ---
# 1. Input file to be cleaned (the original SQL dump)
INPUT_FILENAME = "C:/Users/jordi/Documents/09miar/Wiki_original_files/eswiki-latest-categorylinks.sql"

# 2. Output file (the one we will feed to SQLite)
OUTPUT_FILENAME = "C:/Users/jordi/Documents/09miar/Wiki_original_files/categorylinks_clean.sql"

# 3. Input Encoding (Changed to 'latin-1' to bypass decoding errors)
# 'latin-1' can decode every possible byte, preventing the crash.
INPUT_ENCODING = 'latin-1'

# 4. Output Encoding (Keep as UTF-8 for modern compatibility)
OUTPUT_ENCODING = 'utf-8'


def clean_for_sqlite(input_filepath, output_filepath):
    """
    Cleans a MySQL dump file for compatibility with SQLite, writing ONLY the INSERT statements.
    It uses aggressive filtering to discard all schema, index, and command lines.
    """
    start_time = time.time()
    total_lines = 0
    insert_lines_count = 0

    print(f"Starting aggressive cleaning process for: {input_filepath}")

    try:
        # We use 'errors=ignore' to handle residual encoding issues without crashing.
        with open(input_filepath, 'r', encoding='utf-8', errors='ignore') as infile, \
                open(output_filepath, 'w', encoding='utf-8') as outfile:

            # --- 1. Define Patterns for DISCARDING and KEEPING ---

            # Pattern to KEEP: Must be a line starting with INSERT INTO
            insert_pattern = re.compile(r'^\s*INSERT\s+INTO', re.IGNORECASE)

            # Patterns to DISCARD: All schema, command, and comment lines
            discard_patterns = [
                re.compile(r'^\s*DROP\s+TABLE', re.IGNORECASE),
                re.compile(r'^\s*CREATE\s+TABLE', re.IGNORECASE),
                re.compile(r'^\s*ALTER\s+TABLE', re.IGNORECASE),
                re.compile(r'^\s*UNLOCK\s+TABLES', re.IGNORECASE),
                re.compile(r'^\s*SET\s+.*', re.IGNORECASE),
                re.compile(r'^\s*/\*!.*?\*/;$'),  # MySQL version comments
                re.compile(r'^\s*--.*$'),  # Standard SQL comments
                re.compile(r'^\s*$', re.IGNORECASE),  # Blank lines
            ]

            # --- 2. Processing Loop (Discard or Keep) ---
            for line in infile:
                total_lines += 1

                # Check if the line should be explicitly discarded (Schema commands)
                if any(p.search(line) for p in discard_patterns):
                    continue

                # Check if the line is an INSERT statement
                if insert_pattern.match(line):
                    # Step 1: Replace MySQL-style escaped single quotes (\') with SQLite-style ('')
                    cleaned_line = line.replace("\\'", "''")

                    # Step 2: Remove backticks from table/column names if they interfere
                    cleaned_line = cleaned_line.replace('`', '')

                    # Step 3: Write the cleaned INSERT line
                    outfile.write(cleaned_line.strip() + '\n')
                    insert_lines_count += 1

                # Any other lines (like remaining metadata) are ignored/discarded.

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except Exception as e:
        print(f"An error occurred during cleaning on line {total_lines}: {e}")
        return

    end_time = time.time()
    print("\n--- Summary ---")
    print(f"Input file lines read: {total_lines}")
    print(f"INSERT statements written (Cleaned): {insert_lines_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Successfully created clean data file: {output_filepath}")


if __name__ == "__main__":
    clean_for_sqlite(INPUT_FILENAME, OUTPUT_FILENAME)