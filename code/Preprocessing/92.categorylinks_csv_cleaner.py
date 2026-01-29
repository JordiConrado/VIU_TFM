import csv
import os

"""
It gets the categorylinks.csv file from SQLite export and deletes the second column as it is binary"""

def remove_second_column(input_filename, output_filename):
    """
    Reads a CSV file, deletes the second column (index 1), and saves the
    remaining data to a new file.

    Args:
        input_filename (str): The name of the original CSV file.
        output_filename (str): The name for the new, cleaned CSV file.
    """
    print(f"Starting to process '{input_filename}'...")

    # We use 'with open' to ensure files are properly closed, even if errors occur.
    try:
        with open(input_filename, mode='r', newline='', encoding='utf-8') as infile, \
                open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:

            # Create CSV reader and writer objects
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            row_count = 0

            for row in reader:
                # Ensure the row has at least 2 columns before attempting to delete
                if len(row) >= 2:
                    # Deleting the second element (index 1) of the list
                    del row[1]

                # Write the (potentially modified) row to the new file
                writer.writerow(row)
                row_count += 1

                # Print progress for large files
                if row_count % 100000 == 0:
                    print(f"Processed {row_count:,} rows...")

            print(f"Finished processing. Total rows: {row_count:,}")
            print(f"Successfully saved clean data to '{output_filename}'")

    except FileNotFoundError:
        print(f"\nERROR: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # Clean up the output file if the process failed midway
        if os.path.exists(output_filename):
            os.remove(output_filename)
            print(f"Removed incomplete output file: {output_filename}")


# --- Configuration ---
INPUT_FILE = '../../data/02_csv/categorylinks.csv'
OUTPUT_FILE = '../../data/02_csv/categorylinks_clean.csv'

# Run the function
# Note: Ensure you place your 'categorylinks.csv' in the same directory as this script!
remove_second_column(INPUT_FILE, OUTPUT_FILE)