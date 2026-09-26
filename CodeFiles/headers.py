import csv
import glob
from pathlib import Path
import pandas as pd
import gzip

def get_headers():
    # Find all CSV files in the target directory
    directory = Path("../x-24-us-election")
    files = list(directory.glob("**/*.csv.gz"))

    # Dictionary to map {filename: list_of_headers}
    all_headers = {}

    for file in files:
        try:
            with gzip.open(file, mode="rt", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                headers = next(reader)  # Reads only the first line
                all_headers[file] = headers
        except (StopIteration, csv.Error):
            all_headers[file] = []  # Handles empty files

    # Print the results
    for file, headers in all_headers.items():
        print(f"{file}: {headers}")
    df = pd.DataFrame.from_dict(all_headers, orient="index")
    df.to_csv("headers.csv", index=False)
    return


def fix_headers():
    directory = Path("../x-24-us-election")
    files = list(directory.glob("**/*.csv.gz"))
    file_list = [pd.read_csv(file, low_memory=False,nrows=1) for file in files]
    combined_df = pd.concat(file_list, ignore_index=True)

    # Export to a new massive CSV file without repeating indices
    combined_df.to_csv("combined_output.csv", index=False)




if __name__ == '__main__':
    fix_headers()

