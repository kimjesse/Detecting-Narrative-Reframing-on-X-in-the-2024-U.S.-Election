import logging
import sys
from pathlib import Path
import pandas as pd

START_DATE = pd.to_datetime("2024-05-01")

def all_df():
    directory = Path("../x-24-us-election")
    files = list(directory.glob("**/*.csv.gz"))
    min_date = pd.to_datetime("2026-05-01")
    for file in files:
        print(f'running {file}')
        try:
            df = pd.read_csv(file, low_memory=False)
            df['date'] = pd.to_datetime(df['date'])
            sheet_min = df['date'].min()
            if pd.to_datetime("2023-01-01") <= sheet_min < min_date:
                min_date = sheet_min
            print(min_date)
        except IndexError:
            logging.exception("Error with file %s", file)
        except KeyError:
            logging.exception("Error with file %s", file)
    return min_date

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),  # Print to console
            logging.FileHandler("../x-24-us-election/part_47/milestone2.log")  # Save to file
        ]
    )
    data_min = all_df()
    print(r'Starting date in data is {data_min}')
