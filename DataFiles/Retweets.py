from pathlib import Path
import pandas as pd
from glob import glob
import duckdb
import logging
import re
import pytest

from IPython.testing.tools import full_path

COLS = ['id', 'text', 'epoch', 'media', 'retweetedTweet',
            'lang', 'rawContent','cleanedTweet', 'replyCount', 'retweetCount',
            'likeCount', 'quoteCount', 'hashtags',
            'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name',
            'location', 'user', 'date','week']
START_DATE = pd.to_datetime("2024-05-01")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler to write to a file
file_handler = logging.FileHandler("../x-24-us-election/part_47/milestone2.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and tie it to the handler
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def process_df(df):
    try:
        df = df[(df['lang'] == 'en') & (df['retweetedTweet'] == False)]
        df['rawContent'] = df['rawContent'].astype(str)
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        df['cleanedTweet'] = df['cleanedTweet'].astype(str).str.replace(r'(\n)', lambda m: "",regex=True).str.strip()
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",lambda m: "",regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r'\s+', ' ', regex=True)
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = (df['date'] - START_DATE).dt.days // 7
    except:
        logging.error("Error with cleaning tweet")
    return df

def all_df():
    directory = Path("../x-24-us-election")
    #folders = {"part_45","part_46","part_47"}
    files = list(directory.glob("**/*.csv.gz"))
    output_dir = Path("../fixed")
    min_date = pd.to_datetime("2026-05-01")
    for file in files:
        print(f'running {file}')
    #for file in Path(directory).rglob('*.csv.gz'):
    #    if any(part in folders for part in file.parts):
        try:
            df = pd.read_csv(file, low_memory=False)
            df['date'] = pd.to_datetime(df['date'])
            mindate = df['date'].min()
            if (mindate < min_date) and (mindate >= pd.to_datetime("2023-01-01")):
                min_date = mindate
            #rts = df[df['retweetedTweet'] == True]
            print(min_date)
            #if rts.empty:
            #    print('No retweeted tweets')
            #else:
            #    print('retweeted tweets')
        except:
            logging.error(f"Error with file {file.name}")
    return min_date

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler("../x-24-us-election/part_47/milestone2.log")  # Save to file
        ]
    )
    mindate = all_df()
    print(r'Starting date in data is {mindate}')



