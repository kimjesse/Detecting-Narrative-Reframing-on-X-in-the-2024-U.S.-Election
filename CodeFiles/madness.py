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
file_handler = logging.FileHandler("../milestone2.log")
file_handler.setLevel(logging.INFO)

# Create a formatter and tie it to the handler
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


def process_df(df):
    try:
        df = df[(df['lang'] == 'en') & (df['retweetedTweet'] == False)]
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= pd.to_datetime('2024-05-01')]
        df['rawContent'] = df['rawContent'].astype(str)
        df = df[df['rawContent'] != ""]
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        df['cleanedTweet'] = df['cleanedTweet'].astype(str).str.replace(r'(\n)', lambda m: "",regex=True).str.strip()
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",lambda m: "",regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r'\s+', ' ', regex=True)
        df['week'] = (df['date'] - START_DATE).dt.days // 7
    except:
        logging.error("Error with cleaning tweet")
    return df

def fix_headers():
    directory = Path("../x-24-us-election")
    #folders = {"part_45","part_46","part_47"}
    files = list(directory.glob("**/*.csv.gz"))
    output_dir = Path("../fixed")
    for file in files:
    #for file in Path(directory).rglob('*.csv.gz'):
    #    if any(part in folders for part in file.parts):
        try:
            df = pd.read_csv(file, low_memory=False)
            df = process_df(df)
            name = re.search(r'(.*).csv.gz', file.name).group(1)
            if "inReplyToUser" in df.columns:
                df = df.rename(columns={"inReplyToUser": "in_reply_to_screen_name", "place": "location"})
            df = df[COLS]
            output_file_path = output_dir / name
            df.to_csv(f"{output_file_path}.csv", index=False)
            logging.info(f"Successfully processed {file.name}")
        except:
            logging.error(f"Error with file {file.name}")
    return

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler("../milestone2.log")  # Save to file
        ]
    )
    fix_headers()



