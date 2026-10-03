import logging
import re
from pathlib import Path
import pandas as pd


COLS = ['id', 'text', 'epoch', 'media', 'retweetedTweet',
            'lang', 'rawContent','cleanedTweet', 'replyCount', 'retweetCount',
            'likeCount', 'quoteCount', 'hashtags',
            'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name',
            'location', 'user', 'date','week']
START_DATE = pd.to_datetime("2024-05-01")

logger = logging.getLogger(__name__)

def process_df(df):
    try:
        df = df[(df['lang'] == 'en') & (df['retweetedTweet'] is False)]
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= pd.to_datetime('2024-05-01')]
        df['rawContent'] = df['rawContent'].astype(str)
        df = df[df['rawContent'] != ""]
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        df['cleanedTweet'] = (df['cleanedTweet'].astype(str).
                              str.replace(r'(\n)', lambda m: "",regex=True).str.strip())
        df['cleanedTweet'] = (df['cleanedTweet'].
                              str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",
                                          lambda m: "",regex=True))
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r'\s+', ' ', regex=True)
        df['week'] = (df['date'] - START_DATE).dt.days // 7
    except IndexError:
        logging.exception("Error with execution")
    except KeyError:
        logging.exception("Error with execution")

def fix_headers():
    directory = Path("../x-24-us-election")
    #folders = {"part_45","part_46","part_47"}
    files = list(directory.glob("**/*.csv.gz"))
    output_dir = Path("../fixed")
    for file in files:
        try:
            df = pd.read_csv(file, low_memory=False)
            process_df(df)
            name = re.search(r'(.*).csv.gz', file.name).group(1)
            if "inReplyToUser" in df.columns:
                df = df.rename(columns={"inReplyToUser": "in_reply_to_screen_name",
                                        "place": "location"})
            df = df[COLS]
            output_file_path = output_dir / name
            df.to_csv(f"{output_file_path}.csv", index=False)
            logger.info("Successfully processed %s", file.name)
        except IndexError:
            logger.exception("Error with file %s", file)
        except KeyError:
            logger.exception("Error with file %s", file)

if __name__ == '__main__':
    fix_headers()
