from pathlib import Path
import pandas as pd
import re
import logging
import fastparquet

def clean_tweet(df):
    try:
        df = df[(df['lang'] == 'en') & (df['retweetedTweet'] == False)]
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        df['cleanedTweet'] = df['cleanedTweet'].astype(str).str.replace(r'(\n)', lambda m: "",regex=True).str.strip()
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",lambda m: "",regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r'\s+', ' ', regex=True)
    except:
        logging.exception("Error with cleaning tweet")
    return df

def run_pipeline():
    directory = Path("../x-24-us-election")
    output_dir = Path("../data")
    files = list(directory.glob("**/*.csv.gz"))
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        name = re.search(r'(.*).csv.gz', file.name).group(1)
        try:
            processed_df = clean_tweet(df)
            output_file_path = output_dir / name
            processed_df.to_csv(f"{output_file_path}.csv", index=False)
            logging.info(f"Successfully processed {name}")
        except:
            logging.exception(f"Error with file {name}")
