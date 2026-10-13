from pathlib import Path
import logging
import re
import pandas as pd
import polars as pl

cur_directory = Path(__file__).resolve().parent
directory = cur_directory.parent / "x-24-us-election"
logger = logging.getLogger(__name__)

COLS = ['tweet_id', 'text', 'epoch', 'media', 'retweetedTweet',
            'lang', 'rawContent','replyCount', 'retweetCount',
            'likeCount', 'quoteCount', 'hashtags',
            'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name',
            'user', 'date','week']
START_DATE = pd.to_datetime("2024-05-01")

month_list = ["aug_chunk","may_july",
             "november","october","september"]

#weekly_counts = []

def sampling():
    files = list(directory.glob("**/*.csv.gz"))
    output_dir = cur_directory.parent / "fixed"
    total_dir = cur_directory.parent / "weekly_orig_totals.csv"
    exemplar = pd.read_csv("F:/Milestone 2/exemplar_file.csv")
    target_cols = exemplar.columns
    total_df = pd.read_csv(total_dir)
    sampling_scheme = total_df['chunk_sample'].to_dict()

    for file in files:
        try:
            df = pd.read_csv(file,
                             low_memory=False,
                             dtype=object
                             )
            name = re.search(r'(.*).csv.gz', file.name).group(1)
            if "inReplyToUser" in df.columns:
                df = df.rename(columns={"inReplyToUser": "in_reply_to_screen_name",
                                    "place": "location"})
            df['retweetedTweet'] = df['retweetedTweet'].astype(str)
            is_rt = (df['retweetedTweet'].astype(str).str.lower() == 'true') | \
                    df['rawContent'].astype(str).str.startswith('RT @')
            df = df[(df['lang'] == 'en') & (~is_rt)]
            # The real tweet ID is stored as text at the end of the url column
            # so we need to extract it
            df["tweet_id"] = df["url"].astype(str).str.extract(r"status/(\d+)")[0]
            df = df.drop_duplicates(subset='tweet_id', keep='first').drop(columns='id')
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= pd.to_datetime('2024-05-01')]
            df['rawContent'] = df['rawContent'].astype(str)
            df = df[df['rawContent'] != ""]
            df['week'] = (df['date'] - START_DATE).dt.days // 7
            df = df[COLS]
            #weekly_counts.append(df.groupby('week').size())
            strat = df.groupby('week', group_keys=False).apply(
                lambda x:x.sample(frac=sampling_scheme[x.name], random_state=26)
                           )
            strat['week'] = (strat['date'] - START_DATE).dt.days // 7
            output_file = process_df(strat)
            output_file = output_file.reindex(columns=target_cols)
            output_file_path = output_dir / name
            output_file.to_csv(f"{output_file_path}.csv", index=False)
            logger.info("Successfully processed %s", file)
        except IndexError:
            logger.exception("Error with file %s", file)
    #totals = (pd.concat(weekly_counts)
    #          .groupby(level=0).sum()
    #          .rename('weekly_orig_total'))
    #totals.to_csv(total_dir)
def process_df(df):
    try:
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        df['cleanedTweet'] = (df['cleanedTweet'].astype(str).
                              str.replace(r'(\n)', lambda m: "", regex=True).str.strip())
        df['cleanedTweet'] = (df['cleanedTweet'].
                              str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",
                                          lambda m: "", regex=True))
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+s\s+", "'s ", regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+t\s+", "'t ", regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+u\s+", " you ", regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r'\s+', ' ', regex=True)
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = (df['date'] - START_DATE).dt.days // 7
        return df

    except IndexError:
        logging.exception("Error with execution")
        return pd.DataFrame(["Error with execution"])

    except KeyError:
        logging.exception("Error with execution")
        return pd.DataFrame(["Error with execution"])

def concat_samples():
    samples_dir = cur_directory.parent / "fixed"
    csv_files=list(samples_dir.glob("*.csv"))
    output_file = cur_directory.parent / "sampleddate.csv"

    with open(output_file, "w", encoding="utf-8") as outfile:
        keep_header = True

        for filename in csv_files:
            with open(filename, "r", encoding="utf-8") as infile:
                # Read the header row of the current file
                header = infile.readline()

                # Write the header only once for the very first file
                if keep_header:
                    outfile.write(header)
                    keep_header = False

                # Stream the rest of the lines efficiently [0.15]
                for line in infile:
                    outfile.write(line)
def make_parquet():
    try:
        (pl.scan_csv("F:/Milestone 2/sampleddate.csv",
                    schema_overrides= {"user": pl.String, 'mentionedUsers': pl.String,
                                       'quotedTweet':pl.String,'retweetedTweet':pl.String})
         .drop(['epoch']).sink_parquet("../sampleddata.parquet"))

    except BaseException:
        logger.exception("Error with execution")
        raise
if __name__ == '__main__':
    sampling()
    #make_parquet()
