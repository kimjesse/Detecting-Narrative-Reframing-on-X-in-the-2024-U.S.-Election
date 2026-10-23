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
    """This module goes through each folder in the original twitter files
    and preprocesses each file from the original files and creates a new cleaned file
    """
    #Setting up the directories that the script needs to read from and write to.
    files = list(directory.glob("**/*.csv.gz"))
    output_dir = cur_directory.parent / "fixed"
    total_dir = cur_directory.parent / "weekly_orig_totals.csv"
    #This file contains the correct columns when run correctly that the other files will use
    #for correct column alignment.
    exemplar = pd.read_csv("/DataFiles/exemplar_file.csv")
    target_cols = exemplar.columns
    total_df = pd.read_csv(total_dir)
    sampling_scheme = total_df['chunk_sample'].to_dict()

    for file in files:
    #Iterate through each file and try to run the preprocessing steps, if it fails log it and
    #throw an exception.
        try:
            df = pd.read_csv(file,
                             low_memory=False,
                             dtype=object
                             )
            name = re.search(r'(.*).csv.gz', file.name).group(1)
            #If the files have this column then the whole file will be different, s
            #set the column names to the standard names
            if "inReplyToUser" in df.columns:
                df = df.rename(columns={"inReplyToUser": "in_reply_to_screen_name",
                                    "place": "location"})
            df['retweetedTweet'] = df['retweetedTweet'].astype(str)
            #Create a column which determines if the tweet is a retweeted tweet
            is_rt = (df['retweetedTweet'].astype(str).str.lower() == 'true') | \
                    df['rawContent'].astype(str).str.startswith('RT @')
            #Filter out non-english language tweets and retweets.
            df = df[(df['lang'] == 'en') & (~is_rt)]
            # The real tweet ID is stored as text at the end of the url column
            # so we need to extract it
            df["tweet_id"] = df["url"].astype(str).str.extract(r"status/(\d+)")[0]
            df = df.drop_duplicates(subset='tweet_id', keep='first').drop(columns='id')
            df['date'] = pd.to_datetime(df['date'])
            #Filter out columns which are after when the tweets should start.
            df = df[df['date'] >= pd.to_datetime('2024-05-01')]
            df['rawContent'] = df['rawContent'].astype(str)
            df = df[df['rawContent'] != ""]
            #Creating a week column for use in weekly analysis.
            df['week'] = (df['date'] - START_DATE).dt.days // 7
            df = df[COLS]
            #This line builds the weekly_orig_totals.csv file which only needs to be built once.
            #weekly_counts.append(df.groupby('week').size())
            strat = df.groupby('week', group_keys=False).apply(
                lambda x:x.sample(frac=sampling_scheme[x.name], random_state=26)
                           )
            strat['week'] = (strat['date'] - START_DATE).dt.days // 7
            #Once we have sampled the rows and filtered down the data,
            # run the tweet processing function.
            output_file = process_df(strat)
            output_file = output_file.reindex(columns=target_cols)
            output_file_path = output_dir / name
            output_file.to_csv(f"{output_file_path}.csv", index=False)
            logger.info("Successfully processed %s", file)
        except IndexError:
            logger.exception("Error with file %s", file)
    #These lines create the weekly_orig_totals file which only needs to be built once.
    #totals = (pd.concat(weekly_counts)
    #          .groupby(level=0).sum()
    #          .rename('weekly_orig_total'))
    #totals.to_csv(total_dir)
def process_df(df):
    try:
        df['cleanedTweet'] = df['rawContent'].str.normalize('NFKD')
        #removing all instances of the explicit newline symbol and replacing with nothing
        df['cleanedTweet'] = (df['cleanedTweet'].astype(str).
                              str.replace(r'(\n)', lambda m: "", regex=True).str.strip())
        #If there is user being @ in the first 5 characters, remove that
        df['cleanedTweet'] = (df['cleanedTweet'].
                              str.replace(r"^.{0,5}(@\S*\b\s)+|(https:\S*\b)",
                                          lambda m: "", regex=True))
        #The original twitter scrape was inconsistent on contractions being kept
        #The following lines try to correct them
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+s\s+", "'s ", regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+t\s+", "'t ", regex=True)
        df['cleanedTweet'] = df['cleanedTweet'].str.replace(r"\s+u\s+", " you ", regex=True)
        #The text cleaning would leave different whitespace characters, replacing them all with
        #a standard space.
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
    """This function reads each cleaned file and feeds them iteratively into one big file
    while not having to open them all in memory.
    """
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
    """This function takes the consolidated file made in concat_samples() and turns it into
    a parquet.
    """
    try:
        #This implementation streams the data into the parquet file without having to open the
        #entire file in memory
        (pl.scan_csv("F:/Milestone 2/sampleddate.csv",
                    schema_overrides= {"user": pl.String, 'mentionedUsers': pl.String,
                                       'quotedTweet':pl.String,'retweetedTweet':pl.String})
         .drop(['epoch']).sink_parquet("../DataFiles/sampleddata.parquet"))

    except BaseException:
        logger.exception("Error with execution")
        raise
if __name__ == '__main__':
    sampling()
    #make_parquet()
