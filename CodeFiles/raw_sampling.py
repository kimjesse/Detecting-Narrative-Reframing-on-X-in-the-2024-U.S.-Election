from pathlib import Path
import logging
import pandas as pd



cur_directory = Path(__file__).resolve().parent
directory = cur_directory.parent / "x-24-us-election"
output_dir = cur_directory.parent / "raw_sample_data.csv"
sample_dir = cur_directory.parent / "sampleddata.csv"
total_dir = cur_directory.parent / "weekly_orig_totals.csv"
logger = logging.getLogger(__name__)

COLS = ['id', 'text', 'epoch', 'media', 'retweetedTweet',
            'lang', 'rawContent','replyCount', 'retweetCount',
            'likeCount', 'quoteCount', 'hashtags',
            'mentionedUsers', 'links', 'viewCount', 'quotedTweet', 'in_reply_to_screen_name',
            'location', 'user', 'date','week']
START_DATE = pd.to_datetime("2024-05-01")

month_list = ["aug_chunk","may_july",
             "november","october","september"]

weekly_counts = []

def sampling():
    samples =[]
    files = list(directory.glob("**/*.csv.gz"))
    for file in files:
        try:
            df = pd.read_csv(file,
                             low_memory=False,
                             parse_dates=["date"]
                             )
            if "inReplyToUser" in df.columns:
                df = df.rename(columns={"inReplyToUser": "in_reply_to_screen_name",
                                    "place": "location"})

            df = df[(df['lang'] == 'en') & (df['retweetedTweet'] != 'True')]
            df = df.drop_duplicates(subset='id', keep='first')
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= pd.to_datetime('2024-05-01')]
            df['rawContent'] = df['rawContent'].astype(str)
            df = df[df['rawContent'] != ""]
            df['week'] = (df['date'] - START_DATE).dt.days // 7
            df = df[COLS]
            weekly_counts.append(df.groupby('week').size())
            strat = (df.groupby('week', group_keys=False).
                     apply(func=lambda x:
            x.sample(n=min(len(x), 20), random_state=26)))
            strat['week'] = (strat['date'] - START_DATE).dt.days // 7
            samples.append(strat)
            logger.info("Successfully processed %s", file)
        except IndexError:
            logger.exception("Error with file %s", file)
    final_df = pd.concat(samples,ignore_index=True,axis=0)
    final_df = final_df.drop_duplicates(subset='id', keep='first')
    final_df.to_csv(output_dir, index=False)
    totals = (pd.concat(weekly_counts)
              .groupby(level=0).sum()
              .rename('weekly_orig_total'))
    totals.to_csv('weekly_orig_totals.csv')
    return final_df
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
if __name__ == '__main__':
    raw_sample = sampling()
    output = process_df(raw_sample)
    output.to_csv(sample_dir, index=False)
