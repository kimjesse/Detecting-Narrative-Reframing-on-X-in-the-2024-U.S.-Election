import ast
import pandas as pd
import logging
from pathlib import Path
import re
import fastparquet

def get_mentioned_users(df):
    en = df[['mentionedUsers']]
    en['mentionedUsers'] = en['mentionedUsers'].apply(lambda x: ast.literal_eval(x))
    # 1. Explode the list (missing rows are safely kept as NaN)
    exploded_df = en.explode('mentionedUsers').reset_index(drop=True)

    # 2. Fill NaN rows with an empty dict so json_normalize doesn't fail
    dict_series = exploded_df['mentionedUsers'].apply(lambda x: x if isinstance(x, dict) else {})

    # 3. Normalize the dictionaries into columns
    normalized_cols = pd.json_normalize(dict_series)

    # 4. Combine the ID with the extracted values
    #flat_df = pd.concat(exploded_df[exploded_df['id'],normalized_cols],axis=1)
    #flat_df['final_handle'] = flat_df['screen_name'].fillna(flat_df['username'])
    #result = flat_df[[id,'final_handle']]
    #users = result.groupby(['screen_name', 'name']).agg(count=('screen_name', 'size'))
    other_cols = exploded_df.drop(columns=['mentionedUsers'])
    return pd.concat([other_cols, normalized_cols], axis=1)

    #return users



def run_pipeline():
    directory = Path("../x-24-us-election/")
    output_dir = Path("../testing")
    files = list(directory.glob("**/*.csv.gz"))
    #files = list(directory.glob("*.csv.gz"))
    user_list = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        name = re.search(r'(.*).csv.gz', file.name).group(1)
        try:
            en = df[df['lang'] == 'en']
            users = get_mentioned_users(en)
            output_file_path = output_dir / name
            if not users.empty:
                user_list.append(users)

            output_file_path = output_dir / name
            users.to_csv(f"{output_file_path}.csv", index=False)
            logging.info(f"Successfully processed {name}")
        except:
            logging.exception(f"Error with file {name}")

    return user_list



def find_frequency(user_list):
    flats = pd.concat(user_list, axis=0)
    fastparquet.write('outfile.parq', flats)
    flats['handle'] = flats['screen_name'].fillna(flats['username'])
    flats['master_id'] = flats['id_str'].fillna(flats['id'])
    result = flats[['master_id', 'handle']]
    countlist = result.groupby(['handle']).agg(count=('handle', 'size'))
    countlist.to_csv("countlist.csv")

if __name__ == '__main__':
    user_list = run_pipeline()
    find_frequency(user_list)
