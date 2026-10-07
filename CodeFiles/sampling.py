from pathlib import Path
import logging
import pandas as pd



cur_directory = Path(__file__).resolve().parent
directory = cur_directory.parent / "fixed"
output_dir = cur_directory.parent / "sampleddata.csv"
logger = logging.getLogger(__name__)

month_list = ["aug_chunk","may_july",
             "november","october","september"]
def sampling():
    samples =[]

    for month in month_list:
        files = list(directory.glob(f"*{month}*.csv"))
        for file in files:
            try:
                df = pd.read_csv(file, low_memory=False)

                strat = (df.groupby('week', group_keys=False).
                         apply(func=lambda x:
                x.sample(n=min(len(x), 20), random_state=26)))

                samples.append(strat)
                logger.info("Successfully processed %s", file)
            except IndexError:
                logger.exception("Error with file %s", file)
    final_df = pd.concat(samples,ignore_index=True,axis=0)

    final_df.to_csv(output_dir, index=False)

if __name__ == '__main__':
    #madness.fix_headers()
    sampling()
