import pandas as pd
from pathlib import Path
import logging

directory = Path("fixed")

monthlist = ["aug_chunk","may_july",
             "november","october","september"]
def sampling():
    samples =[]

    for month in monthlist:
        files = list(directory.glob(f"*{month}*.csv"))
        for file in files:
            try:
                df = pd.read_csv(file, low_memory=False)

                strat = df.groupby('week', group_keys=False).apply(lambda x: x.sample(n=min(len(x),20), random_state=26))

                samples.append(strat)
                logging.info(f"Successfully processed {file.name}")
            except:
                logging.error(f"Error with file {file.name}")
    final_df = pd.concat(samples,ignore_index=True,axis=0)

    final_df.to_csv("./fixed/working_sample.csv")
    return

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # Print to console
            logging.FileHandler("milestone2.log")  # Save to file
        ]
    )
    sampling()