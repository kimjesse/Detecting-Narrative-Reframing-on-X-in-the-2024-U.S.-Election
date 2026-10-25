"""
Make the label packets.

A packet is what a labeler reads to decide if an issue's story changed
from one week to the next. One packet = one issue in one week (an
issue-week). It shows up to 30 random tweets about the issue from last
week and up to 30 from this week. The labeler (LLM and Shay) compares
the two lists and gives a label: the right answer the model learns from
(1 = narrative shift, 0 = no shift).

The labeler never sees the week number, dates, or any scores to prevent leakage.

How to run (from the repo folder):
    python CodeFiles/make_packets.py

Outputs (in DataFiles/labeling/):
    packets/row_001.md ..one packet per issue-week
    packets.jsonl        the same packets in a file label_with_llm_simple.py can read
    answer_key.csv       which row_id is which issue and week
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# File locations

REPO = Path(__file__).resolve().parent.parent
SAMPLE_FILE = REPO / "DataFiles" / "sampleddata.parquet"
OUT_FOLDER = REPO / "DataFiles" / "labeling"


sys.path.append(str(REPO / "CodeFiles"))
from shift_scoring import (  # pylint: disable=wrong-import-position
    ISSUE_KEYWORDS, filter_issue_tweets, tag_issues,
)


# Settings

SEED = 42
TWEETS_PER_SIDE = 30
MIN_TWEETS = 10       # fewer than this on either side = too few to judge


def pick_tweets(df, issue, issue_number, week):
    """Randomly pick up to 30 tweets about one issue in one week.

    Each issue-week gets its own seed, built from its issue number and week.
    That way, changing one issue's tweets never changes another issue's picks.
    """
    tweets = filter_issue_tweets(df, issue)
    tweets = tweets[tweets["week"] == week]["cleanedTweet"].tolist()

    if len(tweets) <= TWEETS_PER_SIDE:
        return tweets

    rng = np.random.default_rng([SEED, issue_number, week])
    chosen = rng.choice(len(tweets), size=TWEETS_PER_SIDE, replace=False)
    return [tweets[i] for i in sorted(chosen)]


def write_packet(packet):
    """Save one packet as a markdown file you can read in VS Code
    (right-click the file, then choose Open Preview)."""
    lines = [f"# {packet['row_id']}: {packet['issue'].replace('_', ' ')}", ""]
    if packet["too_thin"]:
        lines += ["> Too few tweets on one side. Mark this row **unsure**.", ""]

    lines += ["## Last week", ""]
    lines += [f"{n}. {tweet}" for n, tweet in enumerate(packet["before_tweets"], start=1)]
    lines += ["", "## This week", ""]
    lines += [f"{n}. {tweet}" for n, tweet in enumerate(packet["after_tweets"], start=1)]

    file_path = OUT_FOLDER / "packets" / f"{packet['row_id']}.md"
    file_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Build one packet per issue-week and save them."""
    # Stop if the packets were already built so they can't change by accident.
    # (To rebuild, delete the labeling folder first.)
    if (OUT_FOLDER / "answer_key.csv").exists():
        print("STOP: packets already exist. Delete DataFiles/labeling/ to rebuild.")
        return

    # 1. Load the tweets and tag each tweet's issues with the shared function
    df = pd.read_parquet(SAMPLE_FILE, columns=["cleanedTweet", "week"])
    df = tag_issues(df)

    # 2. Build one packet per issue-week. The issues come from the shared
    #    keyword list. Week 0 is skipped because it has no "last week".
    issues = sorted(ISSUE_KEYWORDS)
    weeks = sorted(df["week"].unique())[1:]
    packets = []
    for issue_number, issue in enumerate(issues):
        for week in weeks:
            before = pick_tweets(df, issue, issue_number, week - 1)
            after = pick_tweets(df, issue, issue_number, week)
            packets.append({
                "issue": issue,
                "week": week,
                "before_tweets": before,
                "after_tweets": after,
                "too_thin": min(len(before), len(after)) < MIN_TWEETS,
            })
    packets = pd.DataFrame(packets)

    # 3. Give each packet a random ID (row_001, row_002, ...) so the labeler
    #    reads them in mixed-up order and can't guess the week from the order.
    #    The shuffle uses the fixed seed, so each issue-week keeps the same ID
    #    every time the script runs.
    shuffled_order = np.random.default_rng(SEED).permutation(len(packets))
    packets["row_id"] = [f"row_{n + 1:03d}" for n in shuffled_order]
    packets = packets.sort_values("row_id")

    # 4. Save
    (OUT_FOLDER / "packets").mkdir(parents=True, exist_ok=True)
    for _, packet in packets.iterrows():
        write_packet(packet)

    # The LLM's file leaves out the week so there's no label leakage.
    llm_columns = ["row_id", "issue", "too_thin", "before_tweets", "after_tweets"]
    packets[llm_columns].to_json(OUT_FOLDER / "packets.jsonl", orient="records", lines=True)

    # The answer key is the only file that links a row_id back to its week.
    packets[["row_id", "issue", "week"]].to_csv(OUT_FOLDER / "answer_key.csv", index=False)

    print(f"Made {len(packets)} packets. Too thin to label: {packets['too_thin'].sum()}")


if __name__ == "__main__":
    main()
