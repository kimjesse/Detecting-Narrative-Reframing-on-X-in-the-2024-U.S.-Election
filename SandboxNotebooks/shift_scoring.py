"""
Shared issue tagging and shift scoring (from tk_unsupervised_learning.ipynb).
Used by TK's and Shay's notebooks. TK edits keywords and scoring here directly.

Source: copied from SandboxNotebooks/tk_unsupervised_learning.ipynb (Tanisha).
Functions from commit bb56842. ISSUE_KEYWORDS from her Oct 15 update, commit c2bb1ea. 
Logic and settings are unchanged. The only edits:
  - Settings are pulled out as named constants so every notebook uses the same values.
  - The NMF fit is pulled out into fit_frames() so run_issue() and the noise floor
    fit topics the same way.
  - run_issue() and tag_issues() take df as an argument instead of reading a global df.

Example of how to import in a notebook:
    import sys
    sys.path.append("../SandboxNotebooks")

    from shift_scoring import (
        K, ISSUE_KEYWORDS, tag_issues, filter_issue_tweets, make_issue_vectorizer,
        fit_frames, shift_between, run_issue,
    )

    df = tag_issues(df)
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# UNSUPERVISED SETTINGS

K = 5                      # frames (NMF topics) fit per issue per week
MATCH_THRESHOLD = 0.2      # cosine similarity needed for two frames to count as "the same"
PERIOD_WEEKS = 1
MIN_WORDS = 5              # drop tweets shorter than this (too little text to model)
MIN_TWEETS_PER_FRAME = 10  # a week needs K * this many tweets to be scored (5 * 10 = 50)
RANDOM_STATE = 42
MAX_ITER = 500

# Typical stop words plus common filler words
CUSTOM_STOPS = list(ENGLISH_STOP_WORDS.union(
    {"amp", "don", "didn", "just", "like", "did", "know", "let", "said", "people"}
))



ISSUE_KEYWORDS = {
    "immigration":    r"immigra|border|migrant|asylum|deport|illegals|illegal alien|refugee",
    "economy":        r"econom|inflation|prices|jobs|tax|wages|gas price|groceries",
    "abortion": r"abortion|\broe\b|roevember|restoreroe|pro.?life|pro.?choice|reproductive|ivf",
    "democracy":      r"democracy|january 6|jan 6|insurrection|stolen election|fascist",
    "foreign_policy": r"ukraine|russia|israel|gaza|china|\bnato\b",
}



# ISSUE TAGGING FUNCTIONS (keyword patterns)

def tag_issues(df, text_col="cleanedTweet"):
    """Add one True/False column per issue, named issue_<name>.
    A tweet can match more than one issue."""
    text = df[text_col].fillna("").str.lower()
    for issue, pattern in ISSUE_KEYWORDS.items():
        df[f"issue_{issue}"] = text.str.contains(pattern, regex=True)
    return df


def filter_issue_tweets(df, issue, text_col="cleanedTweet", min_words=MIN_WORDS):
    """Return only tweets tagged with this issue that have at least min_words words.
    Requires tag_issues() to have been run on df first."""
    issue_tweets = df[df[f"issue_{issue}"]]
    return issue_tweets[issue_tweets[text_col].str.split().str.len() >= min_words]


def make_issue_vectorizer():
    """TF-IDF settings used for every issue. Fit it on ALL of one issue's tweets,
    so every week of that issue shares one vocabulary and weeks can be compared."""
    return TfidfVectorizer(stop_words=CUSTOM_STOPS, max_df=0.95, min_df=2)



# FRAME FITTING, MATCHING, AND SHIFT SCORING FUNCTIONS

def fit_frames(x_rows, k=K):
    """Fit one NMF on a group of tweets (one week, or one half of a week).
    Returns what shift_between needs:
      H     = each frame's word weights
      props = each frame's share of the group's tweets"""
    nmf = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=MAX_ITER)
    w = nmf.fit_transform(x_rows)
    return {"H": nmf.components_, "props": w.sum(axis=0) / w.sum()}


def match_topics(h_a, h_b, threshold=MATCH_THRESHOLD):
    """Pair each frame in A with its most similar frame in B (cosine similarity of
    word weights; 1 = identical, 0 = no overlap). Pairs below the threshold are
    'unmatched': a frame that appeared or disappeared."""
    sim = cosine_similarity(h_a, h_b)
    rows, cols = linear_sum_assignment(-sim)   # best one-to-one pairing
    return [(a, b, sim[a, b], sim[a, b] >= threshold) for a, b in zip(rows, cols)]


def shift_between(frames_a, frames_b, threshold=MATCH_THRESHOLD):
    """Score how much framing changed from A to B. 
    All scores: 0 = no change, 1 = complete change."""
    pairs = match_topics(frames_a["H"], frames_b["H"], threshold)

    # Content shift: how much matched frames changed their wording
    matched = [s for _, _, s, ok in pairs if ok]
    content_shift = np.mean([1 - s for s in matched]) if matched else 1.0

    # Proportion shift: how much attention moved between frames (JS distance, base 2)
    a_idx = [a for a, _, _, _ in pairs]
    b_idx = [b for _, b, _, _ in pairs]
    proportion_shift = jensenshannon(frames_a["props"][a_idx], frames_b["props"][b_idx], base=2)

    return {
        "content_shift": round(content_shift, 3),
        "proportion_shift": round(proportion_shift, 3),
        "new_frames": sum(1 for _, _, _, ok in pairs if not ok),
        "shift_score": round((content_shift + proportion_shift) / 2, 3),
    }

def run_issue(df, issue, period_weeks=PERIOD_WEEKS, k=K, threshold=MATCH_THRESHOLD):
    """Score every week of one issue against the week before it (TK's run_issue).

    Returns one row per period, scoring the move into that period from the
    previous scored period. The first period has nothing to compare to, so it
    gets no row. Periods with fewer than k * MIN_TWEETS_PER_FRAME tweets are skipped.
    """
    # This issue's tweets with 5+ words, each assigned to a period (1 period = 1 week)
    issue_tweets = filter_issue_tweets(df, issue).reset_index(drop=True)
    issue_tweets["period"] = issue_tweets["week"] // period_weeks

    # One shared vocabulary for the whole issue, so different weeks can be compared
    x_issue = make_issue_vectorizer().fit_transform(issue_tweets["cleanedTweet"])

    # Fit frames for each period that has enough tweets
    issue_models = {}
    for period, rows in sorted(issue_tweets.groupby("period").indices.items()):
        if len(rows) < k * MIN_TWEETS_PER_FRAME:
            continue
        issue_models[period] = fit_frames(x_issue[rows], k)

    # Score each scored period against the one before it
    fitted = sorted(issue_models)
    results = [
        {"period": p_b, "from_period": p_a,
         **shift_between(issue_models[p_a], issue_models[p_b], threshold)}
        for p_a, p_b in zip(fitted[:-1], fitted[1:])
    ]
    columns = ["period", "from_period", "content_shift", "proportion_shift",
               "new_frames", "shift_score"]
    return pd.DataFrame(results, columns=columns)
