"""
This is a placeholder for the supervised pipeline using fake data.

Purpose: To test the pipeline from feature table -> models -> results before the 
real feature table is available. Will swap with the real feature table, real N_TOPICS,
real issues, and noise floor when they're ready. Everything else should run unchanged.
"""

# %%
# SECTION 1: SETUP

# This is needed because our repo's GitHub check runs Python 3.8, which crashes on
# type hints (labels like list[str] that say what kind of value a function expects).
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, 
)


# The fake data will come from random numbers. setting random seed for reproducibility.
RANDOM_SEED = 42

N_WEEKS = 31
N_TOPICS = 5
FAKE_ISSUES = ['economy','immigration', 'democracy', 'abortion']
NOISE_FLOOR =0.05




# %%
# SECTION 2: FAKE FEATURE TABLE AND PREP

def make_fake_feature_table(
    n_weeks: int = N_WEEKS,
    issues: list[str] = FAKE_ISSUES,
    n_topics: int = N_TOPICS,
    noise_floor: float = NOISE_FLOOR,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Fake table. One row = one issue-week. Shift weeks (label = 1) get 
    bigger JS divergence."""

    rng = np.random.default_rng(random_state)
    week_starts = pd.date_range("2024-05-06", periods=n_weeks, freq="W-MON")

    # Every issue paired with every week. sorted by issue then week
    df = pd.MultiIndex.from_product(
        [issues, week_starts], names=["issue", "week_start"]
    ).to_frame(index=False)
    n_rows = len(df)

    # Label first: about 15% of issue-weeks are shifts
    df["label"] = (rng.random(n_rows) < 0.15).astype(int)

    # JS divergence: shift weeks are usually higher. The ranges overlap a little.
    df["js_divergence"] = np.where(
        df["label"] == 1, rng.uniform(0.05, 0.30, n_rows), rng.uniform(0.00, 0.10, n_rows)
    )
    df.loc[df["week_start"] == week_starts[0], "js_divergence"] = np.nan  # week 1 has no "last week"
    df["js_minus_noise"] = df["js_divergence"] - noise_floor

    # z-score vs. this issue's past weeks
    # We'll say it needs at least 3 past weeks to be meaningful.
    past_js = df.groupby("issue")["js_divergence"].shift(1)
    past_mean = past_js.groupby(df["issue"]).transform(lambda s: s.expanding(min_periods=3).mean())
    past_std = past_js.groupby(df["issue"]).transform(lambda s: s.expanding(min_periods=3).std())
    df["js_zscore"] = (df["js_divergence"] - past_mean) / past_std

    # Volume: weekly_orig_total is one number per week (all issues combined)
    # orig_tweet_count = that total x this issue's share of the week (made up: 5%-25%)
    weekly_totals = pd.Series(rng.integers(100_000, 400_000, n_weeks), index=week_starts)
    df["weekly_orig_total"] = df["week_start"].map(weekly_totals)
    df["orig_tweet_count"] = (df["weekly_orig_total"] * rng.uniform(0.05, 0.25, n_rows)).round()

    # Topic shares: random numbers that add up to 1 in each row (Dirichlet distribution)
    shares = rng.dirichlet(np.ones(n_topics), size=n_rows)
    for t in range(n_topics):
        df[f"topic_share_top{t + 1}"] = shares[:, t]

    return df


def prepare_table(df: pd.DataFrame) -> pd.DataFrame:
    """Steps that run on BOTH the fake and the real table.
    This is the swap point: later, pass in the real parquet instead."""

    df = df.sort_values(["issue", "week_start"]).copy()

    # week_idx (0, 1, 2, ...) made from week_start, so the real table doesn't need it
    df["week_idx"] = df["week_start"].rank(method="dense").astype(int) - 1

    # Lag features: this issue's JS from 1 and 2 weeks earlier.
    # groupby("issue") keeps each issue's history separate.
    df["lag1_js"] = df.groupby("issue")["js_divergence"].shift(1)
    df["lag2_js"] = df.groupby("issue")["js_divergence"].shift(2)
    return df


# %% Test Section 2
df = prepare_table(make_fake_feature_table())
# To use the real table later:  df = prepare_table(pd.read_parquet("feature_table.parquet"))

# The first few weeks of each issue have no history (no lags / z-score). Can't be used.
rows_before = len(df)
df = df.dropna(subset=["js_divergence", "js_zscore", "lag1_js", "lag2_js"]).reset_index(drop=True)

print(f"{len(df)} rows kept, {rows_before - len(df)} early rows dropped")
print(f"Shift rate: {df['label'].mean():.1%}")
print(df.groupby("label")["js_divergence"].mean().round(3))  # shifts should be higher



# %%
# SECTION 3: WEEK-BASED CROSS-VALIDATION FOLDS
# https://scikit-learn.org/stable/modules/cross_validation.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

def week_based_folds(
    df: pd.DataFrame, n_splits: int = 5, gap: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_rows, test_rows) pairs, one per fold.
    Each fold trains on earlier weeks and tests on later weeks."""

    # Get the unique weeks, in time order
    unique_weeks = np.sort(df["week_idx"].unique())

    # TimeSeriesSplit splits the weeks (not the rows)
    # Training comes before testing. training window grows each fold.
    # gap=1 skips one week between train and test, so neighboring weeks
    # don't make the test too easy (b/c they tend to look alike)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    folds = []
    for fold_num, (train_pos, test_pos) in enumerate(tscv.split(unique_weeks), start=1):
        train_weeks = unique_weeks[train_pos]
        test_weeks = unique_weeks[test_pos]

        # Turn weeks back into rows (every issue in those weeks)
        train_rows = df.index[df["week_idx"].isin(train_weeks)].to_numpy()
        test_rows = df.index[df["week_idx"].isin(test_weeks)].to_numpy()

        # Warn if a test fold has no shifts b/c F1 and recall can't be calculated
        n_test_shifts = df.loc[test_rows, "label"].sum()
        if n_test_shifts == 0:
            print(f"WARNING: fold {fold_num} has no shifts in its test weeks")

        folds.append((train_rows, test_rows))

    return folds


# %% Test Section 3
folds = week_based_folds(df)

for fold_num, (train_rows, test_rows) in enumerate(folds, start=1):
    train_weeks = df.loc[train_rows, "week_idx"]
    test_weeks = df.loc[test_rows, "week_idx"]
    print(
        f"Fold {fold_num}: train weeks {train_weeks.min()}-{train_weeks.max()} "
        f"({len(train_rows)} rows), test weeks {test_weeks.min()}-{test_weeks.max()} "
        f"({len(test_rows)} rows, {df.loc[test_rows, 'label'].sum()} shifts)"
    )






# %%
# SECTION 4: MODELS AND BASELINES
# Columns the models learn from. Everything else (week_start, week_idx, issue, label)
# is an ID or the answer, not a feature.
FEATURES = [
    "weekly_orig_total",
    "orig_tweet_count",
    "js_divergence",
    "js_minus_noise",
    "js_zscore",
    "lag1_js",
    "lag2_js",
] + [f"topic_share_top{t + 1}" for t in range(N_TOPICS)]


# https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
def make_models(random_state: int = RANDOM_SEED) -> dict:
    """Return the three model families and the 'always no shift' baseline, """

    models = {
        # Baseline 1: always predicts the most common class (0 = no shift)
        "Baseline: always no shift": DummyClassifier(strategy="most_frequent"),

        # Family 1: LR -probabilistic and linear
        # Features come in very different sizes (tweet counts in the 100,000s, JS under 1).
        # StandardScaler converts them all to the same units so big numbers don't dominate.
        # Pipeline calculates that scaling from training weeks only sotest weeks stay unseen.
        # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]),

        # Family 2: GBT - trees don't need scaling.
        # HistGradientBoostingClassifier is sklearn's newer boosted-trees model. Unlike
        # GradientBoostingClassifier it accepts class_weight="balanced" directly, so shift rows
        # count more during fitting (no separate sample_weight step needed).
        # min_samples_leaf = the fewest rows allowed in each branch of a tree.
        # The default 20 is too big for our small folds (fold 1 has only 24 training rows)
        # so the trees couldn't split at all. 5 lets them learn. **NOTE to revisit this when tuning.**
        # https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
        "Gradient-Boosted Trees": HistGradientBoostingClassifier(
            max_depth=2, max_iter=100, learning_rate=0.1,
            class_weight="balanced", min_samples_leaf=5, random_state=random_state,
        ),

        # Family 3: SVM -non-probabilistic, kernel-based
        # https://scikit-learn.org/stable/modules/svm.html
        # https://scikit-learn.org/stable/glossary.html#term-class_weight
        # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        "SVM (RBF kernel)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight="balanced")),
        ]),
    }
    return models


def js_baseline_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Baseline 2: identify the weeks with the biggest JS increases.
    If 15% of training weeks are shifts, identify the top 15%.
"""

    train_shift_rate = train_df["label"].mean()
    cutoff = train_df["js_divergence"].quantile(1 - train_shift_rate)

    scores = test_df["js_divergence"].to_numpy()
    predictions = (scores >= cutoff).astype(int)
    return predictions, scores





# %%
# SECTION 5: RUN EVERY MODEL ON EVERY FOLD AND GET RESULTS
# https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

def shift_scores(model, X_test: pd.DataFrame) -> np.ndarray:
    """How strongly the model thinks each row is a shift (higher = more likely).
    Needed for PR-AUC. SVM has no probabilities, so we use its decision_function
    (distance from the boundary), which ranks rows just as well."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)


METRICS = ["precision", "recall", "f1", "pr_auc"]


def score_fold(y_test: pd.Series, predictions: np.ndarray, scores: np.ndarray) -> dict:
    """Compute the metrics for one fold. If the fold has no shifts,
    recall, F1, and PR-AUC can't be computed, so they are recorded as missing."""

    if y_test.sum() == 0:
        return {metric: np.nan for metric in METRICS}

    return {
        # Of the weeks flagged as shifts, how many really were?
        "precision": precision_score(y_test, predictions, zero_division=0),
        # Of the real shifts, how many were flagged?
        "recall": recall_score(y_test, predictions),
        # A single score balancing precision and recall
        "f1": f1_score(y_test, predictions),
        # How well the model RANKS shifts above non-shifts, across all cutoffs
        "pr_auc": average_precision_score(y_test, scores),
    }


def evaluate_all(df: pd.DataFrame, folds: list, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """Train and score every model and baseline on every fold.
    Returns one row per (model, fold)."""

    results = []
    for fold_num, (train_rows, test_rows) in enumerate(folds, start=1):
        train_df, test_df = df.loc[train_rows], df.loc[test_rows]
        X_train, y_train = train_df[FEATURES], train_df["label"]
        X_test, y_test = test_df[FEATURES], test_df["label"]

        # New untrained models for every fold, so nothing carries over
        for name, model in make_models(random_state).items():
            model.fit(X_train, y_train)
            fold_scores = score_fold(y_test, model.predict(X_test), shift_scores(model, X_test))
            results.append({"model": name, "fold": fold_num, **fold_scores})

        # The JS baseline doesn't learn so it's handled separately
        predictions, scores = js_baseline_predict(train_df, test_df)
        fold_scores = score_fold(y_test, predictions, scores)
        results.append({"model": "Baseline: biggest JS jumps", "fold": fold_num, **fold_scores})

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per model: mean and std of each metric across folds.
    Kept as numbers (not "0.4 ± 0.1" text) so it's easy to plot and sort later.
    Folds with no shifts (missing values) are skipped automatically."""

    summary = results.groupby("model")[METRICS].agg(["mean", "std"]).round(2)
    summary["folds_used"] = results.groupby("model")["f1"].count()
    return summary


# %% Test Section 5
results = evaluate_all(df, folds)
print(summarize(results).to_string())