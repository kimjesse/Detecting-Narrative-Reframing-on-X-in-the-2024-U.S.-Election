# Team 10 - Detecting Narrative Reframing on X in the 2024 U.S. Election
Authors: Shay Gooding, Tanisha Garret, Jesse Kim

This project contains the code files and intermediary data files for our
project on identifying topics and detecting
narrative reframing events during the 2024 US election. With the addition of
the original data files
from https://github.com/sinking8/x-24-us-election, users will be able to
reproduce our methods and results contained in
our report.

## Installation

```bash
# Clone the repository
git clone https://github.com/kimjesse/Detecting-Narrative-Reframing-on-X-in-the-2024-U.S.-Election.git

# Navigate to the project directory
cd Detecting-Narrative-Reframing-on-X-in-the-2024-U.S.-Election

# Install dependencies
pip install -r requirements.txt

# Clone the X data repository as a submodule
git submodule add https://github.com/sinking8/x-24-us-election.git x-24-us-election
```

## Usage

This project has three main parts: Preprocessing, Unsupervised Learning, and
Supervised Learning which should be run in that order.

### Preprocessing

```bash
cd CodeFiles
python run_preprocessing.py
```

This pipeline may take some time to run as it is cleaning the original data
files and creating our representative
sample that was used in our analyses. It will automatically create the
necessary directories to hold the intermediary
data files and produces [sampleddata.parquet](/DataFiles/sampleddata.parquet) 
which is the representative sample used in other parts.
The intermediate output files have not been included as a part of this repo 
to avoid unnecessarily adding to the repo size.

[first_100_lines_of_sampleddata.csv](first_100_lines_of_sampleddata.csv) is 
included in this repo which contains the top 100 lines of sampleddata.parquet.

### Unsupervised Learning

The unsupervised learning portion of the project is all contained within
[tk_unsupervised_learning.py](/Notebooks/tk_unsupervised_learning.ipynb). The
notebook itself contains instructions
to run the codeblocks.

This portion generates the files:

- [tk_issue_week_features.csv](/DataFiles/tk_issue_week_features.csv) this file
  contains the:
    - Tweet Counts
    - Week Info
    - Issue share: what fraction of that week's tweets mention the issue, and
      its weekly change
    - Shift scores: every week in a period gets that periods scores (blank in
      each issue's first period and in periods with too few tweets to fit
      topics)

- Topic shares: among this issue's tweets that week, the average share of each
  overall NMF topic (Biden, Trump, ...). These topics have the same meaning
  every week, unlike the per-period topics.

- [tk_topic_top_words.csv](/DataFiles/tk_topic_top_words.csv) this file
  contains the Top 20 words for every topic in the LDA and NMF runs at 5, 20
  and 30 topics

Also included within the notebook are various charts which break down the
topics on a weekly basis on various metrics.

### Supervised Learning

The supervised part predicts which issue-weeks have a narrative shift.
Each row in the model table is one issue in one week. The label is 1 (shift) or
0 (no shift).\
The features come from the unsupervised topic model, the weekly shift scores,
the tweet volume, and the noise floor.

#### Running the Supervised Pipeline

##### :warning: Before you start

- The unsupervised notebook must be run first. The supervised steps
  need [tk_issue_week_features](/DataFiles/tk_issue_week_features.csv).
- Steps 1 and 2 may be skipped if using the repository files. The labels are
  already saved
  in [labeling](DataFiles/labeling/)
  and [labels_for_repo.csv](DataFiles/labels_for_repo.csv),
  and the report results use these saved labels. Step 2 uses the paid Anthropic
  API. Start at step 3.
  Steps 1 and 2 are listed to show how the labels were made. To reproduce the
  results, start at step 3.

Run the scripts from the base folder of the repo. Run the notebooks from
the [Notebooks Folder](/Notebooks).

1. [make_packets.py](/CodeFiles/make_packets.py)
   Makes one label packet (a sample of tweets) for each issue-week.\
   Reads: [sampleddate.parquet](/DataFiles/sampleddata.parquet)\
   Writes: [packets folder](/DataFiles/labeling/packets/), [packets.json](/DataFiles/labeling/packets.jsonl), [answer_key.csv](/DataFiles/labeling/answer_key.csv)

1. [label_with_llm_simple.py](/CodeFiles/label_with_llm_simple.py)
   Labels each packet as shift or no shift with Claude Sonnet 5.5 (3 runs per
   row, majority vote), using the prompt in
   llm_prompt_simple.txt.\
   Reads: [packets.jsonl](DataFiles/labeling/packets.jsonl)\
   Writes: [llm_labels_simple.csv](/DataFiles/labeling/llm_labels_simple.csv)\
   :warning: **Needs an Anthropic API key. Set it in PowerShell before 
   running: $env:
   ANTHROPIC_API_KEY = "<your-api-key>"**

1. [check_labels.ipynb](Notebooks/check_labels.ipynb)
   Compares a blind hand check of random rows with the LLM labels, and saves
   the final labels.\
   Reads: [llm_labels_simple.csv](DataFiles/labeling/llm_labels_simple.csv),
   [hand_labels.csv](/DataFiles/labeling/hand_labels.csv)\
   Writes: [hand_check_rows.csv](/DataFiles/labeling/hand_check_rows.csv), [labels_final.csv](/DataFiles/labeling/labels_final.csv), [labels_for_repo.csv](DataFiles/labels_for_repo.csv)

1. [noise_floor.ipyb](Notebooks/noise_floor.ipynb)
   and [volume_summary.ipynb](Notebooks/volume_summary.ipynb) (either
   order)
   - noise_floor.ipynb estimates the noise floor for each issue-week.\
   Reads: [sampleddata.parquet](DataFiles/sampleddata.parquet), [answer_key.
      csv](/DataFiles/answer_key.csv), [tk_issue_week_features.csv](/DataFiles/tk_issue_week_features.csv)\
   Writes: [noise_floor.csv](DataFiles/noise_floor.csv)

   - volume_summary.ipynb adds weekly tweet volume for each issue.\
   Reads: [tk_issue_week_features.csv](DataFiles/tk_issue_week_features.csv)
      , [weekly_orig_totals.csv](/DataFiles/weekly_orig_totals.csv)\
   Writes: [volume_summary](DataFiles/volume_summary.csv)

1. [supervised_model_table.ipynb](Notebooks/supervised_model_table.ipynb)
   Joins the labels and all features into one table.\
   Reads: outputs of steps 3 and
   4, [tk_issue_week_features.csv](/DataFiles/tk_issue_week_features.csv), 
   [answer_key.csv](/DataFiles/answer_key.csv)\
   Writes: [supervised_model_table.csv](DataFiles/supervised_model_table.csv)

1. [supervised_models.ipynb](Notebooks/supervised_models.ipynb)
   Trains, tunes, and evaluates the models, and runs the failure analysis.\
   Reads: [supervised_model_table.csv](DataFiles/supervised_model_table.csv)\
   Writes: [tuning_results.csv](DataFiles/tuning_results.csv)

#### Models and evaluation

[supervised_models.ipynb](/Notebooks/supervised_models.ipynb) compares Logistic
Regression, Gradient-Boosted Trees, and
a Support Vector Machine (SVM) with two baselines. It uses 5-fold time-based
cross-validation. The main metric is
PR-AUC. The failure analysis uses SHAP. The random seed is set to 42.
