# pylint: disable=invalid-name
"""
SIMPLE-PROMPT TEST: label each packet with an LLM (Claude Sonnet 5.5)
using the simple shift / no-shift prompt (no Narrative Policy Framework).
This is a copy of 3_label_with_llm.py. Only the prompt, the answer shape,
the categories, and the output file names are different.

What this script does, step by step:
  1. Reads the prompt
  2. Reads the packets (last week's tweets vs. this week's tweets).
  3. Sends each packet to the LLM, one at a time, and gets back a label.
  4. Saves every label to a CSV file as it goes.

How to run from repo:
  - Prompt test: put 5 row_ids in TEST_ROWS, then run
        python supervised/3_label_with_llm_simple.py
    Answers go to llm_test_labels_simple.csv. Read the "reason" column.
  - Full run: make TEST_ROWS empty again ([]), then run the same command.
    Answers go to llm_labels_simple.csv.
  - If the script stops run it again. It skips rows that already have a label.

Majority vote (RUNS):
  The LLM has some randomness, so the same row can get a different answer on
  different runs. To even this out, each row is labeled RUNS times (3 by
  default) and we keep the answer that comes up most often (a majority vote,
  also called self-consistency). The "agreement" column shows how many runs
  agreed, e.g. "3/3" (all agreed) or "2/3" (one run disagreed).
  If no answer gets a majority, the row gets "no shift" (label 0), the same
  default our hand-labeling rules use when labelers can't agree.
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from anthropic import Anthropic  # pylint: disable=import-error

# Settings

FOLDER = Path(__file__).resolve().parent
LABEL_FOLDER = FOLDER / "labeling"
PROMPT_FILE = FOLDER / "2_llm_prompt_simple.txt"
MODEL = "claude-sonnet-5-5"

# Price in dollars per million tokens (a token is a word piece, about 3/4 of a word).
# Sonnet 5.5: $2 for input (what we send), $10 for output (what it writes back,
# including any "thinking" it does before answering).
PRICE_IN = 2.00
PRICE_OUT = 10.00

EXAMPLE_ROWS = ["row_015", "row_063", "row_123"]
TEST_ROWS = []

# How many times to label each row. Use an odd number (3 or 5) so a
# majority is more likely. RUNS = 1 works like the old one-run script.
RUNS = 3

# The simple prompt has only 3 answers. "unsure" gets no label (None).
CATEGORY_TO_LABEL = {
    "shift": 1,
    "no shift": 0,
    "unsure": None,
}

# JSON schema: the shape the LLM's answer must have.
# Simpler than the v3 shape: no setting/characters/plot/moral parts.
ANSWER_SHAPE = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORY_TO_LABEL)},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["evidence", "reason", "category", "confidence"],
    "additionalProperties": False,
}


def make_message(packet):
    """Put one packet into the format the prompt expects: the issue, then
    last week's and this week's tweets, each list wrapped in its own tag."""
    last_week = "\n".join(f"{n}. {tweet}" for n, tweet in
                          enumerate(packet["before_tweets"], start=1))
    this_week = "\n".join(f"{n}. {tweet}" for n, tweet in
                          enumerate(packet["after_tweets"], start=1))
    issue = packet["issue"].replace("_", " ")
    return (f"<issue>{issue}</issue>\n\n"
            f"<last_week>\n{last_week}\n</last_week>\n\n"
            f"<this_week>\n{this_week}\n</this_week>")


def ask_llm(client, prompt, packet):
    """Send one packet to the LLM and return its answer as a dictionary."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,     # this gives room for the LLM to think and answer
        system=prompt,
        messages=[{"role": "user", "content": make_message(packet)}],
        output_config={"format": {"type": "json_schema", "schema": ANSWER_SHAPE}},
    )
    # The reply can have a few parts. the answer is in the text.
    text = ""
    for part in response.content:
        if part.type == "text":
            text += part.text
    answer = json.loads(text)
    # The API can change capital letters so we lowercase them
    answer["category"] = answer["category"].lower()
    answer["confidence"] = answer["confidence"].lower()
    # The API tells us how many tokens this call used. Turn that into dollars.
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    answer["cost"] = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000
    return answer


def majority_vote(answers):
    """Pick the answer most of the runs agreed on.

    answers: a list of answer dictionaries, one per run.
    Returns (the chosen answer, an agreement string like "2/3").
    """
    categories = [answer["category"] for answer in answers]
    # Counter counts how often each category shows up,
    # e.g. Counter({"no shift": 2, "shift": 1})
    top_category, top_count = Counter(categories).most_common(1)[0]

    # A majority means MORE than half of the runs picked it
    if top_count > len(answers) / 2:
        # Keep the reason/evidence/confidence from the first run
        # that gave the winning answer
        chosen = next(a for a in answers if a["category"] == top_category)
    else:
        # No majority (for example 1 shift, 1 no shift, 1 unsure):
        # default to "no shift", like our hand-labeling rule
        top_count = categories.count("no shift")
        chosen = {
            "category": "no shift",
            "confidence": "low",
            "evidence": "",
            "reason": "No majority across runs; defaulted to no shift.",
        }
    return chosen, f"{top_count}/{len(answers)}"


def main():
    """Label every packet that doesn't have a label yet, and save each answer."""
    # 1. Read the prompt, and stop if a [FILL] spot is empty
    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    if "[FILL" in prompt:
        print("STOP: fill in every [FILL] spot in the prompt first.")
        return
    if len(EXAMPLE_ROWS) != 3:
        print("STOP: put your 3 example row_ids in EXAMPLE_ROWS first.")
        return

    # 2. Read the packets and pick which rows to label
    packets = pd.read_json(LABEL_FOLDER / "packets.jsonl", lines=True)
    packets = packets[~packets["row_id"].isin(EXAMPLE_ROWS)]
    if TEST_ROWS:
        packets = packets[packets["row_id"].isin(TEST_ROWS)]
        out_file = LABEL_FOLDER / "llm_test_labels_simple.csv"
    else:
        out_file = LABEL_FOLDER / "llm_labels_simple.csv"

    # 3. Skip rows that have a label (from an earlier run that stopped)
    done = set()
    if out_file.exists():
        old = pd.read_csv(out_file)
        # Files from before RUNS was added have no "agreement" column.
        # Mixing the two formats would break the CSV, so stop here.
        if "agreement" not in old.columns:
            print(f"STOP: {out_file.name} is from the old one-run version.")
            print("Rename or delete it, then run again.")
            return
        done = set(old["row_id"])
    print(f"Labeling {len(packets) - len(done)} rows with {MODEL}, {RUNS} runs each ...")

    client = Anthropic()
    total_cost = 0.0    # dollars spent so far in this run

    for _, packet in packets.iterrows():
        if packet["row_id"] in done:
            continue

        # Ask the LLM RUNS times about the same row
        answers = []
        try:
            for _ in range(RUNS):
                answers.append(ask_llm(client, prompt, packet))
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Any problem (no internet, API error): skip this row for now.
            # Running the script again will retry it (all runs).
            print(f"  {packet['row_id']}: error, will retry next run ({error})")
            continue

        answer, agreement = majority_vote(answers)
        row_cost = sum(a["cost"] for a in answers)

        # 4. Save this row so nothing is lost if the script stops
        row = pd.DataFrame([{
            "row_id": packet["row_id"],
            "issue": packet["issue"],
            "category": answer["category"],
            "label": CATEGORY_TO_LABEL[answer["category"]],
            "agreement": agreement,
            # Every run's answer, e.g. "no shift | shift | no shift"
            "all_runs": " | ".join(a["category"] for a in answers),
            "confidence": answer["confidence"],
            "evidence": answer["evidence"],
            "reason": answer["reason"],
        }])
        row.to_csv(out_file, mode="a", header=not out_file.exists(), index=False)
        total_cost += row_cost
        print(f"  {packet['row_id']}: {answer['category']}  "
              f"[{agreement} agree]  (${row_cost:.4f})")

    print(f"Done. Labels are in {out_file.name}")
    print(f"Total cost this run: ${total_cost:.2f}")


if __name__ == "__main__":
    main()
