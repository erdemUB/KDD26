# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Answer the generated questions with global search on one index (input for judge.py).

Usage
    python evaluation/answers.py --root ./podcast_M2hC --community-level LF \
        --questions questions/podcast.json --out answers/gpt35/podcast/M2hC_LF.json

--community-level: LF, L1 (k-core indexes) or an integer level (2 = Leiden C2).
Writes a JSON list of {"id", "question", "answer"} in question order. A failed
query is stored with an empty answer and is skipped by judge.py.
"""

import argparse
import json
import os
from pathlib import Path

from graphrag.cli.query import run_global_search


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="GraphRAG workspace (contains settings.yaml)")
    ap.add_argument("--community-level", required=True)
    ap.add_argument("--questions", required=True, help="output of question_generation.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--response-type", default="Multiple Paragraphs")
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = [q["question"] for q in json.load(f)["questions"]]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for i, question in enumerate(questions):
        try:
            response, _ = run_global_search(None, None, args.root, args.community_level, False,
                                            args.response_type, False, question, False)
        except Exception as e:
            print(f"Question {i} failed: {e}")
            response = ""
        rows.append({"id": str(i), "question": question, "answer": str(response)})
        with open(args.out, "w") as f:  # rewrite after every question
            json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} answers -> {args.out}")


if __name__ == "__main__":
    main()
