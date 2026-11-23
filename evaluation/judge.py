# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Head-to-head evaluation of two sets of GraphRAG answers with multiple LLM judges.

For every question and metric, each judge evaluates the pair NUM_RUNS times with
the answer order randomized per run; the judge's vote is the majority over its
runs, and the final outcome is the strict majority over judges (tie otherwise).

Usage
    python evaluation/judge.py \
        --answers1 answers/gpt35/podcast/M2hC_LF.json \
        --answers2 answers/gpt35/podcast/C2.json \
        --out results/gpt35/podcast/M2hC_LF_C2.json

Answer files are JSON lists of {"id", "question", "answer"} in the same question
order (as written by answers.py); "answer" is the global search response (with
older graphrag CLI output, the text after "Global Search Response:" is used).
In the output, winner 1 = answers1, 2 = answers2, 0 = tie. Each call writes
<out>_<k>.json for the first unused k, so repeated evaluations of the same
pair are kept side by side.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from llm_judge import llm_judge

JUDGES = [
    "deepseek-ai/deepseek-v3.2-maas",
    "qwen/qwen3-next-80b-a3b-instruct-maas",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
    "azure",  # GPT-5-mini
]
NUM_RUNS_PER_JUDGE = 3
METRICS = ["comprehensiveness", "diversity"]  # also available: "empowerment", "directness"
RESPONSE_MARKER = "Global Search Response:"


def safe_json_load(text):
    """Extracts the first JSON object from LLM output and parses it."""
    match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1).strip())

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0).strip())

    print("Raw LLM output:\n", text)
    raise ValueError("Could not find valid JSON in LLM output")


def judge_with_retry(question, a1, a2, metric, judge="azure", max_retries=5):
    base_delay, max_delay = 1.0, 30.0
    for attempt in range(1, max_retries + 1):
        try:
            return safe_json_load(llm_judge(question, a1, a2, metric, judge))
        except Exception:
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += random.uniform(0, delay * 0.1)
            print(f"Retry {attempt}/{max_retries} failed for {judge}. Sleeping {delay:.2f}s before retry...")
            time.sleep(delay)

    raise RuntimeError(f"llm_judge failed after {max_retries} retries for judge={judge}")


def judge_multiple_runs(question, answer1, answer2, metric, judge, num_runs=NUM_RUNS_PER_JUDGE):
    """One judge, num_runs evaluations with randomized answer order; majority vote."""

    def single_run(i):
        a1, a2, flipped = (answer2, answer1, True) if random.random() < 0.5 else (answer1, answer2, False)
        assessment = judge_with_retry(question, a1, a2, metric, judge=judge)
        winner = assessment.get("winner", 0)
        r = assessment.get("reasoning", "No reasoning or failed")
        if flipped and winner in [1, 2]:
            winner = 1 if winner == 2 else 2
            r = r.replace("Answer 1", "TEMP").replace("Answer 2", "Answer 1").replace("TEMP", "Answer 2")
        return i, winner, r

    run_results = []
    with ThreadPoolExecutor(max_workers=num_runs) as executor:
        futures = [executor.submit(single_run, i) for i in range(num_runs)]
        for future in as_completed(futures):
            try:
                run_results.append(future.result())
            except Exception as e:
                print(f"Single run failed for judge {judge}: {e}. Counting as tie (0).")
                run_results.append((-1, 0, "Run failed (FAILED_PARALLEL), counted as tie."))
    run_results.sort(key=lambda x: x[0])

    votes = [w for _, w, _ in run_results]
    per_run_results = [{"winner": w, "reasoning": r} for _, w, r in run_results]
    vote_counts = Counter(votes)
    final_winner = vote_counts.most_common(1)[0][0]
    final_reasoning = "\n########\n".join(r for _, w, r in run_results if w == final_winner)

    return {"winner": final_winner, "votes": votes, "vote_counts": dict(vote_counts),
            "reasoning": final_reasoning, "all_reasoning": per_run_results}


def majority_vote_judge_parallel(question, answer1, answer2, metric, judges=JUDGES,
                                 num_runs_per_judge=NUM_RUNS_PER_JUDGE):
    """All judges in parallel; strict majority over judges, tie otherwise."""
    per_judge_winners, per_judge_all_reasoning = {}, {}

    def call_judge(jm):
        try:
            return jm, judge_multiple_runs(question, answer1, answer2, metric, jm, num_runs_per_judge)
        except Exception as e:
            print(f"JUDGE {jm} FAILED COMPLETELY: {e}")
            return jm, {"winner": 0, "reasoning": "JUDGE FAILED (FAILED_PARALLEL)",
                        "all_reasoning": "JUDGE FAILED (FAILED_PARALLEL)"}

    with ThreadPoolExecutor(max_workers=len(judges)) as executor:
        for future in as_completed([executor.submit(call_judge, jm) for jm in judges]):
            jm, result = future.result()
            per_judge_winners[jm] = result["winner"]
            per_judge_all_reasoning[jm] = result.get("all_reasoning", result["reasoning"])

    final_vote_counts = Counter(per_judge_winners.values())
    max_count = max(final_vote_counts.values())
    majority_winners = [w for w, c in final_vote_counts.items() if c == max_count]
    final_winner = majority_winners[0] if len(majority_winners) == 1 else 0

    return {"final_winner": final_winner, "all_votes": per_judge_winners,
            "all_reasoning": per_judge_all_reasoning}


def process_single_question(i, task, metrics=METRICS):
    question, answer1, answer2 = task
    question_result = {"id": str(i), "question": question, "answer1": answer1, "answer2": answer2,
                       "lengths": (len(answer1), len(answer2)), "metrics": {}}
    for metric in metrics:
        try:
            result = majority_vote_judge_parallel(question, answer1, answer2, metric)
        except Exception as e:
            print(f"Metric {metric} failed for question {i}: {e}")
            result = {"final_winner": 0, "all_reasoning": "FAILED (FAILED_PARALLEL)"}
        question_result["metrics"][metric] = result
    return question_result


def load_tasks(answers1_path, answers2_path):
    """Pair answers by position; questions must match and both queries must have succeeded."""
    with open(answers1_path) as f:
        rows1 = json.load(f)
    with open(answers2_path) as f:
        rows2 = json.load(f)

    if len(rows1) != len(rows2) or any(a["question"] != b["question"] for a, b in zip(rows1, rows2)):
        raise ValueError("The two answer files must contain the same questions in the same order.")

    def response(answer):  # text after the marker if present (older graphrag CLI output)
        return (answer or "").split(RESPONSE_MARKER)[-1].strip()

    tasks = []
    for q1, q2 in zip(rows1, rows2):
        a1, a2 = response(q1["answer"]), response(q2["answer"])
        if not a1 or not a2:
            print(f"Skipping question without a response: {q1['question']}")
            continue
        tasks.append((q1["question"], a1, a2))
    return tasks


def unique_filename(path):
    """<base>_<k><ext> for the first unused k (repeated evaluations of the same pair)."""
    base, ext = os.path.splitext(path)
    counter = 0
    while os.path.exists(f"{base}_{counter}{ext}"):
        counter += 1
    return f"{base}_{counter}{ext}"


def compare(answers1_path, answers2_path, out_path, metrics=METRICS):
    tasks = load_tasks(answers1_path, answers2_path)
    out_path = unique_filename(out_path)
    print(f"{len(tasks)} questions -> {out_path}")

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_question, i, t, metrics): (i, t) for i, t in enumerate(tasks)}
        for idx, future in enumerate(as_completed(futures), 1):
            i, (question, answer1, answer2) = futures[future]
            try:
                res = future.result(timeout=300)
            except Exception as e:
                print(f"Question {i} failed: {e}")
                res = {"id": str(i), "question": question, "answer1": answer1, "answer2": answer2,
                       "lengths": (len(answer1), len(answer2)), "metrics": {}}
            results.append(res)
            print(f"Processed {idx}/{len(tasks)}")

    results.sort(key=lambda x: int(x["id"]))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answers1", required=True, help="answers of the row method (e.g. a k-core heuristic)")
    p.add_argument("--answers2", required=True, help="answers of the column method (e.g. Leiden C2)")
    p.add_argument("--out", required=True, help="output JSON (suffixed _<k> to keep repeated runs)")
    p.add_argument("--metrics", nargs="+", default=METRICS, choices=list(METRICS) + ["empowerment", "directness"])
    p.add_argument("--judges", nargs="+", default=list(JUDGES), choices=JUDGES, help="default: all five judges")
    args = p.parse_args()

    # Fail fast: a judge without credentials would otherwise retry for a long time and count as a tie.
    need = (["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"] if "azure" in args.judges else []) + (
        ["GOOGLE_CLOUD_PROJECT"] if any(j != "azure" for j in args.judges) else [])
    missing = [v for v in need if not os.getenv(v)]
    if missing:
        p.error(f"set {', '.join(missing)} for the selected judges (see evaluation/README.md)")
    JUDGES[:] = args.judges  # used as the default judge list
    compare(args.answers1, args.answers2, args.out, args.metrics)


if __name__ == "__main__":
    main()
