# Copyright (c) 2026 Jakir Hossain, Ahmet Erdem Sariyuce.
# Licensed under the MIT License

"""Generate global sensemaking questions for a corpus.

Following Edge et al., an LLM (GPT-5-mini) first describes the corpus from a sample
of its community reports, then creates K personas, N tasks per persona and M
questions per (persona, task) pair: K = N = M = 5 gives 125 questions.

Usage
    python evaluation/question_generation.py \
        --reports <index>/output/community_reports.parquet \
        --out questions/<dataset>.json

Report sample: the top --top-by-rank reports by rank plus --random-from-rest
reports sampled from the rest (seeded), shuffled; each report contributes its
title and the first --summary-char-cap characters of its summary.

Credentials (environment): AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
AZURE_OPENAI_DEPLOYMENT (default "gpt-5-mini"), AZURE_OPENAI_API_VERSION.
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

import pandas as pd

from question_prompts import (
    CORPUS_DESCRIPTION_SYSTEM,
    CORPUS_DESCRIPTION_USER_TEMPLATE,
    PERSONAS_SYSTEM,
    PERSONAS_USER_TEMPLATE,
    QUESTIONS_FOR_PAIR_USER_TEMPLATE,
    QUESTIONS_SYSTEM,
    TASKS_SYSTEM,
    TASKS_USER_TEMPLATE,
)

_client = None


def select_top_and_random_reports(df: pd.DataFrame, top_n: int, random_n: int, seed: int) -> pd.DataFrame:
    """Select top_n by rank (if available), then sample random_n from remaining rows.

    If rank is missing, uses a random sample of size (top_n + random_n).
    """
    random.seed(seed)
    df_work = df.copy()

    if "rank" in df_work.columns:
        df_sorted = df_work.sort_values("rank", ascending=False)
        top_df = df_sorted.head(top_n)
        rest_df = df_sorted.iloc[top_n:]
        sample_n = min(random_n, len(rest_df))
        rand_df = rest_df.sample(n=sample_n, random_state=seed) if sample_n > 0 else rest_df.head(0)
        selected = pd.concat([top_df, rand_df], ignore_index=True)
    else:
        total = min(top_n + random_n, len(df_work))
        selected = df_work.sample(n=total, random_state=seed)

    # shuffle to distribute themes
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_reports_snippets(df: pd.DataFrame, summary_char_cap: int) -> str:
    snippets: List[str] = []
    for _, r in df.iterrows():
        title = str(r.get("title", "")).strip()
        # prefer summary; fallback to full_content
        summary = str(r.get("summary") or r.get("full_content") or "").replace("\n", " ").strip()
        if summary_char_cap > 0:
            summary = summary[:summary_char_cap]
        if title or summary:
            snippets.append(f"- {title}: {summary}".strip())
    return "\n".join(snippets)


def chat_json(system: str, user: str, *, name: str) -> Dict[str, Any]:
    """Calls Azure OpenAI and expects the model to return strict JSON text."""
    global _client
    if _client is None:
        from openai import AzureOpenAI

        _client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    resp = _client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    text = resp.choices[0].message.content or ""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[{name}] Model did not return valid JSON. Error: {e}\nRAW:\n{text}")


def get_description(df_all: pd.DataFrame, top_n: int, random_n: int, seed: int, summary_char_cap: int):
    """1) Corpus description from a sample of community reports."""
    selected_reports = select_top_and_random_reports(df_all, top_n=top_n, random_n=random_n, seed=seed)
    snippets = build_reports_snippets(selected_reports, summary_char_cap=summary_char_cap)
    corpus_desc = chat_json(
        system=CORPUS_DESCRIPTION_SYSTEM,
        user=CORPUS_DESCRIPTION_USER_TEMPLATE.format(snippets=snippets),
        name="corpus_description",
    )
    return selected_reports, corpus_desc


def get_persona_task(corpus_desc, K: int, N: int):
    """2) K personas, 3) N tasks per persona."""
    corpus_json = json.dumps(corpus_desc, ensure_ascii=False, indent=2)
    personas = chat_json(
        system=PERSONAS_SYSTEM,
        user=PERSONAS_USER_TEMPLATE.format(K=K, corpus_description_json=corpus_json),
        name="personas",
    )
    print("Personas done:", len(personas.get("personas", [])))

    tasks = chat_json(
        system=TASKS_SYSTEM,
        user=TASKS_USER_TEMPLATE.format(
            N=N,
            personas_json=json.dumps(personas, ensure_ascii=False, indent=2),
            corpus_description_json=corpus_json,
        ),
        name="tasks",
    )
    print("Tasks done:", len(tasks.get("tasks", [])))
    return personas, tasks


def get_questions(corpus_desc, personas, tasks, M: int) -> List[Dict[str, Any]]:
    """4) M questions per (persona, task) pair."""
    corpus_json = json.dumps(corpus_desc, ensure_ascii=False, indent=2)
    tasks_by_persona: Dict[Any, list] = {}
    for t in tasks.get("tasks", []):
        tasks_by_persona.setdefault(t["persona_id"], []).append(t)

    all_questions = []
    for p in personas.get("personas", []):
        pid = p["persona_id"]
        for t in tasks_by_persona.get(pid, []):
            pair_resp = chat_json(
                system=QUESTIONS_SYSTEM,
                user=QUESTIONS_FOR_PAIR_USER_TEMPLATE.format(
                    M=M,
                    persona_json=json.dumps(p, ensure_ascii=False, indent=2),
                    task_json=json.dumps(t, ensure_ascii=False, indent=2),
                    corpus_description_json=corpus_json,
                ),
                name=f"questions_persona{pid}_task{t['task_id']}",
            )
            for i, q in enumerate(pair_resp.get("questions", []), start=1):
                all_questions.append({"persona_id": pid, "task_id": t["task_id"], "question_id": i,
                                      "question": q.get("question", "")})
    print("Questions done:", len(all_questions))
    return all_questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True, help="community_reports.parquet of an index of the corpus")
    ap.add_argument("--out", required=True, help="output JSON")
    ap.add_argument("--count", type=int, default=5, help="K personas = N tasks = M questions (default 5 -> 125)")
    ap.add_argument("--top-by-rank", type=int, default=50)
    ap.add_argument("--random-from-rest", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-char-cap", type=int, default=700)
    args = ap.parse_args()
    K = N = M = args.count

    df_all = pd.read_parquet(args.reports)
    selected_reports, corpus_desc = get_description(
        df_all, args.top_by_rank, args.random_from_rest, args.seed, args.summary_char_cap
    )
    personas, tasks = get_persona_task(corpus_desc, K, N)
    questions = get_questions(corpus_desc, personas, tasks, M)

    out = {
        "algorithm": "persona_task_question",
        "params": {"K": K, "N": N, "M": M},
        "fewshot_examples_included": True,
        "fewshot_source": "Podcast transcripts, News articles examples",
        "report_selection": {
            "strategy": "top_by_rank_plus_random_from_rest_then_shuffle",
            "top_by_rank": args.top_by_rank,
            "random_from_rest": args.random_from_rest,
            "random_seed": args.seed,
            "summary_char_cap": args.summary_char_cap,
        },
        "source": {
            "community_reports_parquet": args.reports,
            "total_reports_in_file": int(len(df_all)),
            "selected_reports_count": int(len(selected_reports)),
        },
        "corpus_description": corpus_desc,
        "personas": personas.get("personas", []),
        "tasks": tasks.get("tasks", []),
        "questions": questions,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} | reports(total={len(df_all)} selected={len(selected_reports)}) | "
          f"personas={len(out['personas'])} tasks={len(out['tasks'])} questions={len(questions)}")


if __name__ == "__main__":
    main()
