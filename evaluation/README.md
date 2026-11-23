# Head-to-head evaluation

LLM-as-judge evaluation used in the paper.

- `question_generation.py` and `question_prompts.py`: generate 125 global sensemaking questions per corpus.
- `answers.py`: answers the questions with global search on one index.
- `llm_judge.py`: evaluation prompt, criteria (comprehensiveness, diversity, empowerment, directness), and judge backends.
- `judge.py`: compares two answer files. Five judges each evaluate every question 3 times with the answer order randomized. Each judge votes by majority over its runs, and the final outcome is a strict majority over judges (tie otherwise).
- `win_rates.py`: win/loss/tie rates and Wilcoxon p-values (with Holm-Bonferroni correction) from the judge outputs.

Judges: GPT-5-mini (Azure OpenAI), Gemini 3 Pro Preview, Gemini 2.5 Pro, Qwen3 Next 80B and DeepSeek v3.2 (Vertex AI).

## Setup

```bash
pip install -e .                          # from the repository root
pip install -r evaluation/requirements.txt
gcloud auth application-default login     # Gemini, Qwen, DeepSeek

export GOOGLE_CLOUD_PROJECT=<gcp-project>
export AZURE_OPENAI_ENDPOINT=<https://...openai.azure.com/>
export AZURE_OPENAI_API_KEY=<key>
export AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
```

## Layout

Evaluation data is not included in the repository; the steps below create it. `win_rates.py` builds its tables from the file names, so keep this layout (`<model>` e.g. `gpt35`, `<dataset>` e.g. `podcast`):

```
questions/<dataset>.json                           # step 0
answers/<model>/<dataset>/<method>.json            # step 1: RkH_LF, RkH_L1, M2hC_LF, ..., C2, C3
results/<model>/<dataset>/<method>_<baseline>_<k>.json   # step 2: e.g. M2hC_LF_C2_0.json (k = repeated run)
```

The baseline name must not contain `_` (use `C2`, `C3`).

## 0. Questions

```bash
python evaluation/question_generation.py \
    --reports ./kevin_scott_podcasts_leiden/output/community_reports.parquet \
    --out questions/podcast.json
```

Following Edge et al., GPT-5-mini (via the Azure variables above) first describes the corpus. It works from 100 community reports: the top 50 by rank plus 50 sampled from the rest (`--seed 42`). It then creates 5 personas, 5 tasks per persona, and 5 questions per (persona, task) pair, which gives 125 questions. The output JSON also records the corpus description, personas, tasks and sampling settings.

## 1. Answers

Index the corpus once per method (one workspace each, see the main [README](../README.md)), then answer the same questions on each index and level. `--community-level` is `LF` / `L1` for k-core indexes and `2` / `3` for Leiden C2 / C3:

```bash
python evaluation/answers.py --root ./kevin_scott_podcasts_M2hC --community-level LF \
    --questions questions/podcast.json --out answers/gpt35/podcast/M2hC_LF.json
python evaluation/answers.py --root ./kevin_scott_podcasts_leiden --community-level 2 \
    --questions questions/podcast.json --out answers/gpt35/podcast/C2.json
```

The output is a JSON list `[{"id": "0", "question": "...", "answer": "..."}, ...]` in question order. Failed queries get an empty answer and are skipped by the judge.

## 2. Judge

```bash
python evaluation/judge.py \
    --answers1 answers/gpt35/podcast/M2hC_LF.json \
    --answers2 answers/gpt35/podcast/C2.json \
    --out results/gpt35/podcast/M2hC_LF_C2.json
```

Winner `1` is `--answers1` (row method) and `2` is `--answers2` (baseline). Each call writes `results/gpt35/podcast/M2hC_LF_C2_<k>.json` for the first unused `k`; run it again to repeat the evaluation. Add `--metrics empowerment directness` for the empowerment and directness metrics. `--judges` selects a subset of the judges (default: all five); the script stops if their credentials are not set. All comparisons of one dataset:

```bash
RUNS=3   # repeated evaluations per pair
for m in RkH_L1 RkH_LF M2hC_L1 M2hC_LF MRC_L1 MRC_LF; do
  for b in C2 C3; do
    for k in $(seq $RUNS); do
      python evaluation/judge.py --answers1 answers/gpt35/podcast/$m.json \
          --answers2 answers/gpt35/podcast/$b.json --out results/gpt35/podcast/${m}_$b.json
    done
  done
done
```

## 3. Win rates

```bash
python evaluation/win_rates.py results/gpt35/podcast
```

For each pair (all `<pair>_<k>.json` runs together):

- **Win / Loss / Tie**: percentages over all evaluations.
- **p**: two-sided Wilcoxon signed-rank test (`--alternative greater` for one-sided, `--answers1` better). It uses per-question scores (win = 100, tie = 50, loss = 0) averaged over runs.
- **p_holm**: p after Holm-Bonferroni correction within each dataset folder and metric (default). `--holm folder` corrects over both metrics of a folder, `--holm all` over everything passed in.

Rows are the methods and columns `<dataset> <baseline>`. `--verbose` also prints one line per pair and metric with the number of runs and questions. Both metrics are printed by default. Several dataset folders can be passed at once:

```bash
python evaluation/win_rates.py results/gpt35/podcast results/gpt35/news results/gpt35/semiconductor
```
