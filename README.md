# Core-based Hierarchies for Efficient GraphRAG

This repository contains the code accompanying the paper  
**“Core-based Hierarchies for Efficient GraphRAG.”**

Our implementation builds directly on the official GraphRAG benchmarking framework released by Microsoft. We introduce k-core–based hierarchical community construction algorithms as drop-in replacements for Leiden-based community detection. The overall pipeline, datasets, and evaluation procedures remain unchanged.

## Code Base and Attribution

The code follows the same setup and execution procedure as the original GraphRAG benchmarking repository [based on GraphRAG release v2.7.0.]:

https://github.com/microsoft/graphrag-benchmarking-datasets 

We retain:
- The same dataset format  
- The same indexing and query-time execution flow  
- The same evaluation and head-to-head comparison framework  

Our contributions are limited to modifications in the community construction and hierarchy generation components.

## Running the Code

The code is executed using the standard GraphRAG commands. No additional steps are required beyond those described in the original GraphRAG repository.

### Installing Necessary Packages

Run the following from the root directory (Python 3.10–3.12):

```bash
pip install -e .
```

### Prepare Your Workspace

Create a new directory to hold your input files (e.g., for the Kevin Scott podcast benchmark):

```bash
mkdir -p ./kevin_scott_podcasts/input
```

Place the unzipped input files into the `input` folder.

### Initialize the GraphRAG Workspace

```bash
python graphrag/cli/main.py init  --root kevin_scott_podcasts/
```

This command creates two files inside `./kevin_scott_podcasts/`:

* `.env`: contains the `GRAPHRAG_API_KEY` environment variable. Set it to your OpenAI or Azure OpenAI API key.
* `settings.yaml`: configures the GraphRAG pipeline. You may edit this file to customize pipeline behavior.

Set the chat model under `models`. We use 600-token chunks with 100-token overlap:

```yaml
chunks:
  size: 600
  overlap: 100
```

### Indexing

Indexing constructs the knowledge graph and hierarchical communities:

```bash
python graphrag/cli/main.py index --root ./kevin_scott_podcasts --community RkH
```

Here, `RkH` specifies the community construction algorithm. If `--community` is omitted, the original GraphRAG Leiden algorithm (`leiden`) is used. You may alternatively select one of our proposed heuristics:

* `RkH`
* `M2hC`
* `MRC`

Indexing time varies depending on dataset size and may take several minutes to several hours.

The index is written to `<root>/output`, so use one workspace per method (e.g. copy the initialized workspace to `kevin_scott_podcasts_leiden`, `kevin_scott_podcasts_RkH`, ...).

### Query

You can now run queries against your indexed dataset. For example, to perform global search on leaf-level (`LF`) communities:

```bash
python graphrag/cli/main.py query \
  --root ./kevin_scott_podcasts \
  --method global \
  --community-level LF \
  --query "What recurring topics do tech leaders emphasize in their discussions?"
```

`--community-level` accepts `LF` (leaf level), `L1` (the level immediately above the leaf), or an integer level (e.g., `2` or `3` for Leiden C2/C3).

### Evaluation

Question generation, answering, the head-to-head LLM-judge evaluation (five judges, randomized answer order, majority vote) and the win-rate / Wilcoxon computation are in [`evaluation/`](evaluation/README.md).

### RRTC

Round-Robin Token-Constrained Selection is in `graphrag/index/operations/rrtc.py`. It selects edges from the leaf-level communities of an index under an edge budget.

To use it during indexing, set the budget in `settings.yaml`; community reports are then generated from the selected edges:

```yaml
community_reports:
  rrtc_budget: 0.8   # keep 80% of the leaf-level community edges
```

It can also be called directly:

```python
import pandas as pd
from graphrag.index.operations.rrtc import rrtc_select

out = "./kevin_scott_podcasts/output"
selected = rrtc_select(
    pd.read_parquet(f"{out}/communities.parquet"),
    pd.read_parquet(f"{out}/relationships.parquet"),
    budget=0.8,  # keep 80% of the edges
)
```
