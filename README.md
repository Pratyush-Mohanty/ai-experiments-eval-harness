# ai-experiments-eval-harness

Experimental LLM evaluation harness over a small golden dataset. The goal: never change a prompt, a chunker, a model, or a retrieval strategy without measuring it first. Scores faithfulness, relevance, and accuracy both cheaply (offline heuristics) and, when you have an API key, with an LLM-as-judge. Personal learning project for a data engineer learning AI engineering.

```
datasets/golden_dataset.json
        |
        v
run_eval.py  ------------------------->  your RAG/LLM answer_fn (injectable)
        |                                        |
        v                                        v
EvalRunner scores every golden item    answer + context + golden answer
        |
        v
results/
  evals_latest.json   (raw per-item metrics + summary)
  evals_latest.csv    (flat table for Excel / BI tools)
  evals_latest.md     (human-readable summary table)

compare_runs.py before.json after.json  ->  metric diffs + regression flags
```

## Quick Start

```bash
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
py run_eval.py                 # full LLM run against the golden dataset
py compare_runs.py results/evals_latest.json results/evals_previous.json
```

If `OPENAI_API_KEY` is not set, `run_eval.py` still works: it swaps in a stub `answer_fn` that returns `"I don't know."`, so you can smoke-test the pipeline offline. Every metric used in the default run is an offline heuristic — no API calls needed.

## The Golden Dataset

`datasets/golden_dataset.json` is a JSON array of objects shaped like:

```json
{
  "question": "What is an LSM-tree?",
  "answer": "A write-optimized storage structure that appends writes to a memtable and compacts them into sorted runs.",
  "context": "LSM-trees are write-optimized data structures used by databases like Bigtable and Cassandra. ..."
}
```

The golden set is your lab exam: every prompt/retriever/model change must pass the same exam so results stay comparable. Extend it freely — the more items, the more signal. Items are validated through a pydantic `GoldenItem` model (`src/eval_harness/dataset.py`), and an optional `tags` list lets you slice results by topic later (e.g. compare only the "storage engines" questions).

## Metrics Explained

### Accuracy (exact)

`exact_accuracy(answer, golden)` returns `1.0` when the answer equals the golden answer (case-insensitive, whitespace-stripped), else `0.0`. Cheap, offline, harsh — great for fact-based Q&A where the golden answer is a fixed string.

### Accuracy (contains)

`contains_accuracy(answer, golden)` returns `1.0` when the golden answer appears inside the answer (case-insensitive), else `0.0`. Softer than exact match and useful when answers are free-form sentences.

### Faithfulness

Is the answer grounded in the retrieved context, or did the model hallucinate? The offline `faithfulness_check(answer, context)` splits the answer into sentences, tokenizes each sentence and the context, and measures token overlap. A sentence is "supported" when at least 50% of its content tokens appear in the context; the score is the fraction of supported sentences. The `LLMJudge.judge_faithfulness` variant uses the model to answer YES/NO per claim when an API key is available.

### Relevance

Does the answer actually address the question? `LLMJudge.judge_relevance(answer, question)` asks the model to rate 0.0-1.0 and parses the number. It is the closest thing to a free-text correctness score and needs an API key.

### Hit-Rate

Fraction of queries that retrieved at least one relevant result inside the top-K. Given a per-query boolean `hits` list, `hit_rate(hits)` returns the fraction of `True` values. Use `has_hit(retrieved, relevant, k)` to turn retrieval lists into booleans.

### MRR (Mean Reciprocal Rank)

`mrr_at_k(retrieved, relevant, k)` returns `1 / rank` of the first relevant result found within the top-K (0 if none found). It rewards not just *whether* you retrieved something relevant, but *how high* it ranked — a relevant hit at rank 1 scores 1.0, at rank 3 scores 0.33.

## Comparing Runs

```bash
py compare_runs.py results/evals_before.json results/evals_after.json
```

`compare_runs` normalizes both result files, diffs every summary metric, and flags each as `improved`, `regressed`, or `unchanged` (using a small epsilon so floating point noise does not count). The CLI prints a table and exits with code 1 if any metric regressed, so you can wire it into a CI-style gate. Programmatically, `compare_runs(run_a, run_b)` returns a dict with per-metric `{before, after, delta, status}` — easy to feed into a notebook or report.

## Plugging In Your Own RAG System

`EvalRunner.run(dataset, answer_fn)` accepts any `answer_fn(question, context) -> str`. Swapping your production RAG pipeline in is a two-line change:

```python
from eval_harness.runner import EvalRunner, default_metrics
from eval_harness.reporting import write_json

def my_rag(question, context):
    chunks = retriever.retrieve(question, k=5)
    return generator.generate(question, chunks)

run = EvalRunner(metrics=default_metrics()).run(dataset, my_rag)
write_json(run, "results/my_rag.json")
```

## Rules of the Lab

1. Evaluate BEFORE and AFTER every change (prompt, chunk size, model, temperature).
2. Change one variable at a time so you know which one moved the metric.
3. Keep every result file — build a before/after table over time.
4. Treat offline heuristics as the cheap gate, LLM-as-judge as the deeper review.

## Roadmap

- [ ] LLM-as-judge scoring wired into the default run (faithfulness + relevance)
- [ ] RAGAS library integration (retrieval hit-rate, MRR over real retrievers)
- [ ] Guardrail checks (refusals, toxicity, PII leakage)
- [ ] CI-style gate: fail the run when metrics regress
- [ ] Per-tag slice analysis over the golden dataset

Intentionally experimental.
