# ai-experiments-eval-harness

Experimental LLM evaluation harness. The goal: never change a prompt, chunker, or model without measuring it. Uses RAGAS-style metrics over a small golden dataset. Personal learning project.

```
golden_dataset.json
      |
      v
run_evals.py -> calls your RAG/LLM -> scores (faithfulness, relevance, accuracy)
      |
      v
results/evals_YYYYMMDD_HHMMSS.json  (compare before/after every change)
```

## Quick Start

```bash
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
python run_evals.py
```

## What You Get

- **Faithfulness** — is the answer grounded in retrieved context?
- **Answer relevance** — does the answer address the question?
- **Accuracy** — exact/contains match against golden answer (cheap + offline)
- A timestamped JSON report you can diff across runs

## The Golden Dataset

`golden_dataset.json` holds 50 question/answer/context triples. Edit it to match YOUR domain. Without a golden set you cannot evaluate anything.

## Golden Rules

1. Eval BEFORE and AFTER every change (prompt, chunk size, model, temperature)
2. One change at a time
3. Keep results — build a before/after table over time

## Roadmap

- [ ] LLM-as-judge scoring
- [ ] RAGAS library integration (retrieval hit-rate, MRR)
- [ ] Guardrail checks (refusals, toxicity)
- [ ] CI-style: fail the run if metrics regress

Intentionally experimental.