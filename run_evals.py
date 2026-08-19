"""Minimal LLM eval harness over a golden dataset.

Scores faithfulness, answer relevance, and exact/contains accuracy.
Writes a timestamped JSON report to results/.

Requires OPENAI_API_KEY. Run:  python run_evals.py
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

client = OpenAI()
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_golden(path="golden_dataset.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def answer_question(question, context):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the context. If the context "
                "does not answer, say 'I don't know'. Be concise.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQ: {question}"},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def faithfulness(answer, context):
    facts = re.split(r"[.!\n]", answer)
    facts = [f.strip() for f in facts if len(f.split()) > 3]
    if not facts:
        return 1.0
    supported = 0
    for fact in facts:
        check = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer only YES or NO: is the following claim "
                    "supported by the context?",
                },
                {
                    "role": "user",
                    "content": f"Claim: {fact}\nContext: {context}",
                },
            ],
            temperature=0,
        )
        if "YES" in check.choices[0].message.content.upper():
            supported += 1
    return supported / len(facts)


def answer_relevance(answer, question):
    check = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rate 0.0-1.0 how well the answer addresses the "
                "question. Output only the number.",
            },
            {"role": "user", "content": f"Q: {question}\nA: {answer}"},
        ],
        temperature=0,
    )
    try:
        return float(check.choices[0].message.content.strip())
    except ValueError:
        return 0.0


def accuracy(answer, golden_answer):
    ga = golden_answer.lower().strip()
    a = answer.lower().strip()
    if a == ga:
        return 1.0
    if a and ga and (ga in a or a in ga):
        return 0.5
    return 0.0


def main():
    golden = load_golden()
    rows = []
    for item in golden:
        ans = answer_question(item["question"], item["context"])
        rows.append(
            {
                "question": item["question"],
                "answer": ans,
                "golden": item.get("answer", ""),
                "faithfulness": faithfulness(ans, item["context"]),
                "relevance": answer_relevance(ans, item["question"]),
                "accuracy": accuracy(ans, item.get("answer", "")),
            }
        )
        print(f"[{len(rows)}/{len(golden)}] {item['question'][:60]}")

    avg = lambda key: sum(r[key] for r in rows) / len(rows)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": len(rows),
        "avg_faithfulness": round(avg("faithfulness"), 3),
        "avg_relevance": round(avg("relevance"), 3),
        "avg_accuracy": round(avg("accuracy"), 3),
        "rows": rows,
    }
    path = RESULTS_DIR / f"evals_{int(time.time())}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {path}")
    print(f"faithfulness={summary['avg_faithfulness']} relevance={summary['avg_relevance']} accuracy={summary['avg_accuracy']}")


if __name__ == "__main__":
    main()
