"""Parse Haiku batch results into final labels + a summary/agreement report.

Usage:
  python3 relabel/collect.py relabel/batch_results_full.jsonl [more.jsonl ...]

Writes results/gemma3_L40_16k_haiku_labels.json and
results/gemma3_L40_16k_haiku_summary.md.
"""

import gzip
import json
import os
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIRNAME = os.environ.get("SAE_DIRNAME", "gemma-3-27b-40-gemmascope-2-res-16k")
OUT_PREFIX = os.environ.get("OUT_PREFIX", "gemma3_L40_16k_haiku")
CATS = ["c1_capability_evals_oversight", "c2_ai_safety_oversight",
        "c3_human_oversight_testing", "c4_ai_training_pipeline",
        "c5_swe_tests", "none"]


def parse_reply(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(re.sub(r",\s*}", "}", m.group(0)))
    except json.JSONDecodeError:
        return None
    if d.get("category") not in CATS or not isinstance(d.get("forget"), bool):
        return None
    return d


def main():
    descs = {}
    expl_root = pathlib.Path(os.environ.get("EXPL_ROOT", ROOT))
    for f in sorted((expl_root / "explanations" / DIRNAME).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                descs[int(r["index"])] = r["description"]

    labels, bad = {}, []
    for path in sys.argv[1:]:
        for line in open(path):
            r = json.loads(line)
            idx = int(r["custom_id"].split("-")[1])
            res = r["result"]
            if res["type"] != "succeeded":
                bad.append((idx, res["type"]))
                continue
            text = " ".join(b.get("text", "") for b in res["message"]["content"])
            d = parse_reply(text)
            if d is None:
                bad.append((idx, "unparseable"))
                continue
            labels[idx] = {"index": idx, "category": d["category"],
                           "forget": d["forget"], "confidence": d.get("confidence"),
                           "rationale": d.get("rationale", ""),
                           "desc": descs.get(idx)}

    out = ROOT / "results" / f"{OUT_PREFIX}_labels.json"
    out.write_text(json.dumps([labels[i] for i in sorted(labels)], indent=1))
    print(f"{len(labels)} labels -> {out}; {len(bad)} failed: {bad[:10]}")

    forget = [l for l in labels.values() if l["forget"]]
    lines = ["# Haiku 4.5 relabel summary\n",
             f"- classified: {len(labels)}, parse/API failures: {len(bad)}",
             f"- forget: {len(forget)} ({100 * len(forget) / max(1, len(labels)):.1f}%)\n",
             "## Forget by category x confidence\n",
             "| category | high | medium | low | total |", "|---|---|---|---|---|"]
    for cat in CATS[:5]:
        cc = Counter(l["confidence"] for l in forget if l["category"] == cat)
        lines.append(f"| {cat} | {cc.get('high', 0)} | {cc.get('medium', 0)} "
                     f"| {cc.get('low', 0)} | {sum(cc.values())} |")

    vpath = ROOT / "results" / "gemma3_L40_16k_verdicts.json"
    if vpath.exists() and "16k" in DIRNAME:
        verdicts = json.load(open(vpath))
        lines += ["\n## Agreement with prior manual verdicts\n",
                  "| prior verdict | n seen | haiku forget | haiku keep |", "|---|---|---|---|"]
        for v in ("core", "soft", "fp"):
            seen = [labels[x["index"]] for x in verdicts
                    if x["verdict"] == v and x["index"] in labels]
            f = sum(1 for l in seen if l["forget"])
            lines.append(f"| {v} | {len(seen)} | {f} | {len(seen) - f} |")
        lines.append("\n### Disagreements (core->keep and fp->forget)\n")
        for x in verdicts:
            l = labels.get(x["index"])
            if not l:
                continue
            if (x["verdict"] == "core" and not l["forget"]) or \
               (x["verdict"] == "fp" and l["forget"]):
                lines.append(f"- {x['index']} [{x['verdict']}] \"{x['desc']}\" -> "
                             f"forget={l['forget']} {l['category']} "
                             f"({l['confidence']}): {l['rationale']}")

    md = ROOT / "results" / f"{OUT_PREFIX}_summary.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"summary -> {md}")


if __name__ == "__main__":
    main()
