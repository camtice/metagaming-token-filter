"""Classify candidate latents as metagaming YES/NO with Claude.

Reads out/evidence.jsonl, sends description + activating examples to
claude-sonnet-5, writes out/classified.jsonl. Resumable: skips latents already
present in the output file.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

OUT = "/home/a5k/cwtice.a5k/sae-exploration/out"
MODEL = "gpt-5-mini"
API_KEY = os.environ["OPENAI_API_KEY"]
SYSTEM = open("/home/a5k/cwtice.a5k/sae-exploration/scripts/prompt_metagaming.txt").read()

done = set()
if os.path.exists(f"{OUT}/classified.jsonl"):
    done = {json.loads(l)["index"] for l in open(f"{OUT}/classified.jsonl")}

cands = [json.loads(l) for l in open(f"{OUT}/evidence.jsonl")
         if json.loads(l)["index"] not in done]
print(f"{len(cands)} to classify ({len(done)} already done)")


def classify(cand):
    ex_lines = "\n".join(f'[act {e["act"]}] "{e["text"]}"' for e in cand["examples"][:8])
    user = (f'Description: {cand["description"]}\n'
            f'Fires on 1 in {1/cand["firing_rate"]:.0f} tokens. '
            f'Top promoted output tokens: {cand["top_promoted_tokens"][:8]}\n'
            f'Activating examples:\n{ex_lines}\n\nAnswer:')
    body = {"model": MODEL, "max_completion_tokens": 1000,
            "reasoning_effort": "low",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}]}
    for attempt in range(6):
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          json=body, timeout=120)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            try:
                verdict = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except Exception:
                verdict = {"answer": "PARSE_ERROR", "reason": text[:100]}
            return {**cand, "verdict": verdict.get("answer"),
                    "verdict_reason": verdict.get("reason")}
        if r.status_code in (429, 500, 529, 503):
            time.sleep(2 ** attempt)
            continue
        return {**cand, "verdict": f"HTTP_{r.status_code}", "verdict_reason": r.text[:200]}
    return {**cand, "verdict": "RETRIES_EXHAUSTED", "verdict_reason": ""}


with ThreadPoolExecutor(max_workers=8) as pool, \
     open(f"{OUT}/classified.jsonl", "a") as out:
    for res in pool.map(classify, cands):
        out.write(json.dumps(res) + "\n")
        out.flush()
        print(res["index"], res["verdict"], "-", res["verdict_reason"])
