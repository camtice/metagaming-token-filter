"""Stream-build and submit the 262k SAE run (Haiku, v8 prompt) without writing
a giant requests file: render each activation batch file's latents, submit in
1000-request chunks, append batch ids to relabel/ids_262k_v8.txt.

Resumable: the highest submitted latent index is checkpointed after every
chunk; on restart, lower indices are skipped.

Usage: python3 relabel/build_262k_stream.py
"""

import gzip
import heapq
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_prompts as bp
import prompt_iter
from run_batch import API, call

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 262k data lives outside the iCloud-synced Desktop (eviction-proof)
CACHE = pathlib.Path.home() / ".cache" / "sae-probe"
DIR = CACHE / "activations" / "gemma-3-27b-40-gemmascope-2-res-262k"
FDIR = CACHE / "features" / "gemma-3-27b-40-gemmascope-2-res-262k"
IDS = ROOT / "relabel" / "ids_262k_v8.txt"
RESUME = ROOT / "relabel" / "262k_v8_resume.txt"
CHUNK = 1000


def submit(reqs):
    for attempt in range(10):
        try:
            b = call(API, {"requests": reqs})
            return b["id"]
        except Exception as e:
            print(f"submit attempt {attempt + 1} failed: {e}", flush=True)
            if attempt == 9:
                raise
            time.sleep(min(120, 20 * (attempt + 1)))


def main():
    d16, f16, a16 = bp.load_data()
    system = prompt_iter.build_variant_system("v8", d16, f16, a16)
    print(f"system (v8): ~{len(system) // 4} tokens", flush=True)

    feats = {}
    for f in sorted(FDIR.glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                feats[int(r["index"])] = {
                    "pos": [bp.clean(t).strip() for t in (r.get("pos_str") or [])[:bp.TOP_LOGITS]],
                    "max_act": r.get("maxActApprox") or 0,
                }
    live_total = sum(1 for v in feats.values() if v["max_act"] > 0)
    print(f"features: {len(feats)} total, {live_total} live", flush=True)

    resume_after = int(RESUME.read_text()) if RESUME.exists() else -1
    if resume_after >= 0:
        print(f"resuming after index {resume_after}", flush=True)

    done = set()
    buf, submitted = [], 0
    # activation files sorted numerically so indices ascend across files
    files = sorted(DIR.glob("batch-*.jsonl.gz"),
                   key=lambda p: int(p.stem.split("-")[1].split(".")[0]))

    def flush():
        nonlocal buf, submitted
        if not buf:
            return
        hi = max(int(r["custom_id"].split("-")[1]) for r in buf)
        bid = submit(buf)
        with open(IDS, "a") as fh:
            fh.write(bid + "\n")
        RESUME.write_text(str(hi))
        submitted += len(buf)
        print(f"submitted {len(buf)} (total {submitted}) as {bid}", flush=True)
        buf = []

    import subprocess
    for f in files:
        acts = {}
        # iCloud may have evicted the file; rematerialize with retries
        for attempt in range(6):
            try:
                acts = {}
                with gzip.open(f, "rt") as fh:
                    for line in fh:
                        r = json.loads(line)
                        idx = int(r["index"])
                        if idx <= resume_after or idx in done:
                            continue
                        s = bp.snippet(r)
                        if s is None:
                            continue
                        h = acts.setdefault(idx, [])
                        item = (r.get("maxValue") or 0, len(h), s)
                        if len(h) < bp.TOP_ACTS:
                            heapq.heappush(h, item)
                        else:
                            heapq.heappushpop(h, item)
                break
            except OSError as e:
                print(f"read {f.name} attempt {attempt + 1} failed ({e}); "
                      "rematerializing from iCloud", flush=True)
                if attempt == 5:
                    raise
                subprocess.run(["brctl", "download", str(f)], check=False)
                time.sleep(30 * (attempt + 1))
        for idx in sorted(acts):
            if feats.get(idx, {}).get("max_act", 0) <= 0:
                continue
            done.add(idx)
            snips = [s for _, _, s in sorted(acts[idx], reverse=True)]
            lines = [f"Feature {idx}", "Top activating snippets:"]
            lines += [f"{n}. {s}" for n, s in enumerate(snips, 1)]
            pos = feats[idx]["pos"]
            if pos:
                lines.append("Top promoted output tokens: " + ", ".join(repr(t) for t in pos))
            buf.append({
                "custom_id": f"latent-{idx}",
                "params": {"model": bp.MODEL, "max_tokens": 150, "temperature": 0,
                           "system": [{"type": "text", "text": system,
                                       "cache_control": {"type": "ephemeral"}}],
                           "messages": [{"role": "user", "content": "\n".join(lines)}]},
            })
            if len(buf) >= CHUNK:
                flush()
    flush()
    print(f"ALL SUBMITTED: {submitted} requests", flush=True)


if __name__ == "__main__":
    main()
