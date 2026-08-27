"""Build Haiku Batch requests for the 65k SAE (gemma-3-27b/40-gemmascope-2-res-65k),
using the winning prompt variant. Few-shot examples render from 16k data (they
are illustrative content); targets render from 65k data.

Usage: python3 relabel/build_65k.py <variant: v1|v2|v3>
"""

import gzip
import heapq
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_prompts as bp
import prompt_iter

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIR65 = "gemma-3-27b-40-gemmascope-2-res-65k"


def load_65k():
    feats = {}
    for f in sorted((ROOT / "features" / DIR65).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                feats[int(r["index"])] = {
                    "pos": [bp.clean(t).strip() for t in (r.get("pos_str") or [])[:bp.TOP_LOGITS]],
                    "max_act": r.get("maxActApprox") or 0,
                }
    acts = {}
    for f in sorted((ROOT / "activations" / DIR65).glob("batch-*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                idx = int(r["index"])
                s = bp.snippet(r)
                if s is None:
                    continue
                heap = acts.setdefault(idx, [])
                item = (r.get("maxValue") or 0, len(heap), s)
                if len(heap) < bp.TOP_ACTS:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)
    acts = {i: [s for _, _, s in sorted(h, reverse=True)] for i, h in acts.items()}
    return feats, acts


def main():
    variant = sys.argv[1]
    d16, f16, a16 = bp.load_data()
    system = prompt_iter.build_variant_system(variant, d16, f16, a16)
    print(f"system ({variant}): ~{len(system) // 4} tokens", flush=True)

    feats, acts = load_65k()
    live = sorted(i for i in acts if feats.get(i, {}).get("max_act", 0) > 0)
    print(f"65k: {len(feats)} features, {len(live)} live to classify", flush=True)

    out = ROOT / "relabel" / "requests_65k.jsonl"
    with open(out, "w") as fh:
        for idx in live:
            fh.write(json.dumps({
                "custom_id": f"latent-{idx}",
                "params": {
                    "model": bp.MODEL, "max_tokens": 150, "temperature": 0,
                    "system": [{"type": "text", "text": system,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user",
                                  "content": bp.render_feature(idx, d16, feats, acts)}],
                },
            }) + "\n")
    print(f"wrote {len(live)} requests -> {out} "
          f"({out.stat().st_size // 2**20} MB)", flush=True)


if __name__ == "__main__":
    main()
