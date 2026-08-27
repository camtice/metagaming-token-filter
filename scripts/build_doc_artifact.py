"""Assemble the document-firing-map artifact HTML.

Injects out/doc_view_<sae>.json for the primary + confirmatory SAEs into
artifact/doc_firing_map.template.html at the /*__DATA__*/ marker.

Also attaches out/token_view_gemma3-l40-16k.json (if present) as DATA.tokenView
for the per-token Rathi-rule zoom, with Neuronpedia descriptions merged in.

Usage: python3 scripts/build_doc_artifact.py
Output: out/doc_firing_map.html
"""
import glob
import gzip
import json
import os

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
SAES = ["gemma3-l40-16k", "olmo3-l32-131k"]

data = {"saes": [json.load(open(f"{ROOT}/out/doc_view_{s}.json")) for s in SAES]}
# Probe texts are injected here, at build time, straight from the dataset file —
# they exist only inside the built HTML, keyed by doc id and shared by both SAEs.
rows = [json.loads(l) for l in open(f"{ROOT}/data/probe_dataset.jsonl")]
data["texts"] = {r["id"]: {"text": r["text"], "context": r.get("context") or None}
                 for r in rows}

tv_path = f"{ROOT}/out/token_view_gemma3-l40-16k.json"
if os.path.exists(tv_path):
    tv = json.load(open(tv_path))
    want = set(int(x) for x in tv["latents"])
    descs = {}
    for fn in glob.glob(f"{ROOT}/out/np_expl_batch-*.jsonl.gz"):
        for line in gzip.open(fn, "rt"):
            d = json.loads(line)
            if d.get("layer") == "40-gemmascope-2-res-16k" and int(d["index"]) in want:
                descs[str(d["index"])] = d["description"]
    tv["descs"] = descs
    tv["np_prefix"] = "https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-16k/"
    # tag the loaded latents: prefer the frozen audited tiers over the old manual triage
    final_path = f"{ROOT}/out/final_features_v1.json"
    tags_path = f"{ROOT}/data/latent_list_2026-07-29.json"
    if os.path.exists(final_path):
        fin = json.load(open(final_path))
        t1 = set(fin["tier1"] + fin["tier1_expansion"])
        t2 = set(fin["tier2"])
        tv["taglist"] = {
            "categories": {"t1": "tier-1 (audited final)", "t2": "tier-2 (audited final)"},
            "tags": {str(l): [["t1", "core"]] if l in t1 else [["t2", "soft"]]
                     for l in tv["latents"] if l in t1 | t2},
        }
    elif os.path.exists(tags_path):
        tv["taglist"] = json.load(open(tags_path))
    data["tokenView"] = tv
    print(f"token view: {len(tv['latents'])} latents, {len(descs)} labeled, "
          f"tags: {len(tv.get('taglist', {}).get('tags', {}))}")

blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")

tpl = open(f"{ROOT}/artifact/doc_firing_map.template.html").read()
assert "/*__DATA__*/" in tpl
html = tpl.replace("/*__DATA__*/", blob, 1)

out = f"{ROOT}/out/doc_firing_map.html"
open(out, "w").write(html)
print(f"wrote {out} ({len(html)/1e6:.2f} MB)")

# sanity: every probe text must have made it into the page (JSON-escaped compare)
missing = [r["id"] for r in rows
           if json.dumps(r["text"])[1:-1].replace("</", "<\\/") not in html]
print(f"probe texts embedded: {len(rows) - len(missing)}/{len(rows)}"
      + (f"  MISSING: {missing}" if missing else ""))
