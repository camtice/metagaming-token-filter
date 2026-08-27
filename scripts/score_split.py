"""Eval protocol — score a feature set on a split harvest.

Metrics (Rathi rule: >=2 features >= mean+4SD of the scored corpus, windowing):
  * MAIN (protocol v5, 2026-08-15): forget = human span tokens PLUS all tokens
    of dense_forget c1-c4 docs (all-forget assumption; c5/SWE excluded per the
    2026-08-12 retain ruling). P / R / F2 / F1 over annotated + dense c1-c4
    docs. Rationale: dense docs register-match the controls, so selection can
    separate forget from retain without chat-template confounds.
  * annotated (secondary report): span/clean-annotated docs only — token
    P / R / F2 / F1 vs human spans, per-register recall, FP rate on
    reviewed-clean docs. The pre-v5 headline; kept as the human-label check.
  * dense_forget docs: coverage per dense category (incl. c5, reported
    separately)
  * unlabeled docs: descriptive coverage only

Validation lock: scoring a harvest whose docs carry role=validation requires
--frozen-set <file>; the access is appended to
data/splits/validation_access_log.jsonl.

Usage:
  python score_split.py out/harvest_test_v1 data/test_docs_v1.jsonl --set og
  python score_split.py out/harvest_val_v1 data/val_docs_v1.jsonl \
      --set og --frozen-set data/latent_list_2026-07-29.json --actor baseline-reference
Output: out/score_<harvest>_<setname>.json + console table
"""
import argparse
import datetime
import json
import os

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"

ap = argparse.ArgumentParser()
ap.add_argument("harvest_prefix")
ap.add_argument("docs_jsonl")
ap.add_argument("--set", dest="set_name", default="og",
                help="'og' (the 222 blocklist baseline) or a path to a JSON with a latent list")
ap.add_argument("--subset", default=None,
                help="set name inside a candidate_sets JSON (format: sets.<name>.latents)")
ap.add_argument("--confidence", default="all", choices=["h", "hm", "all"],
                help="confidence filter for candidate-set latents (h / h+m / all)")
ap.add_argument("--frozen-set", default=None,
                help="required for validation harvests: the frozen feature-set file being scored")
ap.add_argument("--actor", default=None, help="who/what this validation access is for (logged)")
ap.add_argument("--sealed", action="store_true",
                help="write results to out/sealed/ without displaying them (for validation runs "
                     "whose numbers should only be revealed at a milestone comparison)")
ap.add_argument("--unlock-validation", action="store_true",
                help="explicit opt-in required to run on a validation harvest at all; "
                     "validation results are ALWAYS sealed, never displayed")
args = ap.parse_args()

if args.set_name == "og":
    feats = sorted(int(k) for k in json.load(open(f"{ROOT}/data/latent_list_2026-07-29.json"))["tags"])
    set_label = "OG_blocklist_222"
else:
    d = json.load(open(args.set_name))
    if isinstance(d, dict) and "sets" in d:
        assert args.subset, "--subset required for a candidate_sets JSON"
        lat = d["sets"][args.subset]["latents"]
        keep = {"h": {"h"}, "hm": {"h", "m"}, "all": {"h", "m", "l"}}[args.confidence]
        feats = sorted({int(l) for l, cat, conf in lat if conf in keep})
        set_label = f"{args.subset}_{args.confidence}"
    else:
        feats = sorted(d["members"] if isinstance(d, dict) and "members" in d else d)
        set_label = os.path.basename(args.set_name).replace(".json", "")

docs = [json.loads(l) for l in open(args.docs_jsonl)]
by_id = {d["id"]: d for d in docs}
is_validation = any(d.get("role") == "validation" for d in docs)
if is_validation:
    if not args.unlock_validation:
        raise SystemExit("REFUSING: validation does not run by default. Pass --unlock-validation "
                         "(plus --frozen-set) only for a frozen candidate at a milestone. "
                         "Results will be sealed, never displayed.")
    if not args.frozen_set:
        raise SystemExit("REFUSING: validation harvest requires --frozen-set (see split manifest rules)")
    args.sealed = True   # validation output is always sealed
    with open(f"{ROOT}/data/splits/validation_access_log.jsonl", "a") as f:
        f.write(json.dumps({"ts": datetime.datetime.utcnow().isoformat() + "Z",
                            "set": set_label, "frozen_set_file": args.frozen_set,
                            "actor": args.actor or "unspecified",
                            "harvest": args.harvest_prefix}) + "\n")

z = np.load(args.harvest_prefix + ".npz")
meta = json.load(open(args.harvest_prefix + ".meta.json"))
W = meta["width"]
av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
ntok = z["doc_ntok"].astype(np.int64)
nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, dtype=np.int64); np.cumsum(ntok, out=tok_base[1:])
n_tok = int(ntok.sum())
flat = tok_base[z["doc_idx"]] + z["tok_idx"]
offs = z["offsets"].reshape(-1, 2)
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
thr = mu + 4 * sd

doc_ids = [d["id"] for d in meta["docs"]]
lt = np.array([by_id[i]["label_type"] for i in doc_ids])
reg = np.array([by_id[i]["register"] for i in doc_ids])
cfg = np.array([by_id[i].get("config", "") for i in doc_ids])
tok_doc = np.repeat(np.arange(nd), ntok)

gt = np.zeros(n_tok, bool)
for di, did in enumerate(doc_ids):
    for cs, ce in by_id[did].get("char_spans", []):
        o = offs[tok_base[di]:tok_base[di + 1]]
        m = (o[:, 0] < ce) & (o[:, 1] > cs)
        gt[tok_base[di] + np.flatnonzero(m)] = True

members = np.array(feats)
selm = np.isin(z["lat_idx"], members)
li, fl, a = z["lat_idx"][selm], flat[selm], av[selm]
seed = np.bincount(fl[a >= thr[li]], minlength=n_tok) >= 2
posany = np.bincount(fl, minlength=n_tok) > 0
pred = seed.copy()
for d in range(nd):
    s, e = tok_base[d], tok_base[d + 1]
    lab, pv = pred[s:e].copy(), posany[s:e]
    ch = True
    while ch:
        ch = False
        for t in np.flatnonzero(~lab & pv):
            if (t > 0 and lab[t-1]) or (t < len(lab)-1 and lab[t+1]):
                lab[t] = True; ch = True
    pred[s:e] = lab

res = {"set": set_label, "n_features": len(feats), "harvest": args.harvest_prefix,
       "thresholds": "mean+4SD over this harvest", "n_docs": nd, "n_tokens": n_tok}
annm = np.isin(lt[tok_doc], ["spans", "clean"])
dense14 = (lt[tok_doc] == "dense_forget") & (np.char.find(cfg[tok_doc].astype(str), "c5") < 0)
if dense14.any():
    mainm = annm | dense14
    gt_main = gt | dense14
    tpm = int((pred & gt_main & mainm).sum()); fpm_ = int((pred & ~gt_main & mainm).sum())
    fnm = int((~pred & gt_main & mainm).sum())
    Pm = tpm / max(tpm + fpm_, 1); Rm = tpm / max(tpm + fnm, 1)
    res["main"] = {"P": round(Pm, 3), "R": round(Rm, 3),
                   "F2": round(5 * Pm * Rm / max(4 * Pm + Rm, 1e-9), 3),
                   "F1": round(2 * Pm * Rm / max(Pm + Rm, 1e-9), 3),
                   "forget_def": "human spans + dense c1-c4 all-forget (protocol v5)"}
tp = int((pred & gt & annm).sum()); fp = int((pred & ~gt & annm).sum()); fn = int((~pred & gt & annm).sum())
P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
res["annotated"] = {"P": round(P, 3), "R": round(R, 3),
                    "F2": round(5 * P * R / max(4 * P + R, 1e-9), 3),
                    "F1": round(2 * P * R / max(P + R, 1e-9), 3),
                    "fp_rate_clean_docs": round(float(pred[lt[tok_doc] == "clean"].mean()), 3)
                    if (lt == "clean").any() else None}
for rg in sorted(set(reg[np.isin(lt, ["spans", "clean"])])):
    m = annm & (reg[tok_doc] == rg)
    if not (gt & m).any():
        res["annotated"][f"recall_{rg}"] = None
        continue
    res["annotated"][f"recall_{rg}"] = round(float(pred[gt & m].mean()), 3)
for fam_lt, key in (("dense_forget", "dense_coverage"), ("swe_forget", "swe_coverage")):
    if (lt == fam_lt).any():
        cov = {}
        for c in sorted(set(cfg[lt == fam_lt])):
            m = (lt[tok_doc] == fam_lt) & (cfg[tok_doc] == c)
            cov[c] = round(float(pred[m].mean()), 3)
        cov["overall"] = round(float(pred[lt[tok_doc] == fam_lt].mean()), 3)
        res[key] = cov
if (lt == "assumed_clean").any():
    fp = {}
    for c in sorted(set(cfg[lt == "assumed_clean"])):
        m = (lt[tok_doc] == "assumed_clean") & (cfg[tok_doc] == c)
        fp[c] = round(float(pred[m].mean()), 3)
    fp["overall"] = round(float(pred[lt[tok_doc] == "assumed_clean"].mean()), 3)
    res["fp_assumed_clean"] = fp
if (lt == "unlabeled").any():
    res["unlabeled_coverage"] = round(float(pred[lt[tok_doc] == "unlabeled"].mean()), 3)

if args.sealed:
    os.makedirs(f"{ROOT}/out/sealed", exist_ok=True)
    outp = f"{ROOT}/out/sealed/score_{os.path.basename(args.harvest_prefix)}_{set_label}.json"
    json.dump(res, open(outp, "w"), indent=1)
    print(f"SEALED: results written (not displayed) to {outp} — reveal only at a milestone comparison")
else:
    outp = f"{ROOT}/out/score_{os.path.basename(args.harvest_prefix)}_{set_label}.json"
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", outp)
