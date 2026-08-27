"""Simple screening rules vs direct optimization (2026-08-15, TEST v5).

Question: how close can a ONE-KNOB screen over a judge pool get to the
optimized sets, with no combinatorial search? Each rule keeps every latent
the rule admits and evaluates the full Rathi rule on that set.

Rules (all thresholds on a per-latent scalar; sweep = sorted prefix):
  R1 ctl_rate    keep latents with above-threshold fire rate on TEST control
                 tokens <= tau                      (the user's proposed rule)
  R2 corpus_rate keep latents with fire rate on ALL corpus tokens <= tau
                 (fully label-free)
  R3 ratio       keep latents with smoothed (forget rate)/(control rate)
                 >= rho — one statistic per latent, no search
  R4 stop_frac   keep latents whose firings are <= sigma stopword/punct
                 (label-free register screen)

Metrics: protocol v5 main (spans + dense c1-c4) on TEST v5 (manifest sha
0f6f83c7; controls = 2,625 docs incl. the 2026-08-15 refresh TEST half).
Usage: python screen_rules_sweep.py --subset fable_16k --harvest out/harvest_test_v5 --tag fable_16k
"""
import argparse
import json
import re

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
DOCS = f"{ROOT}/data/test_docs_v5.jsonl"
POOL = f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"

ap = argparse.ArgumentParser()
ap.add_argument("--subset", required=True)
ap.add_argument("--harvest", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--embed-scores", default=None,
                help="out/embed_scores_{16k,65k}.json for R5/R6 (Paulo embedding screen)")
args = ap.parse_args()
HARV = f"{ROOT}/{args.harvest}"

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was
were be been being am do does did have has had will would can could should may might must
that this these those it its they them their he she his her you your i we our my me us not
no nor so than too very just only also there here when where which who whom what how why
all any both each few more most other some such own same s t don now""".split())
WORD = re.compile(r"[A-Za-z]")

z = np.load(HARV + ".npz")
meta = json.load(open(HARV + ".meta.json"))
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

docs = [json.loads(l) for l in open(DOCS)]
by_id = {d["id"]: d for d in docs}
doc_ids = [d["id"] for d in meta["docs"]]
lt = np.array([by_id[i]["label_type"] for i in doc_ids])
cfg = np.array([by_id[i].get("config", "") for i in doc_ids])
tok_doc = np.repeat(np.arange(nd), ntok)
gt = np.zeros(n_tok, bool)
for di, did in enumerate(doc_ids):
    for cs, ce in by_id[did].get("char_spans", []):
        o = offs[tok_base[di]:tok_base[di + 1]]
        m = (o[:, 0] < ce) & (o[:, 1] > cs)
        gt[tok_base[di] + np.flatnonzero(m)] = True
annm = np.isin(lt[tok_doc], ["spans", "clean"])
cleanm = lt[tok_doc] == "clean"
ctlm = lt[tok_doc] == "assumed_clean"
dense14 = (lt[tok_doc] == "dense_forget") & (np.char.find(cfg[tok_doc].astype(str), "c5") < 0)
is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True
gt_ann = gt & annm; n_gt = int(gt_ann.sum())
mainm = annm | dense14
gt_main = gt_ann | dense14
n_gtm = int(gt_main.sum())
print(f"corpus: {nd} docs {n_tok:,} tokens | ctl tokens {int(ctlm.sum()):,} | forget {n_gtm:,}")

pool_raw = json.load(open(POOL))["sets"][args.subset]["latents"]
members = np.array(sorted({int(l) for l, _c, _f in pool_raw}))
nm = len(members)
lat2pos = -np.ones(W, np.int64); lat2pos[members] = np.arange(nm)
selm = lat2pos[z["lat_idx"]] >= 0
mp = lat2pos[z["lat_idx"][selm]]
mf = flat[selm]
mabove = av[selm] >= thr[z["lat_idx"][selm]]
order = np.argsort(mp, kind="stable")
mp_s, mf_s, ma_s = mp[order], mf[order], mabove[order]
bounds = np.searchsorted(mp_s, np.arange(nm + 1))
per_flat = [mf_s[bounds[i]:bounds[i+1]] for i in range(nm)]
per_above = [mf_s[bounds[i]:bounds[i+1]][ma_s[bounds[i]:bounds[i+1]]] for i in range(nm)]

def rate(mask):
    k = np.zeros(nm)
    h = mask[mf_s] & ma_s
    np.add.at(k, mp_s[h], 1)
    return k / max(int(mask.sum()), 1)

r_ctl = rate(ctlm)
r_all = rate(np.ones(n_tok, bool))
q_forget = rate(gt_main)
ratio = (q_forget * n_gtm + 0.5) / (r_ctl * int(ctlm.sum()) + 0.5)

# stopword fraction of firings (surface pass)
surface_stop = np.zeros(n_tok, bool)
for di, did in enumerate(doc_ids):
    txt = by_id[did]["text"]
    o = offs[tok_base[di]:tok_base[di + 1]]
    for j in range(len(o)):
        s = txt[o[j, 0]:o[j, 1]].strip().lower()
        surface_stop[tok_base[di] + j] = (not s) or (not WORD.search(s)) or (s in STOP)
stop_cnt = np.zeros(nm); tot_cnt = np.zeros(nm)
np.add.at(tot_cnt, mp_s[ma_s], 1)
np.add.at(stop_cnt, mp_s[ma_s & surface_stop[mf_s]], 1)
stop_frac = np.where(tot_cnt > 0, stop_cnt / np.maximum(tot_cnt, 1), 1.0)

prev_shift = np.zeros(n_tok, bool)

def evaluate(seedcnt, pacnt):
    seeds = seedcnt >= 2
    pa = pacnt > 0
    prev_shift[1:] = pa[:-1]; prev_shift[0] = False
    start = pa & (~prev_shift | is_doc_start)
    run_id = np.cumsum(start) - 1
    nruns = int(start.sum())
    if nruns == 0:
        return dict(Pm=0.0, Rm=0.0, F2m=0.0, R=0.0, fp_ctl=0.0, fp_clean=0.0)
    seed_runs = np.zeros(nruns, bool)
    seed_runs[run_id[seeds]] = True
    pred = pa & seed_runs[run_id]
    tp = int((pred & gt_main & mainm).sum()); fpa = int((pred & mainm & ~gt_main).sum())
    Rm = tp / max(n_gtm, 1); Pm = tp / max(tp + fpa, 1)
    return {"Pm": Pm, "Rm": Rm, "F2m": 5*Pm*Rm/max(4*Pm+Rm, 1e-9),
            "R": float(pred[gt_ann].mean()) if n_gt else 0.0,
            "fp_ctl": float(pred[ctlm].mean()), "fp_clean": float(pred[cleanm].mean())}

def sweep(scalar, keep_low, label, subset_pos=None):
    """Sweep threshold on scalar; keep_low=True keeps latents with LOW scalar.
    subset_pos: optional restriction to these pool positions (for rule combos)."""
    pos = np.arange(nm) if subset_pos is None else np.asarray(sorted(subset_pos))
    sc = scalar[pos]
    order_idx = pos[np.argsort(sc if keep_low else -sc, kind="stable")]
    seedcnt = np.zeros(n_tok, np.int32); pacnt = np.zeros(n_tok, np.int32)
    rows = []
    npool = len(order_idx)
    grid = sorted(set(list(range(2, 60, 2)) + list(range(60, 300, 8))
                      + list(range(300, npool + 1, max(1, npool // 60))) + [npool]))
    for j, li in enumerate(order_idx, 1):
        np.add.at(seedcnt, per_above[li], 1)
        np.add.at(pacnt, per_flat[li], 1)
        if j not in grid:
            continue
        m = evaluate(seedcnt, pacnt)
        knob = float(scalar[order_idx[j - 1]])
        rows.append({"K": j, "knob": round(knob, 6), **{k: round(v, 4) for k, v in m.items()}})
    best = max(rows, key=lambda r: r["F2m"])
    feas = [r for r in rows if r["fp_ctl"] <= 0.10]
    bestc = max(feas, key=lambda r: r["F2m"]) if feas else None
    print(f"  {label}: best F2m={best['F2m']} (K={best['K']} knob={best['knob']} fp={best['fp_ctl']}) | "
          f"best@fp<=.10: {None if not bestc else (bestc['F2m'], bestc['K'], bestc['knob'])}", flush=True)
    return {"rows": rows, "best": best, "best_fp10": bestc}

res = {"subset": args.subset, "harvest": args.harvest, "n_pool": nm,
       "protocol": "v5 main metric, TEST v5 (manifest 0f6f83c7)"}
print(f"== {args.subset} ({nm} latents) ==")
res["R1_ctl_rate"] = sweep(r_ctl, True, "R1 ctl_rate cap")
res["R2_corpus_rate"] = sweep(r_all, True, "R2 corpus_rate cap (label-free)")
res["R3_ratio"] = sweep(ratio, False, "R3 forget/ctl ratio")
res["R4_stop_frac"] = sweep(stop_frac, True, "R4 stop_frac cap (label-free)")
if args.embed_scores:
    es = {int(k): v for k, v in json.load(open(f"{ROOT}/{args.embed_scores}")).items()}
    esc = np.array([es.get(int(l)) if es.get(int(l)) is not None else -1.0 for l in members])
    res["R5_embed"] = sweep(esc, False, "R5 embed-score floor (Paulo)")
    surv = [i for i in range(nm) if esc[i] >= 0.9]
    res["R5_at_0.9_n"] = len(surv)
    if len(surv) >= 2:
        # exact point evaluation at the Rathi threshold 0.9
        sc9 = np.zeros(n_tok, np.int32); pc9 = np.zeros(n_tok, np.int32)
        for i in surv:
            np.add.at(sc9, per_above[i], 1)
            np.add.at(pc9, per_flat[i], 1)
        res["R5_at_0.9"] = {k: round(v, 4) for k, v in evaluate(sc9, pc9).items()}
        print(f"  R5@0.9 exact: n={len(surv)} {res['R5_at_0.9']}", flush=True)
        res["R6_embed09_then_ctl"] = sweep(r_ctl, True, "R6 embed>=0.9 then R1 ctl cap", surv)
full = evaluate(*(lambda: (
    (lambda sc, pc: (sc, pc))(np.zeros(n_tok, np.int32), np.zeros(n_tok, np.int32))))())
# full-pool reference computed properly:
sc = np.zeros(n_tok, np.int32); pc = np.zeros(n_tok, np.int32)
for i in range(nm):
    np.add.at(sc, per_above[i], 1)
    np.add.at(pc, per_flat[i], 1)
res["full_pool"] = {k: round(v, 4) for k, v in evaluate(sc, pc).items()}
print("  full pool:", res["full_pool"])
json.dump(res, open(f"{ROOT}/out/screen_rules_{args.tag}.json", "w"))
print(f"wrote out/screen_rules_{args.tag}.json")
