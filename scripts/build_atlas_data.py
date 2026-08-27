"""Assemble out/atlas_data.json — every quantitative result of the qualitative
program, normalized into one row schema for the Generalization Atlas artifact.

Row: {pool, width, family, method, knob, n, F2_train, F2_in, gap, R_span, R_c2,
      R_in, fp_ctl, fp_clean, note}
Sources: gen_study_*.json (3 split seeds), quali_combo_grid.json,
quali_rathi_ablation.json, quali_docgold_results.json + hardcoded small tables
(agentic X1, width-ensemble X2, threshold provenance H5) from their run logs.
"""
import json

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
POOLS = [("fable_16k", "16k"), ("haiku_v6_16k", "16k"), ("haiku_v8_16k", "16k"),
         ("haiku_v6_65k", "65k"), ("haiku_v8_65k", "65k")]

rows = []
def famof(mname):
    if mname.startswith("R3_train"): return "ratio screen (paper vote k=2)"
    if mname.startswith("R1_train"): return "control-cap screen"
    if mname.startswith("R3_contam"): return "audit: same-corpus stats"
    if mname == "HILLCLIMB_train": return "hill-climb search"
    return "no-training baselines"
def pretty(mname):
    if mname.startswith("R3_train rho="):
        return f"ratio ρ≥{mname.split('=')[1]} · vote k=2"
    if mname.startswith("R1_train tau="):
        return f"control-cap τ≤{mname.split('=')[1]} · vote k=2"
    if mname.startswith("R3_contam rho="):
        return f"[audit] ratio ρ≥{mname.split('=')[1]} same-corpus stats"
    if mname == "HILLCLIMB_train": return "hill-climb on G-TRAIN"
    return mname

for pool, width in POOLS:
    d = json.load(open(f"{ROOT}/out/gen_study_{pool}.json"))
    ts3, ts1 = d.get("train_selected_R3"), d.get("train_selected_R1")
    for mname, m in d["methods"].items():
        knob = mname.split(" ")[-1] if " " in mname else ""
        rows.append({
            "pool": pool, "width": width, "family": famof(mname), "method": pretty(mname),
            "knob": knob, "n": m.get("n"),
            "F2_train": m.get("F2_train"), "F2_in": m.get("F2_in"),
            "F2_all": m.get("F2_all"),
            "gap": (None if m.get("F2_train") is None or m.get("F2_in") is None
                    else round(m["F2_train"] - m["F2_in"], 3)),
            "R_span": m.get("R_span"), "R_c2": m.get("R_c2"),
            "R_in": m.get("R_dense_in"), "fp_ctl": m.get("fp_ctl"),
            "fp_clean": m.get("fp_clean"),
            "sel": "★" if mname in (ts3, ts1) else ""})

# complete pool fair grid (rho x seed, both widths) — family "R3 + seed vote"
fg = json.load(open(f"{ROOT}/out/pool_fairgrid.json"))
for k, m in fg.items():
    if m["seed"] == 2 and m["rho"] in (1, 2, 3, 5, 10) and m["width"] != "262k":
        continue  # duplicates of R3_train rows (262k has no gen_study run — keep its k=2 rows)
    rows.append({"pool": m["pool"], "width": m["width"], "family": "ratio screen · vote k≥4",
                 "method": f"ratio ρ≥{m['rho']} · vote k={m['seed']}",
                 "knob": f"ρ{m['rho']}·s{m['seed']}",
                 "n": m["n"], "F2_train": None, "F2_in": m["F2in"],
                 "F2_all": m["F2all"], "gap": None,
                 "R_span": m["Rspan"], "R_c2": m["Rc2"], "R_in": m["Rin"],
                 "fp_ctl": m["fp"], "fp_clean": None, "sel": ""})

# full-dictionary baselines (no judge pool)
import os
for width, pool in (("16k", "ALL 16k dictionary"), ("65k", "ALL 65k dictionary")):
    p = f"{ROOT}/out/all{width}_baseline.json"
    if not os.path.exists(p):
        continue
    for mname, m in json.load(open(p)).items():
        rows.append({"pool": pool, "width": width, "family": "ratio, ALL features (no judge)"
                     if " R3 " in mname else "control-cap, ALL features",
                     "method": mname, "knob": "", "n": m["n"], "F2_train": None,
                     "F2_in": m["F2in"], "F2_all": m.get("F2all"), "gap": None,
                     "R_span": m["Rspan"], "R_c2": m["Rc2"], "R_in": m["Rin"],
                     "fp_ctl": m["fp"], "fp_clean": None, "sel": ""})

# seed robustness (rules-only reruns)
seeds_tbl = []
for pool in ("haiku_v6_16k", "haiku_v6_65k"):
    for tag, path in (("20260815", f"gen_study_{pool}.json"),
                      ("20260816", f"gen_study_{pool}_s20260816.json"),
                      ("20260817", f"gen_study_{pool}_s20260817.json")):
        d = json.load(open(f"{ROOT}/out/{path}"))
        r10 = d["methods"]["R3_train rho=10"]; r5 = d["methods"]["R3_train rho=5"]
        r1 = d["methods"][d["train_selected_R1"]] if d.get("train_selected_R1") else None
        seeds_tbl.append({"pool": pool, "seed": tag,
                          "rho10_F2": r10["F2_in"], "rho10_Rspan": r10["R_span"],
                          "rho5_F2": r5["F2_in"], "rho5_Rspan": r5["R_span"],
                          "R1_F2": None if r1 is None else r1["F2_in"]})

# rathi ablation
ab = json.load(open(f"{ROOT}/out/quali_rathi_ablation.json"))
ab_rows = []
for w in ("16k", "65k"):
    for k, m in ab[w].items():
        sd = int(k[2]); seed = int(k.split("seed")[1][0]); win = k.endswith("_win")
        ab_rows.append({"width": w, "sd": sd, "seed": seed, "win": win, **m})

# real-world (docgold + agentic hardcoded from run log)
dg = json.load(open(f"{ROOT}/out/quali_docgold_results.json"))
agentic = {"A": {"baseline": .394, "notice": .409, "scratch": .391, "auc_notice": .553},
           "B": {"baseline": .548, "notice": .566, "scratch": .549, "auc_notice": .547},
           "C": {"baseline": .193, "notice": .214, "scratch": .198, "auc_notice": .606},
           "E": {"baseline": .275, "notice": .292, "scratch": .279, "auc_notice": .567}}
RULE_NAMES = {"A": "ratio ρ≥5 · k=2 (65k)", "B": "ratio ρ≥1 · k=4 (65k)", "C": "ratio ρ≥5 · k=2 (16k)",
              "D": "control-cap τ≤.1% (16k)", "E": "D2 hill-climbed (16k)", "F": "full 65k pool"}
real = []
for code in "ABCDEF":
    k = dg["kimi"][code]
    aucs = [v["auc"] for kk, v in k.items() if isinstance(v, dict)]
    g = dg["docgold"][code]
    real.append({"code": code, "name": RULE_NAMES[code],
                 "kimi_mean_auc": round(sum(aucs)/len(aucs), 3),
                 "kimi_umwp": k["anti_hall_umwp:unanswerable-vs-answerable"]["auc"],
                 "kimi_falseqa": k["anti_hall_falseqa:false_premise-vs-true_premise"]["auc"],
                 "kimi_ifc": k["if_conflict:contradictory-vs-satisfiable"]["auc"],
                 "westover_auc": g["westover_auc"], "needham_auc": g["needham_auc"],
                 "agentic_notice_auc": agentic.get(code, {}).get("auc_notice")})

# choice ledger (the summary deltas, curated from all experiments)
ledger = [
 {"choice": "Screen statistic: ratio vs control-cap", "effect": "+.12–.16 held-out F2; the control-cap collapses on the held-out domain (c2 .39–.66 vs .78–.87)", "verdict": "R3", "size": 3},
 {"choice": "Simple ratio rule vs hill-climb search", "effect": "ratio rule wins held-out F2 in 5/5 pools (+.02–.14); smaller train→test gap", "verdict": "simple rule", "size": 3},
 {"choice": "Token vote quorum k (paper k=2 → k=3/4)", "effect": "halves control FP at ~2pp recall; ρ=1–2 + seed≥4 dominates ρ-only frontier by 2–7pp span recall at matched FP", "verdict": "vote k≥3–4, loosen ρ", "size": 3},
 {"choice": "Windowing (adjacency growth)", "effect": "+.09–.13 span recall, up to +.21 held-out-domain recall (16k); costs ~+.03 fp", "verdict": "keep", "size": 3},
 {"choice": "SAE width 16k → 65k", "effect": "+.01–.03 held-out F2; adds refusal/safety/deception detectors; better c2/span transfer at matched ρ", "verdict": "65k", "size": 2},
 {"choice": "Same-corpus (contaminated) statistics", "effect": "inflates held-out F2 by +.02–.04 at matched ρ — the measured cost of the old practice", "verdict": "use disjoint stats", "size": 2},
 {"choice": "Labeler model / prompt (fable, haiku v6, v8)", "effect": "±.01–.02 — noise at every checkpoint of the project", "verdict": "any", "size": 1},
 {"choice": "Threshold statistic (3SD/4SD/5SD)", "effect": "no measurable effect", "verdict": "any", "size": 0},
 {"choice": "Threshold corpus (TEST vs deployment)", "effect": "recall shifts .001 — no effect", "verdict": "any", "size": 0},
 {"choice": "Width ensemble (16k AND 65k)", "effect": "lands on single-width frontier; 2× harvest cost for nothing", "verdict": "refuted", "size": 0},
 {"choice": "Split seed (3 seeds)", "effect": "rankings unchanged ±.02 — conclusions not seed artifacts", "verdict": "robust", "size": 0},
 {"choice": "Judge pool vs ALL dictionary features", "effect": "ALL-dict + scaled ρ looks better on internal G-TEST metrics (F2_all .858) but this is synthetic-register leakage: on real text it flags 83–90% of ALL westover posts and external AUC drops (.76→.68). The dense corpus has no AI-but-not-oversight negatives, so the statistic cannot separate AI-topical from oversight-specific — the judge supplies exactly that distinction. Hard-negative denominator (needham) does not repair it. Judge misses are real though (e.g. \"my own internal goals\", ratio 10,152) — use the statistic to propose additions to the pool, not to replace it.", "verdict": "judge stage necessary", "size": 3},
]

def cell(pool, method):
    for r in rows:
        if r["pool"] == pool and r["method"] == method:
            return r
    return None
def fgc(pool, rho, ks): return fg.get(f"{pool} rho={rho} seed>={ks}")
gs = {p: json.load(open(f"{ROOT}/out/gen_study_{p}.json")) for p, _ in POOLS}
def gsm(pool, m): return gs[pool]["methods"].get(m)
pairs = []
for pool in ("haiku_v8_16k", "haiku_v8_65k"):
    r1k = gs[pool].get("train_selected_R1")
    pairs.append({"finding": "Screen statistic: control-cap → ratio", "pool": pool,
                  "a_lab": "control-cap (train-selected τ)", "a": gsm(pool, r1k)["F2_all"],
                  "b_lab": "ratio ρ≥10", "b": gsm(pool, "R3_train rho=10")["F2_all"]})
for pool in ("fable_16k", "haiku_v8_16k", "haiku_v8_65k"):
    pairs.append({"finding": "Search → simple ratio rule", "pool": pool,
                  "a_lab": "hill-climb on G-TRAIN", "a": gsm(pool, "HILLCLIMB_train")["F2_all"],
                  "b_lab": "ratio ρ≥10", "b": gsm(pool, "R3_train rho=10")["F2_all"]})
for pool in ("fable_16k", "haiku_v8_16k", "haiku_v8_65k"):
    pairs.append({"finding": "Token vote k=2 (paper) → k=4 (ratio ρ≥5 set, unchanged features)",
                  "pool": pool,
                  "a_lab": "vote k=2", "a": fgc(pool, 5, 2)["F2all"],
                  "b_lab": "vote k=4", "b": fgc(pool, 5, 4)["F2all"],
                  "note": "same selected features; only the per-token co-firing quorum changes"})
wj = json.load(open(f"{ROOT}/out/windowing_f2all.json"))
for w in ("16k", "65k"):
    pairs.append({"finding": "Windowing (Rathi adjacency growth) off → on (ρ≥5 set)",
                  "pool": f"haiku_v6_{w}",
                  "a_lab": f"seeds only (Rspan {wj[w+'_nowin']['Rspan']:.2f}, Rc2 {wj[w+'_nowin']['Rc2']:.2f})",
                  "a": wj[w + "_nowin"]["F2all"],
                  "b_lab": f"windowed (Rspan {wj[w+'_win']['Rspan']:.2f}, Rc2 {wj[w+'_win']['Rc2']:.2f})",
                  "b": wj[w + "_win"]["F2all"],
                  "note": "label spreads from seeds across adjacent tokens where any kept feature is active — per the paper"})
pairs.append({"finding": "SAE width 16k → 65k (ratio ρ≥10 · vote k=4)", "pool": "haiku_v8",
              "a_lab": "16k", "a": fgc("haiku_v8_16k", 10, 4)["F2all"],
              "b_lab": "65k", "b": fgc("haiku_v8_65k", 10, 4)["F2all"]})
pairs.append({"finding": "SAE width 65k → 262k (only l0_small ships: 21 vs 60 active latents/token)",
              "pool": "haiku_v8",
              "a_lab": "65k ρ≥10 · k=4", "a": fgc("haiku_v8_65k", 10, 4)["F2all"],
              "b_lab": "262k ρ≥3 · k=2 (its best cell)", "b": fgc("haiku_v8_262k", 3, 2)["F2all"],
              "note": "width and L0 are confounded at 262k — the sparse SAE starves the co-firing vote (k=4 collapses recall: Rc2 .12 at ρ10), so the drop is attributable to L0, not dictionary size"})
pairs.append({"finding": "Judge pool (fable vs haiku-v8, ρ≥5 · k=4)", "pool": "16k",
              "a_lab": "fable", "a": fgc("fable_16k", 5, 4)["F2all"],
              "b_lab": "haiku v8", "b": fgc("haiku_v8_16k", 5, 4)["F2all"],
              "note": "labeler/prompt choice is noise; v6 dropped from reporting (indistinguishable from v8)"})
candidates = []
for pool, label in (("haiku_v8_65k", "h8-65k"), ("haiku_v8_16k", "h8-16k"), ("fable_16k", "fable-16k")):
    for rho, ks in ((10, 4), (5, 4), (3, 4), (2, 4)):
        c = fgc(pool, rho, ks)
        if c:
            candidates.append({"name": f"{label} · ratio ρ≥{rho} · vote k≥4", **c,
                               "wauc": None, "kauc": None})
candidates.sort(key=lambda c: -(c.get("F2all") or 0))
candidates = candidates[:7]

json.dump({"rows": rows, "seeds": seeds_tbl, "ablation": ab_rows, "real": real,
           "ledger": ledger, "pairs": pairs, "candidates": candidates},
          open(f"{ROOT}/out/atlas_data.json", "w"))
print(f"atlas data: {len(rows)} master rows, {len(ab_rows)} ablation rows, "
      f"{len(real)} real-world rows, {len(seeds_tbl)} seed rows")
