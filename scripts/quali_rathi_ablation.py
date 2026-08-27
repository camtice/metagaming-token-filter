"""H6: ablations of the Rathi firing rule itself (2026-08-15).

Fixed feature set (R3 train-stats rho=5 on haiku_v6, both widths); vary:
  - seed requirement: >=1, >=2 (paper), >=3
  - windowing: on (paper) / off (seeds only)
  - threshold: mean+kSD for k in {3, 4 (paper), 5}
Metrics on G-TEST (span recall, c2 recall, dense-in recall, fp_ctl, F2_in).
"""
import json

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]

def run(width, pool_name):
    HARV = {"16k": f"{ROOT}/out/harvest_test_v5", "65k": f"{ROOT}/out/harvest_test_v5_65k"}[width]
    zf = np.load(HARV + ".npz")
    z = {k: zf[k] for k in zf.files}
    meta = json.load(open(HARV + ".meta.json"))
    W = meta["width"]
    av = z["act"].astype(np.float64); av[np.isinf(av)] = 65504.0
    ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
    tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
    n_tok = int(ntok.sum())
    flat = tok_base[z["doc_idx"]] + z["tok_idx"]
    offs = z["offsets"].reshape(-1, 2)
    mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
    sdv = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))

    doc_ids = [d["id"] for d in meta["docs"]]
    rl = np.array([role[i] for i in doc_ids])
    tok_doc = np.repeat(np.arange(nd), ntok)
    trole = rl[tok_doc]
    gt = np.zeros(n_tok, bool)
    for di, did in enumerate(doc_ids):
        for cs, ce in by_id[did].get("char_spans", []):
            o = offs[tok_base[di]:tok_base[di + 1]]
            m = (o[:, 0] < ce) & (o[:, 1] > cs)
            gt[tok_base[di] + np.flatnonzero(m)] = True
    m_span = (trole == "gtest_span") & gt
    m_c2 = trole == "gtest_c2"; m_gd = trole == "gtest_dense"
    m_ctl = trole == "gtest_ctl"
    m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
    is_doc_start = np.zeros(n_tok, bool); is_doc_start[tok_base[:-1]] = True

    members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool_name]["latents"]}))
    lat2pos = -np.ones(W, np.int64); lat2pos[members] = np.arange(len(members))
    selm = lat2pos[z["lat_idx"]] >= 0
    li_all = z["lat_idx"][selm]; mf = flat[selm]; a_all = av[selm]
    mp = lat2pos[li_all]

    # train-stats rho=5 selection uses the 4SD threshold (selection fixed; ablate scoring rule)
    thr4 = mu + 4 * sdv
    ab4 = a_all >= thr4[li_all]
    cf = np.zeros(len(members)); cc = np.zeros(len(members))
    np.add.at(cf, mp[m_trd[mf] & ab4], 1)
    np.add.at(cc, mp[m_trc[mf] & ab4], 1)
    rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
    keep = set(members[rr >= 5].tolist())
    kmask = np.isin(li_all, list(keep))
    li, fl, a = li_all[kmask], mf[kmask], a_all[kmask]
    print(f"{width}: kept {len(keep)} latents (rho=5 train-stats)")

    res = {}
    for k_sd in (3, 4, 5):
        thr = mu + k_sd * sdv
        above = a >= thr[li]
        seedcnt = np.bincount(fl[above], minlength=n_tok)
        pa = np.bincount(fl, minlength=n_tok) > 0
        prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
        start = pa & (~prev | is_doc_start)
        run_id = np.cumsum(start) - 1
        for k_seed in (1, 2, 3):
            seeds = seedcnt >= k_seed
            for windowed in (True, False):
                if windowed:
                    sr = np.zeros(max(int(start.sum()), 1), bool)
                    sr[run_id[seeds]] = True
                    pred = pa & sr[run_id]
                else:
                    pred = seeds.copy()
                tp = int((pred & (gt | m_gd) & (m_span | m_gd | m_ctl)).sum())
                fpa = int((pred & (m_span | m_gd | m_ctl) & ~(gt | m_gd)).sum())
                fn = int((~pred & ((m_span & gt) | m_gd)).sum())
                P = tp / max(tp + fpa, 1); R = tp / max(tp + fn, 1)
                key = f"sd{k_sd}_seed{k_seed}_{'win' if windowed else 'nowin'}"
                res[key] = {"R_span": round(float(pred[m_span].mean()), 3),
                            "R_c2": round(float(pred[m_c2].mean()), 3),
                            "R_in": round(float(pred[m_gd].mean()), 3),
                            "fp_ctl": round(float(pred[m_ctl].mean()), 4),
                            "F2_in": round(5*P*R/max(4*P+R, 1e-9), 3)}
                print(f"  {key:22s} Rspan={res[key]['R_span']:.3f} Rc2={res[key]['R_c2']:.3f} "
                      f"fp={res[key]['fp_ctl']:.3f} F2in={res[key]['F2_in']:.3f}", flush=True)
    return res

out = {}
for width, pool in (("16k", "haiku_v6_16k"), ("65k", "haiku_v6_65k")):
    print(f"==== {width} ====", flush=True)
    out[width] = run(width, pool)
json.dump(out, open(f"{ROOT}/out/quali_rathi_ablation.json", "w"), indent=1)
print("wrote out/quali_rathi_ablation.json")
