import json
import numpy as np
ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]
out = json.load(open(f"{ROOT}/out/pool_fairgrid.json"))
pool, width = "haiku_v8_262k", "262k"
HARV = f"{ROOT}/out/harvest_test_v5_262k"
zf = np.load(HARV + ".npz"); z = {k: zf[k] for k in zf.files}
meta = json.load(open(HARV + ".meta.json")); W = meta["width"]
av = np.clip(z["act"].astype(np.float64), 0, 65504.0)
ntok = z["doc_ntok"].astype(np.int64); nd = len(meta["docs"])
tok_base = np.zeros(nd + 1, np.int64); np.cumsum(ntok, out=tok_base[1:])
n_tok = int(ntok.sum())
flat = tok_base[z["doc_idx"]] + z["tok_idx"]
offs = z["offsets"].reshape(-1, 2)
mu = np.bincount(z["lat_idx"], weights=av, minlength=W) / n_tok
sd = np.sqrt(np.maximum(np.bincount(z["lat_idx"], weights=av**2, minlength=W)/n_tok - mu**2, 0))
thr = mu + 4 * sd
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
m_c2 = trole == "gtest_c2"; m_gd = trole == "gtest_dense"; m_ctl = trole == "gtest_ctl"
m_trd = trole == "train_dense"; m_trc = trole == "train_ctl"
isd = np.zeros(n_tok, bool); isd[tok_base[:-1]] = True
li = z["lat_idx"]; fl = flat; ab = av >= thr[li]
cf = np.zeros(W); cc = np.zeros(W)
np.add.at(cf, li[m_trd[fl] & ab], 1)
np.add.at(cc, li[m_trc[fl] & ab], 1)
rr = ((cf + 0.5) / max(int(m_trd.sum()), 1)) / ((cc + 0.5) / max(int(m_trc.sum()), 1))
members = np.array(sorted({int(l) for l, _c, _f in POOLJ[pool]["latents"]}))
pm = np.zeros(W, bool); pm[members] = True
print(f"pool {pool}: {len(members)} latents; corpus {n_tok:,} tokens", flush=True)
for rho in (3, 5, 10):
    for ks in (2, 4):
        keep = pm & (rr >= rho)
        km = keep[li]
        fl2, ab2 = fl[km], ab[km]
        seeds = np.bincount(fl2[ab2], minlength=n_tok) >= ks
        pa = np.bincount(fl2, minlength=n_tok) > 0
        prev = np.zeros(n_tok, bool); prev[1:] = pa[:-1]
        start = pa & (~prev | isd)
        run_id = np.cumsum(start) - 1
        sr = np.zeros(max(int(start.sum()), 1), bool); sr[run_id[seeds]] = True
        pred = pa & sr[run_id]
        gt_in = m_span | m_gd
        mask_in = (trole=="gtest_span")|(trole=="gtest_clean")|m_gd|m_ctl
        gt_a = gt_in | m_c2; mask_a = mask_in | m_c2
        def f2(gtx, mask):
            tp = int((pred & gtx & mask).sum()); fpa = int((pred & mask & ~gtx).sum())
            fn = int((~pred & gtx).sum())
            P = tp/max(tp+fpa,1); R = tp/max(tp+fn,1)
            return round(5*P*R/max(4*P+R,1e-9),3)
        row = {"pool": pool, "width": width, "rho": rho, "seed": ks, "n": int(keep.sum()),
               "Rspan": round(float(pred[m_span].mean()),3), "Rc2": round(float(pred[m_c2].mean()),3),
               "Rin": round(float(pred[m_gd].mean()),3), "fp": round(float(pred[m_ctl].mean()),4),
               "F2in": f2(gt_in, mask_in), "F2all": f2(gt_a, mask_a)}
        out[f"{pool} rho={rho} seed>={ks}"] = row
        print(f"{pool} rho={rho:<2} k={ks} n={row['n']:6d} F2all={row['F2all']:.3f} "
              f"F2in={row['F2in']:.3f} Rspan={row['Rspan']:.3f} Rc2={row['Rc2']:.3f} fp={row['fp']:.4f}", flush=True)
json.dump(out, open(f"{ROOT}/out/pool_fairgrid.json", "w"), indent=1)
print("FAIRGRID_262K_DONE")
