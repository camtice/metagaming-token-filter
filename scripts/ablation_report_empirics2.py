"""Second empirics pass for the ablation report:
  boundary_k — {seed, window_ab (strict growth), window (paper)} x k{1,2,3,4,6,8}
               x rho{5,10}, judge pool, 65k
  width      — judge-pool rho frontier at k=4 (paper boundary) on the 16k SAE,
               same machinery as the 65k frontier in pass one
Same G-split, train-only statistics, doc-clustered bootstrap SEs (400).
Emits out/ablation_report_empirics2.json.
"""
import json
import sys

import numpy as np

ROOT = "/home/a5k/cwtice.a5k/sae-exploration"
role = json.load(open(f"{ROOT}/data/splits/gen_split_20260815.json"))["roles"]
docs = [json.loads(l) for l in open(f"{ROOT}/data/test_docs_v5.jsonl")]
by_id = {d["id"]: d for d in docs}
POOLJ = json.load(open(f"{ROOT}/data/candidate_sets/haiku_fable_forget_latents_v1.json"))["sets"]


class Ctx:
    def __init__(self, width):
        harv = f"{ROOT}/out/harvest_test_v5" + ("_65k" if width == "65k" else "")
        zf = np.load(harv + ".npz"); z = {k: zf[k] for k in zf.files}
        meta = json.load(open(harv + ".meta.json")); self.W = meta["width"]
        av = np.clip(z["act"].astype(np.float64), 0, 65504.0)
        self.ntok = z["doc_ntok"].astype(np.int64); self.nd = len(meta["docs"])
        self.tok_base = np.zeros(self.nd + 1, np.int64)
        np.cumsum(self.ntok, out=self.tok_base[1:])
        self.n_tok = int(self.ntok.sum())
        self.flat = self.tok_base[z["doc_idx"]] + z["tok_idx"]
        offs = z["offsets"].reshape(-1, 2)
        mu = np.bincount(z["lat_idx"], weights=av, minlength=self.W) / self.n_tok
        sd = np.sqrt(np.maximum(
            np.bincount(z["lat_idx"], weights=av**2, minlength=self.W)/self.n_tok - mu**2, 0))
        self.thr = mu + 4 * sd
        doc_ids = [d["id"] for d in meta["docs"]]
        self.rl = np.array([role[i] for i in doc_ids])
        self.tok_doc = np.repeat(np.arange(self.nd), self.ntok)
        tr = self.rl[self.tok_doc]
        gt = np.zeros(self.n_tok, bool)
        for di, did in enumerate(doc_ids):
            for cs, ce in by_id[did].get("char_spans", []):
                o = offs[self.tok_base[di]:self.tok_base[di + 1]]
                m = (o[:, 0] < ce) & (o[:, 1] > cs)
                gt[self.tok_base[di] + np.flatnonzero(m)] = True
        self.m_span = (tr == "gtest_span") & gt
        self.m_c2 = tr == "gtest_c2"; self.m_gd = tr == "gtest_dense"
        self.m_ctl = tr == "gtest_ctl"
        m_trd = tr == "train_dense"; m_trc = tr == "train_ctl"
        self.isd = np.zeros(self.n_tok, bool); self.isd[self.tok_base[:-1]] = True
        self.li = z["lat_idx"]; self.ab = av >= self.thr[self.li]
        cf = np.zeros(self.W); cc = np.zeros(self.W)
        np.add.at(cf, self.li[m_trd[self.flat] & self.ab], 1)
        np.add.at(cc, self.li[m_trc[self.flat] & self.ab], 1)
        self.rr = (((cf + 0.5) / max(int(m_trd.sum()), 1))
                   / ((cc + 0.5) / max(int(m_trc.sum()), 1)))
        self.gt_in = self.m_span | self.m_gd
        self.mask_in = ((tr == "gtest_span") | (tr == "gtest_clean")
                        | self.m_gd | self.m_ctl)
        self.gt_a = self.gt_in | self.m_c2
        self.mask_a = self.mask_in | self.m_c2
        self.gt_o = self.m_c2; self.mask_o = self.m_c2 | self.m_ctl
        self.role_groups = {r: np.flatnonzero(self.rl == r) for r in np.unique(self.rl)}
        rng = np.random.default_rng(20260829)
        self.B = 400
        self.boot = {r: rng.integers(0, len(ix), (self.B, len(ix)))
                     for r, ix in self.role_groups.items()}

    def f2(self, pred, gtx, mask):
        tp = int((pred & gtx & mask).sum()); fpa = int((pred & mask & ~gtx).sum())
        fn = int((~pred & gtx).sum())
        P = tp / max(tp + fpa, 1); R = tp / max(tp + fn, 1)
        return round(5 * P * R / max(4 * P + R, 1e-9), 3)

    def metrics(self, pred, n_keep):
        pdoc = lambda x: np.bincount(self.tok_doc[x], minlength=self.nd)
        pd = {"span_num": pdoc(pred & self.m_span), "span_den": pdoc(self.m_span),
              "c2_num": pdoc(pred & self.m_c2), "c2_den": pdoc(self.m_c2),
              "ctl_num": pdoc(pred & self.m_ctl), "ctl_den": pdoc(self.m_ctl)}
        for tag, gtx, mask in (("in", self.gt_in, self.mask_in),
                               ("out", self.gt_o, self.mask_o),
                               ("all", self.gt_a, self.mask_a)):
            pd[f"tp_{tag}"] = pdoc(pred & gtx & mask)
            pd[f"fp_{tag}"] = pdoc(pred & mask & ~gtx)
            pd[f"fn_{tag}"] = pdoc((~pred) & gtx)
        boots = {k: np.zeros(self.B) for k in ("Rspan", "Rc2", "fp", "F2in", "F2out", "F2all")}
        for b in range(self.B):
            tot = {k: 0.0 for k in pd}
            for r, ix in self.role_groups.items():
                take = ix[self.boot[r][b]]
                for k, v in pd.items():
                    tot[k] += float(v[take].sum())
            boots["Rspan"][b] = tot["span_num"] / max(tot["span_den"], 1)
            boots["Rc2"][b] = tot["c2_num"] / max(tot["c2_den"], 1)
            boots["fp"][b] = tot["ctl_num"] / max(tot["ctl_den"], 1)
            for tag, key in (("in", "F2in"), ("out", "F2out"), ("all", "F2all")):
                P = tot[f"tp_{tag}"] / max(tot[f"tp_{tag}"] + tot[f"fp_{tag}"], 1)
                Rc = tot[f"tp_{tag}"] / max(tot[f"tp_{tag}"] + tot[f"fn_{tag}"], 1)
                boots[key][b] = 5 * P * Rc / max(4 * P + Rc, 1e-9)
        return {"n": n_keep,
                "Rspan": round(float(pred[self.m_span].mean()), 3),
                "Rc2": round(float(pred[self.m_c2].mean()), 3),
                "Rin": round(float(pred[self.m_gd].mean()), 3),
                "fp": round(float(pred[self.m_ctl].mean()), 4),
                "F2in": self.f2(pred, self.gt_in, self.mask_in),
                "F2out": self.f2(pred, self.gt_o, self.mask_o),
                "F2all": self.f2(pred, self.gt_a, self.mask_a),
                "se": {k: round(float(v.std()), 4) for k, v in boots.items()}}

    def rule(self, keep_mask, kseed, boundary="window"):
        km = keep_mask[self.li]
        fl2, ab2 = self.flat[km], self.ab[km]
        seeds = np.bincount(fl2[ab2], minlength=self.n_tok) >= kseed
        if boundary == "seed":
            return seeds
        base = ab2 if boundary == "window_ab" else np.ones_like(ab2, bool)
        pa = np.bincount(fl2[base], minlength=self.n_tok) > 0
        prev = np.zeros(self.n_tok, bool); prev[1:] = pa[:-1]
        start = pa & (~prev | self.isd)
        run_id = np.cumsum(start) - 1
        sr = np.zeros(max(int(start.sum()), 1), bool)
        sr[run_id[seeds]] = True
        return pa & sr[run_id]


out = {}

# --- boundary x k grid, 65k judge pool ---
c65 = Ctx("65k")
pool65 = np.array(sorted({int(l) for l, _c, _f in POOLJ["haiku_v8_65k"]["latents"]}))
pm65 = np.zeros(c65.W, bool); pm65[pool65] = True
grid = {}
for rho in (5, 10):
    keep = pm65 & (c65.rr >= rho)
    for k in (1, 2, 3, 4, 6, 8):
        for b in ("seed", "window_ab", "window"):
            row = c65.metrics(c65.rule(keep, k, b), int(keep.sum()))
            row.update(rho=rho, k=k, boundary=b)
            grid[f"rho={rho} k={k} {b}"] = row
            print("bk", rho, k, b, row["F2all"], row["fp"], flush=True)
out["boundary_k"] = grid

# --- width: 16k judge frontier at k=4, paper boundary ---
c16 = Ctx("16k")
pool16 = np.array(sorted({int(l) for l, _c, _f in POOLJ["haiku_v8_16k"]["latents"]}))
pm16 = np.zeros(c16.W, bool); pm16[pool16] = True
w = {}
for rho in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 30):
    keep = pm16 & (c16.rr >= rho)
    row = c16.metrics(c16.rule(keep, 4), int(keep.sum()))
    row["rho"] = rho
    w[f"judge16 rho={rho}"] = row
    print("w16", rho, row["F2all"], row["fp"], flush=True)
out["width_16k_frontier_k4"] = w

json.dump(out, open(f"{ROOT}/out/ablation_report_empirics2.json", "w"), indent=1)
print("EMPIRICS2_DONE")
