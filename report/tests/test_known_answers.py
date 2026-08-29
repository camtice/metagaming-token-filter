"""Known-answer tests for this report.

The report's numbers come from a frozen aggregate (results.json) computed from
the TEST-v5 65k harvest under the G-split. Each case pins a calibrated cell: a
failure means the numbers in the report are not the numbers that were
calibrated against the interactive analysis of 2026-08-29.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results.json"


def _load():
    return json.loads(RESULTS.read_text())


def test_champion_cell() -> None:
    """Judge pool, screen rho>=10, vote k=4, paper boundary rule."""
    r = _load()["judge_frontier_k4"]["judge rho=10"]
    assert r["n"] == 682
    assert r["F2all"] == 0.821
    assert r["fp"] == 0.0110
    assert r["Rspan"] == 0.675


def test_full_dictionary_matched_fp_cell() -> None:
    """Full dictionary at its frontier peak (rho>=35)."""
    r = _load()["judge_frontier_k4"]["all rho=35"]
    assert r["n"] == 720
    assert r["F2all"] == 0.868
    assert r["fp"] == 0.0077


def test_vote_k6_cell() -> None:
    """Vote k=6 at screen rho>=5."""
    r = _load()["vote_k"]["rho=5 k=6"]
    assert r["F2all"] == 0.833
    assert r["fp"] == 0.0121


def test_boundary_rules_share_seed_stage() -> None:
    """The paper boundary rule reproduces the champion cell exactly, and all
    boundary variants share the champion's kept-feature count."""
    d = _load()
    champ = d["judge_frontier_k4"]["judge rho=10"]
    win = d["boundary"]["window"]
    assert win["F2all"] == champ["F2all"]
    assert win["fp"] == champ["fp"]
    assert all(v["n"] == champ["n"] for v in d["boundary"].values())


def test_every_cell_carries_clustered_se() -> None:
    """Every plotted estimate must have a bootstrap SE (error-bar rule)."""
    d = _load()
    for block in ("judge_frontier_k4", "vote_k", "boundary"):
        for key, row in d[block].items():
            for m in ("F2all", "F2in", "F2out", "Rspan", "Rc2", "fp"):
                assert m in row and m in row["se"], f"{block}/{key} missing {m}"


def test_boundary_k_strict_growth_tracks_paper() -> None:
    """Strict growth (above-threshold) within one SE of the paper rule at the
    reference cell, and the seed-collapse cell is pinned."""
    d = _load()["boundary_k"]
    win = d["rho=10 k=4 window"]
    strict = d["rho=10 k=4 window_ab"]
    assert abs(win["F2all"] - strict["F2all"]) <= win["se"]["F2all"]
    assert d["rho=10 k=8 seed"]["F2all"] == 0.286


def test_width_16k_peak() -> None:
    """The 16k judged frontier peaks at rho>=6."""
    w = _load()["width_16k_frontier_k4"]
    peak = max(w.values(), key=lambda r: r["F2all"])
    assert peak["rho"] == 6
    assert peak["F2all"] == 0.794


def test_roc_65k_headline_aucs() -> None:
    """Pinned AUCs: screened sets discriminate, unscreened full dict is near
    chance."""
    roc = _load()["roc_65k"]
    assert roc["judge rho>=10"]["cont_auc_span"] == 0.8422
    assert roc["all rho>=35"]["cont_auc_c2"] == 0.9675
    assert roc["full dictionary (no screen)"]["cont_auc_all_forget"] < 0.6
