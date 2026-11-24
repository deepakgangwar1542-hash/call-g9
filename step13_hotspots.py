"""Exposure-adjusted, significant hotspots (H3 resolution 7).

Serious/fatal collisions are divided by traffic exposure (million vehicle-km from DfT AADF count points) so a busy
road is not flagged merely for being busy. Getis-Ord Gi* with Benjamini-Hochberg false-discovery control marks the
significant cells, and each cell carries an exact Poisson confidence interval so thin cells do not look dangerous by
chance. Hotspots found in 2021-2023 are checked against 2024-2025, and compared with a naive count-based method.
Output: data/processed/hotspots.csv, reports/hotspots.json
"""
import json

import h3
import numpy as np
import pandas as pd
from scipy.stats import chi2, false_discovery_control, norm

from common import MULTI_CLEAN, ROOT

RES = 7
PERIOD_A, PERIOD_B = (2021, 2022, 2023), (2024, 2025)
MIN_EXPOSURE_MVKM = 20.0     # per period; below this the rate is too noisy to rank
ALPHA = 0.05
AADF = ROOT / "data/raw/dft_traffic_counts_aadf.csv"


def cell_exposure():
    cols = ["count_point_id", "year", "latitude", "longitude", "link_length_km", "all_motor_vehicles"]
    d = pd.read_csv(AADF, usecols=cols).dropna()
    d = d[d.year.between(2021, 2025)]
    d["mvkm"] = d.all_motor_vehicles * d.link_length_km * 365 / 1e6
    pts = d[["latitude", "longitude"]].drop_duplicates()
    pts["hex"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(pts.latitude, pts.longitude)]
    d = d.merge(pts, on=["latitude", "longitude"])
    return d.groupby(["hex", "year"]).mvkm.sum().unstack(fill_value=0.0)


def cell_counts():
    df = pd.read_csv(MULTI_CLEAN, usecols=["year", "latitude", "longitude", "target"]).dropna()
    pts = df[["latitude", "longitude"]].drop_duplicates()
    pts["hex"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(pts.latitude, pts.longitude)]
    df = df.merge(pts, on=["latitude", "longitude"])
    return df.groupby(["hex", "year"]).target.sum().astype(float).unstack(fill_value=0.0)


def period_table(counts, exposure, years):
    t = pd.DataFrame({"events": counts.reindex(columns=list(years), fill_value=0).sum(axis=1),
                      "mvkm": exposure[list(years)].sum(axis=1)}).dropna()
    t = t[t.mvkm >= MIN_EXPOSURE_MVKM].copy()
    t["rate"] = t.events / t.mvkm
    return t


def gi_star(values, cells):
    """Gi* z-score per cell, binary weights over the cell and its immediate H3 neighbours that are in the table."""
    index = {c: i for i, c in enumerate(cells)}
    x = np.asarray(values, dtype=float)
    n, mean, sd = len(x), x.mean(), x.std()
    z = np.zeros(n)
    for c, i in index.items():
        members = [index[k] for k in h3.grid_disk(c, 1) if k in index]
        w = len(members)
        denom = sd * np.sqrt((n * w - w ** 2) / (n - 1))
        z[i] = (x[members].sum() - mean * w) / denom if denom > 0 else 0.0
    return z


def hot_flags(values, cells):
    z = gi_star(values, cells)
    p = norm.sf(z)
    q = false_discovery_control(p, method="bh")
    return z, q, (q < ALPHA) & (z > 0)


def poisson_ci(events, exposure):
    lower = np.where(events > 0, chi2.ppf(0.025, 2 * np.maximum(events, 1)) / 2, 0.0) / exposure
    upper = chi2.ppf(0.975, 2 * events + 2) / 2 / exposure
    return lower, upper


def mean_rate(table, mask):
    return float(table.events[mask].sum() / table.mvkm[mask].sum())


def bootstrap_ratio(table, hot, rng, reps=2000):
    h, o = table[hot], table[~hot]
    ratios = []
    for _ in range(reps):
        hs, os_ = h.sample(len(h), replace=True, random_state=int(rng.integers(1 << 30))), \
                  o.sample(len(o), replace=True, random_state=int(rng.integers(1 << 30)))
        ratios.append((hs.events.sum() / hs.mvkm.sum()) / (os_.events.sum() / os_.mvkm.sum()))
    return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]


def main():
    rng = np.random.default_rng(0)
    exposure, counts = cell_exposure(), cell_counts()
    a = period_table(counts, exposure, PERIOD_A)
    b = period_table(counts, exposure, PERIOD_B)
    cells = a.index.intersection(b.index)
    a, b = a.loc[cells], b.loc[cells]
    print("cells analysed:", len(cells), "| exposure covers", f"{a.mvkm.sum():.0f} mvkm in 2021-23")

    prior = a.events.sum() / a.mvkm.sum()
    strength = float(a.mvkm.median())
    a["shrunk"] = (a.events + strength * prior) / (a.mvkm + strength)
    methods = {}
    z, q, hot = hot_flags(a.shrunk.values, list(cells))
    methods["exposure_adjusted"] = (z, q, hot)
    methods["naive_counts"] = hot_flags(a.events.values.astype(float), list(cells))

    zb, qb, hot_b_adj = hot_flags(((b.events + strength * prior) / (b.mvkm + strength)).values, list(cells))
    _, _, hot_b_naive = hot_flags(b.events.values.astype(float), list(cells))
    again = {"exposure_adjusted": hot_b_adj, "naive_counts": hot_b_naive}

    report = {"resolution": RES, "cells_analysed": int(len(cells)), "min_exposure_mvkm": MIN_EXPOSURE_MVKM,
              "fdr_alpha": ALPHA, "period_a": list(PERIOD_A), "period_b": list(PERIOD_B),
              "national_rate_a_per_mvkm": float(prior), "national_rate_b_per_mvkm": mean_rate(b, np.ones(len(b), bool))}
    for name, (_, _, flag) in methods.items():
        n_hot = int(flag.sum())
        if n_hot == 0:
            report[name] = {"hotspots": 0}
            continue
        flag_s = pd.Series(flag, index=cells)
        rate_b_hot, rate_b_rest = mean_rate(b, flag_s.values), mean_rate(b, ~flag_s.values)
        report[name] = {
            "hotspots": n_hot,
            "median_mvkm_in_hotspots": float(a.mvkm[flag_s.values].median()),
            "rate_2024_25_inside_per_mvkm": rate_b_hot, "rate_2024_25_outside_per_mvkm": rate_b_rest,
            "rate_ratio_inside_vs_outside": rate_b_hot / rate_b_rest,
            "rate_ratio_ci95": bootstrap_ratio(b, flag_s, rng),
            "share_hot_again_in_2024_25": float(again[name][flag].mean()),
            "share_of_all_cells_hot_in_2024_25": float(again[name].mean()),
            "share_with_above_national_rate_in_2024_25": float((b.rate[flag_s.values] > report["national_rate_b_per_mvkm"]).mean()),
        }

    lo, hi = poisson_ci(a.events.values, a.mvkm.values)
    out = a.copy()
    out["ci_lo"], out["ci_hi"], out["z"], out["q"], out["hotspot"] = lo, hi, methods["exposure_adjusted"][0], \
        methods["exposure_adjusted"][1], methods["exposure_adjusted"][2]
    centres = [h3.cell_to_latlng(c) for c in out.index]
    out["lat"], out["lon"] = [c[0] for c in centres], [c[1] for c in centres]
    out["ci_excludes_national"] = out.ci_lo > prior
    out.index.name = "hex"
    out.reset_index().to_csv(ROOT / "data/processed/hotspots.csv", index=False)
    (ROOT / "reports/hotspots.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
