"""Hex risk map for the control room (H3 resolution 7, about 5 sq km per cell).

Per-cell serious/fatal rates use 2021-2024 only and are shrunk toward the national rate, so thin cells do not look
extreme. 2025 is held out: the script reports how well the 2021-2024 cell ranking predicts 2025 cell rates, so the map
is shown with a measured, not assumed, level of trust.
Output: data/processed/hex_risk.csv, reports/hex_validation.json
"""
import json

import h3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import MULTI_CLEAN, ROOT

RES = 7
SHRINK = 25          # pseudo-observations at the national rate
MIN_CALLS = 8


def main():
    df = pd.read_csv(MULTI_CLEAN, usecols=["year", "latitude", "longitude", "target"]).dropna()
    coords = df[["latitude", "longitude"]].drop_duplicates()
    coords["hex"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(coords.latitude, coords.longitude)]
    df = df.merge(coords, on=["latitude", "longitude"], how="left")

    train, test = df[df.year <= 2024], df[df.year == 2025]
    national = train.target.mean()
    agg = train.groupby("hex").target.agg(n="size", serious="sum").reset_index()
    agg["rate"] = (agg.serious + SHRINK * national) / (agg.n + SHRINK)
    agg = agg[agg.n >= MIN_CALLS].copy()
    centres = [h3.cell_to_latlng(c) for c in agg.hex]
    agg["lat"], agg["lon"] = [c[0] for c in centres], [c[1] for c in centres]

    held = test.groupby("hex").target.agg(n25="size", rate25="mean").reset_index()
    both = agg.merge(held, on="hex")
    both = both[both.n25 >= 10]
    rho = float(spearmanr(both.rate, both.rate25).statistic)
    top = both.nlargest(max(1, len(both) // 10), "rate")
    report = dict(resolution=RES, cells=int(len(agg)), national_rate_2021_24=float(national), cells_compared=int(len(both)),
                  spearman_2021_24_vs_2025=rho, top_decile_cells_2025_rate=float(np.average(top.rate25, weights=top.n25)),
                  all_cells_2025_rate=float(np.average(both.rate25, weights=both.n25)))
    agg.to_csv(ROOT / "data/processed/hex_risk.csv", index=False)
    (ROOT / "reports/hex_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
