"""Pulse of Britain: where serious and fatal collisions are expected, hour by hour across the week.

For each map cell (H3 resolution 7) the yearly count of serious/fatal collisions comes from 2021-2024. The shape of the
week comes from a profile of serious/fatal collisions by hour of week, built per police force and area type (urban or
rural) and shrunk toward the national profile for that area type, so thin groups do not invent patterns.
expected serious/fatal per year in cell c during hour-of-week h  =  cell_serious_per_year[c] * profile[area(c), h]
(The per-force profile is still computed and compared in the validation, but the shipped file holds only the national
profile per area type because the per-force version did not beat it on 2025.)

2025 is held out. The script checks whether the profile finds where 2025 serious collisions happened better than a flat
week, using the cells and time bands ranked from 2021-2024 only, and reports a bootstrap interval.
Outputs: data/processed/pulse_cells.csv, data/processed/pulse_profiles.csv, reports/pulse_validation.json
"""
import json

import h3
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import MULTI_CLEAN, ROOT, SEED

RES = 7
MIN_CALLS = 8
SHRINK_SERIOUS = 150      # pseudo-collisions at the national profile
TRAIN_YEARS = 4
BANDS = 24                # validation bands: 3 day types x 8 three-hour blocks
BOOTSTRAPS = 300


def hour_of_week(day_of_week, hour):
    """STATS19 day_of_week: 1 = Sunday ... 7 = Saturday. Returns 0..167 with Sunday midnight as 0."""
    return (day_of_week.astype(int) - 1) * 24 + hour.astype(int)


def band_of(how):
    day = how // 24                     # 0 Sunday ... 6 Saturday
    kind = np.where(day == 0, 2, np.where(day == 6, 1, 0))          # 0 Mon-Fri, 1 Saturday, 2 Sunday
    return kind * 8 + (how % 24) // 3


def main():
    cols = ["year", "latitude", "longitude", "target", "hour", "day_of_week", "urban_or_rural_area", "police_force"]
    df = pd.read_csv(MULTI_CLEAN, usecols=cols).dropna()
    coords = df[["latitude", "longitude"]].drop_duplicates()
    coords["hex"] = [h3.latlng_to_cell(a, b, RES) for a, b in zip(coords.latitude, coords.longitude)]
    df = df.merge(coords, on=["latitude", "longitude"], how="left")
    df["how"] = hour_of_week(df.day_of_week, df.hour)
    df["area"] = df.urban_or_rural_area.where(df.urban_or_rural_area.isin([1, 2]), 1).astype(int)
    df["force"] = df.police_force.astype(int)
    train, test = df[df.year <= 2024], df[df.year == 2025]
    serious = train[train.target == 1]

    national = {}
    for area, part in serious.groupby("area"):
        counts = np.bincount(part.how, minlength=168).astype(float)
        national[area] = counts / counts.sum()
    profiles = []
    for (force, area), part in serious.groupby(["force", "area"]):
        counts = np.bincount(part.how, minlength=168).astype(float)
        share = (counts + SHRINK_SERIOUS * national[area]) / (counts.sum() + SHRINK_SERIOUS)
        profiles.extend(dict(force=force, area=area, how=h, share=s) for h, s in enumerate(share))
    profiles = pd.DataFrame(profiles)
    profile_lookup = {(f, a): g.sort_values("how").share.to_numpy() for (f, a), g in profiles.groupby(["force", "area"])}

    cells = train.groupby("hex").agg(n=("target", "size"), serious=("target", "sum"),
                                     area=("area", lambda s: int(s.mode().iloc[0])),
                                     force=("force", lambda s: int(s.mode().iloc[0]))).reset_index()
    cells = cells[cells.n >= MIN_CALLS].copy()
    cells["serious_per_year"] = cells.serious / TRAIN_YEARS
    centres = [h3.cell_to_latlng(c) for c in cells.hex]
    cells["lat"], cells["lon"] = [c[0] for c in centres], [c[1] for c in centres]
    cells[["hex", "n", "serious", "serious_per_year", "area", "force", "lat", "lon"]].to_csv(
        ROOT / "data/processed/pulse_cells.csv", index=False)
    # The per-force profile did not beat the national-by-area profile out of time (see pulse_validation.json),
    # so the app ships the simpler national profile for each area type.
    pd.DataFrame([dict(area=a, how=h, share=s) for a, share in national.items() for h, s in enumerate(share)]).to_csv(
        ROOT / "data/processed/pulse_profiles.csv", index=False)

    # --- 2025 check at band level (24 bands) so counts are not hopelessly thin
    def shares(key):
        return np.array([profile_lookup.get((f, a), national[a]) for f, a in key])

    by_how = shares(list(zip(cells.force, cells.area)))                       # cells x 168
    nat_how = np.array([national[a] for a in cells.area])
    flat_how = np.full_like(by_how, 1 / 168)
    band_of_hour = band_of(np.arange(168))
    one_hot = np.eye(BANDS)[band_of_hour]                                      # 168 x 24

    def band_expectation(how_share):
        return (cells.serious_per_year.to_numpy()[:, None] * how_share) @ one_hot       # cells x 24

    expectations = {"hour-of-week profile (force and area)": band_expectation(by_how),
                    "national profile by area only": band_expectation(nat_how),
                    "flat week": band_expectation(flat_how)}
    index = {h: i for i, h in enumerate(cells.hex)}
    held = test[(test.target == 1) & test.hex.isin(index)]
    cell_idx = held.hex.map(index).to_numpy()
    band_idx = band_of(held.how.to_numpy())
    counts = np.zeros((len(cells), BANDS))
    np.add.at(counts, (cell_idx, band_idx), 1)
    total = counts.sum()

    def capture(expectation, top_share=0.10):
        flat = expectation.ravel()
        take = np.argsort(-flat)[: int(len(flat) * top_share)]
        return counts.ravel()[take].sum() / total, take

    result = dict(test_year=2025, cells=int(len(cells)), held_out_serious_collisions=int(total),
                  top_share_of_cell_bands=0.10, capture={}, spearman={})
    takes = {}
    for name, expectation in expectations.items():
        share, take = capture(expectation)
        takes[name] = take
        result["capture"][name] = float(share)
        result["spearman"][name] = float(spearmanr(expectation.ravel(), counts.ravel()).statistic)

    rng = np.random.default_rng(SEED)
    flat_idx = cell_idx * BANDS + band_idx
    diff = []
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(flat_idx), len(flat_idx))
        boot = np.bincount(flat_idx[pick], minlength=len(cells) * BANDS)
        diff.append((boot[takes["hour-of-week profile (force and area)"]].sum()
                     - boot[takes["flat week"]].sum()) / len(pick))
    result["profile_minus_flat_capture_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
    result["profile_minus_flat_capture"] = result["capture"]["hour-of-week profile (force and area)"] - result["capture"]["flat week"]
    result["note"] = ("Capture = share of 2025 serious/fatal collisions falling in the 10% of cell-time bands ranked highest "
                      "from 2021-2024. The 24 bands are weekday/Saturday/Sunday by eight three-hour blocks.")
    (ROOT / "reports/pulse_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
