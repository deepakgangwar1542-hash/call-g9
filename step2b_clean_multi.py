"""Clean the multi-year STATS19 file: one row per collision, all feature tiers, same rules as step2_clean.py."""
import pandas as pd

from common import (CONTEXT_FEATURES, FORCE_FEATURES, JUNCTION, LIGHT, MULTI_CLEAN, MULTI_RAW, ROAD_TYPE,
                    UNKNOWN_CODES, VALID_SPEED, WEATHER)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.rename(columns=lambda c: c.replace("accident_", "collision_"))
    n0 = len(raw)

    df = raw[raw["collision_severity"].isin([1, 2, 3])].copy()
    df["target"] = df["collision_severity"].isin([1, 2]).astype(int)   # Serious/Fatal = 1, Slight = 0

    date = pd.to_datetime(df["date"], dayfirst=True)
    out = pd.DataFrame({
        "year": df["collision_year"].astype(int),
        "road_type": df["road_type"].map(ROAD_TYPE),
        "speed_limit": df["speed_limit"],
        "weather": df["weather_conditions"].map(WEATHER),
        "light_condition": df["light_conditions"].map(LIGHT),
        "junction_detail": df["junction_detail"].map(JUNCTION),
        "hour": pd.to_numeric(df["time"].astype(str).str[:2], errors="coerce"),
        "day_of_week": df["day_of_week"],
        "month": date.dt.month,
        "target": df["target"],
    })
    for col, mapping in [("road_type", ROAD_TYPE), ("weather_conditions", WEATHER),
                         ("light_conditions", LIGHT), ("junction_detail", JUNCTION)]:
        unmapped = sorted(set(df[col].dropna().unique()) - set(mapping))
        if unmapped:
            print(f"WARNING: unmapped codes in {col}: {unmapped} -> treated as Unknown")
    for col in ["road_type", "weather", "light_condition", "junction_detail"]:
        out[col] = out[col].fillna("Unknown")

    # Context and force fields keep their raw STATS19 code as an explicit category (-1 = data missing / out of range).
    for col in CONTEXT_FEATURES + FORCE_FEATURES:
        codes = df[col].fillna(-1).astype(int)
        if col in UNKNOWN_CODES:
            codes = codes.where(~codes.isin(UNKNOWN_CODES[col]), -1)
        out[col] = codes.astype(str)

    bad_speed = ~out["speed_limit"].isin(VALID_SPEED)
    bad_time = out["hour"].isna()
    # 17-Sep-2023 speed-limit recording issue: those rows are excluded from every split (train, validation and test).
    after_sep17 = (out["year"] == 2023) & (date >= "2023-09-17")
    print(f"Rows: {n0:,} -> severity-valid {len(df):,}")
    print(f"Dropped: invalid speed_limit {int(bad_speed.sum()):,}, missing time {int(bad_time.sum()):,}, "
          f"2023 on/after 17-Sep {int(after_sep17.sum()):,}")
    keep = ~(bad_speed | bad_time | after_sep17)
    out = out[keep].copy()
    out["speed_limit"] = out["speed_limit"].astype(int)
    out["hour"] = out["hour"].astype(int)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    if not MULTI_RAW.exists():
        raise SystemExit("Run `python step0_download.py` first.")
    cleaned = clean(pd.read_csv(MULTI_RAW, low_memory=False))
    MULTI_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(MULTI_CLEAN, index=False)
    summary = cleaned.groupby("year").agg(rows=("target", "size"), serious_fatal_share=("target", "mean"))
    print(summary.round(3).to_string())
    print("Saved:", MULTI_CLEAN)
