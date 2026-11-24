"""Fetch hourly weather for every police-force area (Open-Meteo archive, free, no key) and attach it to collisions.

Each force is represented by the mean position of its collisions, so this is area-level weather, not weather at the
exact crash site. Only the weather of the collision's own hour is used (known at call time, so no leakage).
Writes data/processed/weather_by_force.csv and data/processed/clean_multi_dates.csv (row-aligned collision dates).
"""
import time

import pandas as pd
import requests

from common import MULTI_CLEAN, MULTI_RAW, ROOT, VALID_SPEED

OUT = ROOT / "data/processed"
VARS = "temperature_2m,precipitation,snowfall,wind_speed_10m,visibility,cloud_cover"
URL = "https://archive-api.open-meteo.com/v1/archive"


def aligned_dates(clean: pd.DataFrame) -> pd.Series:
    """Re-apply step2b's row filters to the raw file to recover each cleaned row's exact date, and verify alignment."""
    raw = pd.read_csv(MULTI_RAW, low_memory=False).rename(columns=lambda c: c.replace("accident_", "collision_"))
    raw = raw[raw["collision_severity"].isin([1, 2, 3])]
    date = pd.to_datetime(raw["date"], dayfirst=True)
    hour = pd.to_numeric(raw["time"].astype(str).str[:2], errors="coerce")
    keep = raw["speed_limit"].isin(VALID_SPEED) & hour.notna() & ~((raw["collision_year"] == 2023) & (date >= "2023-09-17"))
    date = date[keep].reset_index(drop=True)
    assert len(date) == len(clean), f"row count mismatch {len(date)} vs {len(clean)}"
    assert (date.dt.year.to_numpy() == clean["year"].to_numpy()).all(), "year misaligned"
    assert (date.dt.month.to_numpy() == clean["month"].to_numpy()).all(), "month misaligned"
    return date


def fetch_force(lat: float, lon: float) -> pd.DataFrame:
    params = dict(latitude=round(lat, 2), longitude=round(lon, 2), start_date="2021-01-01", end_date="2025-12-31",
                  hourly=VARS, timezone="Europe/London")
    for attempt in range(6):
        try:
            r = requests.get(URL, params=params, timeout=45)
        except requests.RequestException as error:
            print("  request failed:", type(error).__name__, flush=True)
            time.sleep(5)
            continue
        print("  status", r.status_code, flush=True)
        if r.status_code == 200:
            return pd.DataFrame(r.json()["hourly"])
        time.sleep(15 * (attempt + 1))
    raise RuntimeError("Open-Meteo kept failing")


def main():
    clean = pd.read_csv(MULTI_CLEAN)
    dates = aligned_dates(clean)
    pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d")}).to_csv(OUT / "clean_multi_dates.csv", index=False)
    centres = clean[clean.police_force.astype(str) != "-1"].groupby("police_force")[["latitude", "longitude"]].mean()
    cache = OUT / "weather_cache"
    cache.mkdir(exist_ok=True)
    frames = []
    for force, row in centres.iterrows():
        path = cache / f"{force}.csv"
        if not path.exists():
            weather = fetch_force(row.latitude, row.longitude)
            weather.insert(0, "police_force", str(force))
            weather.to_csv(path, index=False)
            time.sleep(3)
        frames.append(pd.read_csv(path, dtype={"police_force": str}))
        print("ready force", force, flush=True)
    pd.concat(frames).to_csv(OUT / "weather_by_force.csv", index=False)
    print("done", len(frames), "forces")


if __name__ == "__main__":
    main()
