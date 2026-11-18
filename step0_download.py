"""Download the DfT STATS19 'last 5 years' collision file (currently 2021-2025) into data/raw/."""
import argparse
import shutil
import urllib.request

import pandas as pd

from common import MULTI_RAW, MULTI_URL

REQUIRED = {"collision_year", "collision_severity", "speed_limit", "road_type", "junction_detail", "date", "time",
            "police_force"}


def download(force=False):
    if MULTI_RAW.exists() and not force:
        print(f"Already present: {MULTI_RAW} ({MULTI_RAW.stat().st_size / 1e6:.1f} MB). Use --force to re-download.")
        return
    MULTI_RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = MULTI_RAW.with_suffix(".part")
    print("Downloading", MULTI_URL)
    with urllib.request.urlopen(MULTI_URL, timeout=300) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(MULTI_RAW)
    print(f"Saved {MULTI_RAW} ({MULTI_RAW.stat().st_size / 1e6:.1f} MB)")


def verify():
    head = pd.read_csv(MULTI_RAW, nrows=5)
    missing = REQUIRED - set(head.columns)
    if missing:
        raise SystemExit(f"Downloaded file is missing required columns: {sorted(missing)}")
    years = pd.read_csv(MULTI_RAW, usecols=["collision_year"]).collision_year.value_counts().sort_index()
    print("Rows per year:\n" + years.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.force)
    verify()
