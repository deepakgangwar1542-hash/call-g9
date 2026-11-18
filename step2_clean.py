"""Clean STATS19 2023 data for the Phase 1 evaluation protocol."""
import argparse
import pandas as pd
from common import *

parser = argparse.ArgumentParser()
parser.add_argument("--include-after-sep17", action="store_true",
                    help="also retain records dated on/after 17 September 2023")
args = parser.parse_args()

df = load_raw()
n0 = len(df)

# --- Target (Step 5): Serious/Fatal = 1, Slight = 0 ---
df = df[df["accident_severity"].isin([1, 2, 3])].copy()
df["target"] = df["accident_severity"].isin([1, 2]).astype(int)

# --- Leakage guard (SRS D3): number_of_vehicles / number_of_casualties etc. are NEVER used ---
# --- 5 features (Step 6) ---
out = pd.DataFrame({
    "road_type": df["road_type"].map(ROAD_TYPE),
    "speed_limit": df["speed_limit"],
    "weather": df["weather_conditions"].map(WEATHER),
    "light_condition": df["light_conditions"].map(LIGHT),
    "junction_detail": df["junction_detail"].map(JUNCTION),
    "date": pd.to_datetime(df["date"], dayfirst=True),
    "target": df["target"],
})

# --- unmapped codes check (naya code aaya to yahan warning dikhegi) ---
for col, mp in [("road_type", ROAD_TYPE), ("weather_conditions", WEATHER),
                ("light_conditions", LIGHT), ("junction_detail", JUNCTION)]:
    new = sorted(set(df[col].dropna().unique()) - set(mp))
    if new:
        print(f"WARNING: unmapped codes in {col}: {new} -> treated as Unknown")

# --- Missing/unknown handling: explicit "Unknown" category (silently drop nahi karte) ---
for c in ["road_type", "weather", "light_condition", "junction_detail"]:
    out[c] = out[c].fillna("Unknown")

# --- Speed limit: only valid STATS19 categories; -1 / other rows removed (report karte hain) ---
bad = ~out["speed_limit"].isin(VALID_SPEED)
print(f"Invalid/missing speed_limit rows dropped: {bad.sum()}")
out = out[~bad].copy()
out["speed_limit"] = out["speed_limit"].astype(int)

# --- 17-Sep-2023 speed-limit recording issue: retain a flag for sensitivity analysis ---
out["after_sep17"] = out["date"] >= "2023-09-17"
print("Rows after 17-Sep-2023 (flagged):", int(out["after_sep17"].sum()))
for label, mask in (("Pre-17-Sep", ~out["after_sep17"]), ("On/after 17-Sep", out["after_sep17"])):
    subset = out.loc[mask]
    print(f"{label}: n={len(subset):,} | Serious/Fatal share={subset['target'].mean():.3f}")

if not args.include_after_sep17:
    out = out[~out["after_sep17"]].copy()

out = out.reset_index(drop=True)   # duplicate rows (same conditions) legit hain, drop mat karo
CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
output_path = CLEAN_CSV.with_name("clean_2023_all.csv") if args.include_after_sep17 else CLEAN_CSV
out.drop(columns=["date"]).to_csv(output_path, index=False)
print(f"Rows: {n0} -> {len(out)} | Serious/Fatal share: {out['target'].mean():.3f}")
print("Saved:", output_path)
