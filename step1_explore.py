"""STEP 3: Dataset explore - rows/cols, class balance, missing codes (SRS Section 3 ke numbers yahin se aayenge)."""
import pandas as pd
from common import load_raw

df = load_raw()
print("Rows, Cols:", df.shape)
print("\nColumns:", list(df.columns))
print("\nAccident_Severity counts (1=Fatal 2=Serious 3=Slight):")
print(df["accident_severity"].value_counts().sort_index())
print("\nSerious/Fatal share: %.2f%%" % (100 * df["accident_severity"].isin([1, 2]).mean()))
for c in ["road_type", "speed_limit", "weather_conditions", "light_conditions", "junction_detail"]:
    print(f"\n--- {c} ---")
    print(df[c].value_counts(dropna=False).sort_index())
print("\nNull counts (selected):\n",
      df[["road_type", "speed_limit", "weather_conditions", "light_conditions", "junction_detail"]].isna().sum())
d = pd.to_datetime(df["date"], dayfirst=True)
print("\nRows on/after 17-Sep-2023 (speed-limit issue):", (d >= "2023-09-17").sum())