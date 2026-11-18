"""Shared config, STATS19 code maps and metric helpers."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_CSV = ROOT / "data/raw/dft-road-casualty-statistics-collision-2023.csv"
CLEAN_CSV = ROOT / "data/processed/clean_2023.csv"
MODEL_PATH = ROOT / "models/best_model.joblib"
META_PATH = ROOT / "models/best_model_meta.json"
DB_PATH = ROOT / "calls.db"

K_DEFAULT = 300          # shift capacity K (enhanced responses per shift) - change as you like
SEED = 42

# 5 usable features (known at call time) -> SRS FR-1.1
FEATURES = ["road_type", "speed_limit", "weather", "light_condition", "junction_detail"]
VALID_SPEED = [20, 30, 40, 50, 60, 70]

# STATS19 code -> readable label ("Unknown" kept as an explicit category, not dropped)
ROAD_TYPE = {1: "Roundabout", 2: "One way street", 3: "Dual carriageway", 6: "Single carriageway",
             7: "Slip road", 12: "One way street/Slip road", 9: "Unknown", -1: "Unknown"}
WEATHER = {1: "Fine, no high winds", 2: "Raining, no high winds", 3: "Snowing, no high winds",
           4: "Fine, high winds", 5: "Raining, high winds", 6: "Snowing, high winds",
           7: "Fog or mist", 8: "Other", 9: "Unknown", -1: "Unknown"}
LIGHT = {1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
         6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Unknown"}
# 2023 file uses the NEW junction coding (STATS20 2024 spec). Code 0 also covers roundabout, mini-roundabout
# and slip-road collisions (old codes 1, 2, 5 were merged into 0); -1 is the old "99 Unknown".
JUNCTION = {0: "Not at junction / roundabout / slip road", 13: "T or staggered junction", 16: "Crossroads",
            17: "More than 4 arms (not roundabout)", 18: "Private drive or entrance",
            19: "Other junction", 99: "Unknown", -1: "Unknown"}

CHOICES = {  # dropdown options for the UI / validation
    "road_type": sorted(set(ROAD_TYPE.values())),
    "weather": sorted(set(WEATHER.values())),
    "light_condition": sorted(set(LIGHT.values())),
    "junction_detail": sorted(set(JUNCTION.values())),
}


def precision_at_k(y_true, scores, k, seed=SEED):
    """Of the top-k rows ranked by score, fraction that are truly 1.
    Ties are broken randomly (important for baselines that give many equal scores)."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    if k <= 0:
        raise ValueError("K must be > 0")
    k = min(k, len(y_true))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -scores))   # sort by -score, random tie-break
    return float(y_true[order[:k]].mean())


def load_raw():
    """Load STATS19 CSV; supports old (accident_*) and new 2023 (collision_*) column names."""
    df = pd.read_csv(RAW_CSV, low_memory=False)
    df = df.rename(columns={"collision_severity": "accident_severity",
                            "collision_index": "accident_index",
                            "collision_year": "accident_year"})
    return df


# ---------------------------------------------------------------------------
# Multi-year pipeline (v2): DfT "last 5 years" file, temporal validation
# ---------------------------------------------------------------------------
MULTI_URL = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv"
MULTI_RAW = ROOT / "data/raw/dft-road-casualty-statistics-collision-last-5-years.csv"
MULTI_CLEAN = ROOT / "data/processed/clean_multi.csv"
V2_MODEL_PATH = ROOT / "models/model_v2.joblib"
V2_META_PATH = ROOT / "models/model_v2_meta.json"

FLAG_RATE = 0.02         # share of calls escalated; same operating point as K=300 on the 2023 test set (~2%)

# Feature tiers are cumulative. Time fields are known when a call arrives; the context fields need the call-taker
# to ask (or a map lookup); police_force is known from routing but partly encodes force-level severity recording.
TIME_FEATURES = ["hour", "day_of_week", "month"]
# special_conditions_at_site is excluded on purpose: its "data missing" share jumped from ~1% (2023) to 59% (2024)
# and 87% (2025), a recording change that makes older years non-comparable.
CONTEXT_FEATURES = ["urban_or_rural_area", "first_road_class", "junction_control", "pedestrian_crossing",
                    "road_surface_conditions", "carriageway_hazards"]
# Per-field codes meaning "missing" or "unknown"; DfT moves records between these across years, so they are merged.
UNKNOWN_CODES = {"junction_control": {-1, 9}, "pedestrian_crossing": {-1, 99}, "road_surface_conditions": {-1, 9},
                 "carriageway_hazards": {-1, 99}, "urban_or_rural_area": {-1, 3}}
FORCE_FEATURES = ["police_force"]
TIERS = {
    "T1_base5": FEATURES,
    "T2_plus_time": FEATURES + TIME_FEATURES,
    "T3_plus_context": FEATURES + TIME_FEATURES + CONTEXT_FEATURES,
    "T4_plus_force": FEATURES + TIME_FEATURES + CONTEXT_FEATURES + FORCE_FEATURES,
}
MULTI_CATEGORICAL = ["road_type", "speed_limit", "weather", "light_condition", "junction_detail", "day_of_week",
                     "month"] + CONTEXT_FEATURES + FORCE_FEATURES

# Dispatcher-facing labels for the optional v2 fields. Only codes whose meaning is certain are exposed in the UI;
# pedestrian_crossing and carriageway_hazards are not asked for and are sent to the model as "-1" (unknown).
OPTIONAL_LABELS = {
    "urban_or_rural_area": {"1": "Urban", "2": "Rural"},
    "first_road_class": {"1": "Motorway", "2": "A(M) road", "3": "A road", "4": "B road", "5": "C road",
                         "6": "Unclassified"},
    "junction_control": {"1": "Authorised person", "2": "Automatic traffic signal", "3": "Stop sign",
                         "4": "Give way or uncontrolled"},
    "road_surface_conditions": {"1": "Dry", "2": "Wet or damp", "3": "Snow", "4": "Frost or ice", "5": "Flood"},
}
POLICE_FORCE = {
    "1": "Metropolitan Police", "3": "Cumbria", "4": "Lancashire", "5": "Merseyside", "6": "Greater Manchester",
    "7": "Cheshire", "10": "Northumbria", "11": "Durham", "12": "North Yorkshire", "13": "West Yorkshire",
    "14": "South Yorkshire", "16": "Humberside", "17": "Cleveland", "20": "West Midlands", "21": "Staffordshire",
    "22": "West Mercia", "23": "Warwickshire", "30": "Derbyshire", "31": "Nottinghamshire", "32": "Lincolnshire",
    "33": "Leicestershire", "34": "Northamptonshire", "35": "Cambridgeshire", "36": "Norfolk", "37": "Suffolk",
    "40": "Bedfordshire", "41": "Hertfordshire", "42": "Essex", "43": "Thames Valley", "44": "Hampshire",
    "45": "Surrey", "46": "Kent", "47": "Sussex", "48": "City of London", "50": "Devon and Cornwall",
    "52": "Avon and Somerset", "53": "Gloucestershire", "54": "Wiltshire", "55": "Dorset", "60": "North Wales",
    "61": "Gwent", "62": "South Wales", "63": "Dyfed-Powys", "99": "Police Scotland",
}
OPTIONAL_FIELDS = list(OPTIONAL_LABELS) + ["police_force"]

LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1, random_state=SEED, n_jobs=2)
