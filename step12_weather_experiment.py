"""Does real hourly weather improve the model beyond the coded 'weather' category? (wow factor #2)

Same temporal protocol as step4: train 2021-2024, test once on 2025. Two LightGBM models, identical except that the
second also gets hourly weather for the collision's police-force area. Reports a paired bootstrap interval for the
difference in precision at the top 2% and in ROC AUC. Run step11_weather_fetch.py first.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import LGBM_PARAMS, ROOT, TIERS
from step4_train_multiyear import load_clean, metrics, to_frame
from step9_bakeoff import paired_bootstrap

PROCESSED = ROOT / "data/processed"
TIER = "T4_plus_force"
WEATHER_COLS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "visibility", "cloud_cover",
                "precip_prev3h"]


def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_csv(PROCESSED / "weather_by_force.csv", dtype={"police_force": str})
    weather = weather.sort_values(["police_force", "time"]).reset_index(drop=True)
    weather["precip_prev3h"] = (weather.groupby("police_force")["precipitation"]
                                .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()))
    weather["key"] = weather["police_force"] + "|" + weather["time"].str[:13]
    dates = pd.read_csv(PROCESSED / "clean_multi_dates.csv")["date"]
    key = df["police_force"].astype(str) + "|" + dates + "T" + df["hour"].astype(int).astype(str).str.zfill(2)
    joined = key.map(weather.set_index("key")["temperature_2m"]).notna()
    print(f"Weather matched for {joined.mean():.1%} of collisions")
    merged = pd.DataFrame({"key": key}).merge(weather[["key"] + WEATHER_COLS], on="key", how="left")
    return pd.concat([df.reset_index(drop=True), merged[WEATHER_COLS]], axis=1)


def fit_score(train, test, features, categories):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(to_frame(train, features, categories), train["target"])
    return model.predict_proba(to_frame(test, features, categories))[:, 1]


def main():
    df, categories = load_clean()
    df = attach_weather(df)
    test_year = df.year.max()
    train, test = df[df.year < test_year], df[df.year == test_year]
    base_features = TIERS[TIER]
    base = fit_score(train, test, base_features, categories)
    plus = fit_score(train, test, base_features + WEATHER_COLS, categories)

    d, lo, hi = paired_bootstrap(test.target, plus, base)
    rng, y = np.random.default_rng(0), test.target.to_numpy()
    auc_diffs = []
    for _ in range(200):
        idx = rng.integers(0, len(y), len(y))
        auc_diffs.append(roc_auc_score(y[idx], plus[idx]) - roc_auc_score(y[idx], base[idx]))
    result = dict(
        without_weather=metrics(test.target, base), with_weather=metrics(test.target, plus),
        p2_diff=d, p2_diff_ci=[lo, hi], auc_diff=float(np.mean(auc_diffs)),
        auc_diff_ci=[float(x) for x in np.percentile(auc_diffs, [2.5, 97.5])],
        verdict="helps" if lo > 0 else "no clear gain" if hi >= 0 else "hurts")
    (ROOT / "reports/weather_experiment.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("without_weather", "with_weather")}, indent=2))
    for name in ("without_weather", "with_weather"):
        print(name, {k: round(v, 4) for k, v in result[name].items() if k in ("roc_auc", "P@2pct", "P@5pct")})


if __name__ == "__main__":
    main()
