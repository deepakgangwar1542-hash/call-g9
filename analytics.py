"""Model-health analytics: drift, fairness by group, dispatch economics and calibration reliability."""
import numpy as np
import pandas as pd

from common import FLAG_RATE, POLICE_FORCE

DRIFT_FIELDS = ["road_type", "speed_limit", "weather", "light_condition", "junction_detail"]
MIN_DRIFT_CALLS = 30


def psi(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    """Population Stability Index between two categorical samples. About 0.1 is a modest shift, above 0.25 a major one."""
    cats = sorted(set(reference.astype(str)) | set(current.astype(str)))
    ref = reference.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=eps)
    cur = current.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=eps)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def drift_report(reference: pd.DataFrame, logged: pd.DataFrame) -> pd.DataFrame:
    """PSI per core input between a reference sample and the live call log. Empty until enough calls are logged."""
    if len(logged) < MIN_DRIFT_CALLS:
        return pd.DataFrame(columns=["field", "psi", "level"])
    rows = []
    for field in DRIFT_FIELDS:
        value = psi(reference[field], logged[field])
        rows.append(dict(field=field, psi=round(value, 3), level="major" if value > 0.25 else "moderate" if value > 0.1 else "stable"))
    return pd.DataFrame(rows)


def group_fairness(replay: pd.DataFrame, by: str, flag_rate: float = FLAG_RATE, min_n: int = 1000) -> pd.DataFrame:
    """Out-of-time check of how the top `flag_rate` share of calls is spread across groups.

    flag_share: how often the group is escalated. precision: of its escalations, the share truly serious/fatal.
    recall: of the group's serious/fatal calls, the share escalated. Compare groups on all three; none alone is 'fair'."""
    k = max(1, int(np.ceil(flag_rate * len(replay))))
    cutoff = replay["raw_score"].nlargest(k).min()
    frame = replay.assign(flagged=replay["raw_score"] >= cutoff)
    rows = []
    for group, part in frame.groupby(by):
        if len(part) < min_n:
            continue
        flagged = part[part.flagged]
        serious = part[part.target == 1]
        rows.append(dict(group=str(group), n=len(part), base_rate=part.target.mean(), flag_share=part.flagged.mean(),
                         precision=flagged.target.mean() if len(flagged) else np.nan,
                         recall=serious.flagged.mean() if len(serious) else np.nan))
    out = pd.DataFrame(rows)
    if by == "police_force" and not out.empty:
        out["group"] = out["group"].map(lambda g: POLICE_FORCE.get(g, g))
    return out.sort_values("flag_share", ascending=False).reset_index(drop=True)


def net_benefit_curve(y: np.ndarray, scores: np.ndarray, value_serious: float, cost_dispatch: float, points: int = 60) -> pd.DataFrame:
    """Expected net benefit per 1,000 calls if the top `flag_share` of calls are escalated.

    net = (serious cases reached x value) - (dispatches x cost). Units are whatever the user's value and cost use."""
    order = np.argsort(-scores, kind="stable")
    y_sorted = np.asarray(y)[order]
    cum = np.cumsum(y_sorted)
    n = len(y_sorted)
    ks = np.unique(np.linspace(max(1, n // 400), n // 3, points).astype(int))
    caught, sent = cum[ks - 1], ks
    scale = 1000 / n
    return pd.DataFrame(dict(flag_share=ks / n, dispatches_per_1000=sent * scale, serious_reached_per_1000=caught * scale,
                             precision=caught / sent, net_benefit=(caught * value_serious - sent * cost_dispatch) * scale))


def reliability(calibration: pd.DataFrame) -> pd.DataFrame:
    """Reliability-plot points from reports/calibration_v2.csv: mean score in the bin vs the rate actually observed."""
    out = calibration.copy()
    out["score_mid"] = (out.score_low + out.score_high) / 2
    return out
