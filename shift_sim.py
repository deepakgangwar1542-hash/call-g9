"""Shift replay: a deterministic dispatch simulation on held-out 2025 collisions.

Enhanced-response units are scarce. Calls arrive over a shift; whenever a unit is free, the policy picks which waiting
call it goes to. 'model' ranks by risk score (plus a small age bonus so nobody waits forever); 'fifo' is first come,
first served. A waiting call lapses to standard response after `patience` minutes. Outcomes come from the real STATS19
record, so the comparison shows what each policy would have caught, but only in hindsight and on collision data.
"""
import numpy as np
import pandas as pd

POLICIES = ("model", "fifo")


def build_shift(replay: pd.DataFrame, n_calls: int, minutes: int = 480, seed: int = 7) -> pd.DataFrame:
    """Sample `n_calls` collisions and give each a random arrival minute within the shift."""
    n_calls = min(n_calls, len(replay))
    shift = replay.sample(n_calls, random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    shift["arrival"] = np.sort(rng.uniform(0, minutes, n_calls)).round(2)
    return shift


def simulate(shift: pd.DataFrame, units: int, service_min: int, patience_min: int, policy: str,
             until: int | None = None, age_weight: float = 0.002) -> pd.DataFrame:
    """Run minute by minute up to `until` (default: the whole shift). Returns the shift with status columns:
    status in {not_arrived, waiting, dispatched, lapsed}, dispatch_minute, wait_min."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if units < 1 or service_min < 1 or patience_min < 1:
        raise ValueError("units, service_min and patience_min must be at least 1")
    arrival = shift["arrival"].to_numpy(float)
    score = shift["score"].to_numpy(float)
    n = len(shift)
    horizon = int(np.ceil(arrival.max())) + 1 if n else 0
    end = horizon if until is None else min(int(until), horizon)

    status = np.array(["not_arrived"] * n, dtype=object)
    dispatch_at = np.full(n, np.nan)
    free_at = np.zeros(units)
    waiting: list[int] = []
    next_arrival = 0
    for t in range(end + 1):
        while next_arrival < n and arrival[next_arrival] <= t:
            waiting.append(next_arrival)
            status[next_arrival] = "waiting"
            next_arrival += 1
        expired = [i for i in waiting if t - arrival[i] > patience_min]
        for i in expired:
            status[i] = "lapsed"
        if expired:
            waiting = [i for i in waiting if status[i] == "waiting"]
        for u in np.flatnonzero(free_at <= t):
            if not waiting:
                break
            if policy == "model":
                pick = max(waiting, key=lambda i: score[i] + age_weight * (t - arrival[i]))
            else:
                pick = min(waiting, key=lambda i: arrival[i])
            waiting.remove(pick)
            status[pick] = "dispatched"
            dispatch_at[pick] = t
            free_at[u] = t + service_min
    out = shift.copy()
    out["status"] = status
    out["dispatch_minute"] = dispatch_at
    out["wait_min"] = out["dispatch_minute"] - out["arrival"]
    return out


def summarize(run: pd.DataFrame) -> dict:
    """Headline numbers for one run. 'Serious' means the call's recorded outcome was serious or fatal."""
    arrived = run[run.status != "not_arrived"]
    serious = arrived[arrived.target == 1]
    sent = arrived[arrived.status == "dispatched"]
    caught = sent[sent.target == 1]
    return dict(
        arrived=int(len(arrived)), dispatched=int(len(sent)), lapsed=int((arrived.status == "lapsed").sum()),
        serious_arrived=int(len(serious)), serious_caught=int(len(caught)),
        serious_caught_share=float(len(caught) / len(serious)) if len(serious) else float("nan"),
        precision=float(len(caught) / len(sent)) if len(sent) else float("nan"),
        mean_wait_serious=float(caught.wait_min.mean()) if len(caught) else float("nan"),
    )


def compare(shift: pd.DataFrame, units: int, service_min: int, patience_min: int, age_weight: float = 0.002) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        rows.append({"policy": policy, **summarize(simulate(shift, units, service_min, patience_min, policy, age_weight=age_weight))})
    return pd.DataFrame(rows)
