"""Ask-next engine: which unanswered question would move this call's risk the most?

For every optional field the dispatcher has left blank, re-score the call with each possible answer using the
existing v2 model, weight the answers by how often they occur in 2021-2024 calls, and rank the questions by the
expected size of the change. No new model and no language model: it is the same score, asked "what if" repeatedly.

Whether asking the top-ranked question first really helps is measured out of time in step7_ask_next_eval.py,
so the claim shown in the app is a measured one.
"""
import json
from functools import lru_cache

from common import OPTIONAL_LABELS, ROOT
from predict import calibrate, raw_v2_score, validate

PRIORS_PATH = ROOT / "models/ask_priors.json"
ASKABLE = ["urban_or_rural_area", "first_road_class", "junction_control", "road_surface_conditions"]
QUESTIONS = {
    "urban_or_rural_area": "Is the location in a town or in the countryside?",
    "first_road_class": "What class of road is it (motorway, A road, B road)?",
    "junction_control": "Is the junction controlled (signals, stop sign, give way)?",
    "road_surface_conditions": "What is the road surface like (dry, wet, icy)?",
}


@lru_cache(maxsize=1)
def priors() -> dict:
    """How often each answer occurred in 2021-2024. Falls back to equal weights if the file is missing."""
    try:
        stored = json.loads(PRIORS_PATH.read_text())
        return {field: {k: float(v) for k, v in stored[field].items()} for field in ASKABLE}
    except (OSError, KeyError, ValueError):
        return {field: {k: 1 / len(OPTIONAL_LABELS[field]) for k in OPTIONAL_LABELS[field]} for field in ASKABLE}


def _points(raw: float) -> float:
    """Calibrated chance of a serious/fatal outcome, in percentage points (raw score if no calibrator exists)."""
    calibrated = calibrate(raw)
    return 100 * (calibrated["calibrated"] if calibrated else raw)


def ask_next(call: dict, top_n: int = 3) -> list[dict]:
    """Rank the unanswered optional questions by how much the answer could change the score.

    Each item has `expected_points` (average absolute change, weighted by how common each answer is),
    `max_points`, and the answers that would raise and lower the risk most. Returns [] when everything askable is
    already answered or the v2 model is unavailable. Raises ValidationError for an invalid call."""
    clean = validate(call)
    try:
        base_raw = raw_v2_score(clean)
    except Exception:
        return []
    base_points = _points(base_raw)
    weights = priors()
    ranked = []
    for field in ASKABLE:
        if field in clean:
            continue
        labels = OPTIONAL_LABELS[field]
        outcomes = []
        for code in labels:
            raw = raw_v2_score({**clean, field: code})
            outcomes.append((code, raw, _points(raw) - base_points))
        expected = sum(weights[field][code] * abs(delta) for code, _, delta in outcomes)
        expected_raw = sum(weights[field][code] * abs(raw - base_raw) for code, raw, _ in outcomes)
        up = max(outcomes, key=lambda o: o[2])
        down = min(outcomes, key=lambda o: o[2])
        ranked.append(dict(
            field=field, question=QUESTIONS[field], expected_points=round(expected, 2),
            max_points=round(max(abs(up[2]), abs(down[2])), 1),
            raises=dict(answer=labels[up[0]], points=round(up[2], 1)),
            lowers=dict(answer=labels[down[0]], points=round(down[2], 1)),
            _sort=(expected, expected_raw)))
    ranked.sort(key=lambda r: r["_sort"], reverse=True)
    for item in ranked:
        item.pop("_sort")
    return ranked[:top_n]
