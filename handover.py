"""Grounded shift handover brief.

The brief is built only from numbers computed from the call log. A deterministic template always works. If an AI
Gateway key is present, a language model may rewrite the same facts into smoother prose, but its text is accepted only
if every number it contains is one of the supplied facts; otherwise the template is used and the rejection is reported.
The model never sees raw calls, only the computed facts, and it is never trusted to produce a figure.
"""
import json
import os
import re
import urllib.request

import pandas as pd

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
GATEWAY_MODEL = "google/gemini-2.5-flash"
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def facts_from_calls(calls: pd.DataFrame, precision: dict) -> dict:
    """Every figure the brief is allowed to quote. Percentages are stored as percentages."""
    total = len(calls)
    scored = calls[calls["is_estimate"].fillna(0) == 0] if total else calls
    facts = {
        "assessments": total,
        "high_risk_flags": int((scored["predicted_risk"] == "High risk").sum()) if total else 0,
        "enhanced_responses_sent": int(calls["enhanced_response_sent"].fillna(0).sum()) if total else 0,
        "dispatcher_overrides": int(calls["dispatcher_override"].notna().sum()) if total else 0,
        "outcomes_recorded": int(calls["actual_outcome"].notna().sum()) if total else 0,
    }
    if precision.get("ok"):
        facts["precision_at_k_percent"] = round(100 * precision["value"], 1)
        facts["k"] = int(precision["k"])
    return facts


def template_brief(facts: dict) -> str:
    if facts["assessments"] == 0:
        return "No calls were assessed this shift."
    parts = [f"This shift logged {facts['assessments']} assessments. "
             f"{facts['high_risk_flags']} were flagged high risk and {facts['enhanced_responses_sent']} received an enhanced response. "
             f"Dispatchers overrode the score on {facts['dispatcher_overrides']}."]
    if facts["outcomes_recorded"]:
        parts.append(f"{facts['outcomes_recorded']} outcomes have been recorded so far.")
    else:
        parts.append("No outcomes have been recorded yet.")
    if "precision_at_k_percent" in facts:
        parts.append(f"Among the top {facts['k']} flagged calls with a known outcome, precision was {facts['precision_at_k_percent']}%.")
    else:
        parts.append("Precision is not available yet.")
    return " ".join(parts)


def _decimals(token: str) -> int:
    return len(token.split(".")[1]) if "." in token else 0


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in `text` that are not one of the facts (compared at the precision the text uses)."""
    allowed = [float(v) for v in facts.values()]
    bad = []
    for token in NUMBER.findall(text):
        value = float(token.replace(",", ""))
        digits = _decimals(token)
        if not any(round(a, digits) == round(value, digits) for a in allowed):
            bad.append(token)
    return bad


def gateway_rewrite(facts: dict, timeout: float = 20.0) -> str | None:
    """Ask the model to rewrite the facts as a short paragraph. Returns None when no key is set or the call fails."""
    key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
    if not key:
        return None
    prompt = ("Write a three-sentence shift handover for an ambulance dispatch supervisor using ONLY the facts below. "
              "Quote numbers exactly as given, do not compute new numbers, add no numbers of your own, and do not give "
              "medical or operational advice.\nFACTS: " + json.dumps(facts))
    body = json.dumps({"model": GATEWAY_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    request = urllib.request.Request(GATEWAY_URL, data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def write_brief(calls: pd.DataFrame, precision: dict, rewrite=gateway_rewrite) -> dict:
    """Return {text, source, facts, rejected}. `source` is 'model' only when the rewrite passed the number check."""
    facts = facts_from_calls(calls, precision)
    fallback = template_brief(facts)
    if facts["assessments"] == 0 or rewrite is None:
        return dict(text=fallback, source="template", facts=facts, rejected=[])
    candidate = rewrite(facts)
    if not candidate:
        return dict(text=fallback, source="template", facts=facts, rejected=[])
    bad = ungrounded_numbers(candidate, facts)
    if bad:
        return dict(text=fallback, source="template", facts=facts, rejected=bad)
    return dict(text=candidate, source="model", facts=facts, rejected=[])
