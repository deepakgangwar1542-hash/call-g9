import pandas as pd

from handover import facts_from_calls, template_brief, ungrounded_numbers, write_brief


def make_calls(n=40):
    return pd.DataFrame({
        "is_estimate": [0] * n,
        "predicted_risk": ["High risk"] * 7 + ["Lower risk"] * (n - 7),
        "enhanced_response_sent": [1] * 5 + [0] * (n - 5),
        "dispatcher_override": ["x"] * 3 + [None] * (n - 3),
        "actual_outcome": [1.0] * 4 + [0.0] * 6 + [None] * (n - 10),
    })


PRECISION = {"ok": True, "value": 0.4286, "k": 7}


def test_template_quotes_only_facts():
    facts = facts_from_calls(make_calls(), PRECISION)
    assert facts["high_risk_flags"] == 7 and facts["precision_at_k_percent"] == 42.9
    assert ungrounded_numbers(template_brief(facts), facts) == []


def test_invented_number_is_detected():
    facts = facts_from_calls(make_calls(), PRECISION)
    assert ungrounded_numbers("Risk rose 17% this shift and 40 calls were assessed.", facts) == ["17"]


def test_model_text_with_invented_number_is_rejected():
    result = write_brief(make_calls(), PRECISION, rewrite=lambda f: "We logged 40 calls and saved 12 lives.")
    assert result["source"] == "template" and result["rejected"] == ["12"]


def test_grounded_model_text_is_accepted():
    result = write_brief(make_calls(), PRECISION, rewrite=lambda f: "Of 40 assessments, 7 were high risk.")
    assert result["source"] == "model" and result["rejected"] == []


def test_no_key_or_failure_falls_back_to_template():
    assert write_brief(make_calls(), PRECISION, rewrite=lambda f: None)["source"] == "template"
    assert write_brief(make_calls(), PRECISION, rewrite=None)["source"] == "template"


def test_empty_log():
    empty = make_calls().iloc[0:0]
    assert write_brief(empty, {"ok": False})["text"] == "No calls were assessed this shift."
