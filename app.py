"""Call Severity Triage — dispatch desk and shift review."""
import datetime as dt
import json
import time

import streamlit as st

from common import CHOICES, OPTIONAL_LABELS, POLICE_FORCE, ROOT, VALID_SPEED
from intake import extract_fields, redact
from messages import parse_message
from predict import ValidationError, predict_call, validate, what_if
import context
import db
import handover
import insights
import report
import theme
import views


st.set_page_config(page_title="Call Severity Triage", page_icon="🚑", layout="wide")
theme.inject()

st.markdown(theme.hero(), unsafe_allow_html=True)
st.markdown(
    '<div class="safety-note"><span class="safety-dot"></span>'
    '<span><strong>Decision support only.</strong> Use dispatcher judgement and established emergency-response protocols.</span></div>',
    unsafe_allow_html=True,
)

dispatch_tab, pulse_tab, control_tab, capacity_tab, review_tab, health_tab = st.tabs(
    ["Dispatch desk", "Pulse", "Control room", "Capacity planner", "Shift review", "Model health"])

CONTEXT_FIELDS = {"light_condition": "f_light", "weather": "f_weather", "police_force": "opt_police_force",
                  "urban_or_rural_area": "opt_urban_or_rural_area", "road_surface_conditions": "opt_road_surface_conditions"}
CONTEXT_LABELS = {"light_condition": "Light", "weather": "Weather", "police_force": "Police force",
                  "urban_or_rural_area": "Area type", "road_surface_conditions": "Road surface"}


def valid_options(field: str) -> list:
    """Values the form accepts for a field, so a suggestion can never put an invalid value in a widget."""
    if field in CHOICES:
        return CHOICES[field]
    return list(POLICE_FORCE) if field == "police_force" else list(OPTIONAL_LABELS[field])


def apply_location() -> None:
    """Button callback: read an AML or eCall message, then fill light, weather, force, area and surface from place and time."""
    parsed = parse_message(st.session_state.get("loc_text", ""))
    st.session_state["loc_parsed"] = parsed
    st.session_state.pop("loc_filled", None)
    st.session_state.pop("loc_time", None)
    if parsed["lat"] is None or parsed["lon"] is None:
        return
    when = parsed["call_time"] or dt.datetime.now()
    if parsed["call_time"]:
        st.session_state["loc_time"] = parsed["call_time"]
    suggestions = context.auto_context(parsed["lat"], parsed["lon"], when)
    filled = {}
    for field, widget_key in CONTEXT_FIELDS.items():
        item = suggestions.get(field)
        if not item:
            continue
        if item["value"] in valid_options(field):
            st.session_state[widget_key] = item["value"]
            filled[field] = item
    st.session_state["loc_filled"] = filled


def ask_next_summary() -> str | None:
    """One honest line from the out-of-time test of the ask-next ranking, if that report exists."""
    path = ROOT / "reports/ask_next_eval.json"
    table = ROOT / "reports/ask_next_eval.csv"
    if not (path.exists() and table.exists()):
        return None
    import pandas as pd
    rows = pd.read_csv(table)
    ask = rows[(rows.strategy == "ask next (engine order)") & (rows.questions == 1)].iloc[0]
    rand = rows[(rows.strategy == "random order (mean of 5)") & (rows.questions == 1)].iloc[0]
    interval = json.loads(path.read_text())["ask_next_minus_random_one_question"]["ci95"]["log_loss"]
    return (f"Tested on 2025 calls: asking the top-ranked question first gave better-calibrated scores than asking a random one "
            f"(log loss {ask.log_loss:.3f} against {rand.log_loss:.3f}; difference interval {interval[0]:.3f} to {interval[1]:.3f}). "
            f"The effect on how well calls are ranked was small and not statistically clear.")


def apply_intake() -> None:
    """Button callback: redact the description, then pre-fill the form fields the text clearly supports."""
    text = redact(st.session_state.get("intake_text", ""))
    found = extract_fields(text)
    for key, (value, _) in found.items():
        st.session_state[key if key.startswith("opt_") else f"f_{key}"] = value
    st.session_state["intake_found"] = (text, found)


with dispatch_tab:
    st.markdown(theme.page_intro(
        "New call assessment", "Enter the five core conditions to generate a risk flag and log the assessment. Optional detail sharpens the score."
    ), unsafe_allow_html=True)

    intake_col, result_col = st.columns([1.08, 0.92], gap="large")
    with intake_col:
        st.markdown(theme.section_title("Call conditions", "activity"), unsafe_allow_html=True)
        st.caption("All five fields are required. Choose the closest available category.")
        with st.expander("Paste a location message (phone AML or vehicle eCall)"):
            st.text_area("Location message", key="loc_text", height=80, label_visibility="collapsed",
                         placeholder='e.g. A"ML";lt=+51.5074;lg=-0.1278;rd=18;top=20260624113000;lc=95;pm=G')
            st.button("Read message and fill from location", on_click=apply_location)
            parsed = st.session_state.get("loc_parsed")
            if parsed:
                for problem in parsed["problems"]:
                    st.warning(problem)
                if parsed["lat"] is not None:
                    radius = f", within {parsed['radius_m']:.0f} m" if parsed["radius_m"] is not None else ""
                    st.caption(f"{'Phone location' if parsed['kind'] == 'aml' else 'Vehicle crash alert'}: "
                               f"{parsed['lat']:.4f}, {parsed['lon']:.4f}{radius}"
                               + (f". Call time taken from the message: {parsed['call_time']:%d %b %Y %H:%M}" if parsed["call_time"] else "."))
                for label, value in parsed["details"].items():
                    st.caption(f"{label}: {value}")
                for field, item in st.session_state.get("loc_filled", {}).items():
                    shown = POLICE_FORCE.get(item["value"], item["value"]) if field == "police_force" else (
                        OPTIONAL_LABELS[field].get(item["value"], item["value"]) if field in OPTIONAL_LABELS else item["value"])
                    st.caption(f"{CONTEXT_LABELS[field]}: {shown} ({item['source'].lower()}; {item['detail']})")
                if st.session_state.get("loc_filled"):
                    st.caption("These are suggestions. Check each one against what the caller says, and street lighting and road class by voice.")
        with st.expander("Describe the call in words (optional)"):
            st.text_area("Call description", key="intake_text", height=90, label_visibility="collapsed",
                         placeholder="e.g. Car off the road on the A38 near a village, 60mph, raining, no street lights, icy surface")
            st.button("Suggest form fields", on_click=apply_intake)
            if "intake_found" in st.session_state:
                text, found = st.session_state["intake_found"]
                st.caption(f"Stored text (personal details removed): {text or 'nothing entered'}")
                if found:
                    st.caption("Filled in, please check each one: " + "; ".join(f"{key.removeprefix('opt_').replace('_', ' ')} from \"{ev}\"" for key, (_, ev) in found.items()))
                else:
                    st.caption("No fields could be read from that description. Choose them below.")
        with st.form("call", clear_on_submit=False):
            road = st.selectbox("Road type", [""] + CHOICES["road_type"], format_func=lambda value: value or "Select a road type", key="f_road")
            speed = st.selectbox("Speed limit", [""] + [str(value) for value in VALID_SPEED], format_func=lambda value: f"{value} mph" if value else "Select a speed limit", key="f_speed")
            weather = st.selectbox("Weather", [""] + CHOICES["weather"], format_func=lambda value: value or "Select weather conditions", key="f_weather")
            light = st.selectbox("Light condition", [""] + CHOICES["light_condition"], format_func=lambda value: value or "Select a light condition", key="f_light")
            junction = st.selectbox("Junction detail", [""] + CHOICES["junction_detail"], format_func=lambda value: value or "Select junction detail", key="f_junction")
            with st.expander("More detail (optional, improves accuracy)"):
                st.caption("Call time is captured automatically. Leave a field blank if it is not known.")
                optional = {}
                for field, label in [("urban_or_rural_area", "Area type"), ("first_road_class", "Road class"),
                                     ("junction_control", "Junction control"), ("road_surface_conditions", "Road surface")]:
                    labels = OPTIONAL_LABELS[field]
                    optional[field] = st.selectbox(label, [""] + list(labels), format_func=lambda v, m=labels: m.get(v, "Not known"), key=f"opt_{field}")
                optional["police_force"] = st.selectbox("Police force area", [""] + list(POLICE_FORCE), format_func=lambda v: POLICE_FORCE.get(v, "Not known"), key="opt_police_force")
            go = st.form_submit_button("Assess and log call", type="primary", width="stretch")

        if go:
            call = {
                "road_type": road,
                "speed_limit": speed,
                "weather": weather,
                "light_condition": light,
                "junction_detail": junction,
                **{k: v for k, v in optional.items() if v},
                "call_time": st.session_state.pop("loc_time", None) or dt.datetime.now(),
            }
            try:
                started = time.perf_counter()
                prediction = predict_call(call)
                elapsed_ms = (time.perf_counter() - started) * 1000
                call_id = db.log_call(validate(call), prediction)
                st.session_state["last"] = (call_id, prediction, elapsed_ms)
                st.session_state["last_call"] = call
                st.session_state.pop("err", None)
                st.session_state.pop("saved", None)
            except ValidationError as error:
                st.session_state["err"] = str(error)

        if "err" in st.session_state:
            st.error(f"Assessment not saved. {st.session_state.pop('err')}")

    with result_col:
        st.markdown(theme.section_title("Triage assessment", "gauge"), unsafe_allow_html=True)
        if "last" in st.session_state:
            call_id, prediction, elapsed_ms = st.session_state["last"]
            st.markdown(theme.verdict_card(prediction, elapsed_ms, call_id), unsafe_allow_html=True)
            st.markdown(theme.calibration_card(prediction.get("calibration"), views.BASE_RATE), unsafe_allow_html=True)
            st.markdown(theme.reasons_card(prediction.get("reasons", [])), unsafe_allow_html=True)
            if prediction.get("calibration") and "last_call" in st.session_state:
                try:
                    questions = insights.ask_next(st.session_state["last_call"])
                except ValidationError:
                    questions = []
                if questions:
                    with st.container(border=True):
                        st.markdown(theme.section_title("Ask next", "phone"), unsafe_allow_html=True)
                        st.caption("Unanswered questions, ranked by how much the answer could move this call's risk.")
                        for rank, item in enumerate(questions, 1):
                            st.markdown(f"**{rank}. {item['question']}**  \n"
                                        f"The answer moves the score between {item['lowers']['points']:+.1f} points "
                                        f"({item['lowers']['answer']}) and {item['raises']['points']:+.1f} points "
                                        f"({item['raises']['answer']}). Typical shift {item['expected_points']:.1f} points.")
                        if summary := ask_next_summary():
                            st.caption(summary)
            if prediction.get("calibration") and "last_call" in st.session_state:
                with st.expander("What if the conditions change?"):
                    st.caption("Re-score this call with different conditions. Nothing is logged.")
                    changes = {}
                    for field, label in [("light_condition", "Light condition"), ("weather", "Weather"), ("junction_detail", "Junction detail")]:
                        picked = st.selectbox(label, ["Unchanged"] + CHOICES[field], key=f"wi_{field}")
                        if picked != "Unchanged":
                            changes[field] = picked
                    speed_pick = st.selectbox("Speed limit", ["Unchanged"] + [str(v) for v in VALID_SPEED], key="wi_speed")
                    if speed_pick != "Unchanged":
                        changes["speed_limit"] = speed_pick
                    if changes:
                        try:
                            outcome = what_if(st.session_state["last_call"], changes)
                        except ValidationError as error:
                            outcome = {}
                            st.warning(str(error))
                        if outcome:
                            a, b = st.columns(2)
                            a.metric("Similar calls serious/fatal, now", f"{outcome['before']:.0%}")
                            b.metric("With the change", f"{outcome['after']:.0%}", f"{outcome['delta_points']:+.1f} points", delta_color="inverse")
            with st.container(border=True):
                st.markdown(theme.section_title("Dispatcher decision", "siren"), unsafe_allow_html=True)
                st.caption("Record any override and whether enhanced response was sent.")
                with st.form("dispatcher_decision"):
                    override = st.radio(
                        "Override the model flag?",
                        ["No override", "High risk", "Low risk"],
                        horizontal=True,
                    )
                    enhanced_sent = st.checkbox("Enhanced response sent")
                    save_decision = st.form_submit_button("Save dispatch decision", width="stretch")
                if save_decision:
                    db.update_call(
                        call_id,
                        dispatcher_override=None if override == "No override" else override,
                        enhanced_response_sent=int(enhanced_sent),
                    )
                    st.session_state["saved"] = True
                if st.session_state.get("saved"):
                    st.success("Dispatch decision saved to the call log.")
        else:
            st.markdown(theme.placeholder_card(), unsafe_allow_html=True)

    st.markdown(theme.section_title("Recent assessments", "list"), unsafe_allow_html=True)
    recent = db.recent_calls(25)
    if recent.empty:
        st.info("No calls have been logged yet. Your first assessment will appear here.")
    else:
        filter_col, risk_col = st.columns([1.5, 1])
        with filter_col:
            search = st.text_input("Find a call", placeholder="Search call ID or road type", label_visibility="collapsed")
        with risk_col:
            risk_filter = st.selectbox("Filter by result", ["All results", "High risk", "Low risk", "Baseline estimate"], label_visibility="collapsed")
        visible = recent.copy()
        visible["risk_display"] = visible["predicted_risk"]
        if "is_estimate" in visible:
            estimate_rows = visible["is_estimate"].fillna(0).astype(bool)
            visible.loc[estimate_rows, "risk_display"] = "Baseline estimate"
        if search.strip():
            query = search.strip().casefold()
            mask = visible["call_id"].str.casefold().str.contains(query, na=False) | visible["road_type"].str.casefold().str.contains(query, na=False)
            visible = visible[mask]
        if risk_filter != "All results":
            visible = visible[visible["risk_display"] == risk_filter]
        display = visible[["call_id", "timestamp", "risk_display", "road_type", "speed_limit", "enhanced_response_sent"]].copy()
        display["call_id"] = display["call_id"].str.slice(0, 8)
        display = display.rename(columns={
            "call_id": "Call ID", "timestamp": "Logged", "risk_display": "Result",
            "road_type": "Road type", "speed_limit": "Limit (mph)", "enhanced_response_sent": "Enhanced response",
        })
        if display.empty:
            st.info("No recent calls match those filters.")
        else:
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={"Enhanced response": st.column_config.CheckboxColumn("Enhanced response", disabled=True)},
            )


with review_tab:
    calls = db.all_calls()
    st.markdown(theme.page_intro(
        "Shift review", "Review recorded assessments, outcomes, and enhanced-response activity."
    ), unsafe_allow_html=True)

    total_calls = len(calls)
    high_count = int((calls["predicted_risk"] == "High risk").sum()) if total_calls else 0
    enhanced_count = int(calls["enhanced_response_sent"].fillna(0).sum()) if total_calls else 0
    outcomes_pending = int(calls["actual_outcome"].isna().sum()) if total_calls else 0
    metrics = st.columns(4, gap="medium")
    metric_specs = [
        (total_calls, "Assessments logged", "phone", "blue"),
        (high_count, "High-risk flags", "alert", "red"),
        (enhanced_count, "Enhanced responses", "siren", "green"),
        (outcomes_pending, "Outcomes pending", "clock", "amber"),
    ]
    for column, (value, label, icon, tone) in zip(metrics, metric_specs):
        with column:
            st.markdown(theme.metric_tile(value, label, icon, tone), unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    outcome_col, precision_col = st.columns([1.08, 0.92], gap="large")
    with outcome_col:
        with st.container(border=True):
            st.markdown(theme.section_title("Record an actual outcome", "activity"), unsafe_allow_html=True)
            st.caption("Record the known collision outcome: serious/fatal or slight. Outcomes are used in Precision@K.")
            if calls.empty:
                st.info("Log a call assessment before recording an outcome.")
            else:
                choices = calls.sort_values(["timestamp", "call_id"], ascending=False).to_dict("records")
                def call_label(row):
                    call_id = row["call_id"][:8]
                    risk = "Baseline estimate" if bool(row.get("is_estimate", 0)) else row.get("predicted_risk") or "Unscored"
                    road_type = row.get("road_type") or "Road not recorded"
                    timestamp = str(row.get("timestamp") or "")[:16].replace("T", " ")
                    return f"{timestamp} · {call_id} · {risk} · {road_type}"
                with st.form("record_outcome"):
                    selected = st.selectbox("Call", choices, format_func=call_label)
                    outcome_label = st.selectbox("Actual outcome", ["", "Serious / fatal", "Slight"], format_func=lambda value: value or "Select the known outcome")
                    save_outcome = st.form_submit_button("Save actual outcome", width="stretch")
                if save_outcome:
                    if not outcome_label:
                        st.warning("Select an actual outcome before saving.")
                    else:
                        db.update_call(selected["call_id"], actual_outcome=1 if outcome_label == "Serious / fatal" else 0)
                        st.session_state["outcome_saved"] = True
                if st.session_state.pop("outcome_saved", False):
                    st.success("Actual outcome recorded.")

    with precision_col:
        st.markdown(theme.section_title("Triage performance", "gauge"), unsafe_allow_html=True)
        result = db.shift_precision_at_k(calls) if not calls.empty else {"ok": False, "message": "No calls have been logged yet."}
        if result["ok"]:
            st.markdown(theme.pk_card(result["k"], result["value"]), unsafe_allow_html=True)
            st.caption("Precision@K is calculated from calls with a known outcome and risk score.")
        else:
            st.markdown(theme.note_card(f"<strong>Precision@K not available yet.</strong> {result['message']}", "warn"), unsafe_allow_html=True)
            st.caption("K is the number of enhanced-response dispatches recorded in the call log.")

    st.markdown(theme.section_title("Shift call log", "list"), unsafe_allow_html=True)
    st.caption("Download the complete recorded log as CSV for reporting.")
    if calls.empty:
        st.info("The shift log is empty. It will be available for export after a call is assessed.")
    else:
        brief = handover.write_brief(calls, result)
        st.markdown(theme.section_title("Handover brief", "list"), unsafe_allow_html=True)
        st.write(brief["text"])
        st.caption("Written from the shift numbers only; every figure is checked against them."
                   if brief["source"] == "model" else
                   "Built directly from the shift numbers. A model-written version is used only when every figure in it matches them.")
        csv_col, pdf_col, _ = st.columns([1, 1, 2])
        csv_col.download_button("Download shift log (CSV)", calls.to_csv(index=False), "shift_log.csv", mime="text/csv")
        pdf_col.download_button("Download shift report (PDF)", report.build_pdf(calls, result), "shift_report.pdf", mime="application/pdf")
        display_calls = calls.sort_values(["timestamp", "call_id"], ascending=False).copy()
        estimate_rows = display_calls["is_estimate"].fillna(0).astype(bool)
        display_calls.loc[estimate_rows, "predicted_risk"] = "Baseline estimate"
        display_calls["call_id"] = display_calls["call_id"].str.slice(0, 8)
        display_calls = display_calls.rename(columns={
            "call_id": "Call ID", "timestamp": "Logged", "predicted_risk": "Result",
            "confidence_score": "Confidence", "speed_limit": "Limit (mph)",
            "enhanced_response_sent": "Enhanced response", "actual_outcome": "Actual outcome",
        })
        columns = [name for name in ["Call ID", "Logged", "Result", "Confidence", "Limit (mph)", "Enhanced response", "Actual outcome"] if name in display_calls.columns]
        st.dataframe(display_calls[columns], width="stretch", hide_index=True)

with pulse_tab:
    views.pulse()

with control_tab:
    views.control_room()

with capacity_tab:
    views.capacity_planner()

with health_tab:
    views.model_health(db.all_calls())
