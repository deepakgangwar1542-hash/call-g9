"""Call Severity Triage — dispatch desk and shift review."""
import time

import streamlit as st

from common import CHOICES, VALID_SPEED
from predict import ValidationError, predict_call
import db
import theme


st.set_page_config(page_title="Call Severity Triage", page_icon="🚑", layout="wide")
theme.inject()

st.markdown(theme.hero(), unsafe_allow_html=True)
st.markdown(
    '<div class="safety-note"><span class="safety-dot"></span>'
    '<span><strong>Decision support only.</strong> Use dispatcher judgement and established emergency-response protocols.</span></div>',
    unsafe_allow_html=True,
)

dispatch_tab, review_tab = st.tabs(["Dispatch desk", "Shift review"])


with dispatch_tab:
    st.markdown(theme.page_intro(
        "New call assessment", "Enter the five conditions known at call time to generate a risk flag and log the assessment."
    ), unsafe_allow_html=True)

    intake_col, result_col = st.columns([1.08, 0.92], gap="large")
    with intake_col:
        st.markdown(theme.section_title("Call conditions", "activity"), unsafe_allow_html=True)
        st.caption("All five fields are required. Choose the closest available category.")
        with st.form("call", clear_on_submit=False):
            road = st.selectbox("Road type", [""] + CHOICES["road_type"], format_func=lambda value: value or "Select a road type")
            speed = st.selectbox("Speed limit", [""] + [str(value) for value in VALID_SPEED], format_func=lambda value: f"{value} mph" if value else "Select a speed limit")
            weather = st.selectbox("Weather", [""] + CHOICES["weather"], format_func=lambda value: value or "Select weather conditions")
            light = st.selectbox("Light condition", [""] + CHOICES["light_condition"], format_func=lambda value: value or "Select a light condition")
            junction = st.selectbox("Junction detail", [""] + CHOICES["junction_detail"], format_func=lambda value: value or "Select junction detail")
            go = st.form_submit_button("Assess and log call", type="primary", width="stretch")

        if go:
            call = {
                "road_type": road,
                "speed_limit": speed,
                "weather": weather,
                "light_condition": light,
                "junction_detail": junction,
            }
            try:
                started = time.perf_counter()
                prediction = predict_call(call)
                elapsed_ms = (time.perf_counter() - started) * 1000
                call_id = db.log_call({**call, "speed_limit": int(speed)}, prediction)
                st.session_state["last"] = (call_id, prediction, elapsed_ms)
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
        st.download_button("Download shift log (CSV)", calls.to_csv(index=False), "shift_log.csv", mime="text/csv")
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
