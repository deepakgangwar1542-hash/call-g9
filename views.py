"""Streamlit screens beyond the dispatch desk: control room, capacity planner and model health."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import analytics
import shift_sim
import theme
from common import HEX_PATH, OPTIONAL_LABELS, REPLAY_PATH, ROOT, V2_META_PATH

RED, GREEN, AMBER, MUTED, TEXT = "#e11d2e", "#3ecf8e", "#f0bd5e", "#a7a7b2", "#e9e9ee"
SHIFT_MINUTES = 480


def _base_rate() -> float:
    try:
        return float(json.loads(V2_META_PATH.read_text())["base_rate"])
    except Exception:
        return 0.26


BASE_RATE = _base_rate()


def _style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=12, r=12, t=28, b=12), font=dict(color=TEXT, size=12), legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor="#2a2a31", zerolinecolor="#2a2a31")
    fig.update_yaxes(gridcolor="#2a2a31", zerolinecolor="#2a2a31")
    return fig


@st.cache_data(show_spinner=False)
def load_replay() -> pd.DataFrame:
    return pd.read_csv(REPLAY_PATH)


@st.cache_data(show_spinner=False)
def load_hex() -> pd.DataFrame:
    frame = pd.read_csv(HEX_PATH)
    t = ((frame["rate"] - 0.15) / 0.30).clip(0, 1)
    frame["r"] = (40 + t * 185).astype(int)
    frame["g"] = (46 - t * 20).astype(int)
    frame["b"] = (60 - t * 20).astype(int)
    frame["pct"] = (frame["rate"] * 100).round(1)
    return frame


def _load_json(name: str) -> dict:
    path = ROOT / "reports" / name
    return json.loads(path.read_text()) if path.exists() else {}


def _missing(what: str) -> bool:
    st.info(f"{what} is not built yet. Run the training and calibration scripts (step5_calibrate.py, step6_hexmap.py) first.")
    return False


# ----------------------------------------------------------------------------------------------- control room
def _live_body(shift: pd.DataFrame, hexes: pd.DataFrame, units: int, service: int, patience: int) -> None:
    if st.session_state.get("playing"):
        st.session_state["clock"] = min(SHIFT_MINUTES, st.session_state.get("clock", 0) + 15)
        if st.session_state["clock"] >= SHIFT_MINUTES:
            st.session_state["playing"] = False
    clock = st.slider("Shift clock (minutes since start)", 0, SHIFT_MINUTES, key="clock", step=5)
    model = shift_sim.simulate(shift, units, service, patience, "model", until=clock)
    fifo = shift_sim.simulate(shift, units, service, patience, "fifo", until=clock)
    sm, sf = shift_sim.summarize(model), shift_sim.summarize(fifo)

    tiles = st.columns(4, gap="medium")
    for column, (value, label, icon, tone) in zip(tiles, [
        (sm["arrived"], "Calls arrived", "phone", "blue"), (sm["dispatched"], "Enhanced responses sent", "siren", "green"),
        (sm["serious_caught"], "Serious/fatal reached (model order)", "alert", "red"),
        (sf["serious_caught"], "Serious/fatal reached (first come)", "clock", "amber")]):
        with column:
            st.markdown(theme.metric_tile(value, label, icon, tone), unsafe_allow_html=True)

    map_col, queue_col = st.columns([1.35, 1], gap="large")
    with map_col:
        arrived = model[model.status != "not_arrived"].copy()
        colours = {"dispatched": [62, 207, 142, 230], "waiting": [240, 189, 94, 230], "lapsed": [120, 120, 130, 170]}
        arrived["colour"] = arrived.status.map(colours)
        layers = [
            pdk.Layer("H3HexagonLayer", hexes, get_hexagon="hex", get_fill_color="[r, g, b, 150]", pickable=True, extruded=False, stroked=False),
            pdk.Layer("ScatterplotLayer", arrived, get_position="[longitude, latitude]", get_fill_color="colour",
                      get_radius=5, radius_units="pixels", pickable=False),
        ]
        deck = pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=54.2, longitude=-2.6, zoom=4.6),
                        map_style="dark", tooltip={"text": "Historic serious/fatal share: {pct}%"})
        st.pydeck_chart(deck, height=430)
        st.caption("Hex shading: share of collisions that were serious or fatal in 2021-2024 (about 5 sq km cells, shrunk toward the national rate). "
                   "Dots: green dispatched, amber waiting, grey lapsed to standard response.")
    with queue_col:
        st.markdown(theme.section_title("Waiting queue, highest risk first", "list"), unsafe_allow_html=True)
        waiting = model[model.status == "waiting"].sort_values("score", ascending=False).head(8)
        if waiting.empty:
            st.info("No calls are waiting at this point in the shift.")
        else:
            queue = pd.DataFrame({"Waiting (min)": (clock - waiting.arrival).round(0).astype(int), "Risk": (waiting.score * 100).round(0).astype(int),
                                  "Road type": waiting.road_type, "Limit": waiting.speed_limit.astype(int)})
            st.dataframe(queue, hide_index=True, width="stretch",
                         column_config={"Risk": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=60, format="%d"),
                                        "Limit": st.column_config.NumberColumn("Limit (mph)")})
        st.caption("Calls past the patience limit lapse to standard response and leave the queue.")


def control_room() -> None:
    st.markdown(theme.page_intro(
        "Control room",
        "Replay a shift of held-out 2025 collisions and compare risk-ranked dispatch with first come, first served, using the same scarce enhanced-response units."
    ), unsafe_allow_html=True)
    if not (REPLAY_PATH.exists() and HEX_PATH.exists()):
        _missing("The replay data")
        return
    replay, hexes = load_replay(), load_hex()

    with st.container(border=True):
        a, b, c, d, e = st.columns(5)
        n_calls = a.number_input("Calls in shift", 100, 2000, 600, step=100)
        units = b.number_input("Enhanced units", 1, 60, 12)
        service = c.number_input("Minutes per response", 5, 120, 30, step=5)
        patience = d.number_input("Patience (min)", 5, 120, 20, step=5)
        seed = e.number_input("Shift seed", 0, 999, 7)
    shift = shift_sim.build_shift(replay, int(n_calls), SHIFT_MINUTES, int(seed))

    play_col, _ = st.columns([1, 3])
    with play_col:
        st.toggle("Play the shift", key="playing")
    st.session_state.setdefault("clock", 0)
    run_every = 1.0 if st.session_state.get("playing") else None
    st.fragment(run_every=run_every)(_live_body)(shift, hexes, int(units), int(service), int(patience))

    st.markdown(theme.section_title("Whole-shift result", "gauge"), unsafe_allow_html=True)
    result = shift_sim.compare(shift, int(units), int(service), int(patience))
    labels = {"model": "Risk-ranked", "fifo": "First come, first served"}
    chart_col, text_col = st.columns([1.3, 1], gap="large")
    with chart_col:
        fig = go.Figure(go.Bar(x=[labels[p] for p in result.policy], y=result.serious_caught_share * 100,
                               marker_color=[GREEN, MUTED], text=[f"{v:.0%}" for v in result.serious_caught_share], textposition="outside"))
        fig.update_yaxes(title="Serious/fatal calls reached (%)", range=[0, 100])
        st.plotly_chart(_style(fig, 300), width="stretch")
    with text_col:
        m, f = result.iloc[0], result.iloc[1]
        st.markdown(theme.note_card(
            f"With {int(units)} units and {int(n_calls)} calls, risk-ranked dispatch reached <strong>{int(m.serious_caught)}</strong> of "
            f"{int(m.serious_arrived)} serious/fatal calls against <strong>{int(f.serious_caught)}</strong> for first come, first served. "
            f"Mean wait for those calls: {m.mean_wait_serious:.0f} min against {f.mean_wait_serious:.0f} min. "
            "This is a hindsight simulation on collision data, not a prediction of real-world outcomes.", "warn"), unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ capacity planner
def capacity_planner() -> None:
    st.markdown(theme.page_intro(
        "Capacity planner",
        "Choose how many calls to escalate. The curve shows the trade-off between reaching serious calls and spending enhanced responses, on held-out 2025 data."
    ), unsafe_allow_html=True)
    if not REPLAY_PATH.exists():
        _missing("The replay data")
        return
    replay = load_replay()
    with st.container(border=True):
        a, b = st.columns(2)
        value = a.slider("Value of reaching a serious/fatal call", 1.0, 50.0, 10.0, 0.5, help="In any units you like, for example points or pounds.")
        cost = b.slider("Cost of one enhanced response", 0.1, 10.0, 1.0, 0.1, help="Same units as the value.")
    curve = analytics.net_benefit_curve(replay.target.to_numpy(), replay.raw_score.to_numpy(), value, cost)
    best = curve.loc[curve.net_benefit.idxmax()]

    tiles = st.columns(3, gap="medium")
    for column, (val, label, icon, tone) in zip(tiles, [
            (f"{best.flag_share:.1%}", "Best share of calls to escalate", "gauge", "green"),
            (f"{best.precision:.0%}", "Of those escalations are serious/fatal", "alert", "red"),
            (f"{best.serious_reached_per_1000:.0f}", "Serious/fatal reached per 1,000 calls", "siren", "amber")]):
        with column:
            st.markdown(theme.metric_tile(val, label, icon, tone), unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        fig = go.Figure(go.Scatter(x=curve.flag_share * 100, y=curve.net_benefit, mode="lines", line=dict(color=RED, width=3)))
        fig.add_trace(go.Scatter(x=[best.flag_share * 100], y=[best.net_benefit], mode="markers", marker=dict(color=GREEN, size=12), name="Best"))
        fig.update_xaxes(title="Share of calls escalated (%)")
        fig.update_yaxes(title="Net benefit per 1,000 calls")
        fig.update_layout(showlegend=False, title=dict(text="Net benefit", font=dict(size=13)))
        st.plotly_chart(_style(fig), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve.flag_share * 100, y=curve.precision * 100, mode="lines", line=dict(color=GREEN, width=3), name="Precision"))
        fig.add_hline(y=replay.target.mean() * 100, line_dash="dot", line_color=MUTED, annotation_text="Average call", annotation_font_color=MUTED)
        fig.update_xaxes(title="Share of calls escalated (%)")
        fig.update_yaxes(title="Escalations that were serious/fatal (%)")
        fig.update_layout(showlegend=False, title=dict(text="Precision as the net widens", font=dict(size=13)))
        st.plotly_chart(_style(fig), width="stretch")
    st.caption("The best point is where one more escalation stops paying for itself. Precision falls as more calls are escalated, so the answer depends on your value and cost, not on the model alone.")


# ----------------------------------------------------------------------------------------------- model health
def model_health(calls: pd.DataFrame) -> None:
    st.markdown(theme.page_intro(
        "Model health",
        "Accuracy, calibration, drift against live calls, and a fairness check across groups. Everything here is out-of-time: models never saw the year they are scored on."
    ), unsafe_allow_html=True)
    try:
        meta = json.loads(V2_META_PATH.read_text())
    except Exception:
        st.info("Model metadata is not available.")
        return
    m, cal, hexv = meta["metrics_final_test"], _load_json("calibration_report.json"), _load_json("hex_validation.json")

    tiles = st.columns(4, gap="medium")
    specs = [(f"{m['roc_auc']:.3f}", f"ROC-AUC on {meta['test_year']}", "activity", "blue"),
             (f"{m['P@2pct']:.0%}", f"Precision in top 2% (average call {m['base_rate']:.0%})", "gauge", "green"),
             (f"{cal.get('ece_raw', 0):.3f} to {cal.get('ece_calibrated', 0):.3f}", "Calibration error, raw to calibrated", "cpu", "amber"),
             (f"{hexv.get('spearman_2021_24_vs_2025', 0):.2f}", "Hex map rank agreement with 2025", "alert", "red")]
    for column, (val, label, icon, tone) in zip(tiles, specs):
        with column:
            st.markdown(theme.metric_tile(val, label, icon, tone), unsafe_allow_html=True)

    st.markdown("<div class='section-gap'></div>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(theme.section_title("Does a 30% score mean 30%?", "gauge"), unsafe_allow_html=True)
        path = ROOT / "reports/calibration_v2.csv"
        if path.exists():
            rel = analytics.reliability(pd.read_csv(path))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 60], y=[0, 60], mode="lines", line=dict(color=MUTED, dash="dot"), name="Perfect"))
            fig.add_trace(go.Scatter(x=rel.score_mid * 100, y=rel.observed_rate * 100, mode="markers+lines", line=dict(color=RED, width=2), marker=dict(size=8),
                                     error_y=dict(type="data", symmetric=False, array=(rel.ci_high - rel.observed_rate) * 100,
                                                  arrayminus=(rel.observed_rate - rel.ci_low) * 100, color=AMBER), name="Observed"))
            fig.update_xaxes(title="Calibrated score (%)", range=[0, 55])
            fig.update_yaxes(title="Serious/fatal rate observed (%)", range=[0, 55])
            st.plotly_chart(_style(fig), width="stretch")
            st.caption("Points on the dotted line mean the score matches reality. Bars are 95% intervals on the observed rate.")
        else:
            _missing("The calibration table")
    with right:
        st.markdown(theme.section_title("Drift: live calls against 2025", "activity"), unsafe_allow_html=True)
        if REPLAY_PATH.exists():
            logged = calls[["road_type", "speed_limit", "weather", "light_condition", "junction_detail"]] if not calls.empty else pd.DataFrame()
            drift = analytics.drift_report(load_replay(), logged) if len(logged) else pd.DataFrame()
            if drift.empty:
                st.info(f"Drift needs at least {analytics.MIN_DRIFT_CALLS} logged calls; {len(logged)} so far.")
            else:
                st.dataframe(drift, hide_index=True, width="stretch")
                st.caption("PSI below 0.1 is stable, 0.1 to 0.25 is a modest shift, above 0.25 is a major shift worth a model review.")
        override_rate = (calls["dispatcher_override"].notna().mean() if len(calls) else 0.0)
        st.metric("Dispatcher override rate", f"{override_rate:.0%}", help="A rising override rate is an early sign the flags are losing dispatcher trust.")

    st.markdown(theme.section_title("Fairness check: who gets escalated?", "alert"), unsafe_allow_html=True)
    if not REPLAY_PATH.exists():
        _missing("The replay data")
        return
    replay = load_replay()
    area = analytics.group_fairness(replay, "urban_or_rural_area")
    area["group"] = area["group"].map(lambda g: OPTIONAL_LABELS["urban_or_rural_area"].get(str(int(float(g))), g))
    table = {"group": "Group", "n": "Collisions", "base_rate": "Serious/fatal rate", "flag_share": "Escalated", "precision": "Escalations correct", "recall": "Serious reached"}
    config = {name: st.column_config.NumberColumn(name, format="%.1f%%") for name in ["Serious/fatal rate", "Escalated", "Escalations correct", "Serious reached"]}

    def show(frame: pd.DataFrame) -> None:
        shown = frame.rename(columns=table)
        for name in config:
            shown[name] = shown[name] * 100
        st.dataframe(shown, hide_index=True, width="stretch", column_config=config)

    show(area)
    if len(area) == 2:
        hi, lo = area.sort_values("flag_share", ascending=False).iloc[0], area.sort_values("flag_share").iloc[0]
        ratio = hi.flag_share / max(lo.flag_share, 1e-9)
        st.markdown(theme.note_card(
            f"<strong>{hi.group} calls are escalated {ratio:.0f} times as often as {lo.group.lower()} calls</strong> "
            f"({hi.flag_share:.1%} against {lo.flag_share:.1%}). Part of this is real: {hi.group.lower()} collisions are "
            f"{hi.base_rate:.0%} serious/fatal against {lo.base_rate:.0%}. But {lo.recall:.0%} of {lo.group.lower()} serious calls are reached against "
            f"{hi.recall:.0%}, so the model is better at finding serious calls in one setting than the other. Review before relying on the flag for {lo.group.lower()} areas.",
            "warn"), unsafe_allow_html=True)
    with st.expander("By police force area"):
        show(analytics.group_fairness(replay, "police_force"))
        st.caption("Forces with fewer than 1,000 held-out collisions are left out. Police Scotland records severity on a different basis in some years, so its rate is not directly comparable.")
