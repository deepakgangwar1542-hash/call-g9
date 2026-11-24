"""Accessible visual system for the Call Severity Triage Streamlit app."""
from html import escape
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).with_name("styles.css")

_ICONS = {
    "siren": '<path d="M7 12a5 5 0 0 1 10 0v6H7Z"/><path d="M5 20a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2"/><path d="M12 2v2"/><path d="m4.6 5.6 1.4 1.4"/><path d="m19.4 5.6-1.4 1.4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2m6-2v2M9 20v2m6-2v2M2 9h2m-2 6h2m16-6h2m-2 6h2"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "down": '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "list": '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>',
}


_THREE_URL = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"

_SCENE_JS = """
const kind = document.body.dataset.kind;
const canvas = document.getElementById('c');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
if (typeof THREE !== 'undefined') try {
  const W = () => window.innerWidth, H = () => window.innerHeight;
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W(), H());
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(42, W() / H(), 0.1, 100);
  const RED = 0xf04452, CORAL = 0xff6b70, GREEN = 0x32c98b, SLATE = 0x3a4757;
  const box = (w, h, d, color, wire) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshBasicMaterial({ color, wireframe: !!wire }));
  const ring = (r, opacity, width = 0.05) => {
    const m = new THREE.Mesh(new THREE.RingGeometry(r - width, r, 64), new THREE.MeshBasicMaterial({ color: RED, side: THREE.DoubleSide, transparent: true, opacity }));
    m.rotation.x = -Math.PI / 2; scene.add(m);
  };
  const pulsing = [];
  let sway = 0.3, spin = 0, speed = 0.015;
  if (kind === 'terrain') {
    cam.position.set(5.5, 5.5, 6.5); cam.lookAt(0, 0.5, 0);
    for (let x = -3; x <= 3; x++) for (let z = -3; z <= 3; z++) {
      const d = Math.hypot(x, z), h = Math.max(0.25, 2.4 - d * 0.45);
      const m = box(0.55, h, 0.55, h > 1.3 ? RED : SLATE, h <= 1.3);
      m.position.set(x * 0.75, h / 2, z * 0.75); m.userData = { amp: 0.18, phase: d * 0.7 };
      scene.add(m); pulsing.push(m);
    }
    sway = 0.35;
  } else if (kind === 'radar') {
    cam.fov = 45; cam.position.set(0, 7, 11); cam.lookAt(0, 0.5, 0);
    [1.5, 2.8, 4.1, 5.4].forEach((r, i) => ring(r, 0.85 - i * 0.15, 0.06));
    for (let i = 0; i < 18; i++) {
      const h = 0.6 + (i % 6) * 0.55, a = (i / 18) * Math.PI * 2, r = 0.8 + (i % 3) * 0.9;
      const m = box(0.28, h, 0.28, i % 2 ? CORAL : RED);
      m.position.set(Math.cos(a) * r, h / 2, Math.sin(a) * r); m.userData = { amp: 0.3, phase: i * 0.5, base: 0.75 };
      scene.add(m); pulsing.push(m);
    }
    sway = 0; spin = 0.2; speed = 0.02;
  } else if (kind === 'network') {
    cam.fov = 45; cam.position.set(0, 8, 12); cam.lookAt(0, 0, 0);
    scene.add(new THREE.GridHelper(20, 20, 0x27313e, 0x18212d));
    [1.8, 3.6, 5.4].forEach((r) => ring(r, 0.45, 0.04));
    for (let i = 0; i < 12; i++) {
      const h = 0.5 + (i % 4) * 0.5, m = box(0.3, h, 0.3, i % 3 === 0 ? RED : GREEN);
      m.position.set((i - 5.5) * 1.1, h / 2, Math.sin(i) * 3); scene.add(m);
    }
    sway = 0.3; speed = 0.012;
  } else {
    cam.position.set(5, 5, 6); cam.lookAt(0, 0.8, 0);
    for (let x = -2; x <= 2; x++) for (let z = -1; z <= 1; z++) {
      const h = 0.7 + Math.abs(x + z) * 0.45, m = box(0.65, h, 0.65, (x + z) % 2 === 0 ? RED : SLATE);
      m.position.set(x * 1.1, h / 2, z * 1.1); scene.add(m);
    }
    sway = 0.35; speed = 0.015;
  }
  let t = 0;
  const draw = () => {
    scene.rotation.y = spin ? t * spin : Math.sin(t * 0.5) * sway;
    pulsing.forEach((m) => { m.scale.y = (m.userData.base ?? 0.85) + Math.sin(t * 1.5 + m.userData.phase) * m.userData.amp; });
    renderer.render(scene, cam);
  };
  const loop = () => { t += speed; draw(); if (!reduce) requestAnimationFrame(loop); };
  addEventListener('resize', () => { renderer.setSize(W(), H()); cam.aspect = W() / H(); cam.updateProjectionMatrix(); draw(); });
  loop();
} catch (e) {}
"""


def scene_3d(kind: str, height: int = 260, label: str = "") -> None:
    """Decorative three.js scene (terrain, radar, network or bars). Illustrative only: it carries no data."""
    caption = f'<span class="cap">{escape(label)}</span>' if label else ""
    page = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><style>'
        'html,body{margin:0;height:100%;overflow:hidden;background:#090c11}'
        'canvas{display:block;width:100%;height:100%}'
        '.cap{position:absolute;left:12px;bottom:10px;color:#9aa6b5;font:600 10px/14px "JetBrains Mono",ui-monospace,monospace;'
        'letter-spacing:.08em;text-transform:uppercase}'
        f'</style></head><body data-kind="{escape(kind)}"><canvas id="c" aria-hidden="true"></canvas>{caption}'
        f'<script src="{_THREE_URL}"></script><script>{_SCENE_JS}</script></body></html>'
    )
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(page, height=height)


def _svg(name: str) -> str:
    return f'<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>'


@st.cache_data(show_spinner=False)
def _css(mtime: float) -> str:
    return CSS_PATH.read_text()


def inject() -> None:
    st.markdown(f"<style>{_css(CSS_PATH.stat().st_mtime)}</style>", unsafe_allow_html=True)


def brand() -> str:
    return (f'<div class="brand"><div class="logo">{_svg("siren")}</div>'
            f'<div><div class="brand-name">Call Severity Triage</div><div class="brand-sub">Dispatcher decision support</div></div></div>')


def side_label(text: str) -> str:
    return f'<div class="side-label">{escape(text)}</div>'


STATUS_DOT = {"Operational": "dot-ok dot-live", "Degraded": "dot-warn", "Offline": "dot-off"}


def status_card(state: str, rows: list[tuple[str, str]]) -> str:
    """System state plus a few key facts; every value comes from the caller, nothing is invented here."""
    body = "".join(f'<div class="status-row"><span>{escape(k)}</span><b>{escape(str(v))}</b></div>' for k, v in rows)
    return (f'<div class="status-card" role="status" aria-label="System status {escape(state)}">'
            f'<div class="status-row"><span class="status-pill"><span class="dot {STATUS_DOT.get(state, "")}"></span>{escape(state)}</span>'
            f'<b>System</b></div>{body}</div>')


CHEVRON = '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>'


def page_intro(title: str, description: str, group: str = "Operations") -> str:
    """Breadcrumb, title and description on the left; dataset and mode chips on the right."""
    return (
        f'<div class="topbar"><div>'
        f'<nav class="crumbs" aria-label="Breadcrumb"><span>Home</span>{CHEVRON}<span>{escape(group)}</span>{CHEVRON}<b>{escape(title)}</b></nav>'
        f'<h1 class="page-title">{title}</h1><p class="page-description">{description}</p></div>'
        f'<div class="chips"><span class="chip">STATS19 · 2021-2025</span><span class="chip chip-red">MODEL-ASSISTED</span></div></div>'
    )


def section_title(text: str, icon: str = "list") -> str:
    return f'<div class="section-title">{_svg(icon)}<span>{text}</span></div>'


def verdict_card(pred: dict, ms: float, cid: str) -> str:
    estimate = bool(pred.get("is_estimate"))
    unavailable = pred.get("assessment_state") == "unavailable"
    high = pred["risk_flag"] == "High risk" and not estimate
    css_class = "verdict-estimate" if estimate else "verdict-high" if high else "verdict-low"
    label = "ASSESSMENT UNAVAILABLE" if unavailable else "UNSCORED" if estimate else "HIGH RISK" if high else "LOW RISK"
    if unavailable:
        code = escape(str(pred.get("reason_code") or "UNKNOWN"))
        recommendation = f"{escape(pred.get('message') or '')} (reason: {code})"
    elif estimate:
        recommendation = "The model is unavailable. This baseline estimate is not a scored risk classification; follow dispatcher judgement and established protocols."
    elif high:
        recommendation = "Enhanced response recommended. Apply dispatcher judgement and local protocol."
    else:
        recommendation = "The model did not flag elevated risk. Continue to follow standard dispatch protocols."
    confidence = pred.get("confidence")
    if confidence is not None:
        pct = max(0, min(100, confidence * 100))
        meter = (
            f'<div class="meter-wrap"><div class="meter" role="img" aria-label="Model confidence {pct:.1f} percent">'
            f'<div class="meter-fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="meter-val">{pct:.1f}% confidence</div></div>'
        )
    else:
        chip = "No risk level produced" if unavailable else "Fallback estimate · model prediction unavailable"
        meter = (
            '<div style="margin:.35rem 0 .85rem"><span class="estimate-chip">'
            f'{_svg("alert")} {chip}</span></div>'
        )
    return f"""
    <section class="verdict {css_class}" aria-label="Triage result: {label}">
      <div class="verdict-head">{_svg('siren')} Triage assessment</div>
      <div class="verdict-big">{label}</div>
      <div class="verdict-sub">{recommendation}</div>
      {meter}
      <div class="meta-row">
        <span class="meta-chip">{_svg('list')} Call ID · {cid[:8]}</span>
        <span class="meta-chip">{_svg('cpu')} {pred['model_version']}</span>
        <span class="meta-chip">{_svg('clock')} {ms:.0f} ms</span>
      </div>
    </section>"""


def reasons_card(reasons: list) -> str:
    """Top factors behind the score. Empty string when the model could not explain itself (fallback tiers)."""
    if not reasons:
        return ""
    peak = max(abs(r["effect"]) for r in reasons) or 1.0
    rows = []
    for r in reasons:
        raises = r["effect"] > 0
        verb = "raises" if raises else "lowers"
        rows.append(
            f'<div class="why-row {"why-up" if raises else "why-down"}">{_svg("up" if raises else "down")}'
            f'<div class="why-name">{escape(r["title"])}<span>{escape(str(r["value"]))} {verb} the risk</span></div>'
            f'<div class="why-bar" role="img" aria-label="{verb} risk, relative strength {abs(r["effect"]) / peak * 100:.0f} percent">'
            f'<i style="width:{max(8.0, abs(r["effect"]) / peak * 100):.0f}%"></i></div></div>'
        )
    drivers = [escape(str(r["title"]).lower() + " (" + str(r["value"]) + ")") for r in reasons if r["effect"] > 0][:2]
    lead = f'<p class="why-lead">Mainly driven by {" and ".join(drivers)}.</p>' if drivers else ""
    return (
        f'<section class="why" aria-label="Why this score">'
        f'<div class="why-head">{_svg("cpu")} Why this score</div>'
        f'{lead}'
        f'<p class="why-sub">Largest factors in this call, compared with an average call. Fields left blank are not counted.</p>'
        f'{"".join(rows)}</section>'
    )


def calibration_card(cal: dict | None, base_rate: float) -> str:
    """Plain-English probability for similar historic calls, with the observed range and the average call as reference."""
    if not cal:
        return ""
    p, lo, hi = cal["calibrated"] * 100, cal["ci_low"] * 100, cal["ci_high"] * 100
    scale = 60.0
    pos = lambda v: max(0.0, min(100.0, v / scale * 100))
    return (
        f'<section class="cal" aria-label="Probability for similar historic calls">'
        f'<div class="why-head">{_svg("activity")} Similar historic calls</div>'
        f'<div class="cal-big"><b>{p:.0f}%</b><span>were serious or fatal</span></div>'
        f'<div class="cal-track" role="img" aria-label="Estimated {p:.1f} percent, observed range {lo:.1f} to {hi:.1f} percent, average call {base_rate * 100:.1f} percent">'
        f'<div class="cal-range" style="left:{pos(lo):.1f}%;width:{max(1.0, pos(hi) - pos(lo)):.1f}%"></div>'
        f'<div class="cal-base" style="left:{pos(base_rate * 100):.1f}%"></div>'
        f'<div class="cal-dot" style="left:{pos(p):.1f}%"></div></div>'
        f'<div class="cal-scale"><span>0%</span><span>30%</span><span>60%</span></div>'
        f'<p class="cal-note">Based on {cal["n_similar"]:,} out-of-time collisions with a similar score. The dark tick marks the average call ({base_rate * 100:.0f}%). '
        f'This describes the group, not this call: any single call is either serious or not.</p></section>'
    )


def placeholder_card() -> str:
    return f"""
    <section class="placeholder" aria-label="Awaiting call assessment">
      <div class="radar" aria-hidden="true"><i></i><i></i><i></i><b></b></div>
      <div class="verdict-head">{_svg('gauge')} Ready for assessment</div>
      <div class="placeholder-title" style="margin-top:.5rem">No call assessed yet</div>
      <p class="placeholder-copy">Complete the call conditions to see a risk flag. The result is logged for review and should be considered alongside dispatcher judgement and established protocols.</p>
    </section>"""


def metric_tile(value, label: str, icon: str, tone: str) -> str:
    return f"""
    <div class="tile tone-{tone}">
      <div class="tile-icon">{_svg(icon)}</div>
      <div><div class="tile-val">{value}</div><div class="tile-label">{label}</div></div>
    </div>"""


def pk_card(k: int, value: float) -> str:
    return f"""
    <div class="pk" aria-label="Precision at K {value:.1%}">
      <div class="pk-label">Precision@K</div>
      <div class="pk-value">{value:.1%}</div>
      <div class="pk-k">Among the top {k} model-ranked calls with a known outcome.<br>K = enhanced-response dispatches recorded.</div>
    </div>"""


def note_card(msg: str, tone: str = "info") -> str:
    return f'<div class="note note-{tone}" role="status">{msg}</div>'
