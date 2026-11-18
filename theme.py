"""Accessible visual system for the Call Severity Triage Streamlit app."""
import streamlit as st

CSS = """
:root {
  color-scheme: dark;
  --canvas: #0a0a0b;
  --surface: #17171b;
  --surface-raised: #1f1f25;
  --surface-soft: #25252c;
  --line: #3a3a42;
  --text: #e9e9ee;
  --muted: #a7a7b2;
  --subtle: #6b6b76;
  --blue: #e11d2e;
  --blue-deep: #7a0f19;
  --red: #e11d2e;
  --green: #3ecf8e;
  --amber: #f0bd5e;
  --radius: 14px;
}

.stApp { color: var(--text); background: var(--canvas); }
[data-testid="stAppViewContainer"] { background: var(--canvas); }
.block-container { max-width: 1320px; padding: 2rem clamp(1rem, 3vw, 2.6rem) 3.5rem; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
::selection { background: #7a0f19; color: white; }
h1, h2, h3, p, label { color: var(--text); }

.hero { display:flex; align-items:center; justify-content:space-between; gap:1.5rem; flex-wrap:wrap;
  padding:1.25rem 1.45rem; margin-bottom:.8rem; border:1px solid var(--line);
  border-radius:var(--radius); background:var(--surface); }
.hero-l { display:flex; align-items:center; gap:1rem; }
.logo { width:48px; height:48px; flex:0 0 48px; display:flex; align-items:center; justify-content:center;
  color:#fff; background:var(--red); border-radius:13px; }
.logo svg { width:25px; height:25px; }
.hero-title { font-size:1.28rem; line-height:1.2; font-weight:750; letter-spacing:-.02em; color:var(--text); }
.hero-sub { margin-top:.25rem; font-size:.86rem; color:var(--muted); }
.hero-r { display:flex; gap:.55rem; align-items:center; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; min-height:30px; padding:.28rem .68rem; border:1px solid var(--line);
  border-radius:999px; background:var(--surface-raised); color:var(--muted); font-size:.73rem; font-weight:650; }
.chip-blue { color:#ffd8db; border-color:#7a0f19; background:#29161a; }
.safety-note { display:flex; align-items:center; gap:.65rem; margin:.9rem 0 1.35rem; padding:.75rem 1rem;
  border:1px solid #5b4d2c; border-radius:10px; background:#211d14; color:#f5d994; font-size:.84rem; line-height:1.45; }
.safety-note strong { color:#ffe6a7; }
.safety-dot { width:9px; height:9px; flex:0 0 9px; border-radius:50%; background:var(--amber); }
.page-intro { margin:.1rem 0 1.05rem; }
.page-title { margin:0; font-size:1.35rem; line-height:1.25; font-weight:750; letter-spacing:-.025em; color:var(--text); }
.page-description { margin:.35rem 0 0; color:var(--muted); font-size:.9rem; line-height:1.5; }
.section-title { display:flex; align-items:center; gap:.55rem; margin:.2rem 0 .65rem; color:var(--text); font-size:.96rem; font-weight:700; }
.section-title svg { width:17px; height:17px; color:var(--red); flex:0 0 auto; }
.section-gap { height:1.3rem; }

/* Navigation */
[data-testid="stTabs"] [role="tablist"] { gap:.35rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { min-height:43px; padding:.45rem 1rem; border-radius:9px 9px 0 0; color:var(--muted); font-weight:650; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#ffe1e3; background:#2a171b; }
[data-testid="stTabs"] button[role="tab"]:focus-visible { outline:3px solid var(--red); outline-offset:2px; }

/* Surfaces and intake form */
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); }
[data-testid="stForm"] { padding:1.2rem 1.2rem .8rem; }
[data-testid="stVerticalBlockBorderWrapper"] { padding:.75rem .9rem .85rem; }
[data-testid="stSelectbox"] label p, [data-testid="stTextInput"] label p, [data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p { color:var(--muted)!important; font-size:.82rem; font-weight:600; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTextInput"] input { min-height:44px; border-radius:9px; background:#0d1726; border-color:#40526a; color:var(--text); }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stTextInput"] input:hover { border-color:#6e88a5; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stTextInput"] input:focus { border-color:var(--blue); box-shadow:0 0 0 2px #6db6ff3d; }
[role="listbox"] { background:#111b2c; border:1px solid var(--line); }
[role="option"] { color:var(--text); }
[role="option"]:hover, [role="option"][aria-selected="true"] { background:#203955; }

/* Actions and keyboard access */
.stButton button, [data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stDownloadButton"] button {
  min-height:44px; border-radius:9px; font-weight:700; transition:background-color .15s ease, border-color .15s ease, transform .15s ease; }
[data-testid="stBaseButton-primaryFormSubmit"] { color:#fff!important; border:1px solid #ff3b4a; background:var(--red); }
[data-testid="stBaseButton-primaryFormSubmit"]:hover { background:#ff3b4a; border-color:#fff; transform:translateY(-1px); }
.stButton button, [data-testid="stDownloadButton"] button { color:#e7edf5; border:1px solid #40526a; background:#18263a; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover { color:#fff; border-color:#7593b3; background:#21344c; }
button:focus-visible, a:focus-visible, input:focus-visible, [tabindex="0"]:focus-visible { outline:3px solid var(--red)!important; outline-offset:2px!important; }
[data-testid="stRadio"] [role="radiogroup"] { gap:.35rem; }
[data-testid="stRadio"] [role="radiogroup"] label { padding:.25rem .45rem; border-radius:7px; }
[data-testid="stRadio"] [role="radiogroup"] label:hover { background:#192a3e; }

/* Triage verdict */
@keyframes reveal { from { opacity:0; transform:translateY(7px); } to { opacity:1; transform:none; } }
@keyframes meter-fill { from { transform:scaleX(.02); } to { transform:scaleX(1); } }
@keyframes alert-pulse { 50% { box-shadow:0 0 0 5px #e11d2e20; } }
.verdict { position:relative; overflow:hidden; margin:.2rem 0 .9rem; padding:1.2rem 1.25rem;
  border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); animation:reveal .24s ease-out both; }
.verdict::before { position:absolute; inset:0 auto 0 0; width:4px; content:""; background:#8293a9; }
.verdict-high { border-color:#74404a; background:#211820; }
.verdict-high::before { background:var(--red); }
.verdict-high .verdict-big { animation:alert-pulse 2s cubic-bezier(.23,1,.32,1) infinite; }
.verdict-low { border-color:#28594e; background:#14231f; }
.verdict-low::before { background:var(--green); }
.verdict-estimate { border-color:#66562e; background:#211d14; }
.verdict-head { display:flex; align-items:center; gap:.5rem; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:.7rem; font-weight:750; }
.verdict-head svg { width:16px; height:16px; }
.verdict-big { margin:.45rem 0 .15rem; font-size:2rem; line-height:1.1; font-weight:800; letter-spacing:-.025em; }
.verdict-high .verdict-big { color:#ff9b9b; }
.verdict-low .verdict-big { color:#7be8c0; }
.verdict-estimate .verdict-big { color:var(--amber); }
.verdict-sub { margin:.35rem 0 .85rem; color:#ced8e4; font-size:.87rem; line-height:1.5; }
.meter-wrap { display:flex; align-items:center; gap:.7rem; margin:.3rem 0 .9rem; }
.meter { height:8px; flex:1; overflow:hidden; border-radius:99px; background:#27364a; }
.meter-fill { height:100%; border-radius:99px; background:var(--red); animation:meter-fill .3s cubic-bezier(.23,1,.32,1) both; transform-origin:left; }
.verdict-low .meter-fill { background:var(--green); }
.meter-val { min-width:112px; color:#d9e3ee; text-align:right; font-size:.8rem; font-weight:700; }
.meta-row { display:flex; flex-wrap:wrap; gap:.4rem; }
.meta-chip { display:inline-flex; align-items:center; gap:.35rem; padding:.32rem .55rem; border:1px solid #34465c;
  border-radius:7px; background:#111d2d; color:#bdc9d7; font-size:.72rem; font-weight:600; }
.meta-chip svg { width:13px; height:13px; color:var(--blue); }
.estimate-chip { display:inline-flex; align-items:center; gap:.4rem; padding:.4rem .65rem; border:1px solid #66562e;
  border-radius:8px; background:#282316; color:#f4d786; font-size:.76rem; font-weight:700; line-height:1.4; }
.placeholder { min-height:190px; display:flex; flex-direction:column; justify-content:center; padding:1.2rem 1.25rem;
  border:1px dashed #43546b; border-radius:var(--radius); background:#101a29; }
.placeholder-title { margin:.65rem 0 .35rem; color:#e2eaf3; font-size:1.02rem; font-weight:700; }
.placeholder-copy { margin:0; max-width:42ch; color:var(--muted); font-size:.85rem; line-height:1.55; }

/* Shift metrics and performance */
.tile { min-height:101px; display:flex; align-items:center; gap:.8rem; padding:.9rem; border:1px solid var(--line); border-radius:12px; background:var(--surface); }
.tile-icon { width:39px; height:39px; flex:0 0 39px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:#193452; color:#8dc9ff; }
.tile-icon svg { width:19px; height:19px; }
.tone-red .tile-icon { color:#ff9999; background:#3a2229; }
.tone-green .tile-icon { color:#7fe6c0; background:#18352d; }
.tone-amber .tile-icon { color:#ffdb86; background:#392f1b; }
.tile-val { color:var(--text); font-size:1.55rem; line-height:1.1; font-weight:780; }
.tile-label { margin-top:.25rem; color:var(--muted); font-size:.7rem; line-height:1.35; font-weight:650; }
.pk { padding:1.3rem 1rem; border:1px solid var(--line); border-radius:var(--radius); background:var(--surface); text-align:center; }
.pk-label { color:var(--muted); font-size:.72rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
.pk-value { margin:.35rem 0 .2rem; color:#9bd2ff; font-size:2.6rem; line-height:1.1; font-weight:800; }
.pk-k { color:var(--muted); font-size:.82rem; line-height:1.5; }
.note { margin:.3rem 0; padding:.8rem .95rem; border:1px solid #66562e; border-left:4px solid var(--amber); border-radius:9px; background:#211d14; color:#f5d994; font-size:.86rem; line-height:1.5; }

/* Data and statuses */
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:11px; overflow:hidden; }
[data-testid="stAlert"] { border-radius:9px; }
[data-testid="stCaptionContainer"] p, .stCaption { color:var(--muted); line-height:1.5; }
[data-testid="stMarkdownContainer"] a { color:#91caff; }

@media (max-width: 760px) {
  .block-container { padding:1rem .85rem 2rem; }
  .hero { padding:1rem; align-items:flex-start; }
  .hero-r { padding-left:0; }
  .hero-title { font-size:1.1rem; }
  .hero-sub { max-width:34ch; font-size:.79rem; }
  .safety-note { align-items:flex-start; font-size:.8rem; }
  [data-testid="stForm"] { padding:.95rem .85rem .65rem; }
  [data-testid="stTabs"] button[role="tab"] { padding:.4rem .7rem; font-size:.84rem; }
  .verdict-big { font-size:1.7rem; }
  .tile { min-height:88px; padding:.7rem; gap:.55rem; }
  .tile-icon { width:33px; height:33px; flex-basis:33px; }
  .tile-val { font-size:1.28rem; }
  .tile-label { font-size:.64rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior:auto!important; animation-duration:.01ms!important; animation-iteration-count:1!important; transition-duration:.01ms!important; }
}
"""

_ICONS = {
    "siren": '<path d="M7 12a5 5 0 0 1 10 0v6H7Z"/><path d="M5 20a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2"/><path d="M12 2v2"/><path d="m4.6 5.6 1.4 1.4"/><path d="m19.4 5.6-1.4 1.4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2m6-2v2M9 20v2m6-2v2M2 9h2m-2 6h2m16-6h2m-2 6h2"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "list": '<line x1="8" x2="21" y1="6" y2="6"/><line x1="8" x2="21" y1="12" y2="12"/><line x1="8" x2="21" y1="18" y2="18"/><line x1="3" x2="3.01" y1="6" y2="6"/><line x1="3" x2="3.01" y1="12" y2="12"/><line x1="3" x2="3.01" y1="18" y2="18"/>',
}


def _svg(name: str) -> str:
    return f'<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>'


def inject() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hero() -> str:
    return f"""
    <header class="hero">
      <div class="hero-l">
        <div class="logo">{_svg('siren')}</div>
        <div>
          <div class="hero-title">Call Severity Triage</div>
          <div class="hero-sub">Dispatcher decision support · Traffic incident response</div>
        </div>
      </div>
      <div class="hero-r">
        <span class="chip">STATS19 · 2023</span>
        <span class="chip chip-blue">MODEL-ASSISTED</span>
      </div>
    </header>"""


def page_intro(title: str, description: str) -> str:
    return f'<div class="page-intro"><h1 class="page-title">{title}</h1><p class="page-description">{description}</p></div>'


def section_title(text: str, icon: str = "list") -> str:
    return f'<div class="section-title">{_svg(icon)}<span>{text}</span></div>'


def verdict_card(pred: dict, ms: float, cid: str) -> str:
    estimate = bool(pred.get("is_estimate"))
    high = pred["risk_flag"] == "High risk" and not estimate
    css_class = "verdict-estimate" if estimate else "verdict-high" if high else "verdict-low"
    label = "UNSCORED" if estimate else "HIGH RISK" if high else "LOW RISK"
    if estimate:
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
        meter = (
            '<div style="margin:.35rem 0 .85rem"><span class="estimate-chip">'
            f'{_svg("alert")} Fallback estimate · model prediction unavailable</span></div>'
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


def placeholder_card() -> str:
    return f"""
    <section class="placeholder" aria-label="Awaiting call assessment">
      <div class="verdict-head">{_svg('gauge')} Ready for assessment</div>
      <div class="placeholder-title">No call assessed yet</div>
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
