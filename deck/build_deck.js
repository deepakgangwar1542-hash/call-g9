// Call Severity Triage Assistant — Sprint I review deck
// Dark control-room theme carried over from the product UI.
const pptxgen = require("pptxgenjs");

const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "Deepak Gangwar";
p.title = "Call Severity Triage Assistant — Sprint I Review";

const W = 13.33, H = 7.5, M = 0.5;
const BG = "0B1424", CARD = "14203A", CARD2 = "1A2A4A", LINE = "24335A";
const TEXT = "E8EEF8", MUTED = "9AA9C2";
const BLUE = "3B82F6", BLUE_L = "60A5FA", BLUE_XL = "93C5FD", BLUE_D = "1D4ED8";
const RED = "EF4444", RED_L = "F87171", RED_XL = "FECACA";
const GREEN = "34D399", SLATE = "3E4C68";
const F = "Segoe UI";

const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 90, opacity: 0.35 });
const bu = () => ({ code: "25B8", indent: 12 });

function base(slide) { slide.background = { color: BG }; }

function title(slide, kicker, main) {
  slide.addText(kicker, { x: M, y: 0.42, w: W - 2 * M, h: 0.3, margin: 0, fontFace: F,
    fontSize: 12, bold: true, color: BLUE_L, charSpacing: 3 });
  slide.addText(main, { x: M, y: 0.72, w: W - 2 * M, h: 0.62, margin: 0, fontFace: F,
    fontSize: 28, bold: true, color: TEXT });
}

function card(slide, x, y, w, h, fill = CARD) {
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.09,
    fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}

function srcNote(slide, text, y = 7.02) {
  slide.addText(text, { x: M, y, w: W - 2 * M, h: 0.28, margin: 0, fontFace: F,
    fontSize: 12, color: MUTED, italic: true });
}

function rings(slide, cx, cy) { // radar motif for cover / closing
  [2.9, 2.1, 1.3].forEach((r, i) => {
    slide.addShape(p.shapes.OVAL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r,
      fill: { color: BG, transparency: 100 }, line: { color: "22345C", width: 1.25 } });
  });
  slide.addShape(p.shapes.OVAL, { x: cx - 2.1 - 0.06, y: cy - 0.06, w: 0.12, h: 0.12,
    fill: { color: RED }, line: { color: RED, width: 0 } });
}

// ---------------------------------------------------------------- 1 · COVER
let s = p.addSlide(); base(s);
rings(s, 11.1, 2.0);
s.addText("GLA UNIVERSITY, GREATER NOIDA  ·  B.TECH CSE (AI)  ·  SPRINT I REVIEW",
  { x: M, y: 1.15, w: 9.5, h: 0.3, margin: 0, fontFace: F, fontSize: 12.5, bold: true, color: BLUE_L, charSpacing: 2.5 });
s.addText("Call Severity\nTriage Assistant", { x: M, y: 1.6, w: 10.5, h: 2.2, margin: 0,
  fontFace: F, fontSize: 54, bold: true, color: TEXT, lineSpacing: 62 });
s.addText("Deciding which road-collision calls deserve an enhanced response — before help arrives.",
  { x: M, y: 3.95, w: 8.6, h: 0.7, margin: 0, fontFace: F, fontSize: 17, color: MUTED });

const covStats = [
  ["104,258", "STATS19 collisions, 2023"],
  ["0.39 vs 0.31", "Precision@300 — model vs baseline"],
  ["~1 s", "measured flag latency (limit 2 s)"],
];
covStats.forEach((d, i) => {
  const x = M + i * 4.14;
  card(s, x, 5.0, 3.9, 1.15);
  s.addText(d[0], { x: x + 0.25, y: 5.14, w: 3.4, h: 0.5, margin: 0, fontFace: F, fontSize: 24, bold: true, color: i === 1 ? RED_L : BLUE_L });
  s.addText(d[1], { x: x + 0.25, y: 5.62, w: 3.4, h: 0.35, margin: 0, fontFace: F, fontSize: 12, color: MUTED });
});
s.addText("Deepak Gangwar  ·  Roll No. 22515500027  ·  Section NE  ·  Supervisor: Palash Sir  ·  September 2026",
  { x: M, y: 6.75, w: W - 2 * M, h: 0.3, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED });
s.addNotes("Sprint I review of the Call Severity Triage Assistant. The MVP is built, tested and beats the baseline. Stats on the cover are measured from the actual 2023 file and the trained model.");

// ---------------------------------------------------------------- 2 · PROBLEM
s = p.addSlide(); base(s);
title(s, "THE PROBLEM", "One scarce decision, made in seconds");
s.addText([
  { text: "When a 999 call reports a collision, the control room must decide whether to send an enhanced response — an ambulance plus more than one additional unit.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "Capacity is fixed: K enhanced dispatches per shift. Every wrong escalation takes an ambulance off the road; every missed one delays the critically injured.", options: { breakLine: true } },
  { text: "", options: { breakLine: true } },
  { text: "The dispatcher must judge from what the caller can say — road, speed, weather, light, junction — while the outcome-defining details are only known after arrival.", options: {} },
], { x: M, y: 1.7, w: 6.7, h: 4.3, margin: 0, fontFace: F, fontSize: 15.5, color: TEXT, lineSpacingMultiple: 1.15 });

const stats = [
  ["K", "enhanced dispatches available per shift — the capacity precision@K is measured at", BLUE_L],
  ["5", "conditions a dispatcher actually knows at call time", BLUE_L],
  ["24%", "of modelled 2023 collisions involved a serious or fatal injury", RED_L],
];
stats.forEach((d, i) => {
  const y = 1.62 + i * 1.72;
  card(s, 7.6, y, 5.23, 1.52);
  s.addText(d[0], { x: 7.9, y: y + 0.18, w: 1.7, h: 1.15, margin: 0, fontFace: F, fontSize: 44, bold: true, color: d[2], valign: "middle" });
  s.addText(d[1], { x: 9.65, y: y + 0.18, w: 3.0, h: 1.15, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED, valign: "middle" });
});
srcNote(s, "Source: UK DfT STATS19 2023 collision file (Open Government Licence v3.0).");
s.addNotes("Frame the decision: capacity K is fixed, information is asymmetric — the dispatcher knows less than the eventual report. That asymmetry is the modelling constraint.");

// ---------------------------------------------------------------- 3 · HOW IT WORKS
s = p.addSlide(); base(s);
title(s, "HOW IT WORKS", "From phone call to precision@K, in one pass");
const steps = [
  ["Call comes in", "999/112 collision report reaches the control room"],
  ["Dispatcher enters 5 conditions", "road type, speed limit, weather, light, junction"],
  ["Model scores the call", "trained on STATS19 2023 — threshold 0.378"],
  ["Flag + confidence", "HIGH RISK or LOW RISK, with a 0–1 confidence score"],
  ["Logged & evaluated", "SQLite log feeds precision@K for the shift"],
];
const sw = 2.28, gap = 0.25, y0 = 2.15;
steps.forEach((d, i) => {
  const x = M + i * (sw + gap);
  card(s, x, y0, sw, 2.5, i === 3 ? CARD2 : CARD);
  s.addShape(p.shapes.OVAL, { x: x + 0.22, y: y0 + 0.22, w: 0.5, h: 0.5, fill: { color: i === 3 ? RED : BLUE_D }, line: { color: i === 3 ? RED : BLUE, width: 0 } });
  s.addText(String(i + 1), { x: x + 0.22, y: y0 + 0.22, w: 0.5, h: 0.5, margin: 0, align: "center", valign: "middle", fontFace: F, fontSize: 16, bold: true, color: "FFFFFF" });
  s.addText(d[0], { x: x + 0.18, y: y0 + 0.85, w: sw - 0.36, h: 0.75, margin: 0, fontFace: F, fontSize: 13.5, bold: true, color: TEXT });
  s.addText(d[1], { x: x + 0.18, y: y0 + 1.6, w: sw - 0.36, h: 0.8, margin: 0, fontFace: F, fontSize: 12, color: MUTED });
  if (i < steps.length - 1)
    s.addShape(p.shapes.RIGHT_ARROW, { x: x + sw - 0.02, y: y0 + 1.12, w: 0.3, h: 0.26, fill: { color: SLATE }, line: { color: SLATE, width: 0 } });
});
card(s, M, 5.35, W - 2 * M, 1.05, CARD);
s.addText([
  { text: "Dispatcher-safe by design.  ", options: { bold: true, color: RED_L } },
  { text: "The model never sees number of vehicles or casualties — fields a dispatcher would not have during the call. Offline metrics therefore reflect what the tool can really achieve.", options: { color: MUTED } },
], { x: M + 0.3, y: 5.5, w: W - 2 * M - 0.6, h: 0.75, margin: 0, fontFace: F, fontSize: 13.5, valign: "middle" });
s.addNotes("Walk the five steps left to right. Emphasise step 3's threshold: 0.378 was chosen as the K-th riskiest score of the evaluation set, not hand-tuned.");

// ---------------------------------------------------------------- 4 · DATA
s = p.addSlide(); base(s);
title(s, "THE DATA", "STATS19 2023 — 104,258 recorded collisions");
card(s, M, 1.75, 5.7, 4.9, CARD);
s.addChart(p.charts.DOUGHNUT, [{
  name: "Modelled rows", labels: ["Serious / Fatal", "Slight"], values: [17700, 56030],
}], {
  x: 0.85, y: 2.0, w: 5.0, h: 3.7, holeSize: 62,
  chartColors: [RED, SLATE], chartArea: { fill: { color: CARD } },
  showPercent: true, dataLabelColor: "FFFFFF", dataLabelFontSize: 13, dataLabelFontFace: F,
  showLegend: true, legendPos: "b", legendColor: MUTED, legendFontSize: 12, legendFontFace: F,
});
s.addText("73,730", { x: 2.35, y: 3.35, w: 2.0, h: 0.5, margin: 0, align: "center", fontFace: F, fontSize: 24, bold: true, color: TEXT });
s.addText("modelled rows", { x: 2.35, y: 3.82, w: 2.0, h: 0.3, margin: 0, align: "center", fontFace: F, fontSize: 12, color: MUTED });

const dq = [
  ["Unknown is a category, not a gap", "STATS19 marks missing conditions as -1 / Unknown; these rows are kept and encoded explicitly, never dropped silently."],
  ["17 Sep 2023 speed-limit change", "DfT changed how unrestricted-road limits were recorded mid-year. Affected rows are excluded — 73,730 of 104,258 kept."],
  ["Open by licence", "Open Government Licence v3.0: free to reuse with attribution, no login or application required."],
];
dq.forEach((d, i) => {
  const y = 1.75 + i * 1.72;
  s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: y + 0.12, w: 0.52, h: 0.52, rectRadius: 0.12, fill: { color: i === 1 ? RED : BLUE_D }, line: { color: i === 1 ? RED : BLUE, width: 0 } });
  s.addText(String(i + 1), { x: 6.7, y: y + 0.12, w: 0.52, h: 0.52, margin: 0, align: "center", valign: "middle", fontFace: F, fontSize: 16, bold: true, color: "FFFFFF" });
  s.addText(d[0], { x: 7.45, y, w: 5.35, h: 0.4, margin: 0, fontFace: F, fontSize: 15, bold: true, color: TEXT });
  s.addText(d[1], { x: 7.45, y: y + 0.42, w: 5.35, h: 1.0, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED });
});
srcNote(s, "Source: DfT Road Safety Data (STATS19) collisions 2023, gov.uk; class counts computed from the downloaded file.");
s.addNotes("The 24% serious/fatal share matches the national casualty-level picture in the SRS. The September speed-limit change is why we train on one year and clean carefully.");

// ---------------------------------------------------------------- 5 · DECISIONS
s = p.addSlide(); base(s);
title(s, "APPROACH", "Three calls that shaped the model");
const dec = [
  ["D1", "Binary target", "Serious/Fatal = 1 vs Slight = 0.", "Rejected 3-class: only 1,522 fatal collisions in 2023 — too few to train and evaluate a separate class reliably inside the budget."],
  ["D2", "One year only", "2023 file, no year mixing.", "Rejected 2019–2023 pooling: DfT changed speed-limit coding on 17 Sep 2023, so multi-year data would silently mix incompatible definitions."],
  ["D3", "No leakage", "Excludes vehicles & casualty counts.", "Rejected stronger-looking features: both are unknown during a live call, so they would inflate offline metrics beyond what the tool can deliver."],
];
dec.forEach((d, i) => {
  const x = M + i * 4.14;
  card(s, x, 1.8, 3.9, 4.6);
  s.addText(d[0], { x: x + 0.28, y: 2.05, w: 1.4, h: 0.85, margin: 0, fontFace: F, fontSize: 40, bold: true, color: BLUE_L });
  s.addText(d[1], { x: x + 0.28, y: 2.95, w: 3.35, h: 0.45, margin: 0, fontFace: F, fontSize: 18, bold: true, color: TEXT });
  s.addText(d[2], { x: x + 0.28, y: 3.42, w: 3.35, h: 0.65, margin: 0, fontFace: F, fontSize: 13, bold: true, color: i === 2 ? RED_L : BLUE_L });
  s.addText(d[3], { x: x + 0.28, y: 4.15, w: 3.35, h: 2.0, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED, lineSpacingMultiple: 1.12 });
});
srcNote(s, "Design decisions D1–D3 as recorded in the SRS, Section 9.");
s.addNotes("These three decisions trade raw accuracy for honesty: the offline score is an upper bound of what the tool can really do at call time.");

// ---------------------------------------------------------------- 6 · RESULTS
s = p.addSlide(); base(s);
title(s, "RESULTS", "Precision@300 — the number the shift is judged by");
card(s, M, 1.75, 7.6, 4.95, CARD);
s.addChart(p.charts.BAR, [{
  name: "Precision@300",
  labels: ["Level 0 — always Slight", "Level 1 — speed-group rule", "Logistic Regression", "Random Forest", "HistGradientBoosting (saved)"],
  values: [0.0, 0.31, 0.353, 0.39, 0.39],
}], {
  x: 0.8, y: 2.05, w: 7.0, h: 4.35, barDir: "bar",
  varyColors: true, chartColors: [SLATE, BLUE_D, BLUE, BLUE_L, RED],
  chartArea: { fill: { color: CARD } },
  catAxisLabelColor: TEXT, catAxisLabelFontSize: 12, catAxisLabelFontFace: F,
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 11, valAxisLabelFontFace: F,
  valAxisMaxVal: 0.45, valAxisMinVal: 0, valAxisMajorUnit: 0.15,
  valGridLine: { color: "223355", size: 0.5 }, catGridLine: { style: "none" },
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontSize: 12, dataLabelFontFace: F,
  dataLabelFormatCode: "0.00",
  showLegend: false, valAxisHidden: false,
});
card(s, 8.5, 1.75, 4.33, 4.95, CARD2);
s.addText("+8 pts", { x: 8.8, y: 2.1, w: 3.7, h: 0.95, margin: 0, fontFace: F, fontSize: 48, bold: true, color: RED_L });
s.addText("above the Level 1 baseline at the shift's real capacity (0.39 vs 0.31). Level 0 can never flag a call, so its precision@K is 0.",
  { x: 8.8, y: 3.1, w: 3.75, h: 1.3, margin: 0, fontFace: F, fontSize: 13, color: TEXT });
s.addText([
  { text: "95% CI 0.33–0.44 (bootstrap, 300 runs)", options: { bullet: bu(), breakLine: true, color: MUTED } },
  { text: "Base rate 0.24 — the model concentrates escalation on the riskiest calls", options: { bullet: bu(), breakLine: true, color: MUTED } },
  { text: "ROC-AUC 0.59 · PR-AUC 0.30 on the 20% holdout", options: { bullet: bu(), color: MUTED } },
], { x: 8.8, y: 4.55, w: 3.75, h: 1.9, margin: 0, fontFace: F, fontSize: 12.5, paraSpaceAfter: 8 });
srcNote(s, "Source: reports/model_comparison.csv — Sprint I evaluation on 14,746 held-out collisions, K = 300.");
s.addNotes("Random Forest and HistGradientBoosting tie at 0.39; HGB is saved as best on tie-break by PR-AUC. The CI overlaps Level 1 at the upper end — say so honestly if asked; the release target was to beat Level 1, which the point estimate does.");

// ---------------------------------------------------------------- 7 · ARCHITECTURE
s = p.addSlide(); base(s);
title(s, "SYSTEM ARCHITECTURE", "An offline brain, an online reflex");
function archCol(x, w, head, headColor, boxes) {
  s.addText(head, { x, y: 1.72, w, h: 0.32, margin: 0, fontFace: F, fontSize: 12.5, bold: true, color: headColor, charSpacing: 2 });
  boxes.forEach((d, i) => {
    const y = 2.2 + i * 1.18;
    card(s, x, y, w, 0.92, CARD);
    s.addText(d[0], { x: x + 0.25, y: y + 0.1, w: w - 0.5, h: 0.35, margin: 0, fontFace: F, fontSize: 13.5, bold: true, color: TEXT });
    s.addText(d[1], { x: x + 0.25, y: y + 0.45, w: w - 0.5, h: 0.35, margin: 0, fontFace: F, fontSize: 11.5, color: MUTED });
    if (i < boxes.length - 1)
      s.addShape(p.shapes.RIGHT_ARROW, { x: x + w / 2 - 0.13, y: y + 0.94, w: 0.26, h: 0.22, rotate: 90, fill: { color: SLATE }, line: { color: SLATE, width: 0 } });
  });
}
archCol(M, 5.7, "OFFLINE — RUNS ONCE PER RELEASE", BLUE_L, [
  ["STATS19 2023 CSV", "104,258 collisions from gov.uk"],
  ["Clean & encode", "5 condition features + binary target"],
  ["Train & evaluate", "Level 0/1 baselines, 3 models, precision@K"],
  ["best_model.joblib", "HistGradientBoosting · threshold 0.378"],
]);
archCol(7.13, 5.7, "ONLINE — RUNS ON EVERY CALL", RED_L, [
  ["Dispatcher form (Streamlit)", "five conditions, validated by name"],
  ["Predict & log", "flag + confidence in ~1 s, UUID per call"],
  ["SQLite evaluation log", "inputs, prediction, model version kept"],
  ["Supervisor report", "precision@K + CSV export for Power BI"],
]);
s.addShape(p.shapes.LINE, { x: 6.25, y: 6.14, w: 0.83, h: 0, line: { color: RED_L, width: 2.25, endArrowType: "triangle" } });
srcNote(s, "The trained artifact (best_model.joblib) loads once into the prediction service, then scores every call in milliseconds.");
s.addNotes("Left column runs per release; right column per call. The split matters for NFR-1: no network call to DfT at run time — the artifact is local.");

// ---------------------------------------------------------------- 8 · DISPATCHER UI
s = p.addSlide(); base(s);
title(s, "PRODUCT — DISPATCHER", "Enter the call, get the flag in about a second");
const imgW = 6.35, imgH = imgW / 1.6; // 1280x800 screenshots
card(s, 6.45, 1.62, imgW + 0.36, imgH + 0.36, CARD2);
s.addImage({ path: "assets/ui_verdict.png", x: 6.63, y: 1.8, w: imgW, h: imgH });
const disp = [
  ["Five validated inputs", "Dropdowns constrained to STATS19 categories; a missing field is rejected by name and nothing is stored."],
  ["Flag + confidence + provenance", "HIGH RISK / LOW RISK with a 0–1 confidence score, call UUID, model version and latency on every verdict."],
  ["Level 0 fallback (FR-2.4)", "If the model is unavailable, the majority-class flag is shown labelled as an estimate — never an empty screen."],
  ["Dispatcher judgement kept (FR-2.5)", "Override radio and enhanced-response checkbox are recorded alongside the model's original flag."],
];
disp.forEach((d, i) => {
  const y = 1.72 + i * 1.28;
  s.addText(d[0], { x: M, y, w: 5.6, h: 0.35, margin: 0, fontFace: F, fontSize: 14.5, bold: true, color: i === 2 ? RED_L : TEXT });
  s.addText(d[1], { x: M, y: y + 0.36, w: 5.6, h: 0.8, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED });
});
s.addNotes("This is the real UI, built in Streamlit. The verdict card is where the dispatcher lives: flag, confidence meter, provenance chips. Total submit-to-flag measured at about one second.");

// ---------------------------------------------------------------- 9 · SUPERVISOR UI
s = p.addSlide(); base(s);
title(s, "PRODUCT — SUPERVISOR", "Was the shift's escalation spend justified?");
card(s, M, 1.62, imgW + 0.36, imgH + 0.36, CARD2);
s.addImage({ path: "assets/ui_supervisor.png", x: M + 0.18, y: 1.8, w: imgW, h: imgH });
const sup = [
  ["Shift KPI tiles", "Calls logged, High-risk flags and enhanced responses actually sent — the three numbers oversight needs."],
  ["Precision@K, live", "Of the top-K calls ranked by the model, the share that were truly Serious/Fatal — K set by the dispatches made."],
  ["Zero-dispatch guard", "A shift with no dispatches returns a clear message, never a divide-by-zero (FR-3.3 / AC-4)."],
  ["CSV export", "One click hands the shift log to the Power BI / Tableau report (FR-3.4)."],
];
sup.forEach((d, i) => {
  const y = 1.72 + i * 1.28;
  s.addText(d[0], { x: 7.35, y, w: 5.4, h: 0.35, margin: 0, fontFace: F, fontSize: 14.5, bold: true, color: TEXT });
  s.addText(d[1], { x: 7.35, y: y + 0.36, w: 5.4, h: 0.8, margin: 0, fontFace: F, fontSize: 12.5, color: MUTED });
});
s.addNotes("Precision@3 = 66.7% here is a six-call demo batch, not the model metric — point that out before anyone asks. The real number lives on the results slide.");

// ---------------------------------------------------------------- 10 · ACCEPTANCE
s = p.addSlide(); base(s);
title(s, "QUALITY GATES", "Sprint I (27 Sep): every acceptance criterion passes");
const gates = [
  ["AC-1", "Valid five-condition entry returns a High/Low flag with confidence, within 2 s", "PASS — 1.0 s measured"],
  ["AC-2", "Blank weather is rejected naming the field; no call record stored", "PASS"],
  ["AC-3", "With the model unavailable, a Level 0 estimate flag is shown, labelled as such", "PASS"],
  ["AC-4", "Zero-dispatch shift returns a message, not a divide-by-zero", "PASS"],
  ["NFR-3", "Every flagged call logs inputs, confidence and model version against a UUID", "PASS"],
  ["Tests", "Automated suite covering validation, fallback, guard, metric and logging", "PASS — 7/7"],
];
const rows = [[
  { text: "Gate", options: { bold: true, color: "FFFFFF", fill: { color: BLUE_D }, fontFace: F, fontSize: 12.5 } },
  { text: "What is checked", options: { bold: true, color: "FFFFFF", fill: { color: BLUE_D }, fontFace: F, fontSize: 12.5 } },
  { text: "Result", options: { bold: true, color: "FFFFFF", fill: { color: BLUE_D }, fontFace: F, fontSize: 12.5 } },
]];
gates.forEach((g, i) => rows.push([
  { text: g[0], options: { bold: true, color: BLUE_XL, fill: { color: i % 2 ? BG : CARD }, fontFace: F, fontSize: 12.5 } },
  { text: g[1], options: { color: TEXT, fill: { color: i % 2 ? BG : CARD }, fontFace: F, fontSize: 12.5 } },
  { text: g[2], options: { bold: true, color: GREEN, fill: { color: i % 2 ? BG : CARD }, fontFace: F, fontSize: 12.5 } },
]));
s.addTable(rows, { x: M, y: 1.8, w: W - 2 * M, colW: [1.35, 8.0, 2.98], border: { pt: 1, color: "1B2A4E" }, rowH: 0.62, valign: "middle", margin: 0.08 });
srcNote(s, "Evidence: test_mvp.py (7/7 passing) and browser walkthrough of the running app on 04 Oct 2026.");
s.addNotes("Each row maps to an acceptance criterion in SRS Section 10. The measured latency includes logging; the 2-second NFR is at p95.");

// ---------------------------------------------------------------- 11 · ROADMAP
s = p.addSlide(); base(s);
title(s, "BUILD PLAN", "Where Sprint I sits in the release");
const miles = [
  ["27 SEP", "Sprint I — done", "MVP live: entry, validation, flag + fallback, precision@K", GREEN],
  ["17 OCT", "Sprint II", "Stronger validation, per-call IDs, confidence display, supervisor summary", BLUE_L],
  ["7 NOV", "Sprint III", "Last-20 view, override, login + 15-min lockout, guarded P@K, export", BLUE_L],
  ["17 NOV", "Testing", "Full test report and bug log covering every FR/NFR ID", BLUE_L],
  ["30 NOV", "Deployment", "Live Streamlit URL, Power BI report, user + technical docs", RED_L],
];
const ty = 3.6;
s.addShape(p.shapes.LINE, { x: 1.55, y: ty, w: 10.23, h: 0, line: { color: LINE, width: 2 } });
miles.forEach((d, i) => {
  const cx = 1.55 + i * (10.23 / 4);
  s.addShape(p.shapes.OVAL, { x: cx - 0.11, y: ty - 0.11, w: 0.22, h: 0.22, fill: { color: d[3] }, line: { color: BG, width: 2 } });
  s.addText(d[0], { x: cx - 1.0, y: ty - 0.75, w: 2.0, h: 0.35, margin: 0, align: "center", fontFace: F, fontSize: 14, bold: true, color: d[3] });
  card(s, cx - 0.975, ty + 0.45, 1.95, 1.9, CARD);
  s.addText(d[1], { x: cx - 0.795, y: ty + 0.62, w: 1.59, h: 0.55, margin: 0, fontFace: F, fontSize: 13, bold: true, color: TEXT });
  s.addText(d[2], { x: cx - 0.795, y: ty + 1.14, w: 1.59, h: 1.1, margin: 0, fontFace: F, fontSize: 11.5, color: MUTED });
});
s.addNotes("Sprint I closed on 27 Sep with the MVP gates green. The next hard checkpoint is Testing on 17 Nov; deployment targets a reviewer opening a fresh URL with no local setup.");

// ---------------------------------------------------------------- 12 · CLOSING
s = p.addSlide(); base(s);
rings(s, 2.75, 5.9);
s.addText("WHAT TO REMEMBER", { x: M, y: 0.9, w: 8, h: 0.3, margin: 0, fontFace: F, fontSize: 12.5, bold: true, color: BLUE_L, charSpacing: 3 });
s.addText("A working triage assistant,\nmeasured honestly.", { x: M, y: 1.3, w: 10.5, h: 1.7, margin: 0, fontFace: F, fontSize: 40, bold: true, color: TEXT, lineSpacing: 50 });
const takeaways = [
  ["0.39 vs 0.31", "Precision@300 beats the Level 1 baseline at the shift's real capacity — the release target set in the SRS."],
  ["No crystal ball", "Trained only on conditions a dispatcher can know during the call, so the offline score is achievable in live use."],
  ["Running today", "Enter five conditions, get a flagged, logged, auditable verdict in about one second — with the guard rails the SRS demands."],
];
takeaways.forEach((d, i) => {
  const y = 3.35 + i * 1.15;
  s.addText(d[0], { x: M, y, w: 2.9, h: 0.9, margin: 0, fontFace: F, fontSize: 21, bold: true, color: i === 0 ? RED_L : BLUE_L, valign: "middle" });
  s.addText(d[1], { x: 3.6, y, w: 9.2, h: 0.9, margin: 0, fontFace: F, fontSize: 13.5, color: TEXT, valign: "middle" });
  if (i < 2) s.addShape(p.shapes.LINE, { x: M, y: y + 1.05, w: W - 2 * M, h: 0, line: { color: "1B2A4E", width: 1 } });
});
s.addText("Thank you — questions welcome.   ·   Deepak Gangwar 22515500027, Section NE   ·   Supervisor: Palash Sir, GLA University",
  { x: M, y: 6.9, w: W - 2 * M, h: 0.3, margin: 0, fontFace: F, fontSize: 12, color: MUTED });
s.addNotes("Close on the three takeaways. Likely questions: CI overlap with Level 1 (answered on results slide), why Streamlit Community Cloud (SRS assumption, deployment sprint pending).");

p.writeFile({ fileName: "Call_Severity_Triage_Presentation.pptx" }).then(() => console.log("WROTE deck"));
