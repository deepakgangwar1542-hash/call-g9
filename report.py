"""One-page PDF shift report built from the call log (ReportLab)."""
import datetime as dt
import io
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import V2_META_PATH

INK, MUTED, RED, LINE = colors.HexColor("#17171b"), colors.HexColor("#5b5b66"), colors.HexColor("#c4152a"), colors.HexColor("#d4d4da")


def _model_line() -> str:
    try:
        meta = json.loads(V2_META_PATH.read_text())
        m = meta["metrics_final_test"]
        return (f"Model {meta['model_version']}. Out-of-time test on {meta['test_year']}: ROC-AUC {m['roc_auc']:.3f}, "
                f"precision in the top 2% {m['P@2pct']:.1%} against a base rate of {m['base_rate']:.1%}.")
    except Exception:
        return "Model details unavailable."


def build_pdf(calls: pd.DataFrame, precision: dict) -> bytes:
    """Return the PDF bytes. `precision` is db.shift_precision_at_k output; `calls` is db.all_calls()."""
    base = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=INK, alignment=0, spaceAfter=2)
    sub = ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=MUTED, leading=13)
    h2 = ParagraphStyle("h", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=INK, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("b", parent=base["Normal"], fontSize=9.5, textColor=INK, leading=14)

    total = len(calls)
    scored = calls[calls["is_estimate"].fillna(0) == 0] if total else calls
    high = int((scored["predicted_risk"] == "High risk").sum()) if total else 0
    sent = int(calls["enhanced_response_sent"].fillna(0).sum()) if total else 0
    overrides = int(calls["dispatcher_override"].notna().sum()) if total else 0
    known = int(calls["actual_outcome"].notna().sum()) if total else 0
    pk = f"{precision['value']:.1%} at K = {precision['k']}" if precision.get("ok") else "not available yet"

    story = [Paragraph("Call Severity Triage: shift report", title),
             Paragraph(f"Generated {dt.datetime.now():%d %b %Y %H:%M}. Decision support only: dispatcher judgement and "
                       f"established protocols always take precedence.", sub),
             Paragraph("Shift summary", h2)]
    summary = [["Assessments logged", total], ["High-risk flags", high], ["Enhanced responses sent", sent],
               ["Dispatcher overrides", overrides], ["Outcomes recorded", known], ["Precision@K", pk]]
    table = Table(summary, colWidths=[70 * mm, 60 * mm])
    table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 10),
                               ("TEXTCOLOR", (0, 0), (0, -1), MUTED), ("TEXTCOLOR", (1, 0), (1, -1), INK),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [table, Paragraph("Model", h2), Paragraph(_model_line(), body)]

    story.append(Paragraph("Most recent assessments", h2))
    if total:
        recent = calls.sort_values("timestamp", ascending=False).head(15)
        rows = [["Logged", "Result", "Road type", "Limit", "Sent", "Outcome"]]
        for _, r in recent.iterrows():
            result = "Baseline" if r.get("is_estimate") else r.get("predicted_risk") or "Unscored"
            outcome = "" if pd.isna(r["actual_outcome"]) else ("Serious/fatal" if r["actual_outcome"] == 1 else "Slight")
            rows.append([str(r["timestamp"])[:16].replace("T", " "), result, r["road_type"], f"{r['speed_limit']} mph",
                         "Yes" if r["enhanced_response_sent"] else "No", outcome])
        log = Table(rows, repeatRows=1, colWidths=[34 * mm, 24 * mm, 42 * mm, 18 * mm, 14 * mm, 28 * mm])
        log.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                                 ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                 ("BACKGROUND", (0, 0), (-1, 0), RED), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f6")]),
                                 ("GRID", (0, 0), (-1, -1), 0.25, LINE), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story.append(log)
    else:
        story.append(Paragraph("No calls were logged this shift.", body))

    story += [Spacer(1, 10), Paragraph("Data note", h2),
              Paragraph("Trained on STATS19, which records police-attended injury collisions in Great Britain. It is not a record of "
                        "emergency calls, so scores describe how collisions with similar conditions turned out historically. "
                        "Precision@K uses only calls with a recorded outcome.", body)]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title="Call Severity Triage shift report").build(story)
    return buffer.getvalue()
