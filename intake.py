"""Free-text call intake: remove personal details, then suggest form fields from the description.

Suggestions only. The dispatcher always reviews and confirms the form; nothing is submitted from here. The original
text is never stored: only the redacted version is kept for display. Redaction is pattern-based and cannot catch
every name or address, so it reduces rather than removes the privacy risk.
"""
import re

from common import CHOICES, OPTIONAL_LABELS, VALID_SPEED

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I), "[postcode]"),
    (re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)"), "[phone]"),
    (re.compile(r"\b(?:my name is|this is|i am|i'm|name's)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", re.I), "[name]"),
]


def redact(text: str) -> str:
    for pattern, token in _PATTERNS:
        text = pattern.sub(token, text)
    return text


def _first(text: str, rules):
    """First rule whose regex matches wins; returns (value, evidence) or None."""
    for regex, value in rules:
        m = re.search(regex, text, re.I)
        if m:
            return value, m.group(0)
    return None


_ROAD = [(r"\bone[- ]way\b.*\bslip\b|\bslip\b.*\bone[- ]way\b", "One way street/Slip road"), (r"\broundabout\b", "Roundabout"),
         (r"\bdual carriageway\b|\bdual[- ]carriageway\b", "Dual carriageway"), (r"\bslip road\b|\bsliproad\b", "Slip road"),
         (r"\bone[- ]way\b", "One way street"), (r"\bsingle carriageway\b|\bcountry (?:road|lane)\b|\bhigh street\b|\bresidential\b", "Single carriageway")]
_WEATHER = [(r"\bsnow\w*\b.*\b(?:high |strong )?wind|\bblizzard\b", "Snowing, high winds"), (r"\bsnow\w*\b", "Snowing, no high winds"),
            (r"\b(?:rain\w*|downpour|pouring)\b.*\b(?:high |strong )?wind\w*|\bstorm\b", "Raining, high winds"),
            (r"\b(?:rain\w*|drizzl\w*|downpour|pouring|wet weather)\b", "Raining, no high winds"), (r"\bfog\w*\b|\bmist\w*\b", "Fog or mist"),
            (r"\b(?:high|strong) winds?\b|\bgales?\b", "Fine, high winds"), (r"\b(?:clear|sunny|fine weather|dry)\b", "Fine, no high winds")]
_LIGHT = [(r"\b(?:no|without) (?:street ?)?lights?\b|\bunlit road\b|\bpitch (?:black|dark)\b", "Darkness - no lighting"),
          (r"\b(?:street ?)?lights? (?:are |is )?(?:off|out|broken)\b", "Darkness - lights unlit"),
          (r"\b(?:dark|night|dusk|evening|street ?lights?)\b", "Darkness - lights lit"), (r"\b(?:daylight|daytime|morning|afternoon|midday|bright)\b", "Daylight")]
_JUNCTION = [(r"\bcross ?roads?\b", "Crossroads"), (r"\bt[- ]junction\b|\bstaggered junction\b|\bt junction\b", "T or staggered junction"),
             (r"\b(?:private )?(?:drive|driveway|entrance)\b", "Private drive or entrance"), (r"\b(?:five|5|six|6)[- ]way\b", "More than 4 arms (not roundabout)"),
             (r"\bjunction\b|\bturning\b|\bintersection\b", "Other junction"), (r"\b(?:straight road|open road|not at a junction)\b", "Not at junction / roundabout / slip road")]
_AREA = [(r"\bvillage\b|\brural\b|\bcountry (?:road|lane)\b|\bfarm\b", "2"), (r"\btown\b|\bcity\b|\burban\b|\bhigh street\b|\bresidential\b", "1")]
_CLASS = [(r"\bm\d{1,3}\b|\bmotorway\b", "1"), (r"\ba\d{1,4}\(m\)", "2"), (r"\ba\d{1,4}\b|\ba[- ]road\b", "3"), (r"\bb\d{1,4}\b|\bb[- ]road\b", "4"), (r"\bc[- ]road\b", "5")]
_SURFACE = [(r"\bflood\w*\b", "5"), (r"\b(?:ice|icy|black ice|frost\w*)\b", "4"), (r"\bsnow\w*\b", "3"), (r"\b(?:wet|damp|slippery)\b", "2"), (r"\bdry road\b", "1")]


def extract_fields(text: str) -> dict:
    """Return {form_key: (value, evidence)} for each field the text clearly supports. Unmatched fields are omitted."""
    found = {}
    m = re.search(r"\b(\d{2})\s?mph\b|\b(?:limit|speed limit)\s?(?:is|of)?\s?(\d{2})\b", text, re.I)
    if m:
        speed = int(m.group(1) or m.group(2))
        if speed in VALID_SPEED:
            found["speed"] = (str(speed), m.group(0))
    for key, rules in [("road", _ROAD), ("weather", _WEATHER), ("light", _LIGHT), ("junction", _JUNCTION)]:
        hit = _first(text, rules)
        field = {"road": "road_type", "weather": "weather", "light": "light_condition", "junction": "junction_detail"}[key]
        if hit and hit[0] in CHOICES[field]:
            found[key] = hit
    for field, rules in [("urban_or_rural_area", _AREA), ("first_road_class", _CLASS), ("road_surface_conditions", _SURFACE)]:
        hit = _first(text, rules)
        if hit and hit[0] in OPTIONAL_LABELS[field]:
            found[f"opt_{field}"] = hit
    return found
