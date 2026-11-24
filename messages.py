"""Parse machine-sent incident messages (handset location SMS and decoded eCall data) into form-ready facts.

Two inputs are understood:
  * AML (Advanced Mobile Location): the SMS a phone sends to the emergency number, a semicolon-separated list of
    key=value pairs. Keys used: lt (latitude), lg (longitude), rd (radius in metres), top (time of positioning, UTC,
    YYYYMMDDHHMMSS), lc (confidence, percent), pm (positioning method).
  * eCall: a vehicle crash alert. The real minimum data set is binary; this accepts its decoded fields as text
    (activation, vehicle type, latitude, longitude, timestamp, passengers, direction). The vehicle identification
    number is dropped on purpose and never stored.

These messages carry location and time, not road conditions, so they can fill the call time and the location
lookups but cannot be model features on their own. The parser is tolerant of key spelling and spacing, reports what
it could not read instead of guessing, and checks the position falls inside Great Britain (the data's coverage).
"""
import datetime as dt
import re
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
GB_BOUNDS = (49.8, 60.9, -8.7, 1.9)   # lat_min, lat_max, lon_min, lon_max
_PAIR = re.compile(r"([A-Za-z_]+)\s*[=:]\s*([^;,\n]+)")
_VEHICLES = {"passenger": "Car", "car": "Car", "bus": "Bus or coach", "coach": "Bus or coach",
             "light": "Light goods vehicle", "lgv": "Light goods vehicle", "van": "Light goods vehicle",
             "heavy": "Heavy goods vehicle", "hgv": "Heavy goods vehicle", "motorcycle": "Motorcycle",
             "motorbike": "Motorcycle", "moped": "Motorcycle"}
_METHODS = {"G": "GNSS (satellite)", "W": "Wi-Fi", "C": "Cell tower", "H": "Hybrid", "M": "Mixed"}


def _pairs(text: str) -> dict:
    return {key.lower(): value.strip() for key, value in _PAIR.findall(text)}


def _number(value: str | None) -> float | None:
    try:
        return float(str(value).replace("+", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_time(value: str | None) -> dt.datetime | None:
    """Return a timezone-aware UTC time from YYYYMMDDHHMMSS, ISO text or epoch seconds."""
    if not value:
        return None
    value = value.strip()
    try:
        if re.fullmatch(r"\d{14}", value):
            return dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
        if re.fullmatch(r"\d{10}", value):
            return dt.datetime.fromtimestamp(int(value), dt.timezone.utc)
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _position_issue(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return "No usable latitude and longitude were found."
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "Latitude or longitude is out of range."
    lat_min, lat_max, lon_min, lon_max = GB_BOUNDS
    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
        return "The position is outside Great Britain, where this model's data applies."
    return None


def parse_message(text: str) -> dict:
    """Detect and parse an AML or eCall message.

    Returns kind ('aml', 'ecall' or 'unknown'), lat, lon, radius_m, confidence_pct, time_utc, call_time (UK local,
    naive, ready for the form), details (readable extras) and problems (a list of things that could not be used)."""
    text = (text or "").strip()
    pairs = _pairs(text)
    problems: list[str] = []
    if not text:
        return dict(kind="unknown", lat=None, lon=None, radius_m=None, confidence_pct=None, time_utc=None,
                    call_time=None, details={}, problems=["Paste a message first."])

    is_aml = {"lt", "lg"} <= pairs.keys() or "aml" in text.lower()[:12]
    kind = "aml" if is_aml else ("ecall" if {"lat", "lon"} <= pairs.keys() or "activation" in pairs else "unknown")
    lat = _number(pairs.get("lt") or pairs.get("lat") or pairs.get("latitude"))
    lon = _number(pairs.get("lg") or pairs.get("lon") or pairs.get("lng") or pairs.get("longitude"))
    radius = _number(pairs.get("rd") or pairs.get("radius"))
    confidence = _number(pairs.get("lc") or pairs.get("confidence"))
    when = _parse_time(pairs.get("top") or pairs.get("timestamp") or pairs.get("time"))
    details: dict = {}

    if kind == "unknown":
        problems.append("This does not look like an AML location message or eCall data.")
    if kind == "aml" and (pm := pairs.get("pm", "").upper()[:1]):
        details["Positioning method"] = _METHODS.get(pm, pm)
    if kind == "ecall":
        activation = pairs.get("activation", "").lower()
        if activation:
            details["Activation"] = "Automatic (crash sensors)" if activation.startswith("auto") else "Manual (button)"
        vehicle = pairs.get("vehicle") or pairs.get("vehicle_type") or pairs.get("vehicletype") or ""
        for needle, label in _VEHICLES.items():
            if needle in vehicle.lower():
                details["Vehicle"] = label
                break
        if (passengers := _number(pairs.get("passengers"))) is not None:
            details["Passengers reported"] = int(passengers)
        if (heading := _number(pairs.get("direction") or pairs.get("heading"))) is not None:
            details["Heading"] = f"{heading:.0f} degrees"
        if "vin" in pairs:
            details["Vehicle identification number"] = "removed, never stored"

    if (issue := _position_issue(lat, lon)) and kind != "unknown":
        problems.append(issue)
        lat = lon = None if "outside" in issue or "out of range" in issue else lat
    if pairs.get("top") or pairs.get("timestamp") or pairs.get("time"):
        if when is None:
            problems.append("The time in the message could not be read.")
    if radius is not None and radius > 5000:
        problems.append("The position is very imprecise (radius over 5 km). Confirm the location by voice.")

    call_time = when.astimezone(LONDON).replace(tzinfo=None, second=0, microsecond=0) if when else None
    return dict(kind=kind, lat=lat, lon=lon, radius_m=radius, confidence_pct=confidence, time_utc=when,
                call_time=call_time, details=details, problems=problems)
