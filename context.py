"""Auto-context: fill call conditions from a position and a time.

  * light: the sun's height above the horizon at that place and time (computed, no network)
  * weather and road surface: Open-Meteo hourly weather for that place and hour (free, no key; may be unavailable)
  * police force and area type: the 15 nearest recorded 2025 collisions (computed, no network)

Everything returned is a suggestion with its source. The dispatcher confirms or overrides each one. Only information
available at the time of the call is used.
"""
import datetime as dt
import math
from collections import Counter
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from common import POLICE_FORCE, REPLAY_PATH

LONDON = ZoneInfo("Europe/London")
SUN_DAYLIGHT_DEG = -0.833        # sun's upper edge at the horizon, the usual sunrise and sunset definition
HIGH_WIND_KMH = 50               # gusts at or above this are treated as "high winds" (about 31 mph)
NEIGHBOURS = 15
MAX_NEIGHBOUR_KM = 10
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,precipitation,snowfall,weather_code,wind_gusts_10m,visibility"


def sun_elevation(lat: float, lon: float, when_utc: dt.datetime) -> float:
    """Sun elevation in degrees (NOAA solar position formulas, accurate to a fraction of a degree)."""
    when_utc = when_utc.astimezone(dt.timezone.utc)
    day = when_utc.timetuple().tm_yday
    hours = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    gamma = 2 * math.pi / 365 * (day - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma) - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    true_solar_minutes = (hours * 60 + eqtime + 4 * lon) % 1440
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    cos_zenith = (math.sin(math.radians(lat)) * math.sin(decl)
                  + math.cos(math.radians(lat)) * math.cos(decl) * math.cos(hour_angle))
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def _to_utc(when_local: dt.datetime) -> dt.datetime:
    return (when_local if when_local.tzinfo else when_local.replace(tzinfo=LONDON)).astimezone(dt.timezone.utc)


def light_for(lat: float, lon: float, when_local: dt.datetime) -> dict:
    elevation = sun_elevation(lat, lon, _to_utc(when_local))
    label = "Daylight" if elevation > SUN_DAYLIGHT_DEG else "Darkness - lights lit"
    note = "Sun above the horizon." if label == "Daylight" else "Sun below the horizon. Street lighting is not known: confirm it."
    return dict(value=label, source="Computed from sun position", detail=f"Sun height {elevation:.1f} degrees. {note}")


def weather_label(code: int, gusts_kmh: float) -> str:
    """Map a WMO weather code and gust speed to the form's weather categories."""
    windy = gusts_kmh >= HIGH_WIND_KMH
    if code in (45, 48):
        return "Fog or mist"
    if code in (71, 73, 75, 77, 85, 86):
        return "Snowing, high winds" if windy else "Snowing, no high winds"
    if code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return "Raining, high winds" if windy else "Raining, no high winds"
    return "Fine, high winds" if windy else "Fine, no high winds"


def surface_code(temp_c: float, precipitation_mm: float, snowfall_cm: float) -> str | None:
    """Suggested road surface code ('3' snow, '4' frost or ice, '2' wet). None means dry or not inferable."""
    if snowfall_cm > 0:
        return "3"
    if precipitation_mm > 0:
        return "4" if temp_c <= 0 else "2"
    return None


def weather_for(lat: float, lon: float, when_local: dt.datetime, timeout: float = 6.0) -> dict | None:
    """Hourly weather at the place and hour from Open-Meteo. Returns None if the service cannot be reached."""
    when_local = when_local.replace(tzinfo=None)
    age_days = (dt.datetime.now() - when_local).days
    url = ARCHIVE_API if age_days > 80 else FORECAST_API
    day = when_local.date().isoformat()
    try:
        response = requests.get(url, timeout=timeout, params=dict(
            latitude=lat, longitude=lon, hourly=HOURLY, timezone="Europe/London", start_date=day, end_date=day))
        response.raise_for_status()
        hourly = response.json()["hourly"]
        i = when_local.hour
        code, gusts = int(hourly["weather_code"][i]), float(hourly["wind_gusts_10m"][i])
        temp, rain, snow = float(hourly["temperature_2m"][i]), float(hourly["precipitation"][i]), float(hourly["snowfall"][i])
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return None
    result = dict(weather=dict(value=weather_label(code, gusts), source="Open-Meteo hourly weather",
                               detail=f"{temp:.0f} C, rain {rain:.1f} mm, gusts {gusts:.0f} km/h"))
    if (surface := surface_code(temp, rain, snow)) is not None:
        result["road_surface_conditions"] = dict(value=surface, source="Inferred from the hour's weather",
                                                 detail="A guess from rain, snow and temperature. Confirm it.")
    return result


@lru_cache(maxsize=1)
def _neighbour_index():
    """KD-tree over recent collision positions, with their force and area labels (km on a flat approximation)."""
    from scipy.spatial import cKDTree
    frame = pd.read_csv(REPLAY_PATH, usecols=["latitude", "longitude", "police_force", "urban_or_rural_area"]).dropna()
    xy = np.column_stack([frame.longitude * 111.32 * math.cos(math.radians(54)), frame.latitude * 110.57])
    return cKDTree(xy), frame.police_force.astype(int).astype(str).to_numpy(), frame.urban_or_rural_area.astype(int).astype(str).to_numpy()


def nearby_context(lat: float, lon: float) -> dict:
    """Police force and area type from the nearest recorded collisions. Empty if there are none within 10 km."""
    try:
        tree, forces, areas = _neighbour_index()
    except (OSError, ValueError, KeyError):
        return {}
    point = [lon * 111.32 * math.cos(math.radians(54)), lat * 110.57]
    distances, index = tree.query(point, k=NEIGHBOURS)
    near = index[distances <= MAX_NEIGHBOUR_KM]
    if len(near) < 5:
        return {}
    out = {}
    force, votes = Counter(forces[near]).most_common(1)[0]
    if force in POLICE_FORCE and votes / len(near) >= 0.6:
        out["police_force"] = dict(value=force, source="Nearest recorded collisions",
                                   detail=f"{votes} of {len(near)} nearby records are in this force area.")
    area, votes = Counter(areas[near]).most_common(1)[0]
    if area in ("1", "2") and votes / len(near) >= 0.7:
        out["urban_or_rural_area"] = dict(value=area, source="Nearest recorded collisions",
                                          detail=f"{votes} of {len(near)} nearby records are {'urban' if area == '1' else 'rural'}.")
    return out


def auto_context(lat: float, lon: float, when_local: dt.datetime, use_network: bool = True) -> dict:
    """Suggested form values keyed by the form's field names, each with value, source and detail."""
    suggestions = {"light_condition": light_for(lat, lon, when_local), **nearby_context(lat, lon)}
    if use_network and (weather := weather_for(lat, lon, when_local)):
        suggestions.update(weather)
    return suggestions
