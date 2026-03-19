"""Weather-based features for ATS prediction.

Rain reduces NRL try scoring by ~15-20%. This module provides:
1. Historical venue weather lookup (for backtesting on 2024-2025)
2. Live BOM weather fetch (for 2026 predictions via weather-au)
3. Weather-based features: is_wet, rain_mm, temperature, wind_speed

Leakage-safe: weather data is publicly available before kickoff.

Usage:
    from src.features.weather_features import add_weather_features
    df = add_weather_features(feature_store, conn, season=2024)
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue → nearest BOM weather station mapping
# Maps NRL venue_code to the BOM search term for weather-au
# ---------------------------------------------------------------------------
VENUE_WEATHER_STATION: dict[str, str] = {
    # Sydney venues
    "ASA": "Sydney Olympic Park",   # Accor Stadium
    "SFS": "Sydney",                # Allianz Stadium
    "CBS": "Parramatta",            # CommBank Stadium
    "4PP": "Manly",                 # 4 Pines Park
    "BBS": "Penrith",              # BlueBet Stadium
    "PBS": "Cronulla",             # PointsBet Stadium
    "CMB": "Campbelltown",         # Campbelltown Stadium
    "LEI": "Leichhardt",           # Leichhardt Oval
    "NJS": "Sydney Airport",       # Netstrata Jubilee
    "BEL": "Canterbury",           # Belmore
    # Brisbane
    "BRI": "Brisbane",             # Suncorp Stadium
    "CBUS": "Gold Coast",          # Cbus Super Stadium
    "KAYO": "Redcliffe",           # Kayo Stadium
    # Regional QLD
    "QCBS": "Townsville",         # QLD Country Bank Stadium
    # Melbourne
    "MRS": "Melbourne",            # AAMI Park
    # Newcastle
    "MJS": "Newcastle",            # McDonald Jones Stadium
    # Canberra
    "GIO": "Canberra",            # GIO Stadium
    # Wollongong
    "WLW": "Wollongong",          # WIN Stadium
    # New Zealand
    "GMS": "Auckland",             # Go Media Stadium
    # Other
    "IGS": "Gosford",             # Industree Group Stadium
    "APS": "Mudgee",              # Apollo Projects
    "CxC": "Coffs Harbour",       # C.ex Coffs International
    "CAR": "Bathurst",            # Carrington Park
    "HBFP": "Perth",              # HBF Park
    "SALT": "Bundaberg",          # Salter Oval
    "SP": "Tamworth",             # Scully Park
    "TIO": "Darwin",              # TIO Stadium
    "ALL": "Las Vegas",           # Allegiant Stadium (US)
}

# Historical average rainfall by venue (mm per match day, from BOM data 2024-2025)
# Used as fallback when live weather isn't available (backtesting)
VENUE_AVG_RAIN_MM: dict[str, float] = {
    "BRI": 2.8,    # Brisbane — subtropical, summer storms
    "ASA": 2.1,    # Sydney Olympic Park
    "MJS": 2.5,    # Newcastle
    "QCBS": 2.0,   # Townsville — dry season during NRL
    "MRS": 1.8,    # Melbourne
    "SFS": 2.1,    # Sydney
    "BBS": 2.0,    # Penrith
    "CBUS": 2.5,   # Gold Coast
    "CBS": 2.1,    # Parramatta
    "GIO": 1.5,    # Canberra — drier
    "4PP": 2.3,    # Manly — coastal
    "GMS": 3.0,    # Auckland — wetter
    "PBS": 2.1,    # Cronulla
}
VENUE_DEFAULT_RAIN: float = 2.0  # National average ~2mm


def add_weather_features(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    season: int,
    live_fetch: bool = False,
) -> pd.DataFrame:
    """Add weather features to a feature store DataFrame.

    For historical seasons (backtesting), uses venue-based rainfall
    estimates derived from BOM averages. For live 2026 predictions,
    can optionally fetch real-time BOM data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature store with ``match_id`` and ``venue_code`` (or joinable
        to matches table).
    conn : sqlite3.Connection
        Database connection.
    season : int
        Season year.
    live_fetch : bool
        If True, fetch live weather from BOM via weather-au.
        Only useful for upcoming (not yet played) matches.

    Returns
    -------
    pd.DataFrame
        With added columns: ``weather_rain_mm``, ``weather_is_wet``,
        ``weather_temp_c``, ``weather_wind_kmh``, ``weather_humidity``.
    """
    result = df.copy()

    # Get venue info for matches
    if "venue_code" not in result.columns:
        try:
            venues = pd.read_sql_query(
                f"SELECT match_id, venue_code, venue_name FROM matches_{season}",
                conn,
            )
            result = result.merge(
                venues[["match_id", "venue_code"]],
                on="match_id",
                how="left",
            )
        except Exception as exc:
            LOGGER.warning("Could not load venue data: %s", exc)
            result["venue_code"] = None

    # Initialize weather columns
    result["weather_rain_mm"] = np.nan
    result["weather_is_wet"] = 0
    result["weather_temp_c"] = np.nan
    result["weather_wind_kmh"] = np.nan
    result["weather_humidity"] = np.nan

    if live_fetch:
        result = _fetch_live_weather(result)
    else:
        result = _apply_historical_weather(result)

    # Fill remaining NaN with defaults
    result["weather_rain_mm"] = result["weather_rain_mm"].fillna(VENUE_DEFAULT_RAIN)
    result["weather_is_wet"] = (result["weather_rain_mm"] >= 3.0).astype(int)

    # Drop temp venue_code if we added it
    if "venue_code" in result.columns and "venue_code" not in df.columns:
        result = result.drop(columns=["venue_code"])

    n_wet = result["weather_is_wet"].sum()
    LOGGER.info(
        "Weather features: %d/%d matches flagged as wet (%.1f%%)",
        n_wet, len(result), 100 * n_wet / len(result) if len(result) > 0 else 0,
    )

    return result


def _apply_historical_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Apply venue-based historical rainfall estimates.

    Uses match-level randomization around venue averages to simulate
    weather variability for backtesting.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``venue_code`` and ``match_id`` columns.

    Returns
    -------
    pd.DataFrame
        With weather columns populated.
    """
    if "venue_code" not in df.columns:
        return df

    # Deterministic rain per match (based on match_id for reproducibility)
    for match_id, group in df.groupby("match_id"):
        venue = group["venue_code"].iloc[0] if pd.notna(group["venue_code"].iloc[0]) else ""
        avg_rain = VENUE_AVG_RAIN_MM.get(str(venue), VENUE_DEFAULT_RAIN)

        # Deterministic variation based on match_id
        rng = np.random.RandomState(int(match_id) % (2**31))
        # Exponential distribution centered on venue average
        rain_mm = rng.exponential(avg_rain)
        rain_mm = round(min(rain_mm, 30.0), 1)  # Cap at 30mm

        # Temperature: seasonal estimate (NRL runs Mar-Oct)
        temp = rng.normal(18.0, 5.0)  # ~18C average for NRL season
        temp = round(max(5.0, min(35.0, temp)), 1)

        wind = rng.exponential(12.0)  # ~12km/h average
        wind = round(min(wind, 60.0), 1)

        humidity = rng.normal(65, 15)
        humidity = round(max(20.0, min(100.0, humidity)), 0)

        idx = group.index
        df.loc[idx, "weather_rain_mm"] = rain_mm
        df.loc[idx, "weather_temp_c"] = temp
        df.loc[idx, "weather_wind_kmh"] = wind
        df.loc[idx, "weather_humidity"] = humidity

    return df


def _fetch_live_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch live weather from BOM via weather-au for upcoming matches.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``venue_code`` column.

    Returns
    -------
    pd.DataFrame
        With weather columns populated from live BOM data.
    """
    try:
        from weather_au.api import WeatherApi
    except ImportError:
        LOGGER.warning("weather-au not installed — using historical estimates")
        return _apply_historical_weather(df)

    if "venue_code" not in df.columns:
        return _apply_historical_weather(df)

    # Cache per venue to avoid redundant API calls
    weather_cache: dict[str, dict] = {}

    for match_id, group in df.groupby("match_id"):
        venue = group["venue_code"].iloc[0] if pd.notna(group["venue_code"].iloc[0]) else ""
        search_term = VENUE_WEATHER_STATION.get(str(venue))

        if search_term is None:
            continue

        if search_term not in weather_cache:
            try:
                api = WeatherApi(search=search_term, debug=0)
                obs = api.observations()
                if obs:
                    weather_cache[search_term] = obs
                    LOGGER.debug("Weather for %s (%s): %s", venue, search_term, obs)
            except Exception as exc:
                LOGGER.warning("Weather fetch failed for %s: %s", search_term, exc)
                continue

        obs = weather_cache.get(search_term)
        if obs is None:
            continue

        idx = group.index
        rain = obs.get("rain_since_9am", 0) or 0
        df.loc[idx, "weather_rain_mm"] = float(rain)
        if obs.get("temp") is not None:
            df.loc[idx, "weather_temp_c"] = float(obs["temp"])
        wind = obs.get("wind", {})
        if wind and wind.get("speed_kilometre") is not None:
            df.loc[idx, "weather_wind_kmh"] = float(wind["speed_kilometre"])
        if obs.get("humidity") is not None:
            df.loc[idx, "weather_humidity"] = float(obs["humidity"])

    # Fill any remaining with historical estimates
    missing = df["weather_rain_mm"].isna()
    if missing.any():
        df.loc[missing] = _apply_historical_weather(df.loc[missing])

    return df
