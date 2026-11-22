"""Build-time constants of the WP1 data pipeline (ARCHITECTURE §0 and §2).

Simulation constants live in ``sim/constants.py``. The few values the offline build shares
with the simulation (``CROSSOVER_S``, ``TURNBACK_S``, ``LAYOVER_S``, ``ACCEL_DECEL_LOSS_S``,
``V_MAX_CAP``, the profile windows) are mirrored here so that ``scripts`` never imports the
``sim`` package; ``tests/test_network_artifacts.py`` asserts the mirrors stay equal.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = 1
INFRASTRUCTURE_SCHEMA_VERSION: Final = 1
MANIFEST_SCHEMA_VERSION: Final = 1

# --------------------------------------------------------------------------- HTTP (§2.1)
USER_AGENT: Final = "subway-domino-build/1.0"
HTTP_TIMEOUT_S: Final = 120.0
HTTP_RETRIES: Final = 3
HTTP_BACKOFF_S: Final = 2.0
# The ridership $group aggregation takes 20-95 s (92 s on 2026-11-19), so it gets a longer
# per-attempt timeout than the 120 s default.
RIDERSHIP_TIMEOUT_S: Final = 300.0

GTFS_METADATA_URL: Final = "https://data.ny.gov/api/views/fgm6-ccue.json"
GTFS_FALLBACK_URL: Final = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
GTFS_URL_MARKER: Final = "nyct/subway"
GTFS_REQUIRED_MEMBERS: Final = ("stops.txt", "stop_times.txt", "trips.txt", "routes.txt", "shapes.txt")
COMPLEXES_URL: Final = "https://data.ny.gov/resource/5f5g-n3cz.json?$limit=5000"
# Optional enrichment: per-station structure (Subway / Elevated / ...). Multi-station
# complexes report a mixed structure ("Elevated; Subway"), so the complex row alone cannot
# say which station is which. When unavailable the build falls back to the complex field.
STATIONS_URL: Final = "https://data.ny.gov/resource/39hk-dx4f.json?$limit=5000"
RIDERSHIP_URL: Final = "https://data.ny.gov/resource/5wq4-mkjj.json"
RIDERSHIP_LOOKBACK_DAYS: Final = 56
BASEMAP_URL: Final = "https://data.cityofnewyork.us/api/geospatial/gthc-hcne?method=export&format=GeoJSON"

RAW_GTFS_ZIP: Final = "gtfs_subway.zip"
RAW_COMPLEXES: Final = "complexes_5f5g-n3cz.json"
RAW_STATIONS: Final = "stations_39hk-dx4f.json"
RAW_RIDERSHIP: Final = "ridership_5wq4-mkjj.json"
RAW_BASEMAP: Final = "boroughs_gthc-hcne.geojson"

NETWORK_FILE: Final = "network.json"
GEO_FILE: Final = "network_geo.json"
MANIFEST_FILE: Final = "build_manifest.json"
INFRASTRUCTURE_FILE: Final = "infrastructure.json"

# --------------------------------------------------------------------------- scope (§0, §2.2)
A_DIVISION_ROUTES: Final = ("1", "2", "3", "4", "5", "6", "6X", "7", "7X", "GS")
B_DIVISION_ROUTES: Final = (
    "A", "B", "C", "D", "E", "F", "FX", "G", "J", "L", "M", "N", "Q", "R", "W", "Z", "FS", "H",
)
IN_SCOPE_ROUTES: Final = A_DIVISION_ROUTES + B_DIVISION_ROUTES
SHUTTLE_ROUTES: Final = ("GS", "FS", "H")
WEEKDAY_SERVICE_ID: Final = "Weekday"
DAY_S: Final = 86_400
EXPECTED_ROUTES: Final = 28
EXPECTED_STATIONS: Final = 475
EXPECTED_COMPLEXES: Final = 424
BOROUGHS_IN_SCOPE: Final = ("M", "Bk", "Q", "Bx")

# --------------------------------------------------------------------------- projection (§2.6)
LAT0: Final = 40.7128
LON0: Final = -73.95
KY: Final = 111132.954
KX: Final = 84384.0  # 111319.488 * cos(40.7128 deg), rounded to 0.1 m
# Midtown reference used to orient 3-track groups "toward Manhattan" (Times Sq-42 St).
MANHATTAN_REF_LATLON: Final = (40.7559, -73.9871)

# --------------------------------------------------------------------------- profiles (§2.3.13)
PROFILES: Final = ("am_peak", "midday", "pm_rush", "late_night")
PROFILE_WINDOWS_H: Final = {"am_peak": (7, 10), "midday": (10, 16), "pm_rush": (16, 19), "late_night": (0, 5)}
# Destination attraction uses the mirror hour: people who arrive in the AM leave in the PM.
ATTRACTION_SOURCE_PROFILE: Final = {
    "am_peak": "pm_rush", "pm_rush": "am_peak", "midday": "midday", "late_night": "late_night",
}
# Middle tracks of 3-track lines run toward the peak direction of the morning except in the
# PM rush. Midday and late night keep the morning setup (one direction per profile, §3.3).
PEAK_PROFILE_SENSE: Final = {"am_peak": "am", "midday": "am", "pm_rush": "pm", "late_night": "am"}

# --------------------------------------------------------------------------- topology inference (§2.3)
MIN_PATTERN_TRIPS: Final = 3
MIN_OVERTAKES: Final = 3
# A bypass chain is rejected when its station-to-station length exceeds the direct
# hop by more than this factor (guards against chains taken from a detouring pattern).
MAX_CHAIN_DETOUR: Final = 1.8
# Stopping hops between two X runs of one pattern that are filled with X (same trunk).
MAX_GAP_FILL_HOPS: Final = 2
# Shared stopping hops an X run may be extended over up to a divergence station.
MAX_TAIL_HOPS: Final = 2
# Gap fills and tails never extend X over a hop longer than this (bridges and river tunnels
# are 2-track unless the timetable proves otherwise).
MAX_EXTENSION_HOP_M: Final = 1500.0
# Two patterns run "alongside" when their hourly departures overlap by at least this share
# of the smaller pattern's trips.
MIN_CONCURRENT_FRACTION: Final = 0.25

# --------------------------------------------------------------------------- timing (§2.3, mirrors of §3.2)
CROSSOVER_S: Final = 20.0  # sim CROSSOVER_S
TURNBACK_S: Final = 240.0  # sim TURNBACK_S: pocket turnback base time (randomized per event by the sim)
LAYOVER_S: Final = 180.0  # sim LAYOVER_S: terminal relay base time
ACCEL_DECEL_LOSS_S: Final = 13.0  # sim ACCEL_DECEL_LOSS_S
V_MAX_CAP: Final = 24.6  # sim V_MAX_CAP
V_MAX_FLOOR_MS: Final = 8.0
MIN_RUNNING_S_FOR_VMAX: Final = 20.0
NOMINAL_DWELL_S: Final = 25.0
MIN_STOP_HOP_S: Final = 30.0
# GTFS stop times are rounded to 30 s. Hop times are de-quantized over a window of
# 2 * DEQUANTIZE_HALF_WINDOW + 1 consecutive stop hops, shared in proportion to a kinematic
# weight (dwell + crossovers + accel/brake loss + length / DEQUANTIZE_REF_SPEED_MS).
DEQUANTIZE_HALF_WINDOW: Final = 1
DEQUANTIZE_REF_SPEED_MS: Final = 15.0
# Only for an edge without any scheduled traversal and no parallel edge to copy from.
FALLBACK_SPEED_MS: Final = 12.0

# --------------------------------------------------------------------------- geometry (§2.1.4, §2.3.10)
SEGMENT_SIMPLIFY_EPS_M: Final = 2.0
BASEMAP_SIMPLIFY_EPS_M: Final = 25.0
BASEMAP_MIN_RING_AREA_M2: Final = 20_000.0
STATION_SHAPE_MAX_DIST_M: Final = 60.0  # projection sanity bound (stations lie on shapes)
# A shape may stop short of its terminal's station point (the Q's shapes end 103 m south of the
# 96 St point): the first/last station of a pattern may match the shape's end within this bound.
TERMINAL_SHAPE_MAX_DIST_M: Final = 250.0
# Snapping a cut's ends onto the station points blends the displacement linearly over this much
# arc (at least twice the displacement), so a shape vertex a few meters from a station never
# leaves a short perpendicular leg (a hook in the frontend's offset route bundles).
SNAP_BLEND_M: Final = 120.0

# --------------------------------------------------------------------------- demand (§2.1.3, §2.3.12-13)
CAPACITY_BASE: Final = 350.0
CAPACITY_PER_PEAK_ENTRY: Final = 0.30
CAPACITY_MAX: Final = 3500.0
SYNTH_BASE_ENTRIES: Final = 900.0
SYNTH_ROUTE_EXPONENT: Final = 1.3
SYNTH_CBD_FACTOR: Final = 2.5
SYNTH_BOROUGH_FACTOR: Final = {"M": 1.4, "Bk": 1.0, "Q": 1.0, "Bx": 0.8}
# Weekday system entries per hour measured 2026-11-19 (normalized before use).
HOURLY_CURVE: Final = (
    26397, 10364, 6662, 7803, 20671, 68580, 156526, 323245, 398287, 258058, 176133, 162998,
    173459, 195961, 243161, 304758, 348614, 421687, 323072, 208894, 153213, 125857, 101315, 64142,
)

# --------------------------------------------------------------------------- hubs (§2.5)
HUB_TOP_N_COMPLEXES: Final = 40
HUB_MIN_ROUTES: Final = 4

# --------------------------------------------------------------------------- output rounding (§2.1)
COORD_DECIMALS: Final = 1  # network.json coordinates (0.1 m)
RATE_DECIMALS: Final = 4  # riders/s
# Attraction shares sum to 1 over ~475 stations; 1e-4 would erase small stations.
SHARE_DECIMALS: Final = 7
TIME_DECIMALS: Final = 1
SPEED_DECIMALS: Final = 2
LENGTH_DECIMALS: Final = 1
ENTRIES_DECIMALS: Final = 1
CAPACITY_DECIMALS: Final = 1
FOCUS_ZOOM_DEFAULT_M_PER_PX: Final = 6.0
