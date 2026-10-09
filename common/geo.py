"""Synthetic geography for the simulated MFS market.

District names and approximate centroid coordinates are public geographic
facts about Bangladesh; they are used only to give the synthetic data a
realistic spatial structure (e.g. for impossible-travel detection).
Upazila names are synthetic (``<District>-U03``). Population weights are rough
and illustrative, not official statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class District:
    name: str
    division: str
    lat: float
    lon: float
    weight: float  # relative share of simulated customers
    n_upazilas: int = 6


DISTRICTS: tuple[District, ...] = (
    District("Dhaka", "Dhaka", 23.8103, 90.4125, 14.0, 10),
    District("Gazipur", "Dhaka", 24.0023, 90.4264, 5.0),
    District("Narayanganj", "Dhaka", 23.6238, 90.5000, 4.0),
    District("Tangail", "Dhaka", 24.2513, 89.9167, 3.0),
    District("Faridpur", "Dhaka", 23.6070, 89.8429, 2.0),
    District("Chattogram", "Chattogram", 22.3569, 91.7832, 8.0, 8),
    District("Coxs Bazar", "Chattogram", 21.4272, 92.0058, 2.0),
    District("Cumilla", "Chattogram", 23.4607, 91.1809, 4.0),
    District("Noakhali", "Chattogram", 22.8696, 91.0995, 3.0),
    District("Feni", "Chattogram", 23.0159, 91.3976, 1.5),
    District("Rajshahi", "Rajshahi", 24.3745, 88.6042, 3.0),
    District("Bogura", "Rajshahi", 24.8465, 89.3773, 3.0),
    District("Pabna", "Rajshahi", 24.0064, 89.2372, 2.0),
    District("Khulna", "Khulna", 22.8456, 89.5403, 3.0),
    District("Jashore", "Khulna", 23.1664, 89.2081, 2.5),
    District("Kushtia", "Khulna", 23.9013, 89.1204, 2.0),
    District("Barishal", "Barishal", 22.7010, 90.3535, 2.5),
    District("Patuakhali", "Barishal", 22.3596, 90.3299, 1.2),
    District("Sylhet", "Sylhet", 24.8949, 91.8687, 3.0),
    District("Moulvibazar", "Sylhet", 24.4829, 91.7774, 1.5),
    District("Rangpur", "Rangpur", 25.7439, 89.2752, 3.0),
    District("Dinajpur", "Rangpur", 25.6217, 88.6354, 2.0),
    District("Mymensingh", "Mymensingh", 24.7471, 90.4203, 3.0),
    District("Jamalpur", "Mymensingh", 24.9375, 89.9378, 2.0),
)

DISTRICT_BY_NAME: dict[str, District] = {d.name: d for d in DISTRICTS}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def district_distance_km(a: str, b: str) -> float:
    da, db = DISTRICT_BY_NAME[a], DISTRICT_BY_NAME[b]
    return haversine_km(da.lat, da.lon, db.lat, db.lon)


def upazila_names(district: District) -> list[str]:
    return [f"{district.name}-U{i:02d}" for i in range(1, district.n_upazilas + 1)]
