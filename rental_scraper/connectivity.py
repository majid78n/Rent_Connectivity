"""
Location connectivity scorer using free OpenStreetMap data.
- Geocoding: Nominatim (OSM)
- POI queries: Overpass API
No API keys required.
"""
from __future__ import annotations

import time
import requests

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "RentalConnectivityTool/1.0 (educational project)"}


def geocode(query: str) -> tuple[float, float] | None:
    try:
        resp = requests.get(
            NOMINATIM,
            params={"q": query, "format": "json", "limit": 1},
            headers=HEADERS,
            timeout=10,
        )
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def _overpass_count(query: str) -> int:
    try:
        resp = requests.post(
            OVERPASS, data={"data": query}, headers=HEADERS, timeout=30
        )
        elements = resp.json().get("elements", [])
        if elements and isinstance(elements[0], dict):
            return int(elements[0].get("tags", {}).get("total", 0))
    except Exception:
        pass
    return 0


def _count_transit(lat: float, lng: float, radius: int) -> int:
    q = f"""
    [out:json][timeout:20];
    (
      node["highway"="bus_stop"](around:{radius},{lat},{lng});
      node["railway"~"subway_entrance|station|tram_stop|halt"](around:{radius},{lat},{lng});
      node["public_transport"~"stop_position|station"](around:{radius},{lat},{lng});
    );
    out count;
    """
    return _overpass_count(q)


def _count_food(lat: float, lng: float, radius: int) -> int:
    q = f"""
    [out:json][timeout:20];
    (
      node["amenity"~"restaurant|cafe|bar|fast_food|food_court|bakery"](around:{radius},{lat},{lng});
    );
    out count;
    """
    return _overpass_count(q)


def _count_services(lat: float, lng: float, radius: int) -> int:
    q = f"""
    [out:json][timeout:20];
    (
      node["amenity"~"supermarket|pharmacy|bank|hospital|clinic|atm"](around:{radius},{lat},{lng});
      node["shop"~"supermarket|convenience|bakery"](around:{radius},{lat},{lng});
    );
    out count;
    """
    return _overpass_count(q)


def _count_parks(lat: float, lng: float, radius: int) -> int:
    q = f"""
    [out:json][timeout:20];
    (
      node["leisure"~"park|garden|playground"](around:{radius},{lat},{lng});
      way["leisure"~"park|garden"](around:{radius},{lat},{lng});
    );
    out count;
    """
    return _overpass_count(q)


def get_connectivity_score(
    location: str, lat: float | None = None, lng: float | None = None
) -> dict:
    if lat is None or lng is None:
        coords = geocode(location)
        if not coords:
            return {"score": 0, "label": "Unknown", "error": "Could not geocode location", "breakdown": {}}
        lat, lng = coords

    # Gather counts (sleep between requests to respect Overpass rate limits)
    transit_500 = _count_transit(lat, lng, 500)
    time.sleep(0.8)
    transit_1000 = _count_transit(lat, lng, 1000)
    time.sleep(0.8)
    food_500 = _count_food(lat, lng, 500)
    time.sleep(0.8)
    services_500 = _count_services(lat, lng, 500)
    time.sleep(0.8)
    parks_1000 = _count_parks(lat, lng, 1000)

    # --- Scoring model (max 100) ---
    # Transit (40 pts): rewarded heavily — core of "connectivity"
    transit_score = min(40, transit_500 * 8 + max(0, transit_1000 - transit_500) * 2)

    # Food & drink (25 pts)
    food_score = min(25, food_500 * 2)

    # Daily services (25 pts)
    services_score = min(25, services_500 * 3)

    # Green spaces (10 pts)
    park_score = min(10, parks_1000 * 3)

    total = int(transit_score + food_score + services_score + park_score)
    total = min(100, total)

    if total >= 80:
        label = "Excellent"
    elif total >= 60:
        label = "Good"
    elif total >= 40:
        label = "Fair"
    else:
        label = "Limited"

    return {
        "score": total,
        "label": label,
        "lat": lat,
        "lng": lng,
        "breakdown": {
            "transit": {
                "label": "Public Transit",
                "stops_500m": transit_500,
                "stops_1km": transit_1000,
                "score": int(transit_score),
                "max": 40,
            },
            "food": {
                "label": "Food & Drink",
                "count_500m": food_500,
                "score": int(food_score),
                "max": 25,
            },
            "services": {
                "label": "Daily Services",
                "count_500m": services_500,
                "score": int(services_score),
                "max": 25,
            },
            "parks": {
                "label": "Green Spaces",
                "count_1km": parks_1000,
                "score": int(park_score),
                "max": 10,
            },
        },
    }


if __name__ == "__main__":
    import json, sys
    loc = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Duomo, Milan"
    print(json.dumps(get_connectivity_score(loc), indent=2))
