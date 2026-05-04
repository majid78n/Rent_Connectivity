"""
Property scraper using OpenStreetMap / Overpass API.
Returns real accommodation locations (hotels, apartments, hostels)
with exact coordinates — no scraping, no bot detection, no API keys.
"""
from __future__ import annotations
import json
import re
import sys
import time
import requests

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OVERPASS  = "https://overpass-api.de/api/interpreter"
HEADERS   = {"User-Agent": "RentConnect/1.0 (academic project, Politecnico di Milano)"}

# Price bands by accommodation type (EUR/night estimate)
_PRICE = {
    "hotel":       (80,  250),
    "apartment":   (60,  180),
    "hostel":      (25,   70),
    "guest_house": (50,  130),
    "motel":       (50,  110),
}


def scrape_booking(city: str, n: int = 10) -> list[dict]:
    """Return up to n real accommodation points from OpenStreetMap for the city."""

    # 1 — geocode city centre
    center = _geocode(city)
    if not center:
        print(f"[scraper] Could not geocode '{city}'", file=sys.stderr)
        return []
    clat, clng = center
    print(f"[scraper] City centre: {clat:.4f}, {clng:.4f}", file=sys.stderr)

    # 2 — query Overpass (5 km radius, all accommodation types)
    query = f"""
    [out:json][timeout:30];
    (
      node["tourism"~"^(hotel|apartment|hostel|guest_house|motel)$"](around:5000,{clat},{clng});
      way["tourism"~"^(hotel|apartment|hostel|guest_house|motel)$"](around:5000,{clat},{clng});
    );
    out center {n * 4};
    """
    try:
        resp = requests.post(OVERPASS, data={"data": query}, headers=HEADERS, timeout=35)
        elements = resp.json().get("elements", [])
    except Exception as e:
        print(f"[scraper] Overpass error: {e}", file=sys.stderr)
        return []

    print(f"[scraper] Overpass returned {len(elements)} elements", file=sys.stderr)

    # 3 — build listing objects
    listings = []
    seen: set[str] = set()
    import math

    for el in elements:
        if len(listings) >= n:
            break

        tags = el.get("tags", {})
        name = (tags.get("name")
                or tags.get("brand")
                or tags.get("operator")
                or tags.get("tourism", "").replace("_", " ").title())
        if not name or name in seen:
            continue
        seen.add(name)

        # Coordinates (nodes have lat/lon directly; ways use center)
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lng = el.get("lon") or (el.get("center") or {}).get("lon")
        if not lat or not lng:
            continue

        kind = tags.get("tourism", "hotel")
        stars = tags.get("stars") or tags.get("star_rating")
        address = _build_address(tags, city)

        # Estimated price based on type + distance from centre
        dist_km = _haversine(clat, clng, lat, lng)
        price_str = _estimate_price(kind, stars, dist_km)

        listings.append({
            "id":        str(el.get("id", name)),
            "title":     name,
            "price":     price_str,
            "price_raw": _numeric(price_str),
            "rating":    tags.get("rating") or "N/A",
            "reviews":   0,
            "image":     "",
            "url":       _osm_url(el),
            "location":  address,
            "bedrooms":  tags.get("rooms") or "N/A",
            "bathrooms": "N/A",
            "guests":    tags.get("capacity") or "N/A",
            "lat":       float(lat),
            "lng":       float(lng),
            "type":      kind,
            "stars":     stars or "N/A",
        })

    print(f"[scraper] Returning {len(listings)} listings", file=sys.stderr)
    return listings


def debug_scrape(city: str) -> dict:
    center = _geocode(city)
    if not center:
        return {"error": f"Could not geocode {city}"}
    clat, clng = center
    query = f"""
    [out:json][timeout:20];
    node["tourism"~"hotel|apartment|hostel"](around:3000,{clat},{clng});
    out 5;
    """
    try:
        resp = requests.post(OVERPASS, data={"data": query}, headers=HEADERS, timeout=25)
        elements = resp.json().get("elements", [])
        return {
            "city_centre": {"lat": clat, "lng": clng},
            "sample_elements": elements[:3],
            "total": len(elements),
        }
    except Exception as e:
        return {"error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _geocode(city: str):
    try:
        r = requests.get(NOMINATIM,
                         params={"q": city, "format": "json", "limit": 1},
                         headers=HEADERS, timeout=10)
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def _build_address(tags: dict, city: str) -> str:
    parts = [
        tags.get("addr:street"),
        tags.get("addr:housenumber"),
        tags.get("addr:city") or city,
    ]
    return ", ".join(p for p in parts if p)


def _estimate_price(kind: str, stars, dist_km: float) -> str:
    lo, hi = _PRICE.get(kind, (60, 150))
    # Stars bump: each star adds ~15%
    if stars:
        try:
            lo = int(lo * (1 + int(stars) * 0.15))
            hi = int(hi * (1 + int(stars) * 0.15))
        except (ValueError, TypeError):
            pass
    # Distance discount: >3 km = 20% cheaper
    if dist_km > 3:
        lo = int(lo * 0.8)
        hi = int(hi * 0.8)
    return f"€{lo}–{hi}/night"


def _haversine(lat1, lng1, lat2, lng2) -> float:
    import math
    R = 6371
    d = math.radians
    dlat = d(lat2 - lat1)
    dlng = d(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(d(lat1))*math.cos(d(lat2))*math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _osm_url(el: dict) -> str:
    kind = "node" if el.get("type") == "node" else "way"
    return f"https://www.openstreetmap.org/{kind}/{el.get('id', '')}"


def _numeric(price_str: str) -> float | None:
    m = re.search(r"\d+", str(price_str))
    return float(m.group()) if m else None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    city_arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Milan"
    result = scrape_booking(city_arg)
    print(json.dumps(result, indent=2, ensure_ascii=False))
