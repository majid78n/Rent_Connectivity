"""
RentConnect Flask app.
Flow: city input → scrape 10 listings → run main.py pipeline →
      look up each listing in H3 hexagon data → return enriched results.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response

from scraper import scrape_booking, debug_scrape

app = Flask(__name__)

BASE = Path(__file__).parent.parent          # prog/
MAIN_PY = BASE / "main.py"
DATA_DIR = BASE / "data"
OUTPUTS_DIR = BASE / "outputs"
H3_RES = 9                                  # must match main.py

_jobs: dict[str, dict] = {}                  # job_id → job state


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    body = request.get_json(silent=True) or {}
    city = body.get("city", "").strip()
    current_address = body.get("current_address", "").strip()
    if not city:
        return jsonify({"error": "city is required"}), 400

    job_id = re.sub(r"[^a-z0-9]", "_", city.lower())

    # Re-use a completed job only if city matches; restart otherwise
    existing = _jobs.get(job_id)
    if not existing or existing.get("status") == "error":
        _jobs[job_id] = {
            "status": "starting",
            "log": [],
            "listings": [],
            "current_h3": None,
            "done": False,
            "error": None,
        }
        t = threading.Thread(
            target=_run_pipeline,
            args=(job_id, city, current_address),
            daemon=True,
        )
        t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify({
        "status": job["status"],
        "log": job["log"][-30:],
        "done": job["done"],
        "error": job["error"],
        "listings": job["listings"] if job["done"] else [],
        "current_h3": job.get("current_h3"),
    })


@app.route("/api/debug/scrape")
def api_debug_scrape():
    """Diagnostic endpoint — shows what Airbnb returns without parsing."""
    city = request.args.get("city", "Milan, Italy")
    return jsonify(debug_scrape(city))


@app.route("/api/map/<job_id>")
def api_map(job_id):
    """Serve the Folium map with listing markers injected."""
    map_path = OUTPUTS_DIR / "map.html"
    if not map_path.exists():
        return "Map not ready", 404

    html = map_path.read_text(encoding="utf-8")

    job = _jobs.get(job_id, {})
    listings = job.get("listings", [])

    markers_js = _build_markers_js(listings)
    html = _inject_into_map(html, markers_js)

    return Response(html, mimetype="text/html")


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, city: str, current_address: str):
    job = _jobs[job_id]

    def log(msg: str):
        job["log"].append(msg)

    try:
        # Step 1 ── scrape listings
        log(f"[1/3] Scraping Airbnb listings for '{city}'…")
        job["status"] = "scraping"
        listings = scrape_booking(city)
        listings = listings[:10]
        log(f"      Found {len(listings)} listings.")

        # Step 2 ── run main.py transport poverty pipeline
        log("[2/3] Running transport analysis pipeline (2–5 min)…")
        job["status"] = "analyzing"

        env = {**os.environ, "PIPELINE_CITY": city, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, str(MAIN_PY)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(BASE),
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                log(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"main.py exited with code {proc.returncode}")

        # Step 3 ── enrich listings with H3 connectivity data
        log("[3/3] Looking up connectivity for each listing…")
        job["status"] = "enriching"

        # Prefer the post-ML file (has predicted_prob); fall back to the raw one
        gpkg = OUTPUTS_DIR / "transport_poverty_map.gpkg"
        if not gpkg.exists():
            gpkg = DATA_DIR / "hexagons_labeled.gpkg"
        gdf = None
        if gpkg.exists():
            import geopandas as gpd
            gdf = gpd.read_file(str(gpkg))

        for listing in listings:
            if gdf is not None and listing.get("lat") and listing.get("lng"):
                listing["h3_data"] = _lookup_h3(listing["lat"], listing["lng"], gdf)
            else:
                listing["h3_data"] = None

        # Optional: connectivity for the current address
        current_h3 = None
        if current_address:
            from connectivity import geocode
            coords = geocode(current_address)
            if coords and gdf is not None:
                current_h3 = _lookup_h3(coords[0], coords[1], gdf)
                if current_h3:
                    current_h3["address"] = current_address
                    current_h3["lat"] = coords[0]
                    current_h3["lng"] = coords[1]

        job["listings"] = listings
        job["current_h3"] = current_h3
        job["done"] = True
        job["status"] = "done"
        log("Pipeline complete.")

    except Exception as exc:
        job["error"] = str(exc)
        job["done"] = True
        job["status"] = "error"
        log(f"ERROR: {exc}")


# ── H3 lookup ─────────────────────────────────────────────────────────────────

def _lookup_h3(lat: float, lng: float, gdf) -> dict | None:
    try:
        import h3
        fn = next(f for f in ("latlng_to_cell", "geo_to_h3") if hasattr(h3, f))
        # Detect resolution from the gpkg so it always matches regardless of H3_RES
        res_fn = next(f for f in ("get_resolution", "h3_get_resolution") if hasattr(h3, f))
        res = int(getattr(h3, res_fn)(gdf["h3_index"].iloc[0]))
        cell = getattr(h3, fn)(lat, lng, res)

        row = gdf[gdf["h3_index"] == cell]
        if row.empty:
            return None
        r = row.iloc[0]

        def _get(col, default=0.0):
            return float(r[col]) if col in r.index else default

        conn = _get("connectivity_score")
        prob = _get("predicted_prob", conn)   # prob added after model inference

        return {
            "h3_index": cell,
            "connectivity_score": conn,
            "predicted_prob": prob,
            "connectivity_pct": round(prob * 100),
            "label": "Well-connected" if prob >= 0.5 else "Poorly-connected",
            "stop_count": int(_get("stop_count")),
            "mode_count": int(_get("mode_count")),
            "dist_nearest_stop_m": round(_get("dist_nearest_stop_m")),
            "road_density_m_per_km2": round(_get("road_density_m_per_km2")),
            "ndvi": round(_get("NDVI"), 3),
            "ndvi_pct": round((_get("NDVI") + 1) / 2 * 100),  # NDVI [-1,1] → [0,100]%
        }
    except Exception as exc:
        print(f"[h3 lookup] {exc}", file=sys.stderr)
        return None


# ── Map marker injection ──────────────────────────────────────────────────────

def _build_markers_js(listings: list[dict]) -> str:
    points = []
    for i, l in enumerate(listings, 1):
        if not l.get("lat") or not l.get("lng"):
            continue
        h3d = l.get("h3_data") or {}
        score = h3d.get("connectivity_pct", "?")
        color = (
            "#27ae60" if isinstance(score, int) and score >= 75 else
            "#3498db" if isinstance(score, int) and score >= 50 else
            "#f59e0b" if isinstance(score, int) and score >= 30 else
            "#ef4444"
        )
        popup = (
            f"<b>{i}. {l['title'][:40]}</b><br>"
            f"Price: {l.get('price','N/A')}<br>"
            f"Connectivity: {score}%<br>"
            f"Transit stops: {h3d.get('stop_count','?')}<br>"
            f"Dist. to stop: {h3d.get('dist_nearest_stop_m','?')} m"
        )
        points.append({
            "lat": l["lat"], "lng": l["lng"],
            "label": str(i), "color": color, "popup": popup,
        })

    return f"""
(function addRentConnectMarkers() {{
    // Dynamically find whichever map_XXXX variable Folium created
    var map = window._rentconnect_map
           || (function() {{
                var k = Object.keys(window).find(function(n) {{
                    return /^map_[a-f0-9]+$/.test(n) && window[n] && window[n].addLayer;
                }});
                return k ? window[k] : null;
              }})();

    if (!map) {{
        // Map not ready yet — retry in 150 ms
        setTimeout(addRentConnectMarkers, 150);
        return;
    }}

    var pts = {json.dumps(points)};
    pts.forEach(function(p) {{
        var icon = L.divIcon({{
            html: '<div style="background:' + p.color + ';color:#fff;'
                + 'border-radius:50%;width:30px;height:30px;line-height:30px;'
                + 'text-align:center;font-weight:700;font-size:13px;'
                + 'border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.5);">'
                + p.label + '</div>',
            iconSize: [30, 30], iconAnchor: [15, 15], className: ''
        }});
        L.marker([p.lat, p.lng], {{icon: icon}})
         .bindPopup(p.popup)
         .addTo(map);
    }});
}})();
"""


def _inject_into_map(html: str, markers_js: str) -> str:
    """Inject listing markers into a Folium-generated HTML map."""
    # Insert just before </body>; the JS self-discovers the Folium map variable.
    return html.replace("</body>", markers_js + "\n</body>", 1)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RentConnect → http://localhost:8080")
    app.run(debug=True, port=8080)
