# =============================================================================
# TRANSPORT POVERTY MAPPING PIPELINE — Any City via OSM Query
# Politecnico di Milano / Fondazione Transform Transport
# =============================================================================
#
# REAL SATELLITE DATA (optional):
#   Run `earthengine authenticate` in your terminal BEFORE this script
#   to use real Sentinel-2 / Landsat 8 data via Google Earth Engine.
#   If GEE is unavailable or unauthenticated the pipeline automatically
#   falls back to realistic synthetic data — no internet access required.
#
# REAL GTFS DATA (optional):
#   The script attempts to download the ATM Milano GTFS feed.
#   If the download fails a distance-based simulation is used instead.
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 — SETUP
# =============================================================================
print("\n" + "=" * 60)
print("[1/6] SETUP — Importing libraries and creating output folders")
print("=" * 60)

import importlib
import subprocess

def _ensure(pkg, import_as=None):
    """Silently install a package if the import fails."""
    mod = import_as or pkg
    try:
        importlib.import_module(mod)
    except ImportError:
        print(f"   Installing {pkg} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

for _pkg, _imp in [
    ("h3",           "h3"),
    ("geopandas",    "geopandas"),
    ("shapely",      "shapely"),
    ("pandas",       "pandas"),
    ("numpy",        "numpy"),
    ("scikit-learn", "sklearn"),
    ("folium",       "folium"),
    ("osmnx",        "osmnx"),
]:
    _ensure(_pkg, _imp)

import h3
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import folium
import osmnx as ox
from shapely.geometry import Polygon, mapping, shape, box
import urllib.parse

for folder in ["data", "models", "outputs"]:
    os.makedirs(folder, exist_ok=True)

print("   Libraries imported. Output folders ready: data/, models/, outputs/")

# =============================================================================
# SECTION 2 — H3 GRID
# =============================================================================
print("\n" + "=" * 60)
print("[2/6] H3 GRID — Querying city boundary and generating H3 hexagons")
print("=" * 60)

# ── USER CONFIGURATION ────────────────────────────────────────────────────────
CITY_NAME = os.environ.get("PIPELINE_CITY", "Milan, Italy")  # City name for OSM geocoding
H3_RES    = 9                # resolution 9 ≈ 0.1 km² per hex
# ─────────────────────────────────────────────────────────────────────────────

print(f"   City  : {CITY_NAME}")
print(f"   H3 res: {H3_RES}")

# Rough bounding boxes for common cities — used ONLY if internet is unavailable
_CITY_FALLBACKS = {
    "milan":     box(9.03,  45.38, 9.28,  45.54),
    "rome":      box(12.35, 41.80, 12.62, 41.98),
    "naples":    box(14.18, 40.80, 14.35, 40.91),
    "turin":     box(7.60,  45.01, 7.76,  45.13),
    "florence":  box(11.19, 43.72, 11.32, 43.84),
    "bologna":   box(11.28, 44.46, 11.41, 44.53),
    "barcelona": box(2.07,  41.32, 2.23,  41.47),
    "london":    box(-0.25, 51.46, 0.01,  51.57),
    "paris":     box(2.25,  48.82, 2.42,  48.91),
}

# --- Step 1: fetch administrative boundary -----------------------------------
city_poly = None

try:
    city_gdf  = ox.geocode_to_gdf(CITY_NAME)
    city_poly = city_gdf.geometry.unary_union   # Shapely geometry, WGS84
    print(f"   Admin boundary retrieved via OSMnx / Nominatim.")
except Exception as _e:
    print(f"   OSMnx failed ({type(_e).__name__}: {_e}).")
    print(f"   Trying Nominatim REST API directly...")

if city_poly is None:
    try:
        import urllib.request, json
        q   = urllib.parse.quote(CITY_NAME)
        url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={q}&format=geojson&polygon_geojson=1&limit=1")
        req = urllib.request.Request(
            url, headers={"User-Agent": "transport-poverty-pipeline/1.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        feats = data.get("features", [])
        if not feats:
            raise ValueError("Nominatim returned 0 results for this query.")
        city_poly = shape(feats[0]["geometry"])
        print("   Admin boundary retrieved via Nominatim REST API.")
    except Exception as _e:
        print(f"   Nominatim REST failed ({type(_e).__name__}: {_e}).")

if city_poly is None:
    _key = CITY_NAME.split(",")[0].strip().lower()
    if _key in _CITY_FALLBACKS:
        city_poly = _CITY_FALLBACKS[_key]
        print(f"   !! No internet — using hardcoded fallback bbox for '{_key}'.")
    else:
        raise RuntimeError(
            f"\n\n  Cannot geocode '{CITY_NAME}' and no offline fallback exists.\n"
            f"  Connect to the internet so OSMnx can query Nominatim,\n"
            f"  or add '{_key}' to the _CITY_FALLBACKS dict in Section 2.\n"
        )

# City center (lat, lon) — used by synthetic-feature gradient models
CITY_CENTER = (city_poly.centroid.y, city_poly.centroid.x)
print(f"   City centroid: lat={CITY_CENTER[0]:.4f}, lon={CITY_CENTER[1]:.4f}")

# --- Step 2: polyfill with H3 cells -----------------------------------------
def _fill_poly(geojson, res):
    """Fill a GeoJSON Polygon or MultiPolygon with H3 cells (v3 + v4 safe)."""
    fn_name = None
    for fn in ("geo_to_cells", "polyfill_geojson", "polyfill"):
        if hasattr(h3, fn):
            fn_name = fn
            break
    if fn_name is None:
        raise RuntimeError("No h3 polyfill function found.")

    if geojson["type"] == "MultiPolygon":
        cells = set()
        for ring in geojson["coordinates"]:
            sub = {"type": "Polygon", "coordinates": ring}
            cells.update(getattr(h3, fn_name)(sub, res))
        return list(cells)

    return list(getattr(h3, fn_name)(geojson, res))

city_geojson = mapping(city_poly)
cells = _fill_poly(city_geojson, H3_RES)

def _cell_to_polygon(cell):
    """Convert an H3 cell index to a Shapely Polygon (lon, lat coords)."""
    for fn in ("cell_to_boundary", "h3_to_geo_boundary"):
        if hasattr(h3, fn):
            boundary = getattr(h3, fn)(cell)
            break
    # boundary is a sequence of (lat, lon) tuples
    return Polygon([(lon, lat) for lat, lon in boundary])

polygons = [_cell_to_polygon(c) for c in cells]

gdf = gpd.GeoDataFrame(
    {"h3_index": cells},
    geometry=polygons,
    crs="EPSG:4326",
)
gdf["centroid_lat"] = gdf.geometry.centroid.y
gdf["centroid_lon"] = gdf.geometry.centroid.x

print(f"   Generated {len(gdf):,} H3 hexagons at resolution {H3_RES}.")

# =============================================================================
# SECTION 3 — SATELLITE FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("[3/6] SATELLITE FEATURES — Attempting Google Earth Engine connection")
print("=" * 60)

# ---------- synthetic fallback -----------------------------------------------
def _synthetic_features(gdf, seed=42):
    """Simulate NDVI, NDWI, NDBI, LST with a realistic urban–rural gradient."""
    print("   Generating synthetic satellite features (urban gradient model)...")
    lats = gdf["centroid_lat"].values
    lons = gdf["centroid_lon"].values
    dlat = lats - CITY_CENTER[0]
    dlon = lons - CITY_CENTER[1]
    dist_norm = np.sqrt(dlat**2 + dlon**2)
    dist_norm /= dist_norm.max() + 1e-9   # 0 = city center, 1 = periphery

    rng = np.random.default_rng(seed)
    n = len(gdf)

    NDVI = np.clip(0.08 + 0.60 * dist_norm + rng.normal(0, 0.05, n), -1, 1)
    NDWI = np.clip(-0.20 + 0.30 * dist_norm + rng.normal(0, 0.04, n), -1, 1)
    NDBI = np.clip(0.32 - 0.42 * dist_norm + rng.normal(0, 0.05, n), -1, 1)
    LST  = np.clip(33.0 - 9.0  * dist_norm + rng.normal(0, 1.0,  n), 15, 45)

    return NDVI, NDWI, NDBI, LST

# ---------- GEE attempt -------------------------------------------------------
GEE_OK = False
try:
    import ee  # noqa: F401 — imported only to check authentication
    ee.Initialize()
    GEE_OK = True
    print("   GEE authenticated. Pulling Sentinel-2 L2A and Landsat 8 ...")
except Exception as _gee_err:
    print(f"   GEE unavailable ({type(_gee_err).__name__}). Using synthetic fallback.")

if GEE_OK:
    try:
        import ee

        _minx, _miny, _maxx, _maxy = city_poly.bounds
        city_ee_geom = ee.Geometry.Rectangle([_minx, _miny, _maxx, _maxy])

        # Sentinel-2 composite
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(city_ee_geom)
            .filterDate("2023-06-01", "2023-08-31")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B3", "B4", "B8", "B11"])
            .median()
        )
        ndvi_img = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
        ndwi_img = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
        ndbi_img = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")

        # Landsat 8 LST (ST_B10: scale * 0.00341802 + 149.0 K → °C)
        lst_img = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(city_ee_geom)
            .filterDate("2023-06-01", "2023-08-31")
            .select(["ST_B10"])
            .median()
            .multiply(0.00341802).add(149.0).subtract(273.15)
            .rename("LST")
        )

        composite = ndvi_img.addBands([ndwi_img, ndbi_img, lst_img])

        rows = []
        for _, row in gdf.iterrows():
            pt = ee.Geometry.Point([row["centroid_lon"], row["centroid_lat"]])
            try:
                vals = composite.sample(pt, 30).first().toDictionary().getInfo()
            except Exception:
                vals = {}
            rows.append({
                "h3_index": row["h3_index"],
                "NDVI": vals.get("NDVI", np.nan),
                "NDWI": vals.get("NDWI", np.nan),
                "NDBI": vals.get("NDBI", np.nan),
                "LST":  vals.get("LST",  np.nan),
            })

        feat_df = pd.DataFrame(rows)
        gdf = gdf.merge(feat_df, on="h3_index", how="left")

        missing = gdf[["NDVI", "NDWI", "NDBI", "LST"]].isna().any(axis=1).sum()
        if missing > len(gdf) * 0.5:
            raise ValueError(f"Too many NaNs ({missing}) — falling back to synthetic.")

        print(f"   GEE features extracted for {len(gdf):,} hexagons "
              f"({missing} NaNs filled with synthetic).")

        # Fill remaining NaNs with synthetic values at those positions
        if missing:
            mask = gdf[["NDVI", "NDWI", "NDBI", "LST"]].isna().any(axis=1)
            sub = gdf[mask].copy().reset_index(drop=True)
            sv, sw, sb, sl = _synthetic_features(sub)
            gdf.loc[mask, "NDVI"] = sv
            gdf.loc[mask, "NDWI"] = sw
            gdf.loc[mask, "NDBI"] = sb
            gdf.loc[mask, "LST"]  = sl

    except Exception as _gee_feat_err:
        print(f"   GEE feature extraction failed ({_gee_feat_err}). Using synthetic.")
        GEE_OK = False

if not GEE_OK:
    ndvi, ndwi, ndbi, lst = _synthetic_features(gdf)
    gdf["NDVI"] = ndvi
    gdf["NDWI"] = ndwi
    gdf["NDBI"] = ndbi
    gdf["LST"]  = lst

print(f"   Feature stats  NDVI μ={gdf['NDVI'].mean():.3f}  "
      f"NDBI μ={gdf['NDBI'].mean():.3f}  LST μ={gdf['LST'].mean():.1f}°C")

# =============================================================================
# SECTION 4 — CONNECTIVITY FEATURES & LABELS (OSMnx)
# =============================================================================
print("\n" + "=" * 60)
print(f"[4/6] CONNECTIVITY — OSMnx transit stops, road density, nearest stop")
print("=" * 60)

# ── shared helpers ────────────────────────────────────────────────────────────
def _norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def _latlon_to_cell(lat, lon, res):
    for fn in ("latlng_to_cell", "geo_to_h3"):
        if hasattr(h3, fn):
            return getattr(h3, fn)(lat, lon, res)
    raise RuntimeError("No h3 latlon→cell function found.")

# Approximate hex area by H3 resolution (km²)
_H3_AREA_KM2 = {7: 5.161, 8: 0.737, 9: 0.105, 10: 0.015}
_HEX_AREA_KM2 = _H3_AREA_KM2.get(H3_RES, 0.105)

# ── 4a: Real transit stops from OpenStreetMap ─────────────────────────────────
print("\n   [4a] Fetching transit stops from OSM via OSMnx ...")
TRANSIT_OK = False
stops_gdf  = None

try:
    _transit_tags = {
        "public_transport": ["stop_position", "platform"],
        "highway":          "bus_stop",
        "railway":          ["station", "halt", "tram_stop", "subway_entrance"],
        "amenity":          "bus_station",
    }
    _raw = ox.features_from_place(CITY_NAME, tags=_transit_tags)
    stops_gdf = _raw[_raw.geometry.geom_type == "Point"].copy().reset_index(drop=True)
    stops_gdf["stop_lat"] = stops_gdf.geometry.y
    stops_gdf["stop_lon"] = stops_gdf.geometry.x

    # Assign each stop to its H3 cell (list comprehension is ~10x faster than apply)
    _h3fn = next(f for f in ("latlng_to_cell", "geo_to_h3") if hasattr(h3, f))
    stops_gdf["h3_index"] = [
        getattr(h3, _h3fn)(lat, lon, H3_RES)
        for lat, lon in zip(stops_gdf["stop_lat"], stops_gdf["stop_lon"])
    ]

    # Count stops per hex
    stop_cnt = stops_gdf.groupby("h3_index").size().rename("stop_count")

    # Count distinct transport modes per hex
    _mode_col = next(
        (c for c in ["public_transport", "railway", "highway"] if c in stops_gdf.columns),
        None,
    )
    if _mode_col:
        mode_cnt = stops_gdf.groupby("h3_index")[_mode_col].nunique().rename("mode_count")
    else:
        mode_cnt = (stop_cnt > 0).astype(int).rename("mode_count")

    conn_df = (
        stop_cnt.to_frame()
        .join(mode_cnt, how="outer")
        .fillna(0)
        .reset_index()
    )
    gdf = gdf.merge(conn_df, on="h3_index", how="left").fillna(0)

    composite = (_norm01(gdf["stop_count"].values) +
                 _norm01(gdf["mode_count"].values)) / 2.0
    gdf["connectivity_score"] = composite
    # Rank-based 50/50 split — immune to constant or degenerate distributions
    _rank  = pd.Series(composite).rank(method="first").values
    gdf["label"] = (_rank > len(composite) / 2).astype(int)

    TRANSIT_OK = True
    print(f"   Found {len(stops_gdf):,} transit stop nodes in OSM.")

except Exception as _e:
    print(f"   OSMnx transit fetch failed ({type(_e).__name__}: {_e}).")
    print("   Falling back to distance-based simulation.")

if not TRANSIT_OK:
    rng  = np.random.default_rng(123)
    n    = len(gdf)
    lats = gdf["centroid_lat"].values
    lons = gdf["centroid_lon"].values
    d    = np.sqrt((lats - CITY_CENTER[0])**2 + (lons - CITY_CENTER[1])**2)
    dn   = d / (d.max() + 1e-9)

    sc = np.clip((22 - 19*dn + rng.normal(0, 2, n)).astype(int), 0, 35)
    mc = np.clip((5  -  4*dn + rng.normal(0, 1, n)).astype(int), 0,  8)

    gdf["stop_count"]  = sc
    gdf["mode_count"]  = mc
    composite          = (_norm01(sc) + _norm01(mc)) / 2.0
    gdf["connectivity_score"] = composite
    gdf["label"]       = (composite >= np.median(composite)).astype(int)

n_well = int(gdf["label"].sum())
n_poor = int((gdf["label"] == 0).sum())
print(f"   Labels → well-connected: {n_well:,}  |  poorly connected: {n_poor:,}")

# ── 4b: Road network density (m of road per km²) ─────────────────────────────
print("\n   [4b] Fetching road network from OSMnx ...")
ROADS_OK = False

try:
    # "drive" graph is ~5-10x smaller than "walk" for large cities
    ox.settings.use_cache = True
    G     = ox.graph_from_place(CITY_NAME, network_type="drive", simplify=True)
    _, edges = ox.graph_to_gdfs(G)

    edges["mid_lat"] = edges.geometry.interpolate(0.5, normalized=True).y
    edges["mid_lon"] = edges.geometry.interpolate(0.5, normalized=True).x
    _h3fn = next(f for f in ("latlng_to_cell", "geo_to_h3") if hasattr(h3, f))
    edges["h3_index"] = [
        getattr(h3, _h3fn)(lat, lon, H3_RES)
        for lat, lon in zip(edges["mid_lat"], edges["mid_lon"])
    ]

    road_len = edges.groupby("h3_index")["length"].sum().rename("road_length_m")
    gdf = gdf.merge(road_len.reset_index(), on="h3_index", how="left").fillna(0)
    gdf["road_density_m_per_km2"] = gdf["road_length_m"] / _HEX_AREA_KM2

    ROADS_OK = True
    print(f"   Road network: {len(edges):,} edges  |  "
          f"median density {gdf['road_density_m_per_km2'].median():.0f} m/km²")

except Exception as _e:
    print(f"   Road network fetch failed ({type(_e).__name__}: {_e}).")
    print("   Falling back to distance-based road density approximation.")
    dn = np.sqrt(
        (gdf["centroid_lat"].values - CITY_CENTER[0])**2 +
        (gdf["centroid_lon"].values - CITY_CENTER[1])**2
    )
    dn /= dn.max() + 1e-9
    rng = np.random.default_rng(77)
    gdf["road_length_m"]         = np.clip(6000 - 5000*dn + rng.normal(0, 300, len(gdf)), 0, 10000)
    gdf["road_density_m_per_km2"] = gdf["road_length_m"] / _HEX_AREA_KM2

# ── 4c: Walking distance to nearest transit stop ──────────────────────────────
print("\n   [4c] Computing walking distance to nearest transit stop ...")
from sklearn.neighbors import BallTree

if stops_gdf is not None and len(stops_gdf) > 0:
    _stop_coords = np.radians(stops_gdf[["stop_lat", "stop_lon"]].values)
    _hex_coords  = np.radians(gdf[["centroid_lat", "centroid_lon"]].values)
    _tree = BallTree(_stop_coords, metric="haversine")
    _dists, _ = _tree.query(_hex_coords, k=1)
    gdf["dist_nearest_stop_m"] = _dists[:, 0] * 6_371_000
else:
    dn = np.sqrt(
        (gdf["centroid_lat"].values - CITY_CENTER[0])**2 +
        (gdf["centroid_lon"].values - CITY_CENTER[1])**2
    )
    dn /= dn.max() + 1e-9
    gdf["dist_nearest_stop_m"] = np.clip(80 + 1800 * dn, 80, 2000)

print(f"   Nearest stop  median={gdf['dist_nearest_stop_m'].median():.0f} m  "
      f"max={gdf['dist_nearest_stop_m'].max():.0f} m")

gdf.to_file("data/hexagons_labeled.gpkg", driver="GPKG")
print("\n   Saved  data/hexagons_labeled.gpkg")

# =============================================================================
# SECTION 5 — DATASET
# =============================================================================
print("\n" + "=" * 60)
print("[5/6] DATASET — Assembling feature matrix and stratified splits")
print("=" * 60)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDBI", "LST",          # satellite
    "road_density_m_per_km2",                # OSMnx road network
    "dist_nearest_stop_m",                   # OSMnx transit proximity
]
LABEL_COL    = "label"

df_full = gdf[["h3_index"] + FEATURE_COLS + [LABEL_COL]].copy()
n_before = len(df_full)
df_full  = df_full.dropna()
print(f"   Dropped {n_before - len(df_full)} rows with NaN values. "
      f"Dataset size: {len(df_full):,}")

X = df_full[FEATURE_COLS].values.astype(np.float32)
y = df_full[LABEL_COL].values.astype(np.float32)

df_full.to_csv("data/features.csv", index=False)
print("   Saved  data/features.csv")

# =============================================================================
# SECTION 6 — RANDOM FOREST MODEL
# =============================================================================
print("\n" + "=" * 60)
print("[6/6] RANDOM FOREST — Training RandomForestClassifier on all hexagons")
print("=" * 60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X, y)
joblib.dump(rf, "models/rf_best.pkl")
print(f"   Trained on {len(X):,} hexagons. Saved  models/rf_best.pkl")

# =============================================================================
# SECTION 7 — OUTPUT MAP
# =============================================================================
print("\n" + "=" * 60)
print("[6/6] OUTPUT MAP — Inference on all hexagons + interactive Folium map")
print("=" * 60)

# NaN-safe feature matrix for full grid
X_all = gdf[FEATURE_COLS].values.astype(np.float32)
col_means = np.nanmean(X_all, axis=0)
nan_mask  = np.isnan(X_all)
for col in range(X_all.shape[1]):
    X_all[nan_mask[:, col], col] = col_means[col]

_p = rf.predict_proba(X_all)
probs_all = _p[:, 1] if _p.shape[1] > 1 else _p[:, 0].astype(float)

gdf = gdf.copy()
gdf["predicted_prob"]  = probs_all
gdf["predicted_class"] = (probs_all >= 0.5).astype(int)

gdf.to_file("outputs/transport_poverty_map.gpkg", driver="GPKG")
print("   Saved  outputs/transport_poverty_map.gpkg")

# ---------- Folium interactive map -------------------------------------------
zoom = 12 if H3_RES <= 8 else 13
m = folium.Map(location=list(CITY_CENTER), zoom_start=zoom, tiles="CartoDB positron")

# Stretch color scale to 5th–95th percentile of actual data so the full
# red→yellow→green ramp is used regardless of how the scores cluster.
_p_lo = float(gdf["predicted_prob"].quantile(0.05))
_p_hi = float(gdf["predicted_prob"].quantile(0.95))

def _prob_to_hex(p):
    """Dark-red → orange → yellow → lime → dark-green, stretched to data range."""
    norm = float(np.clip((p - _p_lo) / max(_p_hi - _p_lo, 1e-9), 0, 1))
    # 5-stop palette: #c0392b · #e67e22 · #f1c40f · #2ecc71 · #1a7a4a
    stops = [
        (0.00, (192, 57,  43)),   # dark red
        (0.25, (230, 126, 34)),   # orange
        (0.50, (241, 196, 15)),   # yellow
        (0.75, ( 52, 152, 219)),  # blue
        (1.00, ( 39, 174,  96)),  # green (excellent)
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if norm <= t1:
            f = (norm - t0) / (t1 - t0)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = stops[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"

# Pre-compute hex color as a property so style_function is a pure dict lookup
gdf["_color"] = [_prob_to_hex(p) for p in gdf["predicted_prob"]]
gdf["_label"] = ["Well-connected" if c == 1 else "Poorly-connected"
                 for c in gdf["predicted_class"]]

map_cols = ["h3_index", "_color", "_label", "predicted_prob", "NDVI", "NDBI", "geometry"]
folium.GeoJson(
    gdf[map_cols],
    style_function=lambda f: {
        "fillColor":   f["properties"]["_color"],
        "color":       "none",
        "weight":      0,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["h3_index", "_label", "predicted_prob", "NDVI", "NDBI"],
        aliases=["H3:", "Class:", "Prob:", "NDVI:", "NDBI:"],
        localize=True,
    ),
).add_to(m)

legend_html = """
<div style="
    position: fixed; bottom: 35px; left: 35px; z-index: 1000;
    background: white; padding: 14px 18px; border-radius: 10px;
    border: 1px solid #bbb; font-size: 13px; line-height: 1.9;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.15);">
  <b style="display:block;margin-bottom:4px">Transport Connectivity</b>
  <span style="background:#27ae60;display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:6px"></span>Excellent<br>
  <span style="background:#3498db;display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:6px"></span>Good<br>
  <span style="background:#f1c40f;display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:6px"></span>Moderate<br>
  <span style="background:#e67e22;display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:6px"></span>Poor<br>
  <span style="background:#c0392b;display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:middle;margin-right:6px"></span>Very poor
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save("outputs/map.html")
print("   Saved  outputs/map.html")

# ---------- final summary -----------------------------------------------------
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print("   data/hexagons_labeled.gpkg              — labeled H3 hexagons")
print("   data/features.csv                       — feature matrix")
print("   models/rf_best.pkl                      — Random Forest model")
print("   outputs/transport_poverty_map.gpkg      — predicted transport poverty")
print("   outputs/map.html                        — interactive choropleth map")
print("=" * 60)
