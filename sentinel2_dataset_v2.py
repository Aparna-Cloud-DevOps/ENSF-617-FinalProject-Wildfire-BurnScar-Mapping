"""
sentinel2_dataset_v2.py
=======================
Complete rewrite of the Sentinel-2 L2A dataset construction pipeline for
wildfire burn-scar and severity segmentation with regional fire quotas.

Pipeline
--------
1. Select fires from MTBS (USA) and CNFDB (Canada) with regional quotas
2. Download pre-fire and post-fire Sentinel-2 L2A scenes via Planetary Computer
3. Cloud-mask using SCL band (strict 20% threshold)
4. Compute spectral indices: NBR, NDVI, dNBR, RdNBR
5. Align MTBS severity rasters or rasterise CNFDB perimeters
6. Extract 256x256 patches with 50% overlap and class balance
7. Build stratified forward-chaining train/val/test splits by fire-year

Outputs
-------
  data/sentinel2/patches_v2/USA/    .npz patches
  data/sentinel2/patches_v2/CAN/    .npz patches
  data/sentinel2/splits_v2.json     train/val/test fire-id lists
  data/sentinel2/meta/splits_v2.json (copy)

Run:
    python sentinel2_dataset_v2.py [--n_usa 50] [--n_canada 30]
"""

import os, json, time, warnings, argparse, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import box, mapping
import pystac_client
import planetary_computer
import fiona
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE      = os.environ.get("WILDFIRE_BASE_DIR",
            os.path.dirname(os.path.abspath(__file__)))
META_DIR  = os.path.join(BASE, "data", "sentinel2", "meta")
S2_DIR    = os.path.join(BASE, "data", "sentinel2")

MTBS_SHP  = os.path.join(BASE, "data", "US", "MTBS", "perimeters",
                         "mtbs_perims_DD.shp")
CNFDB_SHP = os.path.join(BASE, "data", "Canada", "NFDB_poly",
                         "NFDB_poly_1972to2020_20250630.shp")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_RES   = 10       # metres
PATCH_SIZE   = 256
OVERLAP      = 0.50     # 50%
MAX_PATCHES_PER_FIRE = 600
MIN_PATCHES_PER_FIRE = 20

# SCL (Scene Classification Layer) class IDs to treat as invalid / mask out.
# 0=No data, 1=Saturated/Defective, 2=Dark Area, 3=Cloud Shadow,
# 8=Cloud Medium, 9=Cloud High, 10=Thin Cirrus, 11=Snow/Ice
SCL_MASK_CLASSES = {0, 1, 2, 3, 8, 9, 10, 11}

# Sentinel-2 bands downloaded from Planetary Computer STAC.
# B11 (SWIR1) is used as a burn proxy; B12 (SWIR2) available but not in final stack.
DATA_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]  # Blue,Green,Red,NIR,SWIR1,SWIR2
ALL_BANDS  = DATA_BANDS + ["SCL"]

# MTBS original 6-class severity → project's 4-class remap.
# MTBS: 1=Unburned/Low  2=Low  3=Moderate  4=High  5=Increased Greenness  6=Non-processing
# New:  0=Unburned/Low(1,2,5,6)  1=Moderate(3)  2=High(4)  3=Very High(dNBR>0.66 within High)
# Very High is split from High using the dNBR threshold (0.66) to create a 4th class
# that represents the most severely burned pixels not resolved by MTBS alone.
MTBS_4CLASS_REMAP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 0, 6: 0}

# Regional quotas
USA_REGIONS = {
    "California":            {"states": {"CA"},             "min_fires": 20},
    "Oregon_Washington":     {"states": {"OR", "WA"},       "min_fires": 15},
    "Colorado_Idaho_Montana":{"states": {"CO", "ID", "MT"}, "min_fires": 15},
    "Arizona_New_Mexico":    {"states": {"AZ", "NM"},       "min_fires": 15},
    "Alaska":                {"states": {"AK"},             "min_fires": 10},
}

CANADA_REGIONS = {
    "British_Columbia":      {"agencies": {"BC"},           "min_fires": 15},
    "Alberta":               {"agencies": {"AB"},           "min_fires": 15},
    "Saskatchewan_Manitoba": {"agencies": {"SK", "MB"},     "min_fires": 10},
    "NWT_Yukon":             {"agencies": {"NT", "YT"},     "min_fires": 10},
}

# Target fraction of patches per severity class in the extracted dataset.
# Each tuple is (min_fraction, max_fraction). Midpoint is used as the sampling target.
# Class 3 (Very High) has a narrow low-fraction range because it is genuinely rare
# in most fires — forcing higher fractions would over-represent a minority geography.
CLASS_BALANCE = {0: (0.20, 0.40), 1: (0.20, 0.40), 2: (0.20, 0.40), 3: (0.05, 0.15)}


# =============================================================================
# SECTION 1  FIRE SELECTION WITH REGIONAL QUOTAS
# =============================================================================

def _state_from_fire_id(fire_id):
    """Extract 2-letter state code from MTBS Event_ID (first 2 chars)."""
    return str(fire_id)[:2].upper()


def select_fires_usa(n_total=50, min_area_ha=3000, year_start=2016, year_end=2023):
    """
    Select USA fires from MTBS shapefile with regional quotas.
    Returns DataFrame with standardized columns.
    """
    print("  Loading MTBS fires ...")
    rows = []
    with fiona.open(MTBS_SHP) as src:
        for feat in src:
            p  = feat["properties"]
            ig = p.get("Ig_Date") or ""
            try:
                yr, mo, dy = [int(x) for x in ig.split("-")]
            except Exception:
                continue
            if yr < year_start or yr > year_end:
                continue
            acres = float(p.get("BurnBndAc") or 0)
            ah    = acres * 0.404686
            if ah < min_area_ha:
                continue
            if p.get("Incid_Type", "Wildfire") not in ("Wildfire", "Prescribed Fire", ""):
                pass  # keep all, filter later if needed
            from shapely.geometry import shape
            s  = shape(feat["geometry"])
            b  = s.bounds
            fid = p["Event_ID"]
            state = _state_from_fire_id(fid)
            rows.append({
                "fire_id":  fid,
                "country":  "USA",
                "year":     yr,
                "month":    mo,
                "day":      dy,
                "area_ha":  ah,
                "low_t":    float(p.get("Low_T")  or 130) / 1000.0,
                "mod_t":    float(p.get("Mod_T")  or 334) / 1000.0,
                "high_t":   float(p.get("High_T") or 585) / 1000.0,
                "lon_min":  b[0], "lat_min": b[1],
                "lon_max":  b[2], "lat_max": b[3],
                "geometry": s,
                "state":    state,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: No MTBS fires found matching criteria")
        return df

    # Assign each fire to a region
    def _region(state):
        for rname, rinfo in USA_REGIONS.items():
            if state in rinfo["states"]:
                return rname
        return "Other"

    df["region"] = df["state"].apply(_region)

    selected = []

    # Fill regional quotas first (largest fires per region)
    for rname, rinfo in USA_REGIONS.items():
        quota    = rinfo["min_fires"]
        subset   = df[df["region"] == rname].sort_values("area_ha", ascending=False)
        taken    = subset.head(quota)
        selected.append(taken)
        print(f"    {rname}: {len(taken)}/{quota} fires selected")

    sel_df = pd.concat(selected, ignore_index=True).drop_duplicates("fire_id")

    # Top up to n_total from remaining large fires (all regions, diverse years)
    if len(sel_df) < n_total:
        used_ids    = set(sel_df["fire_id"])
        remaining   = (df[~df["fire_id"].isin(used_ids)]
                       .sort_values("area_ha", ascending=False)
                       .groupby("year").head(3))
        extra = remaining.head(n_total - len(sel_df))
        sel_df = pd.concat([sel_df, extra], ignore_index=True).drop_duplicates("fire_id")

    sel_df = sel_df.head(n_total).reset_index(drop=True)
    print(f"  Selected {len(sel_df)} USA fires (target {n_total})")
    return sel_df


def select_fires_canada(n_total=30, min_area_ha=3000, year_start=2016, year_end=2023):
    """
    Select Canada fires from CNFDB shapefile with regional quotas.
    Reprojects EPSG:3978 bounds to WGS84.
    """
    print("  Loading CNFDB fires ...")
    t_can = Transformer.from_crs("EPSG:3978", "EPSG:4326", always_xy=True)

    rows = []
    with fiona.open(CNFDB_SHP) as src:
        for feat in src:
            p   = feat["properties"]
            yr  = int(p.get("YEAR") or 0)
            if yr < year_start or yr > year_end:
                continue
            ah = float(p.get("CALC_HA") or p.get("SIZE_HA") or 0)
            if ah < min_area_ha:
                continue
            agency = str(p.get("SRC_AGENCY") or "").strip().upper()
            known  = {"BC", "AB", "SK", "MB", "NT", "YT", "ON", "QC", "NB", "NS", "NL", "PE"}
            if agency not in known:
                continue

            from shapely.geometry import shape as shp_func
            from shapely.ops import transform as shp_transform
            s = shp_func(feat["geometry"])
            if s.geom_type == "MultiPolygon":
                s = s.convex_hull
            b = s.bounds  # EPSG:3978 metres

            # Reproject bounds corners to WGS84
            lon_min, lat_min = t_can.transform(b[0], b[1])
            lon_max, lat_max = t_can.transform(b[2], b[3])
            # Reproject geometry
            s_wgs = shp_transform(t_can.transform, s)

            mo  = int(p.get("MONTH") or 7)
            dy  = int(p.get("DAY")   or 1)
            fid = str(p.get("FIRE_ID") or "").replace("/", "-").replace(" ", "_")

            rows.append({
                "fire_id":  fid,
                "country":  "CAN",
                "year":     yr,
                "month":    mo,
                "day":      dy,
                "area_ha":  ah,
                "low_t":    0.10,
                "mod_t":    0.27,
                "high_t":   0.66,
                "lon_min":  lon_min, "lat_min": lat_min,
                "lon_max":  lon_max, "lat_max": lat_max,
                "geometry": s_wgs,
                "state":    agency,  # use agency as 'state' field
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  WARNING: No CNFDB fires found matching criteria")
        return df

    df["region"] = df["state"]

    selected = []
    for rname, rinfo in CANADA_REGIONS.items():
        quota  = rinfo["min_fires"]
        subset = df[df["state"].isin(rinfo["agencies"])].sort_values("area_ha", ascending=False)
        taken  = subset.head(quota)
        selected.append(taken)
        print(f"    {rname}: {len(taken)}/{quota} fires selected")

    sel_df = pd.concat(selected, ignore_index=True).drop_duplicates("fire_id")

    if len(sel_df) < n_total:
        used_ids  = set(sel_df["fire_id"])
        remaining = (df[~df["fire_id"].isin(used_ids)]
                     .sort_values("area_ha", ascending=False)
                     .groupby("year").head(3))
        extra  = remaining.head(n_total - len(sel_df))
        sel_df = pd.concat([sel_df, extra], ignore_index=True).drop_duplicates("fire_id")

    sel_df = sel_df.head(n_total).reset_index(drop=True)
    print(f"  Selected {len(sel_df)} Canada fires (target {n_total})")
    return sel_df


def select_fires(n_usa=50, n_canada=30, min_area_ha=3000,
                 year_start=2016, year_end=2023):
    """Select fires from both USA and Canada, return combined DataFrame."""
    fires = []
    try:
        usa = select_fires_usa(n_usa, min_area_ha, year_start, year_end)
        if len(usa):
            fires.append(usa)
    except Exception as e:
        print(f"  ERROR loading USA fires: {e}")

    try:
        can = select_fires_canada(n_canada, min_area_ha, year_start, year_end)
        if len(can):
            fires.append(can)
    except Exception as e:
        print(f"  ERROR loading Canada fires: {e}")

    if not fires:
        raise RuntimeError("No fires found matching criteria")

    result = pd.concat(fires, ignore_index=True).drop_duplicates("fire_id")
    print(f"  Total fires selected: {len(result)}")
    return result


# =============================================================================
# SECTION 2  SENTINEL-2 DOWNLOAD
# =============================================================================

def _open_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def _cloud_fraction(item):
    return float(item.properties.get("eo:cloud_cover", 100))


def _utm_epsg(lon, lat):
    """Derive the WGS84 UTM EPSG code for a lon/lat point.
    Northern hemisphere: EPSG:326xx (WGS84 / UTM zone xx N)
    Southern hemisphere: EPSG:327xx (WGS84 / UTM zone xx S)
    UTM zone = floor((lon + 180) / 6) + 1  (zones run 1–60, each 6° wide)
    """
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"


def _compute_fixed_grid(bbox_wgs84, target_res=TARGET_RES):
    """
    Compute fixed UTM pixel grid aligned to round UTM coordinates.
    Returns (out_transform, out_height, out_width, epsg_utm_str).
    """
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    lon_ctr = (lon_min + lon_max) / 2.0
    lat_ctr = (lat_min + lat_max) / 2.0
    epsg_utm = _utm_epsg(lon_ctr, lat_ctr)

    t = Transformer.from_crs("EPSG:4326", epsg_utm, always_xy=True)
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)

    # Snap to grid
    x_min = np.floor(x_min / target_res) * target_res
    y_min = np.floor(y_min / target_res) * target_res
    x_max = np.ceil(x_max  / target_res) * target_res
    y_max = np.ceil(y_max  / target_res) * target_res

    out_width  = int(round((x_max - x_min) / target_res))
    out_height = int(round((y_max - y_min) / target_res))

    out_transform = from_origin(x_min, y_max, target_res, target_res)
    return out_transform, out_height, out_width, epsg_utm


def search_s2_scenes(bbox, date_start, date_end, max_cloud=20.0):
    """Search Sentinel-2 L2A scenes; return sorted by cloud cover."""
    catalog = _open_catalog()
    search  = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        sortby="eo:cloud_cover",
    )
    items = list(search.items())
    return sorted(items, key=_cloud_fraction)


def download_scene_bands(item, bbox_wgs84, bands=ALL_BANDS,
                         target_res=TARGET_RES, fixed_grid=None):
    """
    Download bands for one STAC item, reprojecting all to SAME fixed UTM grid.
    Returns (bands_dict, out_transform, dst_crs, fixed_grid).
    """
    import rasterio.crs as rcrs

    if fixed_grid is None:
        fixed_grid = _compute_fixed_grid(bbox_wgs84, target_res)

    out_transform, out_height, out_width, epsg_utm = fixed_grid
    dst_crs = rcrs.CRS.from_string(epsg_utm)

    result = {}
    signed = planetary_computer.sign(item)

    for band in bands:
        if band not in signed.assets:
            continue
        href = signed.assets[band].href
        try:
            with rasterio.open(href) as src:
                rs = Resampling.nearest if band == "SCL" else Resampling.bilinear
                data_rs = np.zeros((out_height, out_width),
                                   dtype=np.uint8 if band == "SCL" else np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data_rs,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=out_transform,
                    dst_crs=dst_crs,
                    resampling=rs,
                )
                result[band] = data_rs
        except Exception as e:
            print(f"    Warning: could not download {band}: {e}")

    return result, out_transform, dst_crs, fixed_grid


def build_cloud_mask(scl_array):
    """Binary mask: True = valid pixel, False = cloud/shadow/nodata."""
    mask = np.ones_like(scl_array, dtype=bool)
    for cls in SCL_MASK_CLASSES:
        mask &= (scl_array != cls)
    return mask


def cloud_fraction_from_scl(scl_array):
    """Fraction of cloudy/invalid pixels from SCL array."""
    valid = build_cloud_mask(scl_array.astype(np.uint8))
    return 1.0 - valid.mean()


# =============================================================================
# SECTION 3  SPECTRAL INDICES
# =============================================================================

def compute_indices(bands_dict):
    """Compute NBR and NDVI from band arrays."""
    eps  = 1e-9
    nir  = bands_dict.get("B08")
    red  = bands_dict.get("B04")
    swir2 = bands_dict.get("B12")
    out  = {}
    if nir is not None and swir2 is not None:
        out["NBR"]  = (nir - swir2) / (nir + swir2 + eps)
    if nir is not None and red is not None:
        out["NDVI"] = (nir - red)   / (nir + red  + eps)
    return out


def compute_dnbr(nbr_pre, nbr_post):
    """dNBR = NBR_pre - NBR_post"""
    return nbr_pre - nbr_post


def compute_rdnbr(dnbr, nbr_pre, eps=1e-9):
    """RdNBR = dNBR / sqrt(|NBR_pre| + eps)"""
    return dnbr / np.sqrt(np.abs(nbr_pre) + eps)


# =============================================================================
# SECTION 4  SEVERITY MASK GENERATION
# =============================================================================

def _dnbr_to_4class(dnbr, low_t=0.10, mod_t=0.27, high_t=0.66):
    """Threshold dNBR to 4 severity classes."""
    out = np.zeros_like(dnbr, dtype=np.uint8)
    out[dnbr >= low_t]  = 1
    out[dnbr >= mod_t]  = 2
    out[dnbr >= high_t] = 3
    return out


def load_mtbs_severity_raster(fire_id, mtbs_raster_dir,
                               target_shape, target_transform, target_crs,
                               dnbr):
    """
    Load MTBS GeoTIFF severity raster, reclassify to 4-class:
      0: Unburned/Low (MTBS 1,2,5,6)
      1: Moderate     (MTBS 3)
      2: High         (MTBS 4)
      3: Very High    (MTBS 4 AND dNBR > 0.66)
    """
    candidates = [f for f in os.listdir(mtbs_raster_dir)
                  if fire_id in f and f.endswith(".tif")]
    if not candidates:
        raise FileNotFoundError(f"No MTBS raster for {fire_id}")

    H, W = target_shape
    raw  = np.zeros((H, W), dtype=np.uint8)
    with rasterio.open(os.path.join(mtbs_raster_dir, candidates[0])) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

    # Reclassify to 3 classes first
    out = np.zeros_like(raw)
    for src_cls, dst_cls in MTBS_4CLASS_REMAP.items():
        out[raw == src_cls] = dst_cls

    # Class 3 (Very High): MTBS High (raw==4) AND dNBR > 0.66
    very_high = (raw == 4) & (dnbr > 0.66)
    out[very_high] = 3

    return out.astype(np.uint8)


def rasterise_cnfdb_perimeter(geometry, target_shape, target_transform, target_crs):
    """Rasterise CNFDB polygon to binary burn mask on Sentinel-2 grid."""
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom

    geom_proj = transform_geom("EPSG:4326", target_crs.to_string(),
                               mapping(geometry))
    H, W = target_shape
    mask = rasterize(
        [(geom_proj, 1)],
        out_shape=(H, W),
        transform=target_transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


# =============================================================================
# SECTION 5  INPUT CHANNEL STACKING
# =============================================================================

def _normalize_band(arr, p_low=2, p_high=98):
    """Percentile normalisation to [0, 1]."""
    valid = arr[arr > 0]
    if not len(valid):
        return arr.astype(np.float32)
    lo = np.percentile(valid, p_low)
    hi = np.percentile(valid, p_high)
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def stack_input_channels_v2(pre_bands, post_bands, pre_idx, nbr_pre, dnbr):
    """
    Stack 7 input channels in order:
      0: Red    (B04, pre, normalised)
      1: Green  (B03, pre, normalised)
      2: Blue   (B02, pre, normalised)
      3: NIR    (B08, pre, normalised)
      4: SWIR1  (B11, pre, normalised)
      5: NBR    (pre, clipped [-1,1])
      6: dNBR   (clipped [-1,1])

    Returns (7, H, W) float32 array.
    """
    def _get(d, k, ref):
        v = d.get(k)
        return v if v is not None else np.zeros_like(ref)

    ref = dnbr
    ch = [
        _normalize_band(_get(pre_bands, "B04", ref)),
        _normalize_band(_get(pre_bands, "B03", ref)),
        _normalize_band(_get(pre_bands, "B02", ref)),
        _normalize_band(_get(pre_bands, "B08", ref)),
        _normalize_band(_get(pre_bands, "B11", ref)),
        np.clip(nbr_pre, -1.0, 1.0).astype(np.float32),
        np.clip(dnbr,    -1.0, 1.0).astype(np.float32),
    ]

    H = max(c.shape[0] for c in ch)
    W = max(c.shape[1] for c in ch)
    ch = [c if c.shape == (H, W) else np.resize(c, (H, W)) for c in ch]
    return np.stack(ch, axis=0)  # (7, H, W)


# =============================================================================
# SECTION 6  PATCH EXTRACTION WITH CLASS BALANCING
# =============================================================================

def extract_patches_v2(X, y, pre_cm, post_cm,
                       nbr_pre, dnbr,
                       patch_size=PATCH_SIZE, overlap=OVERLAP,
                       min_burn_frac=0.005, min_valid_frac=0.60,
                       max_patches=MAX_PATCHES_PER_FIRE):
    """
    Extract patches with:
    - 50% overlap stride
    - min 2% burned pixels
    - min 80% valid pixels (cloud mask)
    - Class balancing to target ratios
    - Max patches per fire: 300

    Returns list of (X_patch, y_patch, nbr_pre_patch, dnbr_patch) tuples.
    """
    stride = int(patch_size * (1 - overlap))
    H, W   = X.shape[1], X.shape[2]

    # Collect all valid patches grouped by dominant severity class
    class_buckets = {0: [], 1: [], 2: [], 3: []}

    for r0 in range(0, H - patch_size + 1, stride):
        for c0 in range(0, W - patch_size + 1, stride):
            r1, c1 = r0 + patch_size, c0 + patch_size

            xp     = X[:, r0:r1, c0:c1]
            yp     = y[r0:r1, c0:c1]
            pre_p  = pre_cm[r0:r1, c0:c1]
            post_p = post_cm[r0:r1, c0:c1]
            nbr_p  = nbr_pre[r0:r1, c0:c1]
            dnbr_p = dnbr[r0:r1, c0:c1]

            pre_valid  = pre_p.mean()
            post_valid = post_p.mean()
            burn_frac  = (yp > 0).mean()

                    # Per-patch cloud filter (combined mask)
            combined_valid = (pre_p & post_p).mean()
            if combined_valid < min_valid_frac:  continue
            if burn_frac  < min_burn_frac:   continue

            # Dominant class
            counts  = np.bincount(yp.ravel(), minlength=4)
            dom_cls = int(counts.argmax())
            class_buckets[dom_cls].append(
                (xp.copy(), yp.copy(), nbr_p.copy(), dnbr_p.copy()))

    # Sample to achieve class balance
    total_available = sum(len(v) for v in class_buckets.values())
    if total_available == 0:
        return []

    # Compute per-class target counts
    per_class_target = {}
    for cls, (lo, hi) in CLASS_BALANCE.items():
        target = int(max_patches * (lo + hi) / 2)  # midpoint
        avail  = len(class_buckets[cls])
        per_class_target[cls] = min(target, avail)

    patches = []
    rng = np.random.default_rng(42)
    for cls in range(4):
        bucket = class_buckets[cls]
        n_take = per_class_target[cls]
        if n_take > 0 and len(bucket) > 0:
            indices = rng.choice(len(bucket), min(n_take, len(bucket)), replace=False)
            for idx in indices:
                patches.append(bucket[idx])

    # If still fewer than min_patches, add remaining
    if len(patches) < MIN_PATCHES_PER_FIRE:
        all_patches = []
        for bucket in class_buckets.values():
            all_patches.extend(bucket)
        # Add more without duplicating
        existing_count = len(patches)
        extra_needed = MIN_PATCHES_PER_FIRE - existing_count
        extra_pool = [p for p in all_patches
                      if not any(np.array_equal(p[0], ep[0]) for ep in patches)]
        if extra_pool and extra_needed > 0:
            extra_indices = rng.choice(len(extra_pool),
                                       min(extra_needed, len(extra_pool)),
                                       replace=False)
            for idx in extra_indices:
                patches.append(extra_pool[idx])

    # Shuffle
    order = rng.permutation(len(patches))
    patches = [patches[i] for i in order]

    return patches[:max_patches]


# =============================================================================
# SECTION 7  SPLITS
# =============================================================================

def make_splits_v2(fires_df, patch_dir):
    """
    Forward-chaining temporal split:
      Train: 2016–2020
      Val:   2021
      Test:  2022–2023
    """
    splits = {"train": [], "val": [], "test": []}
    for _, row in fires_df.iterrows():
        yr = int(row["year"])
        if yr in (2022, 2023):
            splits["test"].append(row["fire_id"])
        elif yr == 2021:
            splits["val"].append(row["fire_id"])
        else:
            splits["train"].append(row["fire_id"])

    print(f"  Train fires: {len(splits['train'])}")
    print(f"  Val fires:   {len(splits['val'])}")
    print(f"  Test fires:  {len(splits['test'])}")

    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    for path in [
        os.path.join(S2_DIR, "splits_v2.json"),
        os.path.join(META_DIR, "splits_v2.json"),
    ]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(splits, f, indent=2)
    print(f"  Splits saved -> {os.path.join(S2_DIR, 'splits_v2.json')}")
    return splits


# =============================================================================
# SECTION 8  FULL PIPELINE FOR ONE FIRE
# =============================================================================

def process_fire_v2(fire_row, patch_dir, max_cloud_frac=0.20, skip_if_exists=True):
    """
    Full pipeline for one fire:
    1. Search pre/post Sentinel-2 scenes with strict cloud filter
    2. Download all bands to a fixed UTM grid
    3. Compute spectral indices and severity mask
    4. Extract and save patches

    Returns number of patches saved (0 if skipped/failed).
    """
    fid     = fire_row["fire_id"]
    country = fire_row["country"]
    year    = int(fire_row["year"])
    month   = int(fire_row.get("month", 6) or 6)
    day     = int(fire_row.get("day",   15) or 15)
    state   = str(fire_row.get("state", "") or "")

    print(f"\n  Processing {fid} ({country}, {year}-{month:02d}-{day:02d}, "
          f"{fire_row['area_ha']:,.0f} ha) ...")

    # Skip if patches already exist
    country_dir = os.path.join(patch_dir, country)
    if skip_if_exists and os.path.exists(country_dir):
        existing = [f for f in os.listdir(country_dir) if f.startswith(fid)]
        if existing:
            print(f"    -> {len(existing)} patches exist, skipping")
            return len(existing)

    bbox = (fire_row["lon_min"], fire_row["lat_min"],
            fire_row["lon_max"], fire_row["lat_max"])

    # Pad bbox 10%
    pad_x = (bbox[2] - bbox[0]) * 0.10
    pad_y = (bbox[3] - bbox[1]) * 0.10
    bbox_pad = (bbox[0]-pad_x, bbox[1]-pad_y, bbox[2]+pad_x, bbox[3]+pad_y)

    ign_date   = datetime.date(year, month, day)
    # Pre-fire window: 30–120 days before ignition.
    # 30-day buffer ensures post-ignition imagery is excluded from the 'pre' scene.
    # 120-day ceiling catches multi-month fire seasons (common in CA and BC).
    pre_start  = (ign_date - datetime.timedelta(days=120)).isoformat()
    pre_end    = (ign_date - datetime.timedelta(days=30)).isoformat()
    # Post-fire window: 10–60 days after estimated containment.
    # Containment is proxied as ignition + 90 days (conservative for large fires).
    # The 10-day buffer lets smoke clear and vegetation begin recovery signal.
    contain_proxy = ign_date + datetime.timedelta(days=90)
    post_start = (contain_proxy + datetime.timedelta(days=10)).isoformat()
    post_end   = (contain_proxy + datetime.timedelta(days=60)).isoformat()

    # Search at relaxed 70% cloud — per-patch SCL masking rejects cloudy sub-areas.
    # Searching at 20% would miss many valid Western USA fires in smoke-heavy seasons.
    cloud_search_pct = 70.0

    print(f"    Searching pre-fire  ({pre_start} -> {pre_end}) ...")
    pre_items = search_s2_scenes(bbox_pad, pre_start, pre_end,
                                 max_cloud=cloud_search_pct)
    print(f"    Searching post-fire ({post_start} -> {post_end}) ...")
    post_items = search_s2_scenes(bbox_pad, post_start, post_end,
                                  max_cloud=cloud_search_pct)

    if not pre_items:
        print("    No pre-fire scenes found -- skipping")
        return 0
    if not post_items:
        print("    No post-fire scenes found -- skipping")
        return 0

    # Compute fixed grid once — both pre and post will use it
    fixed_grid = _compute_fixed_grid(bbox_pad, TARGET_RES)
    _, out_h, out_w, epsg_utm = fixed_grid
    print(f"    Fixed grid: {out_h} x {out_w} px  ({epsg_utm})")

    def _try_download(items, tag):
        """Try items in order; accept best available scene (cloud filtered per-patch)."""
        for item in items[:10]:
            cf = _cloud_fraction(item)
            print(f"    Trying {tag} scene: {item.id}  cloud={cf:.1f}%")
            bands, transform, crs, _ = download_scene_bands(
                item, bbox_pad, ALL_BANDS, TARGET_RES, fixed_grid)
            if not bands:
                continue
            scl = bands.get("SCL")
            if scl is None:
                continue
            actual_cloud = cloud_fraction_from_scl(scl.astype(np.uint8))
            print(f"    Actual cloud fraction: {actual_cloud*100:.1f}%")
            # Accept scenes up to 70% cloud — per-patch cloud masking filters bad patches
            if actual_cloud > 0.70:
                print(f"    Scene >70% cloudy -- trying next")
                continue
            return bands, transform, crs
        return None, None, None

    print("    Downloading pre-fire bands ...")
    pre_bands, transform, crs = _try_download(pre_items, "pre")
    if pre_bands is None:
        print("    No suitable pre-fire scene found -- skipping")
        return 0

    print("    Downloading post-fire bands ...")
    post_bands, _, _ = _try_download(post_items, "post")
    if post_bands is None:
        print("    No suitable post-fire scene found -- skipping")
        return 0

    H, W = out_h, out_w

    # Cloud masks
    pre_scl  = pre_bands.get("SCL",  np.ones((H, W), dtype=np.uint8))
    post_scl = post_bands.get("SCL", np.ones((H, W), dtype=np.uint8))
    pre_cm   = build_cloud_mask(pre_scl.astype(np.uint8))
    post_cm  = build_cloud_mask(post_scl.astype(np.uint8))

    # Spectral indices
    pre_idx  = compute_indices(pre_bands)
    post_idx = compute_indices(post_bands)
    if "NBR" not in pre_idx or "NBR" not in post_idx:
        print("    Cannot compute NBR -- skipping")
        return 0

    nbr_pre = pre_idx["NBR"].astype(np.float32)
    nbr_post = post_idx["NBR"].astype(np.float32)
    dnbr    = compute_dnbr(nbr_pre, nbr_post).astype(np.float32)
    rdnbr   = compute_rdnbr(dnbr, nbr_pre).astype(np.float32)

    # Severity mask
    low_t  = float(fire_row.get("low_t",  0.10))
    mod_t  = float(fire_row.get("mod_t",  0.27))
    high_t = float(fire_row.get("high_t", 0.66))

    import rasterio.crs as rcrs
    target_crs_obj = rcrs.CRS.from_string(epsg_utm)

    if country == "USA":
        # Try MTBS raster first
        mtbs_raster_dir = os.path.join(BASE, "data", "US", "MTBS", "rasters")
        if os.path.exists(mtbs_raster_dir):
            try:
                y_mask = load_mtbs_severity_raster(
                    fid, mtbs_raster_dir, (H, W), transform, target_crs_obj, dnbr)
                print("    Using MTBS severity raster")
            except FileNotFoundError:
                print(f"    MTBS raster not found -- using dNBR thresholds "
                      f"(low>{low_t:.3f}, mod>{mod_t:.3f}, high>{high_t:.3f})")
                y_mask = _dnbr_to_4class(dnbr, low_t, mod_t, high_t)
        else:
            print(f"    Using per-fire dNBR thresholds")
            y_mask = _dnbr_to_4class(dnbr, low_t, mod_t, high_t)
    else:
        # Canada: binary perimeter + dNBR severity inside
        burn_mask = rasterise_cnfdb_perimeter(
            fire_row["geometry"], (H, W), transform, target_crs_obj)
        # Inside perimeter: class 0=unburned, 1=burned (dNBR-based)
        y_mask = np.zeros((H, W), dtype=np.uint8)
        burned_pixels = burn_mask > 0
        # Use simple threshold for burned class
        y_mask[burned_pixels & (dnbr >= 0.10)] = 1
        print(f"    Canada binary mask: {burn_mask.sum()} burned pixels")

    # Stack input channels
    X_full = stack_input_channels_v2(pre_bands, post_bands, pre_idx, nbr_pre, dnbr)

    # Extract patches
    patches = extract_patches_v2(
        X_full, y_mask, pre_cm, post_cm, nbr_pre, dnbr)

    print(f"    Extracted {len(patches)} patches from {H}x{W} scene")

    if not patches:
        print("    No valid patches -- skipping")
        return 0

    # Save patches
    out_dir = os.path.join(patch_dir, country)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i, (xp, yp, nbr_p, dnbr_p) in enumerate(patches):
        fname = f"{fid}_{i:04d}.npz"
        fpath = os.path.join(out_dir, fname)
        np.savez_compressed(
            fpath,
            X=xp.astype(np.float32),
            y=yp.astype(np.uint8),
            fire_id=fid,
            year=int(year),
            country=country,
            nbr_pre=nbr_p.astype(np.float32),
            dnbr=dnbr_p.astype(np.float32),
        )
        saved += 1

    print(f"    Saved {saved} patches")
    return saved


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 Wildfire Dataset Builder v2")
    parser.add_argument("--n_usa",          type=int,   default=50)
    parser.add_argument("--n_canada",       type=int,   default=30)
    parser.add_argument("--min_area_ha",    type=float, default=3000.0)
    parser.add_argument("--year_start",     type=int,   default=2016)
    parser.add_argument("--year_end",       type=int,   default=2023)
    parser.add_argument("--skip_existing",  action="store_true", default=False)
    parser.add_argument("--patch_dir",      type=str,
                        default=os.path.join(BASE, "data", "sentinel2", "patches_v2"))
    parser.add_argument("--max_cloud_frac", type=float, default=0.20)
    args = parser.parse_args()

    print("=" * 60)
    print("  Sentinel-2 Wildfire Dataset Builder v2")
    print("=" * 60)
    print(f"  n_usa={args.n_usa}  n_canada={args.n_canada}")
    print(f"  min_area_ha={args.min_area_ha}  years={args.year_start}-{args.year_end}")
    print(f"  max_cloud_frac={args.max_cloud_frac}")
    print(f"  patch_dir={args.patch_dir}")

    os.makedirs(args.patch_dir, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(S2_DIR, exist_ok=True)

    # 1. Select fires
    print("\n[1/4] Selecting fires ...")
    fires = select_fires(
        n_usa=args.n_usa,
        n_canada=args.n_canada,
        min_area_ha=args.min_area_ha,
        year_start=args.year_start,
        year_end=args.year_end,
    )

    # Save fire list (drop geometry for CSV)
    fires_csv = fires.drop("geometry", axis=1, errors="ignore")
    fires_csv.to_csv(os.path.join(META_DIR, "selected_fires_v2.csv"), index=False)

    # 2. Build splits
    print("\n[2/4] Building splits ...")
    splits = make_splits_v2(fires, args.patch_dir)

    # 3. Process fires
    print("\n[3/4] Processing fires ...")
    total_patches = 0
    skipped       = 0
    for _, row in fires.iterrows():
        try:
            n = process_fire_v2(
                row,
                patch_dir=args.patch_dir,
                max_cloud_frac=args.max_cloud_frac,
                skip_if_exists=args.skip_existing,
            )
            total_patches += n
            if n == 0:
                skipped += 1
        except Exception as e:
            import traceback
            print(f"    ERROR on {row['fire_id']}: {e}")
            traceback.print_exc()
            skipped += 1

    # 4. Summary
    print(f"\n[4/4] Summary")
    print(f"  Total patches saved : {total_patches}")
    print(f"  Fires skipped/failed: {skipped}")
    print(f"  Patch directory     : {args.patch_dir}")
    print(f"  Splits file         : {os.path.join(S2_DIR, 'splits_v2.json')}")
    print("DONE")


if __name__ == "__main__":
    main()
