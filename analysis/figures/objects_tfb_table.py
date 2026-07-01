"""
tfb_weather_objects.py

Identify which synoptic weather objects (fronts 700/850hPa, anticyclone,
cyclone, WCB) are present over southeast Australia on a fixed list of
Victorian Total Fire Ban (TFB) dates.

Output: a tidy presence table, one row per TFB date, one column per
object type (1 = present somewhere in the SE Australia bbox that day,
0 = not present, NaN = no data file for that year/month).

"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TFB_CSV = Path("/home/565/sm5259/.jupyter-root/g/data/gb02/sm5259/TFB_Objects/TFB_days.csv")
OUTPUT_CSV = Path("tfb_weather_object_presence.csv")

BASE_DIR = Path("/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5")

REGION_BBOX = {
    "lat_min": -39.0,
    "lat_max": -28.0,
    "lon_min": 140.0,
    "lon_max": 154.0,
}


LAT_361 = np.linspace(-90.0, 90.0, 361)          # 0.5 deg spacing
LON_720 = np.arange(-180.0, 180.0, 0.5)           # 720 pts, no dateline dup
LON_721 = np.arange(-180.0, 180.5, 0.5)           # 721 pts, dateline included


OBJECT_CONFIG = {
    "fronts_700": {
        "dir_template": "fronts/cdf.700hPa/{year}",
        "file_template": "F{year}_{month:02d}.nc",
        "dims": ("dimz_INPUT", "dimy_INPUT", "dimx_INPUT"),
        "var": "FRONT",
        "lon_grid": "720",
    },
    "fronts_850": {
        "dir_template": "fronts/cdf.850hPa/{year}",
        "file_template": "F{year}_{month:02d}.nc",
        "dims": ("dimz_INPUT", "dimy_INPUT", "dimx_INPUT"),
        "var": "FRONT",
        "lon_grid": "720",
    },
    "anticyclone": {
        "dir_template": "maxcl/cdf/{year}",
        "file_template": "A{year}_{month:02d}.nc",
        "dims": None,  # already has lat/lon coords (index-based, need overwrite)
        "var": "FLAG",
        "lon_grid": "720",
    },
    "cyclone": {
        "dir_template": "mincl/cdf/{year}",
        "file_template": "C{year}_{month:02d}.nc",
        "dims": ("dimz_INPUT",),  # lat/lon already present, just dimz to squeeze
        "var": "LABEL",
        "lon_grid": "720",
    },
    "wcb": {
        "dir_template": "wcb/cdf.1hourly/{year}",
        "file_template": "hit_{year}_{month:02d}.nc",
        "dims": ("dimz_N", "dimy_N", "dimx_N"),
        "var": "TOTAL",
        "lon_grid": "721",
    },
}



def load_tfb_dates(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    return df.sort_values("Date").reset_index(drop=True)


def prepare_dataset(ds: xr.Dataset, cfg: dict) -> xr.Dataset:
    """Attach real lat/lon coordinates and squeeze singleton dims."""
    lon_arr = LON_720 if cfg["lon_grid"] == "720" else LON_721

    if cfg["dims"] is not None and len(cfg["dims"]) == 3:
        z_dim, y_dim, x_dim = cfg["dims"]
        ds = ds.squeeze(z_dim, drop=True)
        ds = ds.rename({y_dim: "lat", x_dim: "lon"})
        ds = ds.assign_coords(lat=LAT_361, lon=lon_arr)
    elif cfg["dims"] is not None and len(cfg["dims"]) == 1:
        (z_dim,) = cfg["dims"]
        ds = ds.squeeze(z_dim, drop=True)
        ds = ds.assign_coords(lat=LAT_361, lon=lon_arr)
    else:
        ds = ds.assign_coords(lat=LAT_361, lon=lon_arr)

    return ds


def crop_to_region(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(
        lat=slice(REGION_BBOX["lat_min"], REGION_BBOX["lat_max"]),
        lon=slice(REGION_BBOX["lon_min"], REGION_BBOX["lon_max"]),
    )


def open_month(obj_name: str, cfg: dict, year: int, month: int) -> xr.Dataset | None:
    fdir = BASE_DIR / cfg["dir_template"].format(year=year)
    fpath = fdir / cfg["file_template"].format(year=year, month=month)
    if not fpath.exists():
        return None
    ds = xr.open_dataset(fpath)
    ds = prepare_dataset(ds, cfg)
    ds = crop_to_region(ds)
    return ds


def presence_for_date(ds: xr.Dataset, var: str, date: pd.Timestamp) -> int:
    day_ds = ds.sel(time=date.strftime("%Y-%m-%d"))
    if var not in day_ds.data_vars or day_ds[var].size == 0:
        return np.nan
    return int(bool((day_ds[var] != 0).any().values))


def print_value_check(obj_name: str, cfg: dict, ds: xr.Dataset):
    """Print a quick value-distribution sanity check for the chosen variable."""
    var = cfg["var"]
    vals = ds[var].values
    print(f"  [check] {obj_name}.{var}: min={np.nanmin(vals):.3g} "
          f"max={np.nanmax(vals):.3g} nonzero_frac={np.mean(vals != 0):.3%}")


def main():
    tfb = load_tfb_dates(TFB_CSV)
    out = tfb[["Date"]].copy()
    out["date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out = out.drop(columns=["Date"])
    for obj_name in OBJECT_CONFIG:
        out[obj_name] = np.nan

    checked_once = set()

    for (year, month), group in tfb.groupby(["year", "month"]):
        for obj_name, cfg in OBJECT_CONFIG.items():
            ds = open_month(obj_name, cfg, year, month)
            if ds is None:
                print(f"[WARN] {obj_name}: no file for {year}-{month:02d} — "
                      f"{len(group)} date(s) will be NaN")
                continue

            if obj_name not in checked_once:
                print_value_check(obj_name, cfg, ds)
                checked_once.add(obj_name)

            for date in group["Date"]:
                presence = presence_for_date(ds, cfg["var"], date)
                out.loc[out["date"] == date.strftime("%Y-%m-%d"), obj_name] = presence

            ds.close()

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[DONE] Wrote {len(out)} rows to {OUTPUT_CSV}")
    n_missing = out.drop(columns=["date"]).isna().all(axis=1).sum()
    if n_missing:
        print(f"[NOTE] {n_missing} date(s) had no data for ANY object type "
              f"(likely outside 1979/1980-2022 coverage)")


if __name__ == "__main__":
    main()