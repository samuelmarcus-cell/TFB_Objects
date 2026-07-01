
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os


TFB_CSV = "/g/data/gb02/sm5259/TFB_Objects/TFB_days.csv"
FIG_DIR = "/g/data/gb02/sm5259/TFB_Objects/analysis/figures"
WEATHER_OBJECT_DIR = "/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5"

CLIM_START = "1986-01-01"
CLIM_END = "2022-12-31"
FIRE_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4]

# SE Australia bounding box
LON_BOUNDS = slice(140, 154)
LAT_BOUNDS = slice(-39, -28)
MAP_EXTENT = [140, 154, -39, -28]

WEATHER_OBJECTS = {
    "Cyclone":      {"path": f"{WEATHER_OBJECT_DIR}/mincl/cdf/*/C*.nc",         "var": "INPUT"},
    "Anticyclone":  {"path": f"{WEATHER_OBJECT_DIR}/maxcl/cdf/*/A*.nc",         "var": "FLAG"},
    "Front 700":    {"path": f"{WEATHER_OBJECT_DIR}/fronts/cdf.700hPa/*/F*.nc", "var": "FRONT"},
    "Front 850":    {"path": f"{WEATHER_OBJECT_DIR}/fronts/cdf.850hPa/*/F*.nc", "var": "FRONT"},
    "WCB inflow":   {"path": f"{WEATHER_OBJECT_DIR}/wcb/cdf.1hourly/*/hit_*.nc","var": "GT800"},
    "WCB ascent":   {"path": f"{WEATHER_OBJECT_DIR}/wcb/cdf.1hourly/*/hit_*.nc","var": "MIDTROP"},
    "WCB outflow":  {"path": f"{WEATHER_OBJECT_DIR}/wcb/cdf.1hourly/*/hit_*.nc","var": "LT400"},
}


def assign_weatherfeature_coords(raw_ds):

    raw_ds = raw_ds.squeeze()

    y_dim_name = list(raw_ds.dims)[1]
    x_dim_name = list(raw_ds.dims)[2]
    n_lon = raw_ds.sizes[x_dim_name]

    lat_values = np.arange(-90, 90.5, 0.5)
    lon_values = -180 + 0.5 * np.arange(n_lon)

    raw_ds = raw_ds.assign_coords({y_dim_name: (y_dim_name, lat_values)})
    raw_ds = raw_ds.assign_coords({x_dim_name: (x_dim_name, lon_values)})
    raw_ds = raw_ds.rename({y_dim_name: "latitude", x_dim_name: "longitude"})

    if raw_ds.longitude.size != 720:
        raw_ds = raw_ds.sel(longitude=np.arange(-180, 180, 0.5))

    return raw_ds

def plot_three_panel_map(climatology_field, tfb_composite_field, anomaly_field,
                         object_name, n_tfb_days):

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    panels = [
        (climatology_field,    f"Climatology\n(Oct–Apr, 1986–2022)",    "YlOrRd"),
        (tfb_composite_field,  f"TFB composite (n={n_tfb_days})",       "YlOrRd"),
        (anomaly_field,        "Anomaly (TFB − climatology)",           "RdBu_r"),
    ]

    for ax, (field, title, cmap) in zip(axes, panels):
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle="--")

        plot_kwargs = {"cmap": cmap, "transform": ccrs.PlateCarree()}

        if "Anomaly" in title:
            max_absolute_value = float(np.nanmax(np.abs(field.values)))
            plot_kwargs["vmin"] = -max_absolute_value
            plot_kwargs["vmax"] = max_absolute_value
        else:
            plot_kwargs["vmin"] = 0

        mesh = ax.pcolormesh(
            field.longitude, field.latitude, field.values, **plot_kwargs
        )
        ax.set_title(title, fontsize=11)
        plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06,
                     label="Daily occurrence frequency")

    fig.suptitle(f"Weather Object Composite: {object_name}", fontsize=14, y=1.02)
    fig.tight_layout()

    safe_name = object_name.lower().replace(" ", "_")
    output_path = f"{FIG_DIR}/composite_anomaly_{safe_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load TFB dates and filter to climatology period
    tfb_df = pd.read_csv(TFB_CSV, parse_dates=["Date"])
    all_tfb_dates = pd.DatetimeIndex(tfb_df["Date"])
    tfb_dates_in_clim_period = all_tfb_dates[
        (all_tfb_dates >= CLIM_START) & (all_tfb_dates <= CLIM_END)
    ]
    print(f"TFB dates in climatology period: {len(tfb_dates_in_clim_period)} "
          f"of {len(all_tfb_dates)} total")

    for object_name, object_info in WEATHER_OBJECTS.items():
        print(f"\n=== {object_name} ===")

        # Open all years of files at once
        print("  Opening files...")
        raw_ds = xr.open_mfdataset(
            object_info["path"], combine="by_coords", chunks={"time": 744}
        )
        raw_ds = assign_weatherfeature_coords(raw_ds)
        raw_ds = raw_ds.sel(latitude=LAT_BOUNDS, longitude=LON_BOUNDS)
        raw_ds = raw_ds.sel(time=slice(CLIM_START, CLIM_END))

        # Convert to binary hourly presence (1 = object present, 0 = absent)
        hourly_presence = (raw_ds[object_info["var"]] != 0).astype(float)

        # Aggregate to daily: 1 if object present any hour that day, else 0
        print("  Computing daily presence...")
        daily_presence = hourly_presence.resample(time="1D").max()

        # Restrict to fire-season months (Oct-Apr)
        month_numbers = daily_presence.time.dt.month
        fire_season_daily_presence = daily_presence.sel(
            time=month_numbers.isin(FIRE_SEASON_MONTHS)
        )

        # Climatology: mean occurrence across all fire-season days
        print("  Computing climatology...")
        climatology_field = fire_season_daily_presence.mean(dim="time").compute()

        # TFB composite: mean occurrence across TFB days only
        print("  Computing TFB composite...")
        tfb_day_presence = daily_presence.sel(
            time=tfb_dates_in_clim_period, method="nearest"
        )
        tfb_composite_field = tfb_day_presence.mean(dim="time").compute()

        # Anomaly: where is the object more/less frequent on TFB days?
        anomaly_field = tfb_composite_field - climatology_field

        n_tfb_days = len(tfb_dates_in_clim_period)
        n_clim_days = len(fire_season_daily_presence.time)
        print(f"  Fire-season climatology days: {n_clim_days}")
        print(f"  TFB days used: {n_tfb_days}")

        # Plot three-panel map
        print("  Plotting...")
        plot_three_panel_map(
            climatology_field, tfb_composite_field, anomaly_field,
            object_name, n_tfb_days
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
