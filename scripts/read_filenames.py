import xarray as xr

files = {
    "fronts_700": "/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5/fronts/cdf.700hPa/1979/F1979_01.nc",
    "anticyclone": "/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5/maxcl/cdf/1979/A1979_01.nc",
    "cyclone": "/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5/mincl/cdf/1979/C1979_01.nc",
    "wcb": "/g/data/if69/cj0591/GC26_energy_synoptics/data/weatherfeatures.era5/wcb/cdf.1hourly/1980/hit_1980_01.nc",
}

ds = xr.open_dataset(files["anticyclone"])
print("lat:", ds['lat'].values[:5], "...", ds['lat'].values[-5:])
print("lon:", ds['lon'].values[:5], "...", ds['lon'].values[-5:])

ds = xr.open_dataset(files["fronts_700"])
print(ds)
print(ds.attrs)

ds2 = xr.open_dataset(files["wcb"])
print(ds2)
print(ds2.attrs)