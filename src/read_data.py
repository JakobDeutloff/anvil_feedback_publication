"""Readers for the input and intermediate data used by the publication scripts."""

from pathlib import Path
import numpy as np
import xarray as xr


def get_path():
    """Return the local directory containing the downloaded input data."""
    return Path("/work/bu1562/m301049/iwp_dists")


def load_histograms(set="all"):
    """Load, normalize, and harmonize observational and model histograms."""
    path = get_path()

    hists = {}
    obs = ["two_c_ice", "dardar", "ccic", "spare_ice"]
    models = [
        "icon_ap_control",
        "icon_ap_plus4K",
        "icon_ap_plus2K",
        "icon_amip_control",
        "icon_amip_plus4K",
        "rcemip_control",
        "rcemip_plus10K",
        "xshield_control",
        "xshield_plus4K",
    ]

    for name in obs:
        hists[name] = xr.open_dataset(f"{path}/{name}_hist.nc")
    for name in models:
        hists[name] = xr.open_dataarray(f"{path}/{name}_hist.nc")

    hists_norm = {}
    if set == "all":
        hists_norm["two_c_ice"] = hists["two_c_ice"].where(
            hists["two_c_ice"]["size"] > 1.9e6
        )
        hists_norm["dardar"] = hists["dardar"].where(hists["dardar"]["size"] > 1.9e6)
        for key in obs:
            hists_norm[key] = hists[key]["hist"] / hists[key]["size"]
        for key in models:
            hists_norm[key] = hists[key]
        return hists_norm
    elif set == "obs":
        for key in obs:
            hists_norm[key] = hists[key]
        return hists_norm
    elif set == "model":
        for key in models:
            hists_norm[key] = hists[key]
        return hists_norm
    else:
        raise ValueError(
            f"Invalid set: {set}. Must be 'all', 'obs', or 'model'."
        )


def load_slopes():
    """Load monthly histogram slopes, errors, and p-values."""
    path = get_path()
    return (
        xr.open_dataset(f"{path}/slopes_monthly.nc"),
        xr.open_dataset(f"{path}/errors_monthly.nc"),
        xr.open_dataset(f"{path}/p_vals_monthly.nc"),
    )


def load_cre():
    """Load CRE data interpolated onto the observational IWP bins."""
    path = get_path()
    cre = xr.open_dataset(
        f"{path}/jed0011_cre_raw.nc"
    )
    observational_histograms = load_histograms("obs")
    cre["iwp"] = np.log10(cre["iwp"])
    cre = cre.interp(iwp=np.log10(observational_histograms["ccic"].iwp)).drop_vars(
        "iwp"
    )
    cre["iwp"] = observational_histograms["ccic"].iwp
    return cre


def load_feedbacks():
    """Load total, area, and opacity feedback datasets."""
    path = get_path()
    return (
        xr.open_dataset(f"{path}/feedback.nc"),
        xr.open_dataset(f"{path}/feedback_area.nc"),
        xr.open_dataset(f"{path}/feedback_opacity.nc"),
    )


def read_era5_temperature(type='all'):
    """Read gridded and tropical monthly ERA5 temperature."""
    path = get_path()
    tropical_temperature = xr.open_dataset( f"{path}/t2m_tropics_{type}.nc")["t2m"]
    return tropical_temperature

def read_era5_temperature_2d():
    """Read gridded monthly ERA5 temperature."""
    path = get_path()
    temperature_2d = xr.open_dataset(f"{path}/t2m.nc")
    temperature_2d = (temperature_2d.sel(latitude=slice(30, -30)).rename({"valid_time": "time"}))["t2m"]
    return temperature_2d


def read_era5_histograms():
    """Read the monthly ERA5 IWP histogram and normalize it."""
    path = get_path()
    histogram = xr.open_dataset(f"{path}/iwp_hist_monthly_interpolated_all_weighted.nc")
    histogram = histogram.sum("local_time")
    return histogram["hist"] / histogram["size"]
