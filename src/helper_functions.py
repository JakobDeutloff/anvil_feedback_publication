"""Numerical helpers shared by the publication scripts."""

import numpy as np
import xarray as xr


def nan_detrend(da, dim="iwp"):
    """Remove a linear trend along ``dim`` while preserving missing values."""
    out = xr.zeros_like(da)
    for coordinate in da[dim]:
        values = da.sel({dim: coordinate}).values
        mask = np.isfinite(values)
        if np.sum(mask) > 1:
            x_values = np.arange(len(values))
            slope, intercept = np.polyfit(x_values[mask], values[mask], 1)
            out.loc[{dim: coordinate}] = values - (slope * x_values + intercept)
    return out


def interpolate_bins(hist, new_bins, name_old_bins):
    """Interpolate a histogram onto new bins using a log-space CDF."""
    cdf = hist.cumsum(name_old_bins)
    cdf[name_old_bins] = np.log10(hist[name_old_bins])
    cdf_interpolated = cdf.interp({name_old_bins: np.log10(new_bins)}).rename(
        {name_old_bins: "iwp"}
    )
    pdf_interpolated = cdf_interpolated.diff("iwp")
    pdf_interpolated["iwp"] = (new_bins[1:] + new_bins[:-1]) / 2
    return pdf_interpolated

def definitions():

    colors = {
        "icon_ap_control": "#1f948a",
        "rcemip_control": "#ff7f0e",
        "dardar": "#B4A806",
        "two_c_ice": "#006496", 
        "ccic": "purple",
        "spare_ice": "#035827",
        "xshield_control": "#FB06BA",
        "era5": "#03DD6C",
        'icon_amip_control': "#2B00FF",
    }

    labels = {
        "icon_ap_control": "ICON AP",
        "rcemip_control": "RCEMIP",
        "dardar": "DARDAR",
        "two_c_ice": "2C-ICE",
        "ccic": "CCIC",
        "spare_ice": "SPARE-ICE",
        "xshield_control": "X-SHiELD AMIP",
        "era5": "ERA5",
        'icon_amip_control': "ICON-HAMlite AMIP",
    }

    linestyles = {
        "icon_ap_control": "--",
        "rcemip_control": "--",
        "dardar": "-",
        "two_c_ice": "-",
        "ccic": "-",
        "spare_ice": "-",
        "xshield_control": "--",
        "era5": "-",
        'icon_amip_control': "--",
    }

    # use same identifyers for cahnge in simulations
    for name in ['rcemip', 'icon_ap', 'xshield', 'icon_amip']:
        colors[name] = colors[f"{name}_control"]
        labels[name] = labels[f"{name}_control"]
        linestyles[name] = linestyles[f"{name}_control"]

    return colors, labels, linestyles