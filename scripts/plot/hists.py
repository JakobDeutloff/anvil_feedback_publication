# %%
import matplotlib.pyplot as plt
from src.helper_functions import definitions
from src.read_data import load_cre, load_histograms, load_slopes


# %% load data
colors, line_labels, linestyles = definitions()
hists_model = load_histograms('model')
hists_obs = load_histograms('obs')
slopes, errors, pvals = load_slopes()
cre = load_cre()
models_control = ['icon_ap_control', 'rcemip_control', 'xshield_control', 'icon_amip_control']
models = ['icon_ap', 'rcemip', 'xshield', 'icon_amip']
obs = ['ccic', 'spare_ice',  'two_c_ice', 'dardar']

# %% filter 2c_ice and dadar data for size
hists_obs["two_c_ice"] = hists_obs["two_c_ice"].where(
    hists_obs["two_c_ice"]["size"] > 1.9e6
)
hists_obs["dardar"] = hists_obs["dardar"].where(hists_obs["dardar"]["size"] > 1.9e6)

# %% calculate mean hists for obs
for key in hists_obs.keys():
    hists_obs[key] = (hists_obs[key]["hist"] / hists_obs[key]["size"])

# %% concat hists of obs and models
hists = {}
for key in hists_obs.keys():
    hists[key] = hists_obs[key]
hists["rcemip_control"] = hists_model["rcemip_control"]
hists["icon_ap_control"] = hists_model["icon_ap_control"]
hists["xshield_control"] = hists_model["xshield_control"]
hists["icon_amip_control"] = hists_model["icon_amip_control"]

# %% calculate mean cre 
mean_cre = {}
for name in models_control:
    mean_cre[name] = (cre['net'] * hists[name]).sum().item()

for name in obs:
    mean_cre[name] = (cre['net'] * hists[name].mean('time')).sum().item()

# %% plot all distributions and cre for 2016 
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=False, height_ratios=[0.15, 3, 1])


# plot cloud type lables on ax0
position = {
    "Cirrus": 3e-3,
    "Anvil": 1e-1,
    "Deep Convection": 5}

for label, xpos in position.items():
    axes[0].text(
        xpos,
        1.3,
        label,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=axes[0].get_xaxis_transform()
    )

import numpy as np
from matplotlib.collections import LineCollection

# Smooth black -> gray -> light gray gradient in log10(IWP) space
x = np.logspace(-3, np.log10(20), 4000)
y = np.full_like(x, 0.5)

log_x = np.log10(x)

# Transition centers: 1e-2 and 1e0
sharpness = 30
transition_1 = 1 / (1 + np.exp(-sharpness * (log_x + 2)))
transition_2 = 1 / (1 + np.exp(-sharpness * log_x))

# Grayscale values: black -> gray -> light gray
gray = 0.833 - 0.33 * transition_1 - 0.5 * transition_2

points = np.column_stack([x, y]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

color = np.column_stack([gray[:-1], gray[:-1], gray[:-1]])

line = LineCollection(
    segments,
    colors=color,
    linewidths=5,
    transform=axes[0].transData,
)
axes[0].add_collection(line)

for name in models_control:
    axes[1].plot(
        hists[name].iwp,
        hists[name],
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

for name in obs:
    axes[1].plot(
        hists[name].iwp,
        hists[name].sel(time='2016').mean('time'),
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

axes[1].set_ylim(0, 0.013)

axes[2].axhline(0, color="k", linewidth=0.5)
axes[2].plot(
    cre.iwp,
    cre['net'],
    color='k',
)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim([1e-3, 2e1])
    ax.set_xscale("log")

# add mean cre values to legend labels 
labels = []
for name in models_control + obs:
    labels.append(line_labels[name] + f", {mean_cre[name]:.1f} W m$^{{-2}}$")

handles, _ = axes[1].get_legend_handles_labels()

axes[1].legend(frameon=False, labels=labels, bbox_to_anchor=(0.72, 1.1),)
axes[1].set_ylabel(r"$P(I)$")
axes[2].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[2].set_ylabel(r"$C(I)$ / W m$^{-2}$")
axes[2].set_yticks([-100, 0, 40])
axes[1].set_yticks([0, 0.006, 0.012])
axes[0].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].xaxis.set_major_locator(plt.NullLocator())
axes[0].xaxis.set_minor_locator(plt.NullLocator())

fig.tight_layout()
#add letters
for i, ax in enumerate(axes[1:]):
    ax.text(0.02, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.savefig("plots/distributions_cre_2016.pdf", bbox_inches="tight")

# %% 
