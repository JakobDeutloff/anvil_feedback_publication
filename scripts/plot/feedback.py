# %%
import matplotlib.pyplot as plt
import numpy as np
from src.helper_functions import definitions
from src.read_data import load_feedbacks, load_slopes

# %%
colors, line_labels, linestyles = definitions()
feedbacks, feedbacks_area, feedbacks_opacity = load_feedbacks()
slopes, errors, p_vals = load_slopes()
models = ['icon_ap', 'rcemip', 'xshield', 'icon_amip']
obs = ['ccic', 'spare_ice',  'two_c_ice', 'dardar']

# %% plot feedback
fig, axes = plt.subplots(1, 2, figsize=(10, 4), width_ratios=[3, 0.5])
markers = {
    "icon_ap": "x",
    "rcemip": "x",
    "xshield": "x",
    "ccic": "o",
    "two_c_ice": "o",
    "dardar": "o",
    "spare_ice": "o",
    'icon_amip': "x",
}

for name in models + obs:
    axes[0].plot(
        feedbacks[name].iwp,
        feedbacks[name]/2,
        label=line_labels[name],
        color=colors[name],
        linestyle=linestyles[name],
    )

    axes[1].scatter(
        0,
        feedbacks[name].sum().item()/2,
        color=colors[name],
        marker=markers[name],
        label=line_labels[name],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        1,
        feedbacks_area[name].item()/2,
        color=colors[name],
        marker=markers[name],
        s=80,
        alpha=0.7
    )
    axes[1].scatter(
        2,
        feedbacks_opacity[name].item()/2,
        color=colors[name],
        marker=markers[name],
        s=80,
        alpha=0.7
    )

for ax in axes:
    ax.axhline(0, color="k", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


axes[0].set_xscale("log")
axes[0].set_xlim(1e-3, 2e1)
axes[0].set_ylabel(r"$\lambda(I)$ / W m$^{-2}$ K$^{-1}$")
axes[0].set_xlabel(r"$I$ / kg m$^{-2}$")
axes[0].legend(frameon=False, loc="upper left", ncol=2)
axes[0].set_yticks([-0.01, 0, 0.01])
axes[0].set_ylim(-0.011, 0.015)

axes[1].set_xticks([0, 1, 2])
axes[1].set_xlim(-0.5, 2.5)
axes[1].set_xticklabels(["Total", "Amount", "Optical \n Depth"], rotation=45)
axes[1].set_ylabel(r"$\lambda$ / W m$^{-2}$ K$^{-1}$")
axes[1].set_yticks([-0.02, 0, 0.05, 0.1])
axes[1].set_ylim(-0.04, 0.1)


handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=1, bbox_to_anchor=(1.15, 0.98))

# add letters
for i, ax in enumerate(axes):
    ax.text(0.03, 1, chr(97 + i), transform=ax.transAxes, fontsize=14, fontweight='bold')

fig.tight_layout()
fig.savefig("plots/feedback_monthly.pdf", bbox_inches="tight")

# %% calculate mean and std of feedback 
mean_feedback = np.mean([feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
std_feedback = np.std([feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']])
print(f"Mean feedback: {mean_feedback:.4f} W m^-2 K^-1")
print(f"Std feedback: {std_feedback:.4f} W m^-2 K^-1")
print(f"Feedback for ICON: {feedbacks['icon_ap'].sum().item()/2:.4f} W m^-2 K^-1")
print(f"Feedback for RCEMIP: {feedbacks['rcemip'].sum().item()/2:.4f} W m^-2 K^-1")
    
# %% caclculate feedback fro every satellite
total_feedback = [feedbacks[key].sum().item()/2 for key in ['ccic', 'spare_ice', 'two_c_ice', 'dardar']]

