# Confusion-matrix grid: color FP/FN cells by error rate (FPR/FNR)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import seaborn as sns
from matplotlib import patheffects
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# (TN, FP ; FN, TP)
cm_worm = {
    "SVM":           np.array([[38_450,     32],
                               [ 1_863,    322]], dtype=int),
    "Random Forest": np.array([[37_020,  1_462],
                               [ 1_603,    582]], dtype=int),
    "Logistic Reg.": np.array([[30_718,  7_764],
                               [   117,  2_068]], dtype=int),
    "Grad. Boost":   np.array([[38_475,      7],
                               [   597,  1_588]], dtype=int),
    "LightGBM":      np.array([[36_307,  2_175],
                               [   170,  2_015]], dtype=int),
    "LSTM":          np.array([[15_075,  1_204],
                               [304,     234]], dtype=int),
}

cm_ransom = {
    "SVM":           np.array([[121_202,      0],
                               [  1_503,      4]], dtype=int),
    "Random Forest": np.array([[118_602,   2_600],
                               [    642,     865]], dtype=int),
    "Logistic Reg.": np.array([[ 95_681,  25_521],
                               [    220,   1_287]], dtype=int),
    "Grad. Boost":   np.array([[121_181,      21],
                               [  1_340,     167]], dtype=int),
    "LightGBM":      np.array([[ 114_922,   6_280],
                               [    134,    1_373]], dtype=int),
    "LSTM":          np.array([[ 54_507,   1_896],
                               [    319,     264]], dtype=int),
}

cm_trojan = {
    "SVM":           np.array([[221_376,      69],
                               [  4_623,      39]], dtype=int),
    "Random Forest": np.array([[219_392,   2_053],
                               [    940,   3_722]], dtype=int),
    "Logistic Reg.": np.array([[178_423,  43_022],
                               [    196,   4_466]], dtype=int),
    "Grad. Boost":   np.array([[221_427,      18],
                               [  2_593,   2_069]], dtype=int),
    "LightGBM":      np.array([[215_542,   5_903],
                               [    280,   4_382]], dtype=int),
    "LSTM":          np.array([[107_928,     503],
                               [  1_729,   1_246]], dtype=int),
}
cm_spyware = {
    "SVM":           np.array([[138_383,     28],
                               [  2_286,      6]], dtype=int),
    "Random Forest": np.array([[137_922,    489],
                               [    377,  1_915]], dtype=int),
    "Logistic Reg.": np.array([[114_982, 23_499],
                               [    189,  2_103]], dtype=int),
    "Grad. Boost":   np.array([[138_385,     26],
                               [  1_224,  1_068]], dtype=int),
    "LightGBM":      np.array([[136_731,  1_680],
                               [    140,  2_152]], dtype=int),
    "LSTM":          np.array([[ 89_989,  1_813],
                               [    705,    468]], dtype=int),
}

cm_botnet = {
    "SVM":           np.array([[70_875,      0],
                               [ 1_891,      8]], dtype=int),
    "Random Forest": np.array([[70_456,    419],
                               [   401,  1_498]], dtype=int),
    "Logistic Reg.": np.array([[50_970, 19_905],
                               [   114,  1_785]], dtype=int),
    "Grad. Boost":   np.array([[70_851,     24],
                               [ 1_089,    810]], dtype=int),
    "LightGBM":      np.array([[69_216,  1_659],
                               [   140,  1_759]], dtype=int),
    "LSTM":          np.array([[30_919,    950],
                               [ 1_102,    242]], dtype=int),
}

cm_tool = {
    "SVM":           np.array([[44_532,    187],
                               [ 1_276,135_849]], dtype=int),
    "Random Forest": np.array([[44_512,    207],
                               [   405,136_720]], dtype=int),
    "Logistic Reg.": np.array([[44_574,    145],
                               [ 1_500,135_625]], dtype=int),
    "Grad. Boost":   np.array([[44_521,    198],
                               [ 1_040,136_085]], dtype=int),
    "LightGBM":      np.array([[44_591,    128],
                               [   357,136_768]], dtype=int),                             
    "LSTM":          np.array([[     0,      0],
                               [     0,      0]], dtype=int),
}

cm_apt = {
    "SVM":           np.array([[ 6_742,      0],
                               [   407,      0]], dtype=int),
    "Random Forest": np.array([[ 6_408,    334],
                               [   214,    193]], dtype=int),
    "Logistic Reg.": np.array([[ 4_327,  2_415],
                               [    38,    369]], dtype=int),
    "Grad. Boost":   np.array([[ 6_741,      1],
                               [   396,     11]], dtype=int),
    "LightGBM":      np.array([[ 6_013,    729],
                               [    64,    343]], dtype=int),
    "LSTM":          np.array([[19_684,  1_353],
                               [   382,    236]], dtype=int),
}

TYPES = [
    ("Worm",       cm_worm),
    ("Ransomware", cm_ransom),
    ("Trojan",     cm_trojan),
    ("Spyware",    cm_spyware),
    ("Botnet",     cm_botnet),
    ("Tool",       cm_tool),
    ("APT",        cm_apt),
]

MODELS = ["SVM", "Random Forest", "Logistic Reg.", "Grad. Boost", "LightGBM", "LSTM"]

# ---------------------------
# Settings
# ---------------------------
FIGSIZE     = (18, 6)
DPI_OUT     = 240
CMAP_NAME   = "Reds"          
OUTDIR      = Path("cm_heatmaps_error_rate")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- Seaborn style ----------
sns.set_theme(style="white", context="paper", font_scale=0.95)

# ---------- Helpers ----------
def fpr_fnr(cm):
    tn, fp = cm[0,0], cm[0,1]
    fn, tp = cm[1,0], cm[1,1]
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr, fnr

def tile_rates(cm):
    # [[TN, FP],[FN, TP]] -> put rates only on FP/FN, mask TN/TP
    fpr, fnr = fpr_fnr(cm)
    arr  = np.array([[np.nan, fpr],
                     [fnr,    np.nan]], dtype=float)
    mask = np.isnan(arr)
    return arr, mask

def truncated_cmap(name="Reds", low=0.25, high=1.0, N=256):
    base = mpl.colormaps[name]
    colors = base(np.linspace(low, high, N))
    return LinearSegmentedColormap.from_list(f"{name}_trunc_{low}_{high}", colors)

Reds_darker = truncated_cmap("Reds", low=0.15, high=1.0)

def draw_tile(ax, cm, show_xticks=False, show_yticks=False):
    arr, mask = tile_rates(cm)
    ax.set_facecolor("#c2c2c2")  # TN/TP background via mask
    sns.heatmap(
        arr, ax=ax, mask=mask, cmap=Reds_darker, vmin=0, vmax=1, cbar=False,
        square=False, linewidths=0.5, linecolor="white",
        xticklabels=False, yticklabels=False
    )
    
    for i in range(2):
        for j in range(2):
            ax.text(
            j + 0.5, i + 0.5, f"{cm[i, j]:,}",
            ha="center", va="center", fontsize=11, weight="normal", color="white",
            #bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"),
            transform=ax.transData
)
    # Optional ticks only on the first tile
    # Top ticks for Pred 0/1 on selected axes
    if show_xticks:
        ax.set_xticks([0.5, 1.5], ["Pred 0", "Pred 1"])
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(
            axis="x", which="major",
            labelsize=10, pad=0,
            length=3.0, width=0.8, direction="out", color="black"
        )
        # keep right ticks off
        ax.tick_params(axis="x", which="both", bottom=False)

    # Left ticks for True 0/1 on selected axes
    if show_yticks:
        ax.set_yticks([0.5, 1.5], ["True 0", "True 1"])
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(
            axis="y", which="major",
            labelsize=10, pad=2,
            length=3.0, width=0.8, direction="out", color="black"
        )
        # keep right-side ticks off
        ax.tick_params(axis="y", which="both", right=False)

    for s in ax.spines.values():
        s.set_visible(False)

# ---------- Figure ----------
n_rows = len(TYPES)
n_cols = len(MODELS)
FIGSIZE = (2.2 * n_cols, 1.3 * n_rows)

fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE, constrained_layout=False,
    gridspec_kw={'wspace': 0.04, 'hspace': 0.04} )


if n_rows == 1:
    axes = np.array([axes])  # unify indexing to [r,c]

# Titles on top row; row labels on the left margin
for r, (typename, cmdict) in enumerate(TYPES):
    for c, model in enumerate(MODELS):
        ax = axes[r, c]
        cm = cmdict[model]
        if r == 0:
            ax.set_title(model, fontsize=14, pad=12, color="black")
        show_xticks = (r == 0)
        show_yticks = (c == 0)
        draw_tile(ax, cm, show_xticks, show_yticks)
    # Put type label outside leftmost axes
    axes[r, 0].annotate(
        typename, xy=(-0.42, 0.5), xycoords="axes fraction",
        ha="right", va="center", rotation=90, fontsize=14, color="black"
    )


norm = Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(norm=norm, cmap=Reds_darker); sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02, fraction=0.03, aspect=40)
cbar.set_label("Error rates (FP and FN)", fontsize=12, color="black")
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
for t in cbar.ax.get_yticklabels():
    t.set_color("black")

# ---------- Save ----------
OUTDIR = Path("figures_cm"); OUTDIR.mkdir(parents=True, exist_ok=True)
base = OUTDIR / "cm_grid_error_rates"
fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight", facecolor="white")

print("Saved:", f"{base}.png")
# plt.show()
