# rank_overall_performance.py
# ------------------------------------------------------------
# Ranks the OVERALL performance of each model for each malware
# type using *all* metrics (Precision, Recall, F1) and Support.
#
# How the composite score is computed (per malware type, per model):
#   1) Support-weight across classes:
#        P_overall = (P0*S0 + P1*S1) / (S0 + S1)
#        R_overall = (R0*S0 + R1*S1) / (S0 + S1)       # == overall accuracy
#        F1_overall= (F10*S0 + F11*S1) / (S0 + S1)
#   2) Combine metrics into a single score (balanced emphasis):
#        Composite = (P_overall ** wP) * (R_overall ** wR) * (F1_overall ** wF)) ** (1/(wP+wR+wF))
#      (geometric mean resists one metric dominating)
#
# You can adjust weights wP, wR, wF if needed.
#
# Output:
#   • A single heatmap: rows = malware types, cols = models.
#     Each cell shows the RANK (1 = best) and the composite score.
#   • Also saves PNG/PDF.
# ------------------------------------------------------------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# ---------- Global font sizes ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ---------------- Entities 
models = ["SVM", "Random Forest", "Log. Reg.", "Grad. Boost", "LightGBM", "LSTM"]
types  = ["Worm", "Ransomware", "Trojan", "Spyware", "Botnet", "Tool", "APT"]

def k(mal, cls): return f"{mal}-{cls}"

# ---------------- Data 
precision = {
    k("Worm",0):       [0.95378662, 0.958496233, 0.996205611, 0.984720516, 0.99534,       0.98023278],
    k("Worm",1):       [0.90960452, 0.284735812, 0.210333605, 0.995611285, 0.480907,      0.162726],

    k("Ransomware",0): [0.98775111, 0.994616081, 0.997705968, 0.989063099, 0.998835,      0.994181593],
    k("Ransomware",1): [1.0,        0.24963925,  0.048008057, 0.888297872, 0.179407,      0.122222222],

    k("Trojan",0):     [0.979544157,0.995733711, 0.998902692, 0.988425141, 0.998703,      0.984232652],
    k("Trojan",1):     [0.361111111,0.644502165, 0.094044811, 0.99137518,  0.426057,      0.71240708],

    k("Spyware",0):    [0.983749085,0.997274022, 0.998357964, 0.991232657, 0.998977,      0.9922266],
    k("Spyware",1):    [0.176470588,0.796589018, 0.08214202,  0.976234004, 0.561587,      0.2051731],

    k("Botnet",0):     [0.974012588,0.994340714, 0.997768381, 0.984862385, 0.9979814291,  0.965585],
    k("Botnet",1):     [1.0,        0.781429317, 0.082295989, 0.971223022, 0.514628437682,0.203020],

    k("Tool",0):       [0.972144604,0.990983369, 0.967443678, 0.97717346,  0.992057,      0.0],
    k("Tool",1):       [0.998625364,0.998488246, 0.998932017, 0.998547141, 0.999065,      0.0],

    k("APT",0):        [0.943068961,0.967683479, 0.991294387, 0.944514502, 0.9894684,     0.980963],
    k("APT",1):        [0.0,        0.366223909, 0.132543103, 0.916666667, 0.319962686,   0.1485210],
}

recall = {
    k("Worm",0):       [0.999168442, 0.962008212, 0.798243335, 0.999818097, 0.94348,       0.92604],
    k("Worm",1):       [0.147368421, 0.266361556, 0.946453089, 0.726773455, 0.922197,      0.4349442],

    k("Ransomware",0): [1.0,         0.978548209, 0.789434168, 0.999826736, 0.948186,      0.966384767],
    k("Ransomware",1): [0.00265428,  0.573988056, 0.854014599, 0.110816191, 0.911082,      0.452830189],

    k("Trojan",0):     [0.99968841,  0.990729075, 0.805721511, 0.999918716, 0.973343,      0.99536110],
    k("Trojan",1):     [0.008365508, 0.798369798, 0.957957958, 0.443800944, 0.93994,       0.41882352],

    k("Spyware",0):    [0.999797704, 0.996467044, 0.830223031, 0.999812154, 0.987862,      0.98025],
    k("Spyware",1):    [0.002617801, 0.835514834, 0.917539267, 0.465968586, 0.938918,      0.398976982],

    k("Botnet",0):     [1.0,         0.994088183, 0.719153439, 0.999661376, 0.976592592,   0.97019],
    k("Botnet",1):     [0.004212744, 0.78883623,  0.939968404, 0.426540284, 0.9262769878,  0.18005952],

    k("Tool",0):       [0.995818332, 0.995371095, 0.99675753,  0.995572352, 0.997138,      0.0],
    k("Tool",1):       [0.990694622, 0.99704649,  0.989061076, 0.992415679, 0.997397,      0.0],

    k("APT",0):        [1.0,         0.950459804, 0.641797686, 0.999851676, 0.891872,      0.93568474],
    k("APT",1):        [0.0,         0.474201474, 0.906633907, 0.027027027, 0.842751,      0.3818770],
}

f1 = {
    k("Worm",0):       [0.975950251, 0.960249011, 0.886304947, 0.992211878, 0.968716,      0.95236591],
    k("Worm",1):       [0.253643167, 0.275242374, 0.34417908,  0.84021164,  0.632157,      0.2368421],

    k("Ransomware",0): [0.993837815, 0.986516723, 0.881434158, 0.994415792, 0.972852,      0.980086129],
    k("Ransomware",1): [0.005294507, 0.347948512, 0.09090588,  0.197050147, 0.299782,      0.192489974],

    k("Trojan",0):     [0.989513772, 0.993225089, 0.891972284, 0.994138709, 0.98586,       0.9897655],
    k("Trojan",1):     [0.016352201, 0.713231772, 0.171275168, 0.613127871, 0.586338,      0.5275190],

    k("Spyware",0):    [0.991708471, 0.99687037,  0.906560636, 0.995503921, 0.993389,      0.986202],
    k("Spyware",1):    [0.005159071, 0.815587734, 0.150785115, 0.630832841, 0.702809,      0.270990],

    k("Botnet",0):     [0.986835235, 0.994214433, 0.835854672, 0.992206701, 0.9871711675,  0.9678822],
    k("Botnet",1):     [0.008390142, 0.785115304, 0.151341727, 0.592755214, 0.6616513071,  0.190851],

    k("Tool",0):       [0.983839076, 0.993172386, 0.981881863, 0.986287107, 0.994591,      0.0],
    k("Tool",1):       [0.994644184, 0.997766847, 0.993972041, 0.995471969, 0.99823,       0.0],

    k("APT",0):        [0.970700454, 0.958994313, 0.779148285, 0.971395634, 0.9381387,     0.957788],
    k("APT",1):        [0.0,         0.413276231, 0.231275462, 0.052505967, 0.46382691,    0.2138649750],
}

support = {
    k("Worm",0):       [38482, 38482, 38482, 38482, 38482, 16279],
    k("Worm",1):       [2185,  2185,  2185,  2185,  2185,  538],

    k("Ransomware",0): [121202,121202,121202,121202,121202, 56403],
    k("Ransomware",1): [1507,  1507,  1507,  1507,  1507,   583],

    k("Trojan",0):     [221445,221445,221445,221445,221445, 108431],
    k("Trojan",1):     [4662,  4662,  4662,  4662,  4662,   2975],

    k("Spyware",0):    [138411,138411,138411,138411,138411, 91802],
    k("Spyware",1):    [2292,  2292,  2292,  2292,  2292,   1173],

    k("Botnet",0):     [70875, 70875, 70875, 70875, 70875,  31869],
    k("Botnet",1):     [1899,  1899,  1899,  1899,  1899,   1344],

    k("Tool",0):       [44719, 44719, 44719, 44719, 44719,  0],
    k("Tool",1):       [137125,137125,137125,137125,137125, 0],

    k("APT",0):        [6742,  6742,  6742,  6742,  6742,   21037],
    k("APT",1):        [407,   407,   407,   407,   407,    618],
}

# ---------------- Scoring parameters ----------------
""" wP, wR, wF = 1.0, 1.0, 1.0   # equal weights for Precision, Recall(Accuracy), and F1

def sw_metric(mt, j, metric_dict):
    S0, S1 = support[k(mt,0)][j], support[k(mt,1)][j]
    tot = S0 + S1
    if tot == 0: return np.nan
    return (metric_dict[k(mt,0)][j]*S0 + metric_dict[k(mt,1)][j]*S1) / tot

def composite(mt, j):
    P, R, F = sw_metric(mt,j,precision), sw_metric(mt,j,recall), sw_metric(mt,j,f1)
    vals = [v for v in [P,R,F] if not (v is None or np.isnan(v))]
    if not vals: return np.nan
    if any(v <= 0 for v in vals): return float(np.mean(vals))
    return (P**wP * R**wR * F**wF) ** (1.0/(wP+wR+wF))

scores = np.zeros((len(types), len(models)))
for ti, mt in enumerate(types):
    for mj, _m in enumerate(models):
        scores[ti, mj] = composite(mt, mj)
 """

#MCC
EPS = 1e-12  # for numerical safety

def mcc_from_recall_support(R_ben, R_mal, B, M, eps=EPS):
    # Reconstruct confusion counts from recall (R_ben=TNR, R_mal=TPR) and supports
    TP = R_mal * M
    FN = M - TP
    TN = R_ben * B
    FP = B - TN

    denom = np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) + eps
    return float(((TP*TN) - (FP*FN)) / denom)

def composite(mt, j):

    # supports
    B = support[k(mt, 0)][j]  # benign / "not this family"
    M = support[k(mt, 1)][j]  # malicious / "this family"
    tot = B + M
    if tot == 0:
        return np.nan

    # recalls
    R_ben = recall[k(mt, 0)][j]  # TNR
    R_mal = recall[k(mt, 1)][j]  # TPR

    # handle degenerate rows gracefully
    if (B == 0 and M == 0) or not np.isfinite(R_ben) or not np.isfinite(R_mal):
        return np.nan

    return mcc_from_recall_support(R_ben, R_mal, B, M)

# unchanged outer loop
scores = np.zeros((len(types), len(models)))
for ti, mt in enumerate(types):
    for mj, _m in enumerate(models):
        scores[ti, mj] = composite(mt, mj)
scores_rank = np.where(np.isnan(scores), -np.inf, scores)

# ranks per row (1 = best), ties → average rank
ranks = np.zeros_like(scores_rank, dtype=float)
for i in range(ranks.shape[0]):
    row = scores_rank[i]
    order = np.argsort(-row)
    sorted_vals = row[order]
    rvals = np.empty_like(sorted_vals, dtype=float)
    start = 0
    while start < len(sorted_vals):
        end = start
        while end+1 < len(sorted_vals) and np.isclose(sorted_vals[end+1], sorted_vals[start], atol=1e-12):
            end += 1
        avg = (start+1 + end+1)/2.0
        rvals[start:end+1] = avg
        start = end+1
    inv = np.empty_like(rvals); inv[order] = rvals
    ranks[i] = inv

# --------- Discrete colormap (6 bins) ---------
palette = sns.color_palette("RdYlGn_r", 6)  # reversed so low values (1) map to green, high (6) to red
cmap = ListedColormap(palette)

bounds = np.arange(0.5, 6.5 + 1e-9, 1.0)  # bins: (0.5,1.5], (1.5,2.5], ..., (5.5,6.5]
norm = BoundaryNorm(bounds, cmap.N)

# Annotations: "rank • score"
#annot = [[("NA" if not np.isfinite(scores[i,j]) else f"{int(round(ranks[i,j]))} • {scores[i,j]:.3f}")
          #for j in range(ranks.shape[1])] for i in range(ranks.shape[0])]

# Build annot with bold rank via mathtext
def build_annot(scores, ranks):
    annot = []
    for i in range(ranks.shape[0]):
        row = []
        for j in range(ranks.shape[1]):
            if not np.isfinite(scores[i, j]):
                row.append("NA")
            else:
                rank_str  = int(round(ranks[i, j]))
                score_str = f"{scores[i, j]:.3f}"
                # bold only the rank; no TeX needed, mathtext handles \mathbf{}
                row.append(rf"$\mathbf{{{rank_str}}}$ • {score_str}")
        annot.append(row)
    return annot

annot = build_annot(scores, ranks)
# --------- Plot (discrete heatmap of ranks) ---------
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")

hm = sns.heatmap(
    ranks,
    ax=ax,
    cmap=cmap, norm=norm,
    vmin=1, vmax=6,
    cbar=True,
    cbar_kws={"ticks": [1,2,3,4,5,6]},
    linewidths=0.7, linecolor="lightgray",  # light borders
    square=False,
    xticklabels=models, yticklabels=types,
    annot=False
)
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=45,
                   ha='right',
                   rotation_mode='anchor')

for i in range(ranks.shape[0]):
    for j in range(ranks.shape[1]):
        ax.text(
            j + 0.5, i + 0.5, annot[i][j],
            ha="center", va="center",
            fontsize=15, color="black",
            
        )


ax.set_title("Rank • MCC Score", fontsize=16, pad=12)

# Make sure ticks are 14
# Give a bit of padding and keep y labels horizontal
ax.tick_params(axis='x', labelsize=16, pad=6)
ax.tick_params(axis='y', labelsize=16, pad=6)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Colorbar label
cbar = hm.collections[0].colorbar
cbar.ax.minorticks_off()         
cbar.ax.invert_yaxis()  
cbar.set_label("Rank", fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()
plt.savefig("overall_rank_heatmap_seaborn_discrete.png", dpi=300, bbox_inches="tight")
