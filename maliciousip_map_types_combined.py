import os
import math
import pycountry
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.patheffects as pe

# ---------------- CONFIG ----------------
# Map each malware family name to its top-level folder (scan all subfolders within)
MALWARE_DIRS: Dict[str, str] = {
    # "FamilyName": "FolderPath"
    "Spyware": "Spyware",
    "APT": "APT",
    "Ransomware": "Ransomware",
    "Trojan": "Trojan",
    "Worm": "Worm",
    "Botnet": "Botnet",
    "Tool": "Tool",


}
    

# Only process CSVs whose filename contains this token to ensure IPinfo-enriched files
REQUIRED_FILENAME_TOKEN = "-with-ipinfo-"

# Natural Earth shapefile 
WORLD_SHP = "maps/ne_110m_admin_0_countries.shp"

# Where to save the combined PNG
OUTPUT_PNG_PATH = "combined_malware_heatmap.png"

# Label / pie configuration
LABEL_MIN_COUNT = 20                     # annotate country name when total >= this
TOP_N_PIE_COUNTRIES = 40                # draw pies for top N countries by total
MIN_TOTAL_FOR_PIE = 20                   # don't draw pies if total < this
FIXED_PIE_RADIUS = 2
PIE_EDGE_COLOR = "white"
PIE_EDGE_WIDTH = 0.5
FAMILY_CMAP = plt.cm.Set2               

# ----------------- HELPER -----------------
def alpha2_to_alpha3(a2: str) -> str:
    try:
        c = pycountry.countries.get(alpha_2=a2.upper())
        return c.alpha_3 if c else None
    except Exception:
        return None

def safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def draw_pie(ax, x: float, y: float, values: List[float], colors: List[Tuple[float,float,float]], 
             radius: float, edgecolor: str = "white", edgewidth: float = 0.5):
    total = float(sum(values))
    if total <= 0:
        return
    angles = np.cumsum([0] + [v / total * 360.0 for v in values])
    for i in range(len(values)):
        if values[i] <= 0:
            continue
        wedge = Wedge(center=(x, y),
                      r=radius,
                      theta1=angles[i],
                      theta2=angles[i+1],
                      facecolor=colors[i],
                      edgecolor=edgecolor,
                      linewidth=edgewidth)
        ax.add_patch(wedge)

def size_scale(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if vmax <= vmin:
        return (out_min + out_max) / 2.0
    # sqrt scaling for readability
    t = (math.sqrt(value) - math.sqrt(vmin)) / (math.sqrt(vmax) - math.sqrt(vmin))
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)

# --------------- MAIN CLASS ---------------
class CombinedMalwareHeatmap:
    """
    Single PNG:
      - Base choropleth by total malicious flows (log scale)
      - Pie markers on top-N countries showing family composition; size ~ total
    """

    def __init__(self,
                 malware_dirs: Dict[str, str],
                 world_shp: str = WORLD_SHP,
                 output_png: str = OUTPUT_PNG_PATH):
        self.malware_dirs = malware_dirs
        self.world_shp = world_shp
        self.output_png = output_png

        # Data holders
        self.world = None            # GeoDataFrame with polygons
        self.country_family = None   # DataFrame: ISO_A3, family, flow_count
        self.country_totals = None   # DataFrame: ISO_A3, total
        self.centroids = None        # DataFrame: ISO_A3, rep_lon, rep_lat

        # Families & colors
        self.families = list(self.malware_dirs.keys())
        colors = FAMILY_CMAP.colors if hasattr(FAMILY_CMAP, "colors") else [FAMILY_CMAP(i) for i in range(len(self.families))]
        if len(colors) < len(self.families):
            # repeat if needed
            mult = int(np.ceil(len(self.families) / len(colors)))
            colors = (colors * mult)[:len(self.families)]
        self.family_colors = dict(zip(self.families, colors))

    # ---------------- Pipeline ----------------
    def process(self):
        self._load_world()
        self._scan_all_families()
        self._merge_world()
        self._render_map()

    # ---------------- Steps ----------------
    def _load_world(self):
        world = gpd.read_file(self.world_shp)
        world["Country_English"] = world["ADMIN"]
        # ensure we have ISO_A3 column to merge on (Natural Earth commonly uses 'ADM0_A3' or 'iso_a3')
        if "ISO_A3" not in world.columns:
            # Try common fields
            if "ADM0_A3" in world.columns:
                world["ISO_A3"] = world["ADM0_A3"]
            elif "iso_a3" in world.columns:
                world["ISO_A3"] = world["iso_a3"]
            else:
                raise ValueError("Could not determine ISO_A3 column from shapefile.")
        reps = world.geometry.representative_point()
        world["rep_lon"] = reps.x
        world["rep_lat"] = reps.y
        # ensure WGS84 for plotting in lon/lat
        if world.crs is None:
            world.set_crs(epsg=4326, inplace=True)
        elif world.crs.to_epsg() != 4326:
            world = world.to_crs(epsg=4326)

        self.world = world
        self.centroids = world[["ISO_A3", "rep_lon", "rep_lat"]].copy()

    def _scan_family_dir(self, family: str, base_dir: str) -> pd.DataFrame:
        frames = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if not fname.lower().endswith(".csv"):
                    continue
                if REQUIRED_FILENAME_TOKEN and REQUIRED_FILENAME_TOKEN not in fname:
                    continue
                df = safe_read_csv(os.path.join(root, fname))
                if df.empty:
                    continue
                if "malicious" not in df.columns:
                    continue

                m = df[df["malicious"].astype(str).str.lower().isin(["1", "true", "yes", "y"])].copy()
                if m.empty:
                    continue

                # pick country column (ISO2)
                country_col = None
                for c in ["Country", "IPinfo_Country", "IPinfo_CountryCode"]:
                    if c in m.columns:
                        country_col = c
                        break
                if not country_col:
                    continue

                m[country_col] = m[country_col].astype(str).str.strip().str.upper()
                m = m[m[country_col].str.match(r"^[A-Z]{2}$", na=False)]
                if m.empty:
                    continue

                tmp = m[[country_col]].copy()
                tmp["family"] = family
                frames.append(tmp)

        if not frames:
            return pd.DataFrame(columns=["ISO_A3", "family", "flow_count"])

        fam_df = pd.concat(frames, ignore_index=True)
        fam_df["ISO_A3"] = fam_df.iloc[:, 0].apply(alpha2_to_alpha3)
        fam_df = fam_df.dropna(subset=["ISO_A3"])
        fam_counts = (fam_df.groupby(["ISO_A3", "family"], as_index=False)
                             .size()
                             .rename(columns={"size": "flow_count"}))
        return fam_counts[["ISO_A3", "family", "flow_count"]]

    def _scan_all_families(self):
        parts = []
        for family, base_dir in self.malware_dirs.items():
            if not os.path.isdir(base_dir):
                continue
            fam_counts = self._scan_family_dir(family, base_dir)
            if not fam_counts.empty:
                parts.append(fam_counts)
        if parts:
            self.country_family = pd.concat(parts, ignore_index=True)
        else:
            self.country_family = pd.DataFrame(columns=["ISO_A3", "family", "flow_count"])

        # totals
        if self.country_family.empty:
            self.country_totals = pd.DataFrame(columns=["ISO_A3", "total"])
        else:
            self.country_totals = (self.country_family.groupby("ISO_A3", as_index=False)["flow_count"]
                                                 .sum()
                                                 .rename(columns={"flow_count": "total"}))

    def _merge_world(self):
        self.world = self.world.merge(self.country_totals, on="ISO_A3", how="left")
        self.world["total"] = self.world["total"].fillna(0).astype(int)
        self.world["log_total"] = self.world["total"].apply(lambda x: math.log1p(x))

    def _render_map(self):
        fig, ax = plt.subplots(1, 1, figsize=(28, 16))

        # Choropleth setup
        vmin = float(self.world["log_total"].min()) if len(self.world) else 0.0
        vmax = float(self.world["log_total"].max()) if len(self.world) else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Spectral_r

        # Base map
        self.world.plot(
            column="log_total",
            cmap=cmap,
            linewidth=0.6,
            ax=ax,
            edgecolor="0.8",
            legend=False
        )

        # Colorbar with raw tick labels
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.02)
        cbar.set_label("Malicious Flows (log scale)", fontsize=12, color="black")

        max_raw = int(self.world["total"].max()) if len(self.world) else 0
        ticks_raw = [0, 1]
        if max_raw >= 10: ticks_raw.append(10)
        if max_raw >= 100: ticks_raw.append(100)
        if max_raw >= 1000: ticks_raw.append(1000)
        if max_raw not in ticks_raw: ticks_raw.append(max_raw)
        ticks_raw = sorted(set([t for t in ticks_raw if t <= max_raw])) or [0]
        cbar.set_ticks([math.log1p(v) for v in ticks_raw])
        cbar.set_ticklabels([str(v) for v in ticks_raw])
        cbar.ax.tick_params(labelcolor="black")

        # default label offset (replicates your current "+3 lat")
        DEFAULT_OFFSET = (0, 2.8)  # (d_lon, d_lat), degrees

        # per-country custom offsets
        OFFSETS = {
            "United Kingdom": (1.5, 4),   # lon +3, lat +1
            "Netherlands": (1, 4),
            "Moldova": (-1.5, 2.8),
        }

        # Country labels (optional)
        for _, row in self.world.iterrows():
            if row["total"] >= LABEL_MIN_COUNT:
                dlon, dlat = OFFSETS.get(row["Country_English"], DEFAULT_OFFSET)
                label_lon = row["rep_lon"] + dlon
                label_lat = row["rep_lat"] + dlat

                t = ax.text(
                    label_lon,
                    label_lat,
                    row["Country_English"],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                
                )
                t.set_path_effects([pe.withStroke(linewidth=0.75, foreground="white")])

        # ---- Pies for composition on top-N countries ----
        if not self.country_family.empty:
            # pivot to wide for quick access
            pivot = self.country_family.pivot_table(index="ISO_A3", columns="family",
                                                    values="flow_count", aggfunc="sum",
                                                    fill_value=0).reset_index()
            pivot = pivot.merge(self.country_totals, on="ISO_A3", how="left")
            pivot = pivot.merge(self.centroids, on="ISO_A3", how="left")
            pivot = pivot.sort_values("total", ascending=False)

            # restrict to top-N with total >= threshold
            top = pivot[pivot["total"] >= MIN_TOTAL_FOR_PIE].head(TOP_N_PIE_COUNTRIES).copy()

            if not top.empty:
                family_cols = self.families  # keep order stable
                slice_colors = [self.family_colors[f] for f in family_cols]

                for _, r in top.iterrows():
                    vals = [float(r.get(f, 0.0)) for f in family_cols]
                    if sum(vals) <= 0:
                        continue

                    draw_pie(ax, r["rep_lon"], r["rep_lat"], vals, slice_colors, radius=FIXED_PIE_RADIUS,
                              edgecolor=PIE_EDGE_COLOR, edgewidth=PIE_EDGE_WIDTH)

                    # scale radius by total malicious count
                    #size = size_scale(
                        #r["total"],
                        #vmin=top["total"].min(),
                        #vmax=top["total"].max(),
                        #out_min=0.6,   # smallest pies
                        #out_max=3.2    # largest pies
                    #)

                    #draw_pie(ax, r["rep_lon"], r["rep_lat"], vals, slice_colors,
                            #radius=size,
                            #edgecolor=PIE_EDGE_COLOR, edgewidth=PIE_EDGE_WIDTH)
                    


                # Legend for family colors
                family_handles = [Line2D([0], [0], marker="o", color="w",
                                         markerfacecolor=self.family_colors[f], markersize=10,
                                         label=f) for f in family_cols]



                leg1 = ax.legend(handles=family_handles, title="Malware Type",
                                 loc="lower left", bbox_to_anchor=(0.01, 0.02), frameon=True)
                ax.add_artist(leg1)


        ax.set_title("Malicious Flows by Country — Total + Composition",
                     fontsize=18, color="black")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        ax.axis("off")

        plt.savefig(self.output_png, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[+] Static map saved to {self.output_png}")


# ---------------- main ----------------
def run():
    heatmap = CombinedMalwareHeatmap(MALWARE_DIRS, WORLD_SHP, OUTPUT_PNG_PATH)
    heatmap.process()

if __name__ == "__main__":
    run()
