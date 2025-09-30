import os
import math
import pycountry
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Scan everything under this top-level folder:
INPUT_BASE_DIR = "Spyware"

# Where to save the single combined heatmap:
OUTPUT_PNG_PATH = os.path.join(INPUT_BASE_DIR, "Spyware_heatmap.png")

class MaliciousFlowsCountryHeatmap:
    """
    Create a country-level heatmap of *malicious* flows from IPinfo-enriched reporter CSVs
    found anywhere under INPUT_BASE_DIR (all subfolders).
    """

    def __init__(self, base_dir: str, label_min_count: int = 1):
        self.base_dir = base_dir
        self.label_min_count = label_min_count

        # World boundaries shapefile (Natural Earth 1:110m)
        self.world_shp = "maps/ne_110m_admin_0_countries.shp"

        # Output PNG
        self.map_output_path = OUTPUT_PNG_PATH

        # DataFrames
        self.world = None
        self.country_counts = None  # columns: ['ISO_A3','flow_count']

    # ---------------- Pipeline ----------------
    def process(self):
        self._load_world()
        self._scan_and_load_counts()
        self._merge_world_counts()
        self._render_map()

    # ---------------- Steps ----------------
    def _load_world(self):
        self.world = gpd.read_file(self.world_shp)
        # Add a clean English name column and representative points for labels
        self.world["Country_English"] = self.world["ADMIN"]
        reps = self.world.geometry.representative_point()
        self.world["rep_lon"] = reps.x
        self.world["rep_lat"] = reps.y

    def _scan_and_load_counts(self):
        """
        Walk all subfolders, read CSVs, stack malicious rows, aggregate by country code (2-letter).
        Accepts IPinfo-enriched CSVs; requires 'malicious' and one of:
        'Country', 'IPinfo_Country', 'IPinfo_CountryCode'.
        """
        frames = []

        for root, _, files in os.walk(self.base_dir):
            for fname in files:
                if not fname.lower().endswith(".csv"):
                    continue
                if "-with-ipinfo-" not in fname:  # <-- only process enriched CSVs
                    continue

                path = os.path.join(root, fname)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue

                if "malicious" not in df.columns:
                    continue

                # Filter malicious rows
                m = df[df["malicious"].astype(str).str.lower().isin(["1", "true", "yes", "y"])].copy()
                if m.empty:
                    continue

                # Choose country column (2-letter ISO)
                country_col = None
                for c in ["Country", "IPinfo_Country", "IPinfo_CountryCode"]:
                    if c in m.columns:
                        country_col = c
                        break
                if not country_col:
                    continue

                m[country_col] = m[country_col].astype(str).str.strip().str.upper()
                # Keep only non-empty 2-letter codes
                m = m[m[country_col].str.match(r"^[A-Z]{2}$", na=False)]
                if m.empty:
                    continue

                frames.append(m[[country_col]])

        if not frames:
            # No data → create empty counts
            self.country_counts = pd.DataFrame(columns=["ISO_A3", "flow_count"])
            return

        all_mal = pd.concat(frames, ignore_index=True)

        # Convert alpha-2 → alpha-3 for merge with shapefile
        def a2_to_a3(code):
            try:
                c = pycountry.countries.get(alpha_2=code.upper())
                return c.alpha_3 if c else None
            except Exception:
                return None

        all_mal["ISO_A3"] = all_mal.iloc[:, 0].apply(a2_to_a3)
        all_mal = all_mal.dropna(subset=["ISO_A3"])

        self.country_counts = (
            all_mal.groupby("ISO_A3", as_index=False)
                   .size()
                   .rename(columns={"size": "flow_count"})
        )

    def _merge_world_counts(self):
        # Merge counts into world geom
        self.world = self.world.merge(self.country_counts, on="ISO_A3", how="left")
        self.world["flow_count"] = self.world["flow_count"].fillna(0).astype(int)
        self.world["log_flow_count"] = self.world["flow_count"].apply(lambda x: math.log1p(x))

    def _render_map(self):
        fig, ax = plt.subplots(1, 1, figsize=(28, 16))

        # Guard against empty data (all zeros)
        vmin = float(self.world["log_flow_count"].min()) if len(self.world) else 0.0
        vmax = float(self.world["log_flow_count"].max()) if len(self.world) else 1.0
        if vmin == vmax:
            vmax = vmin + 1e-6

        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Spectral_r

        # Plot world polygons shaded by log flows
        self.world.plot(
            column="log_flow_count",
            cmap=cmap,
            linewidth=0.6,
            ax=ax,
            edgecolor="0.8",
            legend=False  # we'll draw a custom colorbar
        )

        # Custom colorbar with readable tick labels on dark background
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # newer Matplotlib prefers set_array over _A hack
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.02)
        cbar.set_label("Malicious Flows (log scale)", fontsize=12, color="black")

        max_raw = int(self.world["flow_count"].max()) if len(self.world) else 0
        if max_raw > 0:
            # nice ticks: 0, 1, 10, 100, max
            tick_vals = [0, 1]
            if max_raw >= 10: tick_vals.append(10)
            if max_raw >= 100: tick_vals.append(100)
            if max_raw not in tick_vals: tick_vals.append(max_raw)
        else:
            tick_vals = [0]
        tick_locs = [math.log1p(v) for v in tick_vals]
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([str(v) for v in tick_vals])
        cbar.ax.tick_params(labelcolor="black")

        # Labels for countries above a threshold
        for _, row in self.world.iterrows():
            if row["flow_count"] >= self.label_min_count:
                ax.annotate(
                    text=row["Country_English"],
                    xy=(row["rep_lon"], row["rep_lat"]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

        ax.set_title("Malicious Flows by Country (All Ransomware folders)", fontsize=18, color="black")
        ax.axis("off")

        
        plt.savefig(self.map_output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"[+] Static map saved to {self.map_output_path}")


# ---------------- main ----------------
def run():
    heatmap = MaliciousFlowsCountryHeatmap(INPUT_BASE_DIR, label_min_count=1)
    heatmap.process()

if __name__ == "__main__":
    run()
