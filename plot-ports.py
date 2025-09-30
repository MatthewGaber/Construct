#!/usr/bin/env python3
"""
nonstandard_ports.py

Detect and visualize non-standard port use from your flow CSVs.

- Loads benign CSVs from a flat folder and malware CSVs from nested subfolders.
- Canonicalizes app from app_proto (fallback to protocol if specific).
- Flags a flow as "non-standard" if NEITHER src nor dst port matches the
  expected set for that app (default). Optionally require destination-only.
- Produces:
    A) Bar: non-standard flow COUNT per family
    B) Bar: non-standard flow PERCENT per family
    C) Heatmap: non-standard COUNT by Family × App
    D) Bar: top non-standard ports observed (global)
- Always orders families with "Benign" first.

Usage:
  python nonstandard_ports.py \
      --benign analysis_output_benign_baseline_labelled \
      --spyware Spyware \
      --dest-only            # (optional) check dst port only
      --top-k 20             # (optional) top K ports to show (default 20)
      --save-prefix out/ns   # (optional) save PNGs/CSVs with this prefix
"""

import os
from glob import glob
import argparse
from typing import Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ----------------------------
# Loaders
# ----------------------------
def load_csvs_flat(folder_path: str, label: str = 'Benign') -> pd.DataFrame:
    files = glob(os.path.join(folder_path, '*.csv'))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['family'] = label
            df['__srcfile'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] loading {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_csvs_nested_all(parent_folder: str) -> pd.DataFrame:
    family_frames = []
    for entry in os.scandir(parent_folder):
        if not entry.is_dir():
            continue
        family = os.path.basename(entry.path)
        files = glob(os.path.join(entry.path, '*.csv'))
        fam_dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                df['family'] = family
                df['__srcfile'] = os.path.join(family, os.path.basename(f))
                fam_dfs.append(df)
                print(f"{f} → loaded {len(df)} rows")
            except Exception as e:
                print(f"[ERROR] loading {f}: {e}")
        if fam_dfs:
            family_frames.append(pd.concat(fam_dfs, ignore_index=True))
    return pd.concat(family_frames, ignore_index=True) if family_frames else pd.DataFrame()


# ----------------------------
# Helpers / Canonicalization
# ----------------------------
TRANSPORT_TOKENS = {'TCP', 'UDP', 'ICMP', 'SCTP', 'ARP', 'OTHER', 'UNKNOWN', 'DATA'}

# Expected ports per application (upper-cased keys)
EXPECTED_PORTS = {
    'HTTP': {80, 8080, 8000, 8081, 8888, 3128},
    'HTTP-ALT': {8080, 8000, 8081, 8888, 80},
    'HTTPS': {443, 8443, 10443},
    'HTTPS-ALT': {8443, 10443, 443},
    'HTTP/3': {443},               # QUIC typically UDP/443
    'DNS': {53},
    'MDNS': {5353},
    'LLMNR': {5355},
    'NBNS': {137},
    'DHCP': {67, 68}, 'BOOTP': {67, 68},
    'SSH': {22},
    'FTP': {21}, 'FTP-DATA': {20},
    'FTPS': {990}, 'FTPS-DATA': {989},
    'SMTP': {25, 587}, 'SMTPS': {465}, 'SMTP-SUBMISSION': {587, 25},
    'POP3': {110}, 'POP3S': {995},
    'IMAP': {143}, 'IMAPS': {993},
    'RDP': {3389},
    'VNC': {5900},
    'SMB': {445, 137, 138, 139},
    'NTP': {123},
    'BGP': {179},
    'LDAP': {389}, 'LDAPS': {636},
    'SNMP': {161}, 'SNMP-TRAP': {162},
    'RTSP': {554},
    'SSDP': {1900},
    'ISAKMP': {500},
    'IPSEC-NAT-T': {4500},
    
}

APP_SYNONYM = {
    'HTTP2': 'HTTP', 'HTTP/2': 'HTTP',
    'HTTP3': 'HTTP/3', 'QUIC': 'HTTP/3',  # treat QUIC as HTTP/3 semantics
    # 'TLS'/'SSL' are generic; omit unless you add EXPECTED_PORTS entries.
    # 'TLS': {...}, 'SSL': {...}
}


def _to_int_port(v) -> Optional[int]:
    try:
        p = int(str(v))
        return p if 0 <= p <= 65535 else None
    except Exception:
        return None


def _canon_app(app_proto: str, protocol: str) -> Optional[str]:
    a = (str(app_proto) or "").strip().upper()
    p = (str(protocol) or "").strip().upper()

    # Ignore artifacts / transports
    if a in TRANSPORT_TOKENS or a == '':
        a = ''
    if p in TRANSPORT_TOKENS or p == 'VSSMONITORING':
        p = ''

    # Prefer app_proto; else fallback to protocol if specific
    candidate = a if a else p
    candidate = APP_SYNONYM.get(candidate, candidate)

    return candidate if candidate in EXPECTED_PORTS else None


def detect_nonstandard(row: pd.Series, dest_only: bool = False) -> Optional[int]:
    app = row.get('app_detected')
    if not app:
        return np.nan  # unknown app → ignore
    sp, dp = row.get('sport_i'), row.get('dport_i')
    if sp is None and dp is None:
        return np.nan
    exp: Set[int] = EXPECTED_PORTS.get(app, set())
    if dest_only:
        return int(not (dp in exp))
    return int(not ((sp in exp) or (dp in exp)))


# ----------------------------
# Plotters
# ----------------------------
def plot_ns_counts(ns_counts: pd.Series, family_order, save_prefix: Optional[str]):
    plt.figure(figsize=(max(12, 1.2*len(ns_counts)), 5.5))
    sns.barplot(x=ns_counts.index, y=ns_counts.values)
    plt.title("Non-standard Port Use — Count per Family")
    plt.xlabel("Family")
    plt.ylabel("Non-standard flow count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_ns_count_per_family.png", dpi=150)
    plt.show()


def plot_ns_percent(ns_pct: pd.Series, save_prefix: Optional[str]):
    plt.figure(figsize=(max(12, 1.2*len(ns_pct)), 5.5))
    sns.barplot(x=ns_pct.index, y=ns_pct.values)
    plt.title("Non-standard Port Use — % of Family Flows")
    plt.xlabel("Family")
    plt.ylabel("% non-standard flows")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_ns_percent_per_family.png", dpi=150)
    plt.show()


def plot_ns_heatmap(apps_ct: pd.DataFrame, save_prefix: Optional[str]):
    if apps_ct.empty:
        print("[INFO] No non-standard rows to plot for Family×App heatmap.")
        return
    # Order apps by global frequency
    apps_ct = apps_ct[apps_ct.sum().sort_values(ascending=False).index]
    plt.figure(figsize=(max(10, 0.28*apps_ct.shape[1]), max(6, 0.6*apps_ct.shape[0])))
    sns.heatmap(apps_ct, annot=True, fmt="g", cmap="Reds", linewidths=.5,
                cbar_kws={"label": "Non-standard count"})
    plt.title("Non-standard Port Use — Counts by Family × App")
    plt.xlabel("App (from app_proto/protocol)")
    plt.ylabel("Family")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_ns_heatmap_family_app.png", dpi=150)
    plt.show()


def plot_top_ports(ns_ports: pd.Series, top_k: int, save_prefix: Optional[str]):
    ns_ports = ns_ports.dropna()
    if ns_ports.empty:
        print("[INFO] No non-standard ports to plot.")
        return
    top_ports = ns_ports.astype(int).value_counts().head(top_k)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_ports.index.astype(str), y=top_ports.values)
    plt.title(f"Top Non-standard Ports Observed (Top {top_k})")
    plt.xlabel("Port")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_top_nonstandard_ports.png", dpi=150)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Detect & plot non-standard port use.")
    parser.add_argument("--benign", default="analysis_output_benign_baseline_labelled",
                        help="Folder with benign *_sessions.csv (flat)")
    parser.add_argument("--spyware", default="Spyware",
                        help="Parent folder with malware family subfolders containing *_sessions.csv")
    parser.add_argument("--dest-only", action="store_true",
                        help="Count as standard only if DEST port matches expected (default: either side matches)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top K non-standard ports to display")
    parser.add_argument("--save-prefix", default=None,
                        help="If set, save plots/CSVs using this prefix (e.g., out/ns)")
    args = parser.parse_args()

    # Load data
    df_benign = load_csvs_flat(args.benign, label='Benign')
    df_spyware = load_csvs_nested_all(args.spyware)

    assert not df_benign.empty, "Benign data is empty"
    assert not df_spyware.empty, "Spyware data is empty"

    df = pd.concat([df_benign, df_spyware], ignore_index=True)

    # Family order: Benign first, then others by frequency
    counts = df['family'].value_counts()
    if 'Benign' in counts.index:
        family_order = ['Benign'] + list(counts.drop('Benign').index)
    else:
        family_order = list(counts.index)

    # Prepare fields
    for col in ['app_proto', 'protocol']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].astype(str).fillna('').str.strip()

    df['sport_i'] = df.get('sport', np.nan).apply(_to_int_port)
    df['dport_i'] = df.get('dport', np.nan).apply(_to_int_port)
    df['app_detected'] = df.apply(lambda r: _canon_app(r.get('app_proto'), r.get('protocol')), axis=1)

    # Detect non-standard
    df['nonstandard_port'] = df.apply(lambda r: detect_nonstandard(r, dest_only=args.dest_only), axis=1)

    # Keep only confirmed non-standard rows (==1)
    ns = df[df['nonstandard_port'] == 1].copy()

    # Summary prints
    print("\n==== Summary ====")
    print("Total rows:", len(df))
    print("Rows with recognized app:", df['app_detected'].notna().sum())
    print("Non-standard rows:", len(ns))
    print("\nNon-standard by family:")
    print(ns['family'].value_counts().reindex(family_order, fill_value=0).to_string())

    if args.save_prefix:
        out_csv = f"{args.save_prefix}_nonstandard_rows.csv"
        ns.to_csv(out_csv, index=False)
        print(f"[Saved] Full non-standard rows → {out_csv}")

    # Plot A: count per family
    ns_counts = ns.groupby('family').size().reindex(family_order, fill_value=0)
    plot_ns_counts(ns_counts, family_order, args.save_prefix)

    # Plot B: percent per family
    tot_counts = df.groupby('family').size().reindex(family_order, fill_value=0)
    ns_pct = (ns_counts / tot_counts.replace(0, np.nan) * 100).fillna(0)
    if args.save_prefix:
        ns_counts.to_csv(f"{args.save_prefix}_ns_counts_per_family.csv")
        ns_pct.to_csv(f"{args.save_prefix}_ns_percent_per_family.csv")
    plot_ns_percent(ns_pct, args.save_prefix)

    # Plot C: heatmap Family × App
    apps_ct = pd.crosstab(ns['family'], ns['app_detected']).reindex(index=family_order)
    if args.save_prefix:
        apps_ct.to_csv(f"{args.save_prefix}_ns_counts_family_app.csv")
    plot_ns_heatmap(apps_ct, args.save_prefix)

    # Plot D: top non-standard ports (prefer dport)
    ns_ports = ns.assign(port_used=ns['dport_i'].fillna(ns['sport_i']))['port_used']
    if args.save_prefix:
        ns_ports.dropna().astype(int).value_counts().to_csv(f"{args.save_prefix}_top_nonstandard_ports.csv")
    plot_top_ports(ns_ports, args.top_k, args.save_prefix)


if __name__ == "__main__":
    main()
