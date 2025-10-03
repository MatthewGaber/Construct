"""
Enrich reporter CSVs with IPinfo ASN-Lite.

Behavior:
- Enrich ALL malicious rows (based on 'malicious' column).
- Write new CSVs with '-with-ipinfo-' appended before the extension.

"""

import os
import csv
import time
from typing import Optional, Dict, Any, List
import json
import requests
import ipaddress

# -----------------------------
# Fixed settings (no argparse)
# -----------------------------
INPUT_DIR = "Tool/analysis_output_mimikatz_baseline_labelled"
TOKEN = ""        # <-- put your token here
SLEEP = 1.0              # seconds between IPinfo requests

# Columns add to the CSV
IPINFO_FIELDS = [
    "IP", "City", "Region", "Country", "Postcode",
    "Timezone", "Latitude", "Longitude", "ISP"
]

# ---------------- helpers ----------------
def _is_true(v: str) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "y")

def _is_malicious_row(row: Dict[str, str]) -> bool:
    return _is_true(row.get("malicious", "0"))

def _peer_ip_for_row(row: Dict[str, str]) -> Optional[str]:
    """
    Choose the remote IP:
      - outbound: dst
      - inbound:  src
      - otherwise: prefer dst, else src
    """
    direction = (row.get("direction") or "").lower()
    src = row.get("src") or ""
    dst = row.get("dst") or ""
    if direction == "outbound":
        return dst or None
    if direction == "inbound":
        return src or None
    return (dst or src) or None

def _is_rfc1918_or_local(ip: str) -> bool:
    try:
        ipobj = ipaddress.ip_address(ip)
        return (
            ipobj.is_private or ipobj.is_loopback or ipobj.is_link_local or
            ipobj.is_multicast or ipobj.is_reserved or ipobj.is_unspecified
        )
    except Exception:
        return False

# ---------------- IPinfo client ----------------
class IPInfoClient:
    """Simple ipinfo.io client with in-memory cache and optional debug printing."""

    def __init__(self, token: str, sleep_between: float = SLEEP, debug: bool = True):
        self.token = token or ""
        self.sleep_between = sleep_between
        self.debug = debug
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def lookup(self, ip: str) -> Optional[Dict[str, Any]]:
        if not ip or _is_rfc1918_or_local(ip):
            return None
        if ip in self._cache:
            return self._cache[ip]

        url = f"https://ipinfo.io/{ip}?token={self.token}"

        try:
            
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if self.debug:
                print(f"[DEBUG] Response for {ip}:\n{json.dumps(data, indent=2)}\n")
        except requests.RequestException as e:
            print(f"[WARN] ipinfo error for {ip}: {e}")
            self._cache[ip] = None
            return None

        # Skip bogons reported by IPinfo
        if data.get("bogon") is True:
            self._cache[ip] = None
            return None

        # Extract fields to keep
        loc = data.get("loc", "")
        if loc:
            try:
                latitude, longitude = loc.split(",", 1)
            except ValueError:
                latitude, longitude = ("", "")
        else:
            latitude, longitude = ("", "")

        out = {
            "IP":        data.get("ip") or ip,
            "City":      data.get("city", "") or "",
            "Region":    data.get("region", "") or "",
            "Country":   data.get("country", "") or "",  
            "Postcode":  data.get("postal", "") or "",
            "Timezone":  data.get("timezone", "") or "",
            "Latitude":  latitude,
            "Longitude": longitude,
            "ISP":       data.get("org", "") or "",      
        }

        self._cache[ip] = out
        # Be polite to the API
        try:
            time.sleep(self.sleep_between)
        except Exception:
            pass
        return out

# ---------------- CSV I/O ----------------
def _out_name_with_ipinfo(path: str) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}-with-ipinfo-.{ext.lstrip('.')}"

def enrich_csv(input_csv: str, client: IPInfoClient) -> str:
    output_csv = _out_name_with_ipinfo(input_csv)

    with open(input_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames: List[str] = list(reader.fieldnames or [])

        # Ensure IPinfo columns exist (at the end)
        for col in IPINFO_FIELDS:
            if col not in fieldnames:
                fieldnames.append(col)

        rows_out: List[Dict[str, str]] = []

        for row in reader:
            # Initialize added columns to empty strings
            for col in IPINFO_FIELDS:
                row.setdefault(col, "")

            # Enrich only malicious rows
            if _is_malicious_row(row):
                ip = _peer_ip_for_row(row)
                info = client.lookup(ip) if ip else None
                if info:
                    for k in IPINFO_FIELDS:
                        row[k] = info.get(k, "")

            rows_out.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[+] Wrote {output_csv}")
    return output_csv

# ---------------- main ----------------
def run():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Not a directory: {INPUT_DIR}")
        return

    client = IPInfoClient(token=TOKEN, sleep_between=SLEEP, debug=True)

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        try:
            enrich_csv(path, client)
        except Exception as e:
            print(f"[WARN] Failed on {fname}: {e}")

if __name__ == "__main__":
    run()
