# baseline_cache.py
"""
NOT USED- FUTURE FOR SPEED
"""

import os, json
from typing import Dict, Any, Set, Tuple

from session_extractor import extract_sessions
from feature_engine import enrich_sessions_with_features
from config import BENIGN_DOMAINS

# ---- config ----
BASELINE_PCAP = "Baseline/NoSample-Background.pcap"
BASELINE_CACHE_PATH = "Baseline/baseline_cache.json"

# ---- helpers ----
def _feat(s: Dict[str, Any]) -> Dict[str, Any]:
    return s.get("features", {}) or {}

def _endswith_any(host: str, suffixes) -> bool:
    host = (host or "").lower()
    return any(host.endswith(sfx) for sfx in suffixes)

def _answer_names(f: Dict[str, Any]):
    names = set()
    q = f.get("dns_qname")
    if q: names.add(str(q).lower())
    for rr in (f.get("dns_answers") or f.get("dns.answers") or []):
        n = (rr.get("name") or rr.get("rrname") or rr.get("owner") or "").lower()
        if n: names.add(n)
        rrtype = (rr.get("type") or rr.get("rrtype") or "").upper()
        if rrtype == "CNAME":
            tgt = (rr.get("data") or rr.get("cname") or rr.get("target") or "").lower()
            if tgt: names.add(tgt)
    return names

def _dns_ips_from_features(f: Dict[str, Any]):
    ips = set()
    for k in ("dns_a","dns.a","dns_aaaa","dns.aaaa"):
        vals = f.get(k) or []
        if isinstance(vals, str): vals = [vals]
        for v in vals:
            if isinstance(v, str) and v: ips.add(v)
    for rr in (f.get("dns_answers") or f.get("dns.answers") or []):
        try:
            rrtype = (rr.get("type") or rr.get("rrtype") or "").upper()
            data = rr.get("data") or rr.get("ip") or rr.get("rdata") or ""
            if rrtype in {"A","AAAA"} and isinstance(data, str) and data:
                ips.add(data)
        except Exception:
            pass
    return ips

def _chain_has_whitelisted_name(f: Dict[str, Any]) -> bool:
    return any(_endswith_any(n, BENIGN_DOMAINS) for n in _answer_names(f))

# ---- main ----
def build_cache():
    os.makedirs(os.path.dirname(BASELINE_CACHE_PATH), exist_ok=True)

    print(f"[+] Reading baseline pcap: {BASELINE_PCAP}")
    sessions, _ = extract_sessions(BASELINE_PCAP)
    sessions = enrich_sessions_with_features(sessions)

    endpoints_bi: Set[Tuple[str,int,str]] = set()
    dns_qnames: Set[str] = set()
    http_hosts: Set[str] = set()
    tls_snis: Set[str] = set()
    dns_answer_ips: Set[str] = set()
    benign_ips_from_whitelisted_dns: Set[str] = set()

    for s in sessions:
        # endpoints (bidirectional)
        t = (s.get("transport") or "").upper()
        try: endpoints_bi.add((s.get("dst",""), int(s.get("dport")), t))
        except Exception: pass
        try: endpoints_bi.add((s.get("src",""), int(s.get("sport")), t))
        except Exception: pass

        f = _feat(s)
        if f.get("dns_qname"): dns_qnames.add(str(f["dns_qname"]).lower())
        if f.get("http_host"):  http_hosts.add(str(f["http_host"]).lower())
        if f.get("tls_sni"):    tls_snis.add(str(f["tls_sni"]).lower())

        if (s.get("protocol") or "").upper() == "DNS":
            ips = _dns_ips_from_features(f)
            dns_answer_ips.update(ips)
            if _chain_has_whitelisted_name(f):
                benign_ips_from_whitelisted_dns.update(ips)

    cache = {
        "endpoints_bi": [(ip, int(port), t) for (ip, port, t) in sorted(endpoints_bi)],
        "dns_qnames": sorted(dns_qnames),
        "http_hosts": sorted(http_hosts),
        "tls_snis": sorted(tls_snis),
        "dns_answer_ips": sorted(dns_answer_ips),
        "benign_ips_from_whitelisted_dns": sorted(benign_ips_from_whitelisted_dns),
        "source_pcap": BASELINE_PCAP,
        "version": 1,
    }

    with open(BASELINE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"[✔] Wrote baseline cache: {BASELINE_CACHE_PATH}")

if __name__ == "__main__":
    build_cache()
