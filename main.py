from __future__ import annotations

import os
import time
import socket
from typing import Iterable, Set, Dict, Any, Tuple, List
from collections import Counter

from utils import is_whitelisted_domain
from session_extractor import extract_sessions
from feature_engine import enrich_sessions_with_features
from scorer import score_sessions
from reporter import generate_reports, SCORE_THRESHOLD
from config import (
    BENIGN_DOMAINS,
    BENIGN_IPS,
    BEACON_EXCLUDED_PROTOCOLS
)

# -----------------------------
# Config
# -----------------------------
PCAP_DIR = "Tool/Mimikatz"
OUTPUT_DIR = "Tool/analysis_output_mimikatz_baseline_labelled"
BASELINE_PCAP = "Baseline/NoSample-Background.pcap"

IS_MALWARE_RUN = (PCAP_DIR.lower() != "benign")

# -----------------------------
# Globals
# -----------------------------
baseline_endpoints_bi: Set[Tuple[str, int, str]] = set()
baseline_dns_qnames: Set[str] = set()
baseline_http_hosts: Set[str] = set()
baseline_tls_snis: Set[str] = set()
# Global benign IPs learned from baseline and pass-1
global_benign_ips: Set[str] = set()
baseline_dns_answer_ips: Set[str] = set()  #all A/AAAA seen in baseline DNS answers


# -----------------------------
# Small helpers
# -----------------------------
def _feat(s: Dict[str, Any]) -> Dict[str, Any]:
    return s.get("features", {})

def _get_qname(f: Dict[str, Any]) -> str:
    return (
        f.get("dns_qname")
        or f.get("dns.qname")
        or f.get("query_name")
        or f.get("dns_query")
        or ""
    ).lower()

def _answer_names(f: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    if f.get("dns_qname"):
        names.add(str(f["dns_qname"]).lower())
    for rr in (f.get("dns_answers") or f.get("dns.answers") or []):
        n = (rr.get("name") or rr.get("rrname") or rr.get("owner") or "").lower()
        if n:
            names.add(n)
        rrtype = (rr.get("type") or rr.get("rrtype") or "").upper()
        if rrtype == "CNAME":
            tgt = (rr.get("data") or rr.get("cname") or rr.get("target") or "").lower()
            if tgt:
                names.add(tgt)
    return names

def _dns_ips_from_features(f: Dict[str, Any]) -> Set[str]:
    ips: Set[str] = set()
    for k in ("dns_a", "dns.a", "dns_aaaa", "dns.aaaa"):
        vals = f.get(k) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            if isinstance(v, str) and v:
                ips.add(v)
    ans = f.get("dns_answers") or f.get("dns.answers") or []
    for rr in ans:
        try:
            rrtype = (rr.get("type") or rr.get("rrtype") or "").upper()
            data = rr.get("data") or rr.get("ip") or rr.get("rdata") or ""
            if rrtype in {"A", "AAAA"} and isinstance(data, str) and data:
                ips.add(data)
        except Exception:
            pass
    return ips

def _chain_has_whitelisted_name(f: Dict[str, Any]) -> bool:
    return any(is_whitelisted_domain(n, BENIGN_DOMAINS) for n in _answer_names(f))

def _is_benign_hostlike(f: Dict[str, Any]) -> bool:
    host = (f.get("http_host") or f.get("tls_sni") or "").lower()
    if not host:
        return False
    return (
        host in baseline_http_hosts
        or host in baseline_tls_snis
        or is_whitelisted_domain(host, BENIGN_DOMAINS)
    )

def _apply_strict_dns_domain_policy(sessions: List[dict], *, score_thresh: int) -> None:
    """
    For DNS flows:
      - If qname (or any name in chain) is in baseline or BENIGN_DOMAINS -> score=0 (benign)
      - Else -> score=score_thresh+1 (malicious)
    """
    for s in sessions:
        if (s.get("protocol") or "").upper() != "DNS":
            continue
        f = _feat(s)
        # Use full chain (qname + rrnames + CNAME targets)
        names = _answer_names(f)
        benign = any(n in baseline_dns_qnames for n in names) or any(is_whitelisted_domain(n, BENIGN_DOMAINS) for n in names)
        if benign:
            s["score"] = 0
            s["reasons"] = ["DNS baseline/whitelist"]
        else:
            s["score"] = score_thresh + 1
            s["reasons"] = ["DNS domain not baseline/whitelist"]

# -----------------------------
# Baseline-aware checks
# -----------------------------
def is_baseline_dns(sess: Dict[str, Any]) -> bool:
    f = _feat(sess)
    qname = _get_qname(f)
    src, dst = sess.get("src", ""), sess.get("dst", "")
    t = (sess.get("transport") or "").upper()

    req_matches_resolver = (dst, 53, "UDP") in baseline_endpoints_bi and t == "UDP"
    resp_matches_resolver = (src, 53, "UDP") in baseline_endpoints_bi and t == "UDP"
    if not (req_matches_resolver or resp_matches_resolver):
        return False

    if qname and (qname in baseline_dns_qnames or is_whitelisted_domain(qname, BENIGN_DOMAINS)):
        return True
    return False

def is_baseline_http_like(sess: Dict[str, Any]) -> bool:
    f = _feat(sess)
    host = (f.get("http_host") or f.get("tls_sni") or "").lower()
    src, dst = sess.get("src", ""), sess.get("dst", "")
    sport, dport = sess.get("sport"), sess.get("dport")
    t = (sess.get("transport") or "").upper()

    sock_matches = ((dst, dport, t) in baseline_endpoints_bi) or ((src, sport, t) in baseline_endpoints_bi)
    if not sock_matches:
        return False

    if host and (
        host in baseline_http_hosts
        or host in baseline_tls_snis
        or is_whitelisted_domain(host, BENIGN_DOMAINS)
    ):
        return True
    return False

def is_baseline_noise(sess: Dict[str, Any]) -> bool:
    proto = (sess.get("protocol") or "").upper()
    if proto == "DNS":
        return is_baseline_dns(sess)
    f = _feat(sess)
    if proto in {"HTTP", "HTTPS"} or f.get("tls_sni") or f.get("http_host"):
        return is_baseline_http_like(sess)
    src, dst = sess.get("src", ""), sess.get("dst", "")
    sport, dport = sess.get("sport"), sess.get("dport")
    t = (sess.get("transport") or "").upper()
    if (
        ((dst, dport, t) in baseline_endpoints_bi)
        or ((src, sport, t) in baseline_endpoints_bi)
    ) and src.startswith("192.168.1.") and dst.startswith("192.168.1."):
        return True
    return False


# -----------------------------
# Baseline loading & seeding
# -----------------------------
def _load_baseline():
    print(f"[+] Loading baseline: {BASELINE_PCAP}")
    baseline_sessions, _ = extract_sessions(BASELINE_PCAP)

    for sess in baseline_sessions:
        try:
            t = (sess.get("transport") or "").upper()
            baseline_endpoints_bi.add((sess.get("dst", ""), sess.get("dport"), t))
        except Exception:
            pass
        try:
            t = (sess.get("transport") or "").upper()
            baseline_endpoints_bi.add((sess.get("src", ""), sess.get("sport"), t))
        except Exception:
            pass

    global baseline_dns_qnames, baseline_http_hosts, baseline_tls_snis
    baseline_dns_qnames = {
        _feat(s).get("dns_qname", "").lower()
        for s in baseline_sessions
        if _feat(s).get("dns_qname")
    }
    baseline_http_hosts = {
        _feat(s).get("http_host", "").lower()
        for s in baseline_sessions
        if _feat(s).get("http_host")
    }
    baseline_tls_snis = {
        _feat(s).get("tls_sni", "").lower()
        for s in baseline_sessions
        if _feat(s).get("tls_sni")
    }

    # DNS: seed whitelisted-domain IPs AND collect all baseline DNS answer IPs (single pass)
    seeded = 0
    for s in baseline_sessions:
        if (s.get("protocol") or "").upper() != "DNS":
            continue
        f = _feat(s)

        # All A/AAAA seen in baseline answers → baseline DNS answer IPs
        ips = _dns_ips_from_features(f)
        baseline_dns_answer_ips.update(ips)

        # If the chain contains a whitelisted domain → also seed global benign IPs
        if _chain_has_whitelisted_name(f):
            global_benign_ips.update(ips)
            seeded += len(ips)

    if seeded:
        print(f"[+] Seeded {seeded} benign IPs from baseline DNS (whitelisted-domain answers).")
    print(f"[+] Baseline DNS answer IPs: {len(baseline_dns_answer_ips)}")

# -----------------------------
# Use PCAP's first-packet time
# -----------------------------
def _assign_flow_start_time(sessions: List[Dict[str, Any]]) -> None:
    """
    For each session, set features['timestamp'] to the FIRST PACKET'S PCAP timestamp (epoch seconds),
    if available. Minimal, robust fallbacks are included.
    """
    for s in sessions:
        f = s.setdefault("features", {})

        # 1) Best: access packets captured from the pcap
        ts = None
        pkts = s.get("packets")
        if pkts:
            # support objects with .time or dicts with 'time'/'ts'
            first = pkts[0]
            ts = getattr(first, "time", None)
            if ts is None and isinstance(first, dict):
                ts = first.get("time") or first.get("ts")

        # 2) Common absolute epoch fields populated by extract/enrich (if any)
        if ts is None:
            ts = f.get("absolute_time") or f.get("frame_time_epoch") or f.get("start_epoch")

        # 3) Last resort: if only a relative offset is present, anchor it to file mtime later
        if ts is None:
            ts = f.get("timestamp")  # may be relative; we'll handle sort fallback later

        # Normalize to float
        try:
            f["timestamp"] = float(ts) if ts is not None else None
        except Exception:
            f["timestamp"] = None


# -----------------------------
# Pass 1: learn global benign IPs & order files by earliest flow start
# -----------------------------
def _collect_globals_and_order_files() -> List[str]:
    print("[+] Pass 1: learning global benign IPs and ordering pcaps...")
    candidates: List[Tuple[str, float]] = []

    for filename in sorted(os.listdir(PCAP_DIR)):
        if not filename.endswith(".pcap"):
            continue
        filepath = os.path.join(PCAP_DIR, filename)
        sessions, _ = extract_sessions(filepath)
        sessions = enrich_sessions_with_features(sessions)

        # Assign start times directly from PCAP packets if available
        _assign_flow_start_time(sessions)

        # Learn benign IPs
        for s in sessions:
            f = _feat(s)
            proto = (s.get("protocol") or "").upper()
            if proto == "DNS" and is_baseline_dns(s) and _chain_has_whitelisted_name(f):
                global_benign_ips.update(_dns_ips_from_features(f))
            if _is_benign_hostlike(f):
                # target ip from direction (outbound→dst, inbound→src)
                direction = s.get("direction")
                ip = s.get("dst") if direction == "outbound" else (
                    s.get("src") if direction == "inbound" else None
                )
                if ip:
                    global_benign_ips.add(ip)

        # Pick earliest absolute timestamp; if missing/relative, fallback to file mtime
        times = [s.get("features", {}).get("timestamp") for s in sessions]
        times = [t for t in times if isinstance(t, (int, float)) and t > 1_000_000_000]  # epoch-ish
        if times:
            anchor = min(times)
        else:
            try:
                anchor = float(os.path.getmtime(filepath))
            except Exception:
                anchor = time.time()
        candidates.append((filename, anchor))

    if global_benign_ips:
        print(f"[+] Learned {len(global_benign_ips)} benign IPs in pass 1.")
    candidates.sort(key=lambda x: x[1])
    ordered = [fn for fn, _ in candidates]
    if ordered:
        print("[+] Analysis order (oldest → newest):")
        for fn in ordered:
            print("   -", fn)
    return ordered


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _load_baseline()
    files_to_process = _collect_globals_and_order_files()
    if not files_to_process:
        files_to_process = [f for f in sorted(os.listdir(PCAP_DIR)) if f.endswith(".pcap")]

    all_results: List[Dict[str, Any]] = []

    for filename in files_to_process:
        print(f"[+] Processing {filename}")
        filepath = os.path.join(PCAP_DIR, filename)

        sessions, stats = extract_sessions(filepath)

        # Mark baseline noise early
        for s in sessions:
            s["baseline_noise"] = is_baseline_noise(s)

        # Basic counters + flow_id tagging
        protocol_counter: Counter[str] = Counter()
        direction_counter: Counter[str] = Counter()
        lan_sessions = 0
        wan_sessions = 0

        for idx, session in enumerate(sessions):
            session["flow_id"] = f"flow_{idx}"

            # Defaults
            session["packet_count"] = session.get("packet_count", 1)
            session["byte_count"] = session.get("byte_count", 0)

            # Directionality
            src, dst = session.get("src", ""), session.get("dst", "")
            if src.startswith("192.168.1.") and dst.startswith("192.168.1."):
                session["direction"] = "internal"
                lan_sessions += 1
            elif src.startswith("192.168.1."):
                session["direction"] = "outbound"
                wan_sessions += 1
            elif dst.startswith("192.168.1."):
                session["direction"] = "inbound"
                wan_sessions += 1
            else:
                session["direction"] = "external"

            protocol = session.get("protocol", "Unknown")
            protocol_counter[protocol] += 1
            direction_counter[session["direction"]] += 1

        stats["protocol_counts"] = dict(protocol_counter)
        stats["direction_counts"] = dict(direction_counter)
        stats["lan_sessions"] = lan_sessions
        stats["wan_sessions"] = wan_sessions

        # Enrich & assign start time from first packet
        enriched_sessions = enrich_sessions_with_features(sessions)
        _assign_flow_start_time(enriched_sessions)

        # Reverse DNS for flows missing a domain and not LAN-only
        for s in enriched_sessions:
            feat = s.setdefault("features", {})
            if feat.get("dns_qname") or feat.get("http_host"):
                continue

            ip_to_lookup = None
            if s.get("direction") == "outbound":
                ip_to_lookup = s.get("dst", "")
            elif s.get("direction") == "inbound":
                ip_to_lookup = s.get("src", "")

            if not ip_to_lookup or ip_to_lookup.startswith("192.168.1."):
                continue

            try:
                rd = socket.gethostbyaddr(ip_to_lookup)[0].lower()
            except Exception:
                rd = ""
            if rd:
                feat["dns_qname"] = rd
                feat["rdns_inferred"] = True

        # Per-file benign IPs only when DNS chain has whitelisted domain
        per_file_benign_ips: Set[str] = set()
        for s in enriched_sessions:
            if (s.get("protocol") or "").upper() != "DNS":
                continue
            f = _feat(s)
            if _chain_has_whitelisted_name(f):
                per_file_benign_ips.update(_dns_ips_from_features(f))

        # Score
        scored_sessions = score_sessions(enriched_sessions)

        # Strict DNS domain policy (overrides general scoring for DNS only)
        _apply_strict_dns_domain_policy(scored_sessions, score_thresh=SCORE_THRESHOLD)

        # Collect per-file IPs from DNS flows marked malicious-by-policy
        per_file_malicious_ips: Set[str] = set()
        for s in scored_sessions:
            if (s.get("protocol") or "").upper() != "DNS":
                continue
            if "DNS domain not baseline/whitelist" in (s.get("reasons") or []):
                per_file_malicious_ips.update(_dns_ips_from_features(_feat(s)))

        # -------- Final whitelisting / zeroing rules (non-DNS only) --------
        for s in scored_sessions:
            proto = (s.get("protocol") or "").upper()
            if proto == "DNS":
                continue  # DNS already handled by strict policy

            feat = s.setdefault("features", {})
            raw_dom = (
                feat.get("http_host")
                or feat.get("tls_sni")
                or (feat.get("dns_qname") if not feat.get("rdns_inferred") else None)
                or ""
            )
            dom = raw_dom.lower()
            direction = s.get("direction")
            target_ip = s.get("dst") if direction == "outbound" else (
                s.get("src") if direction == "inbound" else None
            )

            # escalate any non-DNS flow whose target IP was returned by a malicious-by-policy DNS in THIS PCAP
            if target_ip and target_ip in per_file_malicious_ips:
                s["score"] = max(s.get("score", 0), SCORE_THRESHOLD + 1)
                s["reasons"] = ["Resolved from non-whitelisted domain (same file)"]
                continue

            # A) Benign IP only if learned from whitelisted domains (global/per-file),
            #    or if current host/SNI is benign for a baseline DNS IP.
            hostlike = (feat.get("http_host") or feat.get("tls_sni") or "").lower()
            if target_ip and (
                target_ip in global_benign_ips
                or target_ip in per_file_benign_ips
                or (
                    target_ip in baseline_dns_answer_ips
                    and hostlike
                    and (
                        hostlike in baseline_http_hosts
                        or hostlike in baseline_tls_snis
                        or is_whitelisted_domain(hostlike, BENIGN_DOMAINS)
                    )
                )
            ):
                s["score"] = 0
                s["reasons"] = ["Benign-resolved IP"]
                continue

            # B) Baseline socket noise + benign host/SNI/domain
            if s.get("baseline_noise") and dom and (
                dom in baseline_http_hosts
                or dom in baseline_tls_snis
                or dom in baseline_dns_qnames
                or is_whitelisted_domain(dom, BENIGN_DOMAINS)
            ):
                s["score"] = 0
                s["reasons"] = ["Baseline noise"]
                continue

            # C) Static benign IPs (hardcoded)
            if (s.get("src") in BENIGN_IPS) or (s.get("dst") in BENIGN_IPS):
                s["score"] = 0
                s["reasons"] = ["Static benign IP"]
                continue

            if not s.get("reasons"):
                # DNS is already handled by strict policy; for non-DNS, default to malicious
                s["score"] = max(s.get("score", 0), SCORE_THRESHOLD + 1)
                s["reasons"] = ["Unexplained traffic (not baseline/whitelist)"]

        # -------- Beacon propagation (run ONCE, outside per-flow loop) --------
        if IS_MALWARE_RUN:
            multicast_like = {"224.0.0.252", "239.255.255.250"}
            beacon_ips: Set[str] = set()

            # Seed only from non-excluded, non-benign sessions that matched cadence
            for sess in scored_sessions:
                p = (sess.get("protocol") or "").upper()
                if p in BEACON_EXCLUDED_PROTOCOLS:
                    continue
                if sess.get("score", 0) == 0:
                    continue
                if "Regular interval beaconing pattern" in (sess.get("reasons") or []):
                    if sess.get("dst"):
                        beacon_ips.add(sess.get("dst"))
                    if sess.get("src"):
                        beacon_ips.add(sess.get("src"))

            # Apply only to eligible, non-benign sessions
            for s in scored_sessions:
                p = (s.get("protocol") or "").upper()
                if p in BEACON_EXCLUDED_PROTOCOLS:
                    continue
                if s.get("score", 0) == 0:
                    continue
                if s.get("baseline_noise") or s.get("baseline_tuple_noise"):
                    continue
                reasons = s.get("reasons") or []
                if any(r in reasons for r in (
                    "Whitelisted domain",
                    "Baseline benign content",
                    "Benign-resolved IP",
                    "Static benign IP",
                    "DNS baseline/whitelist",
                )):
                    continue
                if s.get("dst") in multicast_like or s.get("src") in multicast_like:
                    continue

                if s.get("dst") in beacon_ips or s.get("src") in beacon_ips:
                    s["score"] = SCORE_THRESHOLD + 1
                    s["reasons"] = ["IP-level beacon flag"]

        # ---- SORT: by first-packet timestamp; fallback to flow_id index if missing ----
        def _flow_id_as_int(s: Dict[str, Any]) -> int:
            fid = s.get("flow_id", "")
            try:
                return int(str(fid).split("_")[-1])
            except Exception:
                return 1 << 62

        scored_sessions.sort(
            key=lambda s: (
                s.get("features", {}).get("timestamp", float("inf")),
                _flow_id_as_int(s),
            )
        )

        #report
        generate_reports(scored_sessions, filename, OUTPUT_DIR, stats)
        all_results.extend(scored_sessions)

    # -----------------------------
    # Global diagnostics
    # -----------------------------
    print("[✔] Analysis complete. Reports saved in:", OUTPUT_DIR)

    malicious_count = sum(1 for s in all_results if s.get("score", 0) >= SCORE_THRESHOLD)
    print(f"[Stats] Total flows: {len(all_results)}, flagged malicious: {malicious_count}")

    # Suspicious DNS that were baseline-tagged
    susp_baseline_dns: List[str] = []
    for s in all_results:
        if not s.get("baseline_noise"):
            continue
        f = s.get("features", {})
        q = _get_qname(f)
        if q and (q not in baseline_dns_qnames) and (not is_whitelisted_domain(q, BENIGN_DOMAINS)):
            susp_baseline_dns.append(q)

    if susp_baseline_dns:
        print("\n[Diagnostics] Baseline-tagged DNS that looks suspicious:")
        for dom, cnt in Counter(susp_baseline_dns).most_common(20):
            print(f"  {dom}  (x{cnt})")
    else:
        print("\n[Diagnostics] No suspicious baseline-tagged DNS found.")

    overall_reason_counts: Counter[str] = Counter()
    for s in all_results:
        reasons = s.get("reasons") or []
        overall_reason_counts[reasons[0] if reasons else "<none>"] += 1

    print("\n[Summary] Top reasons across all files:")
    for reason, cnt in overall_reason_counts.most_common(15):
        print(f"   {reason}: {cnt}")


if __name__ == "__main__":
    main()
