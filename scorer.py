import re
from collections import defaultdict

from config import SUSPICIOUS_PORTS, SUSPICIOUS_TLDS, DNS_ENTROPY_THRESHOLD, BENIGN_DOMAINS, BENIGN_IPS, BEACON_EXCLUDED_PROTOCOLS



def looks_like_beacon(session, history):
    # Only consider outbound
    if session.get('direction') != 'outbound':
        return False

    # Skip protocols that naturally send periodic traffic
    proto = (session.get('protocol') or '').upper()
    if proto in BEACON_EXCLUDED_PROTOCOLS:
        return False

    # --- cadence logic ---
    key = (session.get('dst'), session.get('dport'))
    ts_list = history.setdefault(key, [])
    ts = (session.get('features') or {}).get('timestamp')
    if not isinstance(ts, (int, float)):
        return False

    intervals = [round(ts - t, 1) for t in ts_list if 0 < ts - t < 60]
    ts_list.append(ts)

    if len(intervals) >= 2:
        mean = sum(intervals) / len(intervals)
        return all(abs(i - mean) < 2 for i in intervals)
    return False

def score_sessions(sessions):
    """
    Assigns a heuristic score and list of reasons to each session.
    Baseline‐noise and whitelisted sessions are zeroed and skipped.
    """
    history = {}  # for looks_like_beacon()

    for session in sessions:
        # 0) If this session is known baseline noise, skip all scoring
        if session.get('baseline_noise'):
            session['score']   = 0
            session['reasons'] = ['Baseline noise']
            continue

        if session['dst'] in BENIGN_IPS or session['src'] in BENIGN_IPS:
            session['score']   = 0
            session['reasons'] = ['Whitelisted IP']
            continue

        # 0b) If this session hits a whitelisted domain, skip all scoring
        qname = session.get('features', {}).get('dns_qname', '').lower()
        if any(qname.endswith(w) for w in BENIGN_DOMAINS):
            session['score']   = 0
            session['reasons'] = ['Whitelisted domain']
            continue

        # Now run through all the heuristics
        features = session.get('features', {})
        score    = 0
        reasons  = []

        # DNS Behavior
        if features.get('dns_entropy', 0) > DNS_ENTROPY_THRESHOLD:
            score   += 2
            reasons.append("High DNS entropy (possible DGA)")

        if features.get('dns_nxdomain'):
            score   += 2
            reasons.append("NXDOMAIN DNS response")

        tld = features.get('dns_tld')
        if tld in SUSPICIOUS_TLDS:
            score   += 1
            reasons.append(f"Uncommon TLD: .{tld}")

        if features.get('dns_sub_len', 0) > 20:
            score   += 2
            reasons.append("Very long DNS subdomain (possible tunneling)")

        if features.get('dns_sub_entropy', 0) > 4.5:
            score   += 2
            reasons.append("High entropy DNS subdomain (possible tunneling)")

        domain = features.get('dns_qname', '')
        if re.match(r"^[A-Za-z0-9]{15,30}\.(?:com|net|org|[a-z]{2,})$", domain):
            score   += 2
            reasons.append("Suspicious domain pattern (possible auto-gen)")

        # HTTP Behavior
        if features.get('http_method') == 'POST' and not features.get('http_user_agent'):
            score   += 2
            reasons.append("HTTP POST with no User-Agent")

        if features.get('uri_base64'):
            score   += 2
            reasons.append("Base64-like pattern in URI")

        if not features.get('server_responded') and features.get('http_method') == 'POST':
            score   += 3
            reasons.append("Beacon attempt with no server response")

        # Timing Pattern
        if looks_like_beacon(session, history):
            score   += 3
            reasons.append("Regular interval beaconing pattern")

        # Port/Protocol Check
        try:
            sport = int(session.get("sport", 0))
            dport = int(session.get("dport", 0))
            proto = (session.get("protocol") or "").upper()
            app_proto = (session.get("app_proto") or "").upper()

            for port, role in [(sport, "sport"), (dport, "dport")]:
                if port in SUSPICIOUS_PORTS:
                    allowed = [p.upper() for p in SUSPICIOUS_PORTS[port]]
                    if allowed and (proto in allowed or app_proto in allowed):
                        continue  # legit use case, skip
                    score += 2
                    reasons.append(f"Suspicious {role}: {port}")
        except Exception:
            pass
        # Assign back
        session['score']   = score
        session['reasons'] = reasons

    return sessions
