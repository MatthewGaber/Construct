import math
import re
import tldextract
from statistics import stdev
from utils import calculate_entropy, is_base64_string

def compute_entropy(data):
    if not data:
        return 0.0
    counts = [data.count(c) for c in set(data)]
    probs  = [c / len(data) for c in counts]
    return -sum(p * math.log2(p) for p in probs)

def _grab_all(layer, field):
    """Return list of values from pyshark field (supports *_all and single)."""
    try:
        multi = getattr(layer, field + '_all', None)
        if multi:
            return list(multi)
        single = getattr(layer, field, None)
        return [single] if single else []
    except Exception:
        return []

def enrich_sessions_with_features(sessions):
    for session in sessions:
        features = {}
        session['features'] = features

        # --- DNS Analysis (first DNS packet) ---
        dns_pkt = session.get('dns_pkt')
        if dns_pkt and hasattr(dns_pkt, 'dns'):
            try:
                dns_layer = dns_pkt.dns

                # qname (what you already had)
                qname = getattr(dns_layer, 'qry_name', None)
                if qname:
                    q    = qname.lower()
                    tld  = tldextract.extract(q).suffix
                    sub  = q.split('.')[0] if '.' in q else ''
                    features.update({
                        'dns_qname':        q,
                        'dns_tld':          tld,
                        'dns_entropy':      calculate_entropy(q),
                        'dns_nxdomain':     getattr(dns_layer, 'rcode', '') == '3',
                        'dns_sub_len':      len(sub),
                        'dns_sub_entropy':  calculate_entropy(sub)
                    })
                else:
                    q = ''  # fallback -- still try to parse answers

                #answers (A/AAAA/CNAME) in a portable schema expected by main.py
                answers = []

                # owner names wireshark exposes for answers (may be many)
                resp_names = _grab_all(dns_layer, 'resp_name')
                # if none, fall back to qname for owner
                if not resp_names and q:
                    resp_names = [q]

                # A/AAAA lists
                a_list    = _grab_all(dns_layer, 'a')
                aaaa_list = _grab_all(dns_layer, 'aaaa')

                # CNAME list (targets)
                cname_list = _grab_all(dns_layer, 'cname')

                # Build normalized dns_answers entries
                for owner in resp_names or [q]:
                    for ip in a_list:
                        if ip:
                            answers.append({'name': owner.lower(), 'type': 'A',    'data': ip})
                    for ip6 in aaaa_list:
                        if ip6:
                            answers.append({'name': owner.lower(), 'type': 'AAAA', 'data': ip6})

                for cn in cname_list:
                    if cn:
                        answers.append({'name': (q or (resp_names[0] if resp_names else '')).lower(),
                                        'type': 'CNAME', 'data': cn.lower()})

                # Also expose flat lists for convenience
                features['dns_answers'] = answers
                features['dns_a']       = [a['data'] for a in answers if a.get('type') == 'A']
                features['dns_aaaa']    = [a['data'] for a in answers if a.get('type') == 'AAAA']

            except Exception as e:
                print(f"[!] DNS feature extraction error: {e}")


        # --- HTTP Analysis (first HTTP packet) ---
        http_pkt   = session.get('http_pkt')
        http_layer = http_pkt.http if http_pkt and hasattr(http_pkt, 'http') else None
        if http_layer:
            try:
                method = getattr(http_layer, 'request_method', '') or ''
                uri    = (getattr(http_layer, 'request_uri', '') or
                          getattr(http_layer, 'request_full_uri', '') or '')
                host   = getattr(http_layer, 'host', '') or ''
                ua     = getattr(http_layer, 'user_agent', '') or ''
                features.update({
                    'http_method':     method,
                    'http_uri':        uri,
                    'http_user_agent': ua,
                    'http_host':       host,
                    'uri_base64':      is_base64_string(uri),
                    'http_uri_entropy':  calculate_entropy(uri),
                    'http_host_entropy': calculate_entropy(host),
                })
            except Exception as e:
                print(f"[!] HTTP feature extraction error: {e}")

        # --- Fallback HTTP Parsing on TCP‐payload text ---
        if not features.get('http_method') and session.get('payload'):
            raw = session['payload']
            fields = {'method':'','uri':'','host':'','ua':''}

            try:
                lines = raw.splitlines()

                # Safely parse the request-line
                if lines:
                    first_parts = lines[0].split()
                    if first_parts and first_parts[0] in ('GET','POST','HEAD','PUT','OPTIONS','DELETE'):
                        if len(first_parts) >= 2:
                            fields['method'], fields['uri'] = first_parts[0], first_parts[1]

                # Headers
                for line in lines[1:]:
                    low = line.lower()
                    if low.startswith('host:'):
                        fields['host'] = line.split(':',1)[1].strip()
                    elif low.startswith('user-agent:'):
                        fields['ua']   = line.split(':',1)[1].strip()

                features.update({
                    'http_method':       fields['method'],
                    'http_uri':          fields['uri'],
                    'http_user_agent':   fields['ua'],
                    'http_host':         fields['host'],
                    'uri_base64':        is_base64_string(uri),
                    'http_uri_entropy':  calculate_entropy(uri),
                    'http_host_entropy': calculate_entropy(host),
                })
                # Override protocol/app_proto if valid HTTP structure is detected
                if fields['method'] and fields['uri']:
                    session['protocol'] = 'HTTP'
                    features['app_proto'] = 'HTTP'
            except Exception as e:
                print(f"[!] Fallback HTTP parsing failed: {e}")


            # --- TLS SNI Extraction (robust) ---
            tls_pkt = session.get('tls_pkt')
            sni = ''

            if tls_pkt:
                # try the 'tls' layer first, then 'ssl'
                tls_layer = getattr(tls_pkt, 'tls', None) or getattr(tls_pkt, 'ssl', None)
                if tls_layer:
                    # two common field names
                    sni = getattr(tls_layer, 'handshake_extensions_server_name', '') or \
                        getattr(tls_layer, 'server_name', '')

            if sni:
                sni = sni.lower()
                features['tls_sni'] = sni
                features['tls_sni_entropy'] = calculate_entropy(sni)

            # FTP command parsing
            payload = session.get('payload','')
            for cmd in ('USER','PASS','RETR'):
                features[f'ftp_{cmd.lower()}'] = bool(
                    re.search(rf'^{cmd} ', payload, re.MULTILINE)
                )

        # --- Behavioral & Timing Features ---
        # 1. Timestamp
        features['timestamp'] = session.get('timestamp', 0.0)
        features['app_proto'] = session.get('app_proto', '')

        # 2. Server response flag
        pkt = session.get('raw_pkt')
        try:
            features['server_responded'] = (
                pkt and hasattr(pkt, pkt.transport_layer.lower()) and
                pkt[pkt.transport_layer].dstport == session['sport']
            )
        except:
            features['server_responded'] = False

        # 3. Inter-packet timing and session duration
        timestamps = session.get('timestamps', [])
        if len(timestamps) > 1:
            deltas = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
            features['inter_packet_timing_stddev'] = stdev(deltas) if len(deltas) > 1 else 0.0
            features['regular_interval']           = int(all(abs(d - deltas[0]) < 0.01 for d in deltas))
            features['session_duration']           = timestamps[-1] - timestamps[0]
        else:
            features['inter_packet_timing_stddev'] = 0.0
            features['regular_interval']           = 0
            features['session_duration']           = 0.0

        # 4. Packet & byte counts, burst count
        sizes = session.get('sizes', [])
        features['packet_count'] = len(sizes)
        features['byte_count']   = sum(sizes)
        features['burst_count']  = sum(1 for s in sizes if s > 1000)

        # 5. Payload-based heuristics
        payload = session.get('payload', '')
        features['uri_entropy']     = calculate_entropy(payload)
        features['contains_exe']    = int("MZ" in payload[:100])
        features['base64_payload']  = int(bool(re.match(r'^[A-Za-z0-9+/=]{16,}$', payload.strip())))
        features['missing_user_agent'] = int("User-Agent" not in payload)
        features['suspicious_referer'] = int("Referer" in payload and "localhost" in payload)
        features['mime_transfer']      = int(any(ext in payload.lower() for ext in ['.exe', '.zip', '.pdf']))

    return sessions
