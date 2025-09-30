import csv
import json
import os
from collections import Counter
from utils import calculate_entropy, is_base64_string

from config import BENIGN_DOMAINS  #  whitelist

# only flows scoring above this get marked malicious (and not whitelisted)
SCORE_THRESHOLD = 4

def is_public_ip(ip):
    return not (
        ip.startswith('192.168.') or
        ip.startswith('10.') or
        ip.startswith('172.16.') or
        ip.startswith('172.17.') or
        ip.startswith('172.18.') or
        ip.startswith('172.19.') or
        ip.startswith('172.2')
    )

def generate_reports(sessions, filename, output_dir, stats=None):
    basename = os.path.splitext(filename)[0]
    csv_path = os.path.join(output_dir, f"{basename}_sessions.csv")
    json_path = os.path.join(output_dir, f"{basename}_summary.json")

    # Write detailed CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            'flow_id', 'src', 'dst', 'domain', 'domain_entropy', 'sport', 'dport', 'protocol', 'transport',
            'direction', 'packet_count', 'byte_count', 'score', 'reasons', 'timestamp', 'malicious',
            'dns_qname', 'dns_tld', 'dns_entropy', 'dns_nxdomain', 'dns_sub_len', 'dns_sub_entropy',
            'http_method', 'http_uri', 'http_user_agent', 'uri_base64','app_proto','http_uri_entropy','http_host_entropy',
            'tls_sni', 'tls_sni_entropy','ftp_user', 'ftp_pass', 'ftp_retr',
            'inter_packet_timing_stddev', 'regular_interval', 'session_duration',
            'burst_count', 'uri_entropy', 'contains_exe', 'base64_payload',
            'missing_user_agent', 'suspicious_referer', 'mime_transfer'
        ])

        for s in sessions:
            features = s.get('features', {})

            # pick a domain for reporting
            domain = (features.get('dns_qname') or features.get('http_host') or "").lower()
            domain_entropy = calculate_entropy(domain)

            # decide malicious flag: must exceed threshold and not be whitelisted
            is_whitelisted = any(domain.endswith(w) for w in BENIGN_DOMAINS)
            malicious_flag = 1 if (s.get('score', 0) > SCORE_THRESHOLD and not is_whitelisted) else 0

            writer.writerow([
                s.get('flow_id', ''),
                s.get('src'),
                s.get('dst'),
                domain,
                domain_entropy,
                s.get('sport'),
                s.get('dport'),
                s.get('protocol', 'Unknown'),
                s.get('transport', 'Unknown'),
                s.get('direction', 'Unknown'),
                s.get('packet_count', 1),
                s.get('byte_count', 0),
                s.get('score', 0),
                '; '.join(s.get('reasons', [])),
                features.get('timestamp', ''),
                malicious_flag,
                features.get('dns_qname', ''),
                features.get('dns_tld', ''),
                features.get('dns_entropy', ''),
                features.get('dns_nxdomain', ''),
                features.get('dns_sub_len', ''),
                features.get('dns_sub_entropy', ''),
                features.get('http_method', ''),
                features.get('http_uri', ''),
                features.get('http_user_agent', ''),
                features.get('uri_base64', ''),
                features.get('app_proto', ''),
                features.get('http_uri_entropy', ''),
                features.get('http_host_entropy', ''),
                features.get('tls_sni', ''),
                features.get('tls_sni_entropy', ''),
                features.get('ftp_user', 0),
                features.get('ftp_pass', 0),
                features.get('ftp_retr', 0),
                features.get('inter_packet_timing_stddev', ''),
                features.get('regular_interval', ''),
                features.get('session_duration', ''),
                features.get('burst_count', ''),
                features.get('uri_entropy', ''),
                features.get('contains_exe', ''),
                features.get('base64_payload', ''),
                features.get('missing_user_agent', ''),
                features.get('suspicious_referer', ''),
                features.get('mime_transfer', ''),
            ])

    # Build JSON summary
    summary = {
        'pcap': filename,
        'total_sessions': len(sessions),
        'suspicious_sessions': sum(
            1 for s in sessions
            if s.get('score', 0) > SCORE_THRESHOLD
               and not any(
                   ((f := s.get('features', {})).get('dns_qname') or f.get('http_host') or "").lower().endswith(w)
                   for w in BENIGN_DOMAINS
               )
        ),
        'top_iocs': list(
            Counter([
                s['dst'] for s in sessions
                if s.get('score', 0) > SCORE_THRESHOLD
                   and not any(
                       ((f := s.get('features', {})).get('dns_qname') or f.get('http_host') or "").lower().endswith(w)
                       for w in BENIGN_DOMAINS
                   )
                   and is_public_ip(s['dst'])
            ]).most_common(5)
        ),
        'offline_beacons': sum(
            1 for s in sessions
            if "no server response" in ';'.join(s.get('reasons', []))
        ),
        'tlds_seen': list(set(
            s.get('features', {}).get('dns_tld')
            for s in sessions
            if s.get('features', {}).get('dns_tld')
        )),
        'dga_domain_count': sum(
            1 for s in sessions
            if 'High DNS entropy' in ';'.join(s.get('reasons', []))
        ),
        'unique_domains': list(set(
            (feat.get('dns_qname') or feat.get('http_host'))
            for s in sessions
            if (feat := s.get('features'))
        )),
        'packet_stats': {
            'total_packets':        stats.get('total_packets', 0),
            'non_ip_packets':       stats.get('non_ip_packets', 0),
            'processed_packets':    stats.get('processed_packets', 0),
            'transportless_packets':stats.get('transportless_packets', 0),
            'protocol_counts':      stats.get('protocol_counts', {}),
            'direction_counts':     stats.get('direction_counts', {}),
            'lan_sessions':         stats.get('lan_sessions', 0),
            'wan_sessions':         stats.get('wan_sessions', 0)
        } if stats else {}
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
