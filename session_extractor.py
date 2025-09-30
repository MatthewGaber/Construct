import pyshark
from collections import defaultdict, Counter
import hashlib

def extract_sessions(pcap_file):
    sessions = []
    session_map = defaultdict(lambda: {
        'timestamps': [],
        'sizes': [],
        'payload': '',
        'packet_count': 0,
        'byte_count': 0
    })
    stats = Counter()

    try:
        cap = pyshark.FileCapture(
            pcap_file,
            use_json=False,
            keep_packets=False,
            override_prefs={
                'tcp.desegment_tcp_streams': 'true',
                'http.desegment_body': 'true'
            }
        )
    except Exception as e:
        print(f"[!] Failed to load PCAP {pcap_file}: {e}")
        return [], stats

    for pkt in cap:
        stats['total_packets'] += 1

        if not hasattr(pkt, 'ip'):
            stats['non_ip_packets'] += 1
            continue

        try:
            src = pkt.ip.src
            dst = pkt.ip.dst

            # Normalize protocol field
            raw_proto = (pkt.highest_layer or "UNKNOWN").upper()
            
            KNOWN_PROTOS = {
                # Core IP/Transport
                'TCP', 'UDP', 'ICMP', 'IGMP', 'SCTP', 'ARP',

                # DNS & name services
                'DNS', 'MDNS', 'LLMNR', 'NBNS',

                # DHCP
                'DHCP', 'BOOTPC', 'BOOTPS',

                # HTTP family
                'HTTP', 'HTTP2', 'QUIC', 'SPDY', 'TLS', 'SSL',

                # FTP family
                'FTP', 'FTP-DATA', 'FTPS', 'FTPS-DATA',

                # Mail protocols
                'SMTP', 'SMTPS', 'POP3', 'POP3S', 'IMAP', 'IMAPS',

                # File transfer / remote shells
                'SSH', 'SFTP', 'TELNET', 'RDP', 'VNC',

                # Streaming / media
                'RTP', 'RTCP', 'RTSP', 'HLS',

                # SNMP
                'SNMP', 'SNMPv2c', 'SNMPv3',

                # Time
                'NTP',

                # VPN / security
                'ISAKMP', 'ESP', 'AH',

                # Discovery / multicast
                'SSDP', 'BROWSER',

                # Windows monitoring
                'VSSMONITORING',

                # Generic
                'DATA', 'URLENCODED-FORM',
            }

            protocol = raw_proto if raw_proto in KNOWN_PROTOS else 'OTHER'

            # Normalize transport layer
            raw_trans = (pkt.transport_layer or "UNKNOWN").upper()
            transport = raw_trans if raw_trans in {'TCP','UDP','UNKNOWN'} else 'OTHER'
            if transport == 'UNKNOWN':
                stats['transportless_packets'] += 1

            # Source and destination ports
            sport = getattr(getattr(pkt, transport, None), 'srcport', 'Unknown')
            dport = getattr(getattr(pkt, transport, None), 'dstport', 'Unknown')

            # ────────────────────────────────
            # Map well-known ports to application protocols
            _app_map = {
                # Web
                80:    'HTTP',
                8080:  'HTTP-ALT',
                443:   'HTTPS',
                8443:  'HTTPS-ALT',

                # File Transfer
                20:    'FTP-DATA',
                21:    'FTP-CONTROL',
                22:    'SSH/SFTP',
                989:   'FTPS-DATA',
                990:   'FTPS',

                # Mail
                25:    'SMTP',
                465:   'SMTPS',
                587:   'SMTP-SUBMISSION',
                110:   'POP3',
                995:   'POP3S',
                143:   'IMAP',
                993:   'IMAPS',

                # Remote Desktop / VNC
                3389:  'RDP',
                5900:  'VNC',

                # Database
                1433:  'MSSQL',
                3306:  'MySQL',
                5432:  'PostgreSQL',

                # Directory / LDAP
                389:   'LDAP',
                636:   'LDAPS',

                # Windows File Sharing
                137:   'NetBIOS-NS',
                138:   'NetBIOS-DGM',
                139:   'NetBIOS-SSN',
                445:   'SMB',

                # DNS & Discovery
                53:    'DNS',
                5353:  'mDNS',
                1900:  'SSDP',
                5355:  'LLMNR',

                # SNMP
                161:   'SNMP',
                162:   'SNMP-TRAP',

                # Time & NTP
                123:   'NTP',

                # VPN / IPsec
                500:   'ISAKMP',
                4500:  'IPsec-NAT-T',

                # BGP / Routing
                179:   'BGP',

                # Logging
                514:   'Syslog',

                # Common HTTPS alternatives
                10443: 'HTTPS-ALT2',
                10000: 'Webmin',
            }

            app_proto = None
            for p in (int(sport) if str(sport).isdigit() else None,
                      int(dport) if str(dport).isdigit() else None):
                if p in _app_map:
                    app_proto = _app_map[p]
                    break

            if not app_proto and raw_proto == 'HTTP':
                app_proto = 'HTTP'
            # ────────────────────────────────

            # Build flow tuple and ID
            flow_tuple = (src, dst, str(sport), str(dport), transport)
            flow_id = hashlib.md5('_'.join(flow_tuple).encode()).hexdigest()
            sess = session_map[flow_id]

            # Stash first HTTP, DNS, TLS
            if hasattr(pkt, 'http') and 'http_pkt' not in sess:
                sess['http_pkt'] = pkt
            if hasattr(pkt, 'dns') and 'dns_pkt' not in sess:
                sess['dns_pkt'] = pkt
            if raw_proto in ('TLS', 'SSL') and 'tls_pkt' not in sess:
                sess['tls_pkt'] = pkt


            # Determine direction
            if src.startswith('192.168.') and dst.startswith('192.168.'):
                direction = 'internal'
            elif src.startswith('192.168.') and not dst.startswith('192.168.'):
                direction = 'outbound'
            elif not src.startswith('192.168.') and dst.startswith('192.168.'):
                direction = 'inbound'
            else:
                direction = 'external'

            # Timestamps and sizes
            ts = float(pkt.sniff_timestamp)
            size = int(pkt.length)

            # Extract payload text
            payload_text = ''
            if hasattr(pkt, 'tcp') and hasattr(pkt.tcp, 'payload'):
                hexstr = pkt.tcp.payload.replace(':', '')
                try:
                    payload_text = bytes.fromhex(hexstr).decode('latin1', errors='ignore')
                except:
                    payload_text = ''

            # Initialze HTTP and DNS packets
            if hasattr(pkt, 'http') and 'http_pkt' not in sess:
                sess['http_pkt'] = pkt
            if hasattr(pkt, 'dns') and 'dns_pkt' not in sess:
                sess['dns_pkt'] = pkt

            # Update session fields
            sess.update({
                'src':       src,
                'dst':       dst,
                'sport':     sport,
                'dport':     dport,
                'protocol':  protocol,
                'transport': transport,
                'flow_id':   flow_id,
                'direction': direction,
                'app_proto': app_proto or transport,
                'timestamp': ts
            })
            sess['timestamps'].append(ts)
            sess['sizes'].append(size)
            sess['packet_count'] += 1
            sess['byte_count']   += size
            sess['payload']      += payload_text

            # Update stats
            stats[f'protocol:{protocol}']   += 1
            stats[f'transport:{transport}'] += 1
            stats[f'app_proto:{sess["app_proto"]}'] += 1
            stats[f'direction:{direction}']  += 1
            stats['processed_packets']      += 1

        except Exception:
            stats['malformed_packets'] += 1
            continue

    cap.close()

    # Build sesions list
    for data in session_map.values():
        sessions.append(data)

    # Summary stats
    proto_counts = {k.split(':')[1]: v for k, v in stats.items() if k.startswith('protocol:')}
    trans_counts = {k.split(':')[1]: v for k, v in stats.items() if k.startswith('transport:')}
    app_counts   = {k.split(':')[1]: v for k, v in stats.items() if k.startswith('app_proto:')}
    dir_counts   = {k.split(':')[1]: v for k, v in stats.items() if k.startswith('direction:')}

    stats_summary = {
        'total_packets': stats.get('total_packets', 0),
        'non_ip_packets': stats.get('non_ip_packets', 0),
        'processed_packets': stats.get('processed_packets', 0),
        'transportless_packets': stats.get('transportless_packets', 0),
        'malformed_packets': stats.get('malformed_packets', 0),
        'protocol_counts': proto_counts,
        'transport_counts': trans_counts,
        'app_protocol_counts': app_counts,
        'direction_counts': dir_counts,
        'lan_sessions': dir_counts.get('internal', 0),
        'wan_sessions': dir_counts.get('outbound', 0) + dir_counts.get('inbound', 0)
    }

    return sessions, stats_summary
