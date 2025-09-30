SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 
    'xyz', 'top', 'club', 'work', 'click', 'link', 'space',
    'online', 'stream', 'site', 'icu', 'pw', 'buzz', 'fit',
    'rest', 'men', 'win', 'live', 'uno', 'host', 'press',
    'today', 'world', 'cam', 'info', 'lol', 'review',
    'vip', 'support', 'biz', 'cc', 'to', 'su', 'ru', 'kim'
}
BEACON_EXCLUDED_PROTOCOLS = {"DNS", "LLMNR", "MDNS", "SSDP", "NTP"}
SUSPICIOUS_PORTS = {
    21:   ["FTP"],
    23:   ["TELNET"],
    25:   ["SMTP"],
    53:   ["DNS"],  # only suspicious if NOT DNS
    69:   ["TFTP"],
    80:   ["HTTP"],
    443:  ["HTTPS", "HTTP/SSL", "TLS"],  # support TLS naming variations
    135:  ["MSRPC"],
    137:  ["NETBIOS"],
    138:  ["NETBIOS"],
    139:  ["SMB"],
    445:  ["SMB"],
    1433: ["MSSQL"],
    3306: ["MYSQL"],
    4444: [],   
    4445: [],
    5555: [],
    6660: ["IRC"],
    6666: ["IRC"],
    6667: ["IRC"],
    6697: ["IRC"],
    12345: [],
    31337: [],
    8080: ["HTTP"],
    8443: ["HTTPS"],
    9001: [],
    9002: [],
    1337: [],
    3131: [],
    65000: []
}
DNS_ENTROPY_THRESHOLD = 4.0
URI_ENTROPY_THRESHOLD = 4.2

# Add known benign CDN/telemetry IPs here
BENIGN_IPS = {
    '192.229.232.200',
    '40.127.169.103',
    '239.255.255.250',
    '224.0.0.251',
    '224.0.0.252',
    '255.255.255.255'
}

BENIGN_DOMAINS = {
    'cloudfront.net',
    'a-msedge.net',
    'static.asianet.co.th',
    'google.com',
    'microsoft.com',
    'bing.com',
    'windowsupdate.com',
    'office365.com',
    'akamaitechnologies.com',
    'dns.msftncsi.com',
    'self.events.data.microsoft.com',
    'events.data.microsoft.com',
    'watson.events.data.microsoft.com',
    'vortex.data.microsoft.com',
    'dmd.metaservices.microsoft.com',
    'tsfe.trafficshaping.dsp.mp.microsoft.com',
    'go.microsoft.com',
    'aka.ms',
    'az700632.vo.msecnd.net',
    'az667904.vo.msecnd.net',
    'visualstudio-devdiv-c2s.msedge.net',
    'targetednotifications-tm.trafficmanager.net',
    'download.visualstudio.microsoft.com',
    'ctldl.windowsupdate.com',
    'edge.microsoft.com',
    'skype.com',
    'oneclient.sfx.ms',
    'googleapis.com',
    'live.com',
    'msn.com',
    'windows.com',
    'adobe.com',
    'msftconnecttest.com',
    '1e100.net',
    'mcast.net',
    'digicert.com',
    'fp.msedge.net',
    'msedge.net',
    'fp-afd.azurefd.net',
    'azurefd.net',
    'azureedge.net',
    'msecnd.net',
    '.arpa',
    '.lencr.org',
    '.identrust.com',
    '.gvt1.com',
    'pki.goog',
    '.nelreports.net'

}
