import math
import re
from typing import Optional, List
import idna

def _normalize_domain(name: str) -> Optional[str]:
    """Lowercase, strip trailing dot, and IDNA-encode to ASCII.
    Returns None if invalid."""
    if not name:
        return None
    name = name.strip().strip('.').lower()
    if not name:
        return None
    try:
        # UTS-46 handling (maps, NFC) then ASCII label encoding
        return idna.encode(name, uts46=True).decode('ascii')
    except idna.IDNAError:
        return None

def _is_subdomain_or_same(domain: str, parent: str) -> bool:
    """True if `domain` == `parent` or ends with '.' + parent (dot-boundary)."""
    d = _normalize_domain(domain)
    p = _normalize_domain(parent)
    if not d or not p:
        return False
    return d == p or d.endswith('.' + p)

def is_whitelisted_domain(domain: str, whitelist: List[str]) -> bool:
    """Supports entries like:
       - 'microsoft.com'      (exact or any subdomain)
       - '*.microsoft.com'    (any subdomain; NOT the apex)
       - '.microsoft.com'     (any subdomain; NOT the apex)"""
    d = _normalize_domain(domain)
    if not d:
        return False

    for raw in whitelist:
        if not raw:
            continue
        s = raw.strip()
        # Wildcard or leading-dot: subdomains only 
        if s.startswith('*.') or s.startswith('.'):
            parent = s.lstrip('*.')
            if _is_subdomain_or_same(d, parent) and d != _normalize_domain(parent):
                return True
        else:
            # Exact or any subdomain 
            if _is_subdomain_or_same(d, s):
                return True
    return False



def calculate_entropy(data):
    """Calculate Shannon entropy of a string."""
    if not data:
        return 0.0
    entropy = 0
    length = len(data)
    symbols = dict((c, data.count(c)) for c in set(data))
    for count in symbols.values():
        p_x = count / length
        entropy -= p_x * math.log2(p_x)
    return round(entropy, 3)

def is_base64_string(s):
    """Check if a string looks like base64 or high-entropy."""
    if not s or len(s) < 8:
        return False
    base64_re = re.compile(r'^[A-Za-z0-9+/=]{8,}$')
    return base64_re.match(s) is not None or calculate_entropy(s) > 4.2
