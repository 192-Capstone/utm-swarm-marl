"""
access.py
---------
Per-operator access for the live UTM server (scripts/utm_server.py): who may see and do what. DEMO-GRADE security
(see docs/access.md): passcodes in a config file as salted hashes, in-memory sessions, no HTTPS. Not for production.

    python -m utm.access hash <passcode>                       # prints a hash line for deployment/access.yaml
    python -m utm.access set <role> [passcode]                  # sets (or adds) a role's passcode in the file (asks if not given)

The file (deployment/access.yaml):
    admin: admin                     # the role that sees everything
    roles: {admin: 'pbkdf2_sha256$...', op0: '...', op1: '...', Hospital: '...'}

Rules (enforced by the server, this module holds the pure parts):
    admin         everything, and may book for any operator
    an operator   only its own flights and their events; books only as itself; 403 on another operator's flight
"""

import hashlib
import hmac
import os
import secrets
import sys
import time

import yaml

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deployment', 'access.yaml')
ITERATIONS = 200_000
SESSION_S = 8 * 3600                         # a session lasts 8 hours (kept in memory: a server restart logs everyone out)
COOKIE = 'utm_session'
NO_ROLE = f'pbkdf2_sha256${ITERATIONS}${"00" * 16}${"00" * 32}'     # checked for an unknown role: one hash, as for a known one
ADMIN = 'admin'


# ------------------------------------------------------------------ passcode hashes
def hash_passcode(passcode, salt=None, iterations=ITERATIONS):
    salt = salt if salt is not None else secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac('sha256', str(passcode).encode(), salt, iterations)
    return f'pbkdf2_sha256${iterations}${salt.hex()}${h.hex()}'


def check_passcode(passcode, stored):
    try:
        algo, it, salt, want = str(stored).split('$')
        if algo != 'pbkdf2_sha256':
            return False
        got = hashlib.pbkdf2_hmac('sha256', str(passcode).encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(got.hex(), want)
    except (ValueError, TypeError):
        return False


# ------------------------------------------------------------------ roles and sessions
class Access:
    """roles: {role name: stored hash}. no_auth: everyone is the admin (the --no-auth demo switch)."""

    def __init__(self, roles=None, admin=ADMIN, no_auth=False, ttl=SESSION_S, clock=time.time):
        self.hashes, self.admin, self.no_auth, self.ttl, self.clock = dict(roles or {}), admin, bool(no_auth), ttl, clock
        self.sessions = {}                   # token -> (role, expires)

    @classmethod
    def load(cls, path=DEFAULT_FILE):
        with open(path, encoding='utf-8') as fh:
            raw = yaml.safe_load(fh) or {}
        roles = raw.get('roles') or {}
        if not roles:
            raise ValueError(f'{path}: no roles')
        admin = raw.get('admin', ADMIN)
        if admin not in roles:
            raise ValueError(f'{path}: the admin role {admin!r} has no passcode')
        return cls(roles, admin)

    def role_names(self, operators=None):
        """The roles that can log in: the admin, and the scenario's operators that have a passcode."""
        ops = [o for o in (operators or self.hashes) if o in self.hashes and o != self.admin]
        return [self.admin] + sorted(ops)

    def is_admin(self, role):
        return role == self.admin

    def login(self, role, passcode):
        """A session token, or None for an unknown role or a wrong passcode (the same work either way)."""
        stored = self.hashes.get(role)
        ok = check_passcode(passcode, stored if stored else NO_ROLE)
        if not (stored and ok):
            return None
        token = secrets.token_urlsafe(32)
        self.sessions[token] = (role, self.clock() + self.ttl)
        return token

    def role_of(self, token):
        """The role of a session token, or None (no session, expired, logged out). With no_auth: the admin."""
        if self.no_auth:
            return self.admin
        s = self.sessions.get(token)
        if s is None:
            return None
        if s[1] < self.clock():
            self.sessions.pop(token, None)
            return None
        return s[0]

    def logout(self, token):
        self.sessions.pop(token, None)


# ------------------------------------------------------------------ what a role may see
def cookie_token(header):
    """The session token in a Cookie header."""
    for part in (header or '').split(';'):
        k, _, v = part.strip().partition('=')
        if k == COOKIE:
            return v
    return None


BIRD_VISIBLE_M = 15.0       # an operator sees the birds this close to one of its flying drones


def filter_state(state, role, admin=False):
    """/api/state for a role: its own flights, and the events of its own flights or addressed to it. The cursor
    (n_events) is the server's, so a polling page keeps its place."""
    if admin:
        return state
    flights = [f for f in state['flights'] if f.get('operator') == role]
    ids = {f['id'] for f in flights}
    events = [e for e in state['events'] if e.get('node') == role or e.get('flight') in ids]
    out = {**state, 'flights': flights, 'events': events}
    if 'high' in state:                         # the flights at high risk: its own only
        out['high'] = [f for f in state['high'] if f in ids]
    if 'birds' in state:                        # birds are environment, not data of anyone: an operator sees those near its drones
        mine = [f['pos'] for f in flights if f.get('status') == 'flying' and f.get('pos')]
        out['birds'] = [b for b in state['birds'] if any(
            (b['pos'][0] - p[0]) ** 2 + (b['pos'][1] - p[1]) ** 2 + (b['pos'][2] - p[2]) ** 2 <= BIRD_VISIBLE_M ** 2 for p in mine)]
    return out


def filter_flightlog(log, role, admin=False):
    """/api/flightlog for a role: its own flights with their events, and the runs they belong to (no run-wide events)."""
    if admin:
        return log
    flights = [f for f in log['flights'] if f.get('operator') == role]
    runs = {f['run_id'] for f in flights} | ({log['active_run']} if log.get('active_run') else set())
    return {**log, 'flights': flights, 'runs': [{**r, 'events': []} for r in log['runs'] if r['run_id'] in runs]}


def may_see_track(track, role, admin=False):
    """/api/track: only the admin, or the operator of that flight."""
    return admin or track['flight'].get('operator') == role


# Pages and scripts that every visitor may load (they hold no data: it all comes from the API). Anything else in the
# dashboard folder (data/*.json: every operator's flights) is for the admin only.
PUBLIC_FILE = ('.html', '.js', '.css', '.ico')


def public_static(path):
    """True for '/', and for a single top-level .html/.js/.css/.ico file name (strict: no folders, no tricks)."""
    p = path.replace('\\', '/')
    if p in ('', '/'):
        return True
    name = p[1:] if p.startswith('/') else p
    ok = name and '/' not in name and all(c.isalnum() or c in '._-' for c in name) and not name.startswith('.')
    return bool(ok and name.lower().endswith(PUBLIC_FILE))


# ------------------------------------------------------------------ command line
def set_passcode(path, role, passcode):
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    raw = yaml.safe_load(text) or {}
    raw.setdefault('roles', {})[role] = hash_passcode(passcode)
    lead = []                                  # only the leading comment block is kept; the rest is rewritten
    for ln in text.splitlines(True):
        if ln.startswith('#') or not ln.strip():
            lead.append(ln)
        else:
            break
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(''.join(lead) + yaml.safe_dump(raw, sort_keys=False, default_flow_style=False))


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) >= 2 and a[0] == 'hash':
        print(hash_passcode(a[1]))
        return 0
    if len(a) >= 2 and a[0] == 'set':
        import getpass
        passcode = a[2] if len(a) > 2 else getpass.getpass(f'passcode for {a[1]}: ')
        set_passcode(DEFAULT_FILE, a[1], passcode)
        print(f'{a[1]}: passcode set in {os.path.normpath(DEFAULT_FILE)}')
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
