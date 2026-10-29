"""
export_dashboard.py
-------------------
E5: turn a scenario recording (utm/flight_log.py) into the compact JSON the dashboard loads
(dashboard/data/<name>.json): the map, drones, routes and their re-plans, booked H3 cells as centres
(the dashboard draws regular hexagons), zones, positions per frame, the UTM's risk per second, events
in plain words (the same wording as scripts/render_scenario.py) and the run's results.

    python scripts/export_dashboard.py eval_results/scenarios/ecity_demo.json.gz
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_scenario import caption, describe, short  # noqa: E402
from utm import flight_log  # noqa: E402


def r(a, d=2):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('recording')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    log = flight_log.read(a.recording)
    cells = {c: np.asarray(p) for c, p in log['cells'].items()}
    centre = {c: p.mean(axis=0) for c, p in cells.items()}
    radius = float(np.mean([np.linalg.norm(p - centre[c], axis=1).mean() for c, p in cells.items()])) if cells else 0.5
    first = next(iter(cells.values()), None)
    angle = float(np.arctan2(*(first[0] - first.mean(0))[::-1])) if first is not None else 0.0
    keys, kidx = [], {}

    def booking(rows):
        out = []
        for c, layer, t0, t1 in rows:
            k = (c, layer)
            if k not in kidx:
                kidx[k] = len(keys)
                keys.append([*r(centre[c]), layer])
            out.append([kidx[k], t0, t1])
        return out

    routes = {d['id']: [{'t': 0.0, 'route': r(d['route'])}] for d in log['drones']}
    books = {f: [{'t': 0.0, 'cells': booking(rows)}] for f, rows in log['bookings'].items()}
    events = []
    node = {d['id']: d['operator'] for d in log['drones']}         # each flight's client node (operator)
    dep = {d['id']: d['t_dep'] for d in log['drones']}            # its booked take-off, as re-plans move it
    arr = {d['id']: d['arrival'] for d in log['drones']}          # and its booked arrival
    for e in log['events']:
        text = describe(e)
        if e['type'] == 'replan' and e['status'] == 'approved' and e.get('route'):
            routes.setdefault(e['flight'], []).append({'t': e['t'], 'route': r(e['route'])})
            books.setdefault(e['flight'], []).append({'t': e['t'], 'cells': booking(e.get('booking', []))})
        if text:
            ev = {'t': e['t'], 'type': e['type'], 'text': text}
            if e.get('flight'):
                ev['flight'] = e['flight']
            if e.get('flight') and node.get(e['flight']):
                ev['node'] = node[e['flight']]
            if e['type'] == 'replan':
                ev.update(status=e['status'], tag=short(e), why=e['why'])
                if e['why'].startswith('displaced by'):          # who it made way for: that flight's node
                    ev['for'] = node.get(e['why'].split()[-1], '')
                old, new = dep.get(e['flight']), e.get('t_dep')
                if e['status'] == 'approved' and old is not None and new is not None and old > e['t']:
                    ev['shift'] = round(float(new) - float(old), 1)   # take-off moved (still on its pad)
                if e['status'] == 'approved' and new is not None:
                    dep[e['flight']] = new
                if e['status'] == 'approved' and e.get('arrival') is not None:
                    if arr.get(e['flight']) is not None:
                        ev['later'] = round(float(e['arrival']) - float(arr[e['flight']]), 1)   # arrives this much later
                    arr[e['flight']] = e['arrival']
            if e['type'] == 'alert':
                ev['risk'] = e['risk']
            if e['type'] == 'zone_on':
                ev['zone'] = e['zone']
            events.append(ev)
    hot = [{'t': k['t'], 'high': sorted(f for f, (s, rk) in k.get('risk', {}).items() if rk == 'high')}
           for k in log['ticks']]
    sc = log['scenario']
    places = []                                   # named sites (scenario places): name, kind, operator, pads
    for name, p in (sc.get('places') or {}).items():
        pads = p['pads'] if p.get('kind') == 'hub' else [p['at']]
        places.append({'name': name, 'kind': p.get('kind', ''), 'operator': p.get('operator', ''), 'pads': r(pads)})

    def named(xy):                               # the place a pad belongs to, or ''
        for pl in places:
            if any(np.hypot(xy[0] - q[0], xy[1] - q[1]) < 0.5 for q in pl['pads']):
                return pl['name']
        return ''

    out = {
        'schema': 'utm-dashboard/1', 'name': log['scenario'].get('name', 'scenario'), 'caption': caption(log),
        'map': {'bounds': log['map']['bounds'], 'layers': log['map']['layers'],
                'buildings': [{'c': r(o['center']), 'h': r(o['half']), 'z': round(float(o['height']), 2),
                               'yaw': round(float(o.get('yaw', 0.0)), 3)} for o in log['map']['obstacles']]},
        'hex': {'radius': round(radius, 3), 'angle': round(angle, 3), 'cells': keys},
        'places': places,
        'drones': [{'id': d['id'], 'kind': d['kind'], 'operator': d['operator'], 'start': r(d['start']),
                    'from': named(d['start']), 'to': named(d['goal']),
                    'goal': r(d['goal']), 't_dep': d['t_dep'], 'arrival': d['arrival'], 'layer': d['layer']}
                   for d in log['drones']],
        'routes': routes, 'bookings': books, 'zones': log['zones'],
        'frames': {'t': log['frames']['t'], 'pos': [r(p) for p in log['frames']['pos']], 'status': log['frames']['status']},
        'hot': hot, 'events': events, 'summary': log['summary'],
    }
    path = a.out or os.path.join('dashboard', 'data', os.path.basename(a.recording).replace('.json.gz', '.json'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {path} ({os.path.getsize(path) / 1024:.0f} KB): {len(out["drones"])} drones, {len(keys)} booked cells, '
          f'{len(out["frames"]["t"])} frames, {len(events)} events')


if __name__ == '__main__':
    main()
