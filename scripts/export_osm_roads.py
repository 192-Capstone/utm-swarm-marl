"""
export_osm_roads.py
-------------------
The streets of the E-City map, for the console to draw (dashboard/osm_roads.json): an Overpass reply of
OpenStreetMap ways -> polylines in the map's local metres, the frame of scripts/import_osm_city.py (UTM 43N,
(0, 0) at the query centre). Only drawn: the simulation and the Planner use the buildings, not the roads.
Data (c) OpenStreetMap contributors, ODbL.

    python scripts/export_osm_roads.py eval_results/osm/roads_raw.json

The raw file: POST to https://overpass-api.de/api/interpreter
    [out:json];(way["highway"](around:560,12.84968,77.66498);way["leisure"~"park|garden|pitch"](around:560,12.84968,77.66498);
    way["natural"="water"](around:560,12.84968,77.66498);way["amenity"="parking"](around:560,12.84968,77.66498););out geom;
"""

import argparse
import json
import os

from pyproj import Transformer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CENTRE = (12.84968, 77.66498)                      # lat, lon of the map's (0, 0)
WIDTH = {'motorway': 12, 'trunk': 12, 'primary': 11, 'secondary': 9, 'tertiary': 8, 'unclassified': 7, 'residential': 6.5,
         'living_street': 5.5, 'service': 4.5, 'pedestrian': 4, 'track': 3, 'footway': 1.8, 'path': 1.6, 'cycleway': 2, 'steps': 1.8}
SKIP = {'construction', 'proposed', 'corridor', 'elevator', 'bus_stop', 'platform'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('raw')
    ap.add_argument('--out', default=os.path.join(ROOT, 'dashboard', 'osm_roads.json'))
    a = ap.parse_args()
    with open(a.raw, encoding='utf-8') as f:
        ways = json.load(f)['elements']
    to_m = Transformer.from_crs('EPSG:4326', 'EPSG:32643', always_xy=True)
    x0, y0 = to_m.transform(CENTRE[1], CENTRE[0])
    roads, areas = [], []
    for w in ways:
        tags, pts = w.get('tags', {}), []
        for p in w.get('geometry', []):
            x, y = to_m.transform(p['lon'], p['lat'])
            pts.append([round(x - x0, 1), round(y - y0, 1)])
        if len(pts) < 2:
            continue
        kind = tags.get('highway')
        if kind:
            kind = kind.replace('_link', '')
            if kind in SKIP or tags.get('tunnel') == 'yes' or (tags.get('area') == 'yes'):
                continue
            width = WIDTH.get(kind, 4)
            try:
                width = min(16.0, max(1.5, float(tags['width'].split()[0])))
            except (KeyError, ValueError):
                if tags.get('lanes', '').isdigit() and width >= 4:
                    width = max(width, min(14.0, 3.2 * int(tags['lanes'])))
            roads.append({'k': kind, 'w': round(width, 1), 'one': tags.get('oneway') == 'yes', 'p': pts})
        elif pts[0] == pts[-1]:
            areas.append({'k': 'water' if tags.get('natural') == 'water' else 'parking' if tags.get('amenity') == 'parking' else 'green', 'p': pts})
    out = {'credit': '(c) OpenStreetMap contributors', 'map': 'osm-full', 'roads': roads, 'areas': areas}
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {a.out}: {len(roads)} roads, {len(areas)} areas, {os.path.getsize(a.out) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
