"""
c3_lidar_compare.py
-------------------
C3: the Webots Lidar scan vs the analytic scan, per ray, from the logs eval_city_routes.py
--lidar writes (<root>/routes_<prefix>_<map>.route.lidar.npz; envs/webots_lidar.py ScanLog).

For every logged (control step, drone, ray) of the 46 rays the policy reads (14 fan + 32 ring):
    analytic   what the policy was trained on (logged; re-cast here from the drone's true position
               along the goal-heading ray to label what it hit: building / drone / ground / nothing)
    lidar      the Webots Lidar's reading of the nearest Lidar ray, analytic conventions (ground and
               beyond range = range)
    reference  the box model cast from the SCAN pose (GPS 4 ms earlier) along the exact direction the
               Lidar ray had (its layer and column, turned by the drone's full roll / pitch / yaw):
               what a perfect Lidar would read in the analytic world. So
                   lidar - reference      = geometry / rendering (building meshes vs boxes, the
                                            Crazyflie vs the 0.3 m box)
                   reference - analytic   = sampling (direction rounding, tilt, the 4 ms age)
               Other drones are taken at the state time (they move <= ~8 mm in 4 ms).
Counted as a hit: reading < range - 1e-3.

Tables (per map and pooled), by what the analytic ray hit, by ray group, by tilt at the scan, and
for drone rays by distance: rays, hit rates, missed (analytic hit, Lidar none), extra (Lidar hit,
analytic none), |range error| median / p95 / max where both hit, mean |proximity error| (proximity
= 1 - d / range, what the policy sees), and the same split by the reference.

Usage (repo root): python scripts/c3_lidar_compare.py --root eval_results/c3 --prefix analytic
Writes <root>/<name>.md and .json (--name, default c3_lidar_compare).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import lidar as lidar_mod                                         # noqa: E402
from envs import webots_lidar as wl                                         # noqa: E402
from envs.webots_adapter import _slab_entry, _cylinder_entry, RAY_START, DRONE_HALF   # noqa: E402

MAPS = ['easy', 'medium', 'hard', 'crop']
RANGE, N_FAN, N_RING = 6.0, 14, 32
HIT = RANGE - 1e-3
KINDS = ['nothing', 'building', 'drone', 'ground']
GROUPS = {'fan horizontal': [0, 1, 2, 3], 'fan up': [4], 'fan down': [5], 'fan up 45': [6, 7, 8, 9],
          'fan down 45': [10, 11, 12, 13], 'ring': list(range(14, 46))}
TILT_BINS = [(0, 2), (2, 5), (5, 10), (10, 90)]
DIST_BINS = [(0.3, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.0)]


def cast(centre, d, layout, dpos, dyaw, owner, rng=RANGE):
    """The analytic rules of webots_adapter.scan_lidar for arbitrary rays.
    centre, d: (R, 3); dpos: (R, n, 3), dyaw: (R, n) the drones for each ray; owner: (R,) the
    drone the ray belongs to. Returns (distance, kind index into KINDS)."""
    R = len(d)
    o = centre + RAY_START * d
    seg = rng - RAY_START
    best, kind = np.full(R, np.inf), np.zeros(R, int)

    def consider(t_in, t_out, k):
        nonlocal best, kind
        hit = (t_in >= 0.0) & (t_in <= t_out) & (t_in <= seg)
        closer = hit & (t_in < best)
        best = np.where(closer, t_in, best)
        kind = np.where(closer, k, kind)

    with np.errstate(divide='ignore', invalid='ignore'):
        tg = np.where(d[:, 2] < 0.0, -o[:, 2] / d[:, 2], np.inf)
    consider(tg, np.full(R, np.inf), 3)
    for ob in layout:
        cx, cy = ob['center']
        h = ob['height']
        if ob['type'] == 'pillar':
            t_in, t_out = _cylinder_entry(o, d, cx, cy, ob['radius'], h)
        else:
            hx, hy = ob['half']
            yaw = ob.get('yaw', 0.0)
            if yaw:
                c, s_ = math.cos(yaw), math.sin(yaw)
                rx, ry = o[:, 0] - cx, o[:, 1] - cy
                ob_o = np.stack([c * rx + s_ * ry, -s_ * rx + c * ry, o[:, 2]], axis=1)
                ob_d = np.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
                t_in, t_out = _slab_entry(ob_o, ob_d, np.array([-hx, -hy, 0.0]), np.array([hx, hy, h]))
            else:
                t_in, t_out = _slab_entry(o, d, np.array([cx - hx, cy - hy, 0.0]), np.array([cx + hx, cy + hy, h]))
        consider(t_in, t_out, 1)
    for j in range(dpos.shape[1]):
        c, s = np.cos(dyaw[:, j]), np.sin(dyaw[:, j])
        rel = o - dpos[:, j]
        ol = np.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1)
        dl = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
        # per-ray box: _slab_entry wants (k,) bounds, so do it by hand
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = 1.0 / dl
            t1, t2 = (-DRONE_HALF - ol) * inv, (DRONE_HALF - ol) * inv
        near, far = np.minimum(t1, t2), np.maximum(t1, t2)
        par = dl == 0.0
        inside = np.abs(ol) <= DRONE_HALF
        near = np.where(par, np.where(inside, -np.inf, np.inf), near)
        far = np.where(par, np.where(inside, np.inf, -np.inf), far)
        t_in, t_out = near.max(axis=1), far.min(axis=1)
        t_in = np.where(owner == j, np.inf, t_in)
        consider(t_in, t_out, 2)
    seen = np.isfinite(best) & (kind != 3)
    return np.where(seen, RAY_START + best, rng), np.where(np.isfinite(best), kind, 0)


def lidar_world_dirs(heading, scan_rpy, layers=5, res=1440):
    """(R, 3) world directions of the Lidar rays the 46 analytic rays were read from."""
    lay, col = wl.ray_index(wl.analytic_dirs(heading, N_RING), scan_rpy[2], layers, res)
    el, az = wl.layer_elevations(layers)[lay], wl.column_azimuths(res)[col]
    body = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
    return body @ wl.rot_rpy(scan_rpy).T


def analyse_map(path):
    z = np.load(path)
    layouts = json.loads(str(z['layouts_json']))
    stats = json.loads(str(z['stats_json']))
    keep = ~z['parked'] & np.isfinite(z['t_scan'])
    out = {k: [] for k in ('ana', 'lid', 'ref', 'kind', 'ref_kind', 'ray', 'tilt', 'ana_check')}
    for ep in np.unique(z['ep'][keep]):
        idx = np.flatnonzero(keep & (z['ep'] == ep))
        layout = layouts[str(int(ep))]
        n_rows = len(idx)
        heading, pos, srpy, spos = z['heading'][idx], z['pos'][idx], z['scan_rpy'][idx], z['scan_pos'][idx]
        all_pos, all_yaw, owner = z['all_pos'][idx], z['all_yaw'][idx], z['drone'][idx]
        nr = N_FAN + N_RING
        d_ana = np.concatenate([wl.analytic_dirs(h, N_RING) for h in heading])
        d_lid = np.concatenate([lidar_world_dirs(h, r) for h, r in zip(heading, srpy)])
        rep = lambda a: np.repeat(a, nr, axis=0)                                  # noqa: E731
        ana, kind = cast(rep(pos), d_ana, layout, rep(all_pos), rep(all_yaw), rep(owner))
        ref, ref_kind = cast(rep(spos), d_lid, layout, rep(all_pos), rep(all_yaw), rep(owner))
        out['ana'].append(z['analytic'][idx].astype(float).ravel())
        out['ana_check'].append(ana)
        out['lid'].append(z['lidar'][idx].astype(float).ravel())
        out['ref'].append(ref)
        out['kind'].append(kind)
        out['ref_kind'].append(ref_kind)
        out['ray'].append(np.tile(np.arange(nr), n_rows))
        out['tilt'].append(rep(np.degrees(np.max(np.abs(srpy[:, :2]), axis=1))))
    d = {k: np.concatenate(v) for k, v in out.items()}
    d['stats'] = stats
    d['steps'] = int(keep.sum())
    return d


def summarise(a, l, mask):
    m = mask
    n = int(m.sum())
    if not n:
        return {'rays': 0}
    ah, lh = a[m] < HIT, l[m] < HIT
    both = ah & lh
    err = np.abs(l[m] - a[m])[both]
    prox = np.abs((1 - l[m] / RANGE) - (1 - a[m] / RANGE))
    r = {'rays': n, 'analytic_hit_pct': 100 * ah.mean(), 'lidar_hit_pct': 100 * lh.mean(),
         'missed_pct_of_analytic_hits': 100 * (ah & ~lh).sum() / max(1, ah.sum()),
         'extra_pct_of_analytic_none': 100 * (lh & ~ah).sum() / max(1, (~ah).sum()),
         'missed': int((ah & ~lh).sum()), 'extra': int((lh & ~ah).sum()), 'both': int(both.sum()),
         'prox_err_mean': float(prox.mean()), 'prox_err_p99': float(np.percentile(prox, 99))}
    if both.any():
        r.update(err_median=float(np.median(err)), err_p95=float(np.percentile(err, 95)),
                 err_max=float(err.max()))
    return r


def tables(d):
    a, l, ref = d['ana'], d['lid'], d['ref']
    t = {'all': summarise(a, l, np.ones_like(a, bool))}
    t['by_kind'] = {k: summarise(a, l, d['kind'] == i) for i, k in enumerate(KINDS)}
    t['by_group'] = {g: summarise(a, l, np.isin(d['ray'], rays)) for g, rays in GROUPS.items()}
    t['by_tilt'] = {f'{lo}-{hi} deg': summarise(a, l, (d['tilt'] >= lo) & (d['tilt'] < hi)) for lo, hi in TILT_BINS}
    dr = d['kind'] == 2
    t['drone_by_dist'] = {f'{lo}-{hi} m': summarise(a, l, dr & (a >= lo) & (a < hi)) for lo, hi in DIST_BINS}
    # split: sampling (reference vs analytic) and geometry (lidar vs reference)
    t['sampling_ref_vs_analytic'] = {k: summarise(a, ref, d['kind'] == i) for i, k in enumerate(KINDS)}
    t['geometry_lidar_vs_ref'] = {k: summarise(ref, l, d['ref_kind'] == i) for i, k in enumerate(KINDS)}
    t['analytic_recast_max_diff_m'] = float(np.max(np.abs(d['ana_check'] - a)))
    t['scan_stats'] = d['stats']
    t['steps_drones'] = d['steps']
    return t


HIT_KINDS, NONE_KINDS = ('building', 'drone'), ('nothing', 'ground')


def fmt_table(title, rows, first='', by_kind=False):
    """by_kind: rows are hit categories; 'extra' is not meaningful for a hit category (its
    denominator is the handful of rays of that category read as no hit) and 'missed' not for
    nothing / ground (no hits to miss): shown as '-'."""
    cols = ['rays', 'analytic hit', 'lidar hit', 'missed', 'extra', '|err| median (m)', '|err| p95 (m)',
            '|err| max (m)', 'mean |prox err|']
    s = [f'**{title}**', '', '| ' + ' | '.join([first] + cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for k, r in rows.items():
        if not r.get('rays'):
            s.append(f'| {k} | 0 |' + ' |' * (len(cols) - 1))
            continue
        g = lambda x: '-' if x is None else f'{x:.3f}'                         # noqa: E731
        missed = (f"{r['missed']} ({r['missed_pct_of_analytic_hits']:.2f}%)"
                  if not (by_kind and k in NONE_KINDS) else '-')
        extra = (f"{r['extra']} ({r['extra_pct_of_analytic_none']:.3f}%)"
                 if not (by_kind and k in HIT_KINDS) else '-')
        s.append(f"| {k} | {r['rays']} | {r['analytic_hit_pct']:.2f}% | {r['lidar_hit_pct']:.2f}% | "
                 f"{missed} | {extra} | {g(r.get('err_median'))} | {g(r.get('err_p95'))} | "
                 f"{g(r.get('err_max'))} | {r['prox_err_mean']:.4f} |")
    return '\n'.join(s + [''])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='eval_results/c3')
    ap.add_argument('--prefix', default='analytic')
    ap.add_argument('--maps', nargs='+', default=MAPS)
    ap.add_argument('--name', default='c3_lidar_compare', help='output file stem in --root')
    ap.add_argument('--from-json', action='store_true', help='only re-write the .md from <name>.json')
    a = ap.parse_args()
    res, pooled = {}, {}
    if a.from_json:
        res = json.load(open(os.path.join(a.root, a.name + '.json')))
    for m in ([] if a.from_json else a.maps):
        p = os.path.join(a.root, f'routes_{a.prefix}_{m}.route.lidar.npz')
        if not os.path.exists(p):
            print(f'missing {p}, skipped')
            continue
        d = analyse_map(p)
        res[m] = tables(d)
        for k in ('ana', 'lid', 'ref', 'kind', 'ref_kind', 'ray', 'tilt', 'ana_check'):
            pooled.setdefault(k, []).append(d[k])
        print(f'{m}: {d["steps"]} drone-steps', flush=True)
    if pooled:
        P = {k: np.concatenate(v) for k, v in pooled.items()}
        P['stats'], P['steps'] = {}, sum(r['steps_drones'] for r in res.values())
        res['ALL'] = tables(P)
    if not a.from_json:
        json.dump(res, open(os.path.join(a.root, a.name + '.json'), 'w'), indent=1)
    md = [f'# C3: Webots Lidar vs analytic scan, per ray (`{a.root}`, prefix `{a.prefix}`)', '',
          'Generated by `scripts/c3_lidar_compare.py`. Rows: 46 rays (14 fan + 32 ring) per drone and '
          'control step, parked drones left out. "missed" = analytic hit, Lidar none (% of analytic hits); '
          '"extra" = Lidar hit, analytic none (% of analytic none). |err| over rays where both hit. '
          'A hit is a reading below range - 1 mm. In the tables split by hit category, "extra" is shown '
          'as "-" for building / drone (its denominator is only the few rays of that category whose '
          'analytic hit lies within 1 mm of the range) and "missed" as "-" for nothing / ground (there '
          'is no hit to miss).', '']
    for m, t in res.items():
        md += [f'## {m}', '', f"drone-steps {t['steps_drones']}; analytic re-cast max diff "
               f"{t['analytic_recast_max_diff_m']:.2e} m; scan stats {t['scan_stats']}", '',
               fmt_table('By what the analytic ray hit', t['by_kind'], 'analytic hit', by_kind=True),
               fmt_table('By ray group', t['by_group'], 'group'),
               fmt_table('By tilt at the scan (max |roll|, |pitch|)', t['by_tilt'], 'tilt'),
               fmt_table('Drone rays by analytic distance', t['drone_by_dist'], 'distance'),
               fmt_table('Sampling: reference (box model, Lidar ray, scan pose) vs analytic, by analytic hit',
                         t['sampling_ref_vs_analytic'], 'analytic hit', by_kind=True),
               fmt_table('Geometry: Lidar vs reference, by what the reference hit', t['geometry_lidar_vs_ref'],
                         'reference hit', by_kind=True)]
    open(os.path.join(a.root, a.name + '.md'), 'w').write('\n'.join(md))
    print(f"wrote {os.path.join(a.root, a.name + '.md')}")


if __name__ == '__main__':
    main()
