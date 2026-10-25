"""Top-down preview of generated city districts (one panel per difficulty level), with sample
flights: spawn (circle), goal (cross) and the shortest free path between them.

    python scripts/preview_city.py --seed 0 --drones 6 --out docs/figures/city_levels.png
"""
import argparse
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.city_generator import DIFFICULTY, generate_city, sample_flights  # noqa: E402


def path(city, a, b):
    """Cells of the shortest 8-connected free path (for drawing)."""
    occ, xs, ys, res = city['occ'], city['xs'], city['ys'], city['res']
    ij = lambda p: (int(round((p[1] - ys[0]) / res)), int(round((p[0] - xs[0]) / res)))
    s, g = ij(a), ij(b)
    dist = np.full(occ.shape, np.inf)
    prev = {}
    dist[s] = 0
    q = deque([s])
    while q:
        i, j = q.popleft()
        for di, dj, w in ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414),
                          (1, -1, 1.414), (1, 1, 1.414)):
            ni, nj = i + di, j + dj
            if 0 <= ni < occ.shape[0] and 0 <= nj < occ.shape[1] and not occ[ni, nj] and dist[i, j] + w < dist[ni, nj]:
                dist[ni, nj] = dist[i, j] + w
                prev[(ni, nj)] = (i, j)
                q.append((ni, nj))
    cells, c = [], g
    while c in prev:
        cells.append(c)
        c = prev[c]
    return [(xs[j], ys[i]) for i, j in reversed(cells + [s])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--drones', type=int, default=6)
    ap.add_argument('--out', default='docs/figures/city_levels.png')
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, len(DIFFICULTY), figsize=(5.2 * len(DIFFICULTY), 5.6))
    for ax, level in zip(axes, DIFFICULTY):
        rng = np.random.RandomState(args.seed)
        city = generate_city(rng, level)
        spawns, goals = sample_flights(rng, city, args.drones)
        W, H = city['size']
        ax.imshow(city['occ'], origin='lower', extent=(-W / 2, W / 2, -H / 2, H / 2), cmap='Greys', alpha=0.12)
        for ob in city['layout']:
            (cx, cy), (hx, hy) = ob['center'], ob['half']
            shade = 0.35 + 0.5 * (ob['height'] - 6) / 14
            ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, color=(shade * 0.3, shade * 0.45, shade * 0.7)))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for k, key in enumerate(spawns):
            a, b = spawns[key][:2], goals[key][:2]
            p = np.array(path(city, a, b))
            ax.plot(p[:, 0], p[:, 1], '-', color=colors[k % 10], lw=1.4, alpha=0.85)
            ax.plot(*a, 'o', color=colors[k % 10], ms=6)
            ax.plot(*b, 'x', color=colors[k % 10], ms=7, mew=2)
        ax.set_xlim(-W / 2, W / 2); ax.set_ylim(-H / 2, H / 2); ax.set_aspect('equal')
        widths = [b - a for a, b in city['streets_x'] + city['streets_y']]
        ax.set_title(f"{level}: {len(city['layout'])} buildings, streets {min(widths):.1f}-{max(widths):.1f} m",
                     fontsize=10)
        ax.set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    fig.suptitle('Procedural city districts (darker = taller, 6-20 m); circle = spawn, x = goal, line = shortest free path',
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
