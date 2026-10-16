"""
webots_l2_sysid.py
------------------
System identification of the Level-2 quadrotor (envs/webots_l2.py): how does
the drone's velocity follow the policy's velocity command?

  1. Velocity steps from hover in x, y (0.5 / 1 / 2 m/s) and z (0.25 / 0.5 /
     1 m/s), held 4 s, then back to 0 for 3 s. Per step: dead time (time to
     10% of the change), time constant tau of a first-order-plus-dead-time fit
     v(t) = A (1 - exp(-(t - L) / tau)), overshoot, 5% settling time, final
     tracking error, and the same for the step back to 0.
  2. Hover jitter: 8 s of hover after settling; speed statistics against the
     0.15 m/s settle threshold (and the settle check's 31-step window).
Writes logs/webots/l2_sysid.json, two plots, and prints a summary with a
suggested discrete lag for PyBullet training (32 ms control step).

Usage (repo root): python scripts/webots_l2_sysid.py [--out logs/webots]
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.webots_l2 import WebotsL2Adapter     # noqa: E402

AX = {'x': 0, 'y': 1, 'z': 2}


def fly(ad, cmd, seconds):
    """Hold a velocity command for some seconds; returns (t, vel, pos) per basic step."""
    ad.trace = []
    for _ in range(int(round(seconds / ad.control_dt))):
        ad.apply_actions({'drone_0': list(cmd) + [0.0]})
        ad.step_simulation()
    tr, ad.trace = ad.trace, None
    return (np.array([r[0] for r in tr]), np.array([r[2] for r in tr]), np.array([r[1] for r in tr]),
            np.array([r[3] for r in tr]))


def fopdt_fit(t, y, y0, y1):
    """Fit y(t) = y0 + (y1 - y0)(1 - exp(-(t - L)/tau)) for t > L (t from the step)."""
    best = (np.inf, 0.0, 0.0)
    for L in np.arange(0.0, 0.3, 0.004):
        for tau in np.arange(0.01, 1.5, 0.005):
            m = y0 + (y1 - y0) * np.where(t > L, 1 - np.exp(-(t - L) / tau), 0.0)
            e = np.sum((m - y) ** 2)
            if e < best[0]:
                best = (e, L, tau)
    return best[1], best[2], float(np.sqrt(best[0] / len(t)))


def analyse(t, v, y0, target):
    """Step metrics of one velocity component; t starts at the command change."""
    t = t - t[0] + (t[1] - t[0])
    final = float(np.mean(v[t > t[-1] - 1.0]))
    span = final - y0
    frac = (v - y0) / span if abs(span) > 1e-6 else np.zeros_like(v)
    def first(th):
        idx = np.where(frac >= th)[0]
        return float(t[idx[0]]) if len(idx) else None
    over = float(max(0.0, (np.max(frac) - 1.0))) * 100 if span > 0 else float(max(0.0, (np.max(-frac + 2) - 2.0 + 1 - 1))) * 100
    peak = float(np.max(frac))
    out_band = np.where(np.abs(frac - 1.0) > 0.05)[0]
    settle = float(t[out_band[-1]]) if len(out_band) else 0.0
    L, tau, rmse = fopdt_fit(t, v, y0, final)
    return {'target': target, 'final': final, 'tracking_error': final - target,
            't10': first(0.1), 't63': first(0.632), 't90': first(0.9),
            'overshoot_pct': max(0.0, (peak - 1.0) * 100), 'settle_5pct_s': settle,
            'fit_delay_s': L, 'fit_tau_s': tau, 'fit_rmse': rmse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config_stage2_ring.yaml')
    ap.add_argument('--out', default='logs/webots')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    cfg['environment']['n_agents'] = 1
    ad = WebotsL2Adapter(cfg)
    res = {'steps': [], 'hover': {}, 'config': {'basic_step_s': ad.dt, 'control_step_s': ad.control_dt}}
    curves = []
    for axis, amps in (('x', (0.5, 1.0, 2.0)), ('y', (0.5, 1.0, 2.0)), ('z', (0.25, 0.5, 1.0))):
        for amp in amps:
            ad.reset_world({'drone_0': [0, 0, 3.0]}, {'drone_0': [20, 0, 3.0]})
            fly(ad, [0, 0, 0], 1.0)
            cmd = [0.0, 0.0, 0.0]; cmd[AX[axis]] = amp
            t1, v1, p1, r1 = fly(ad, cmd, 4.0)
            t2, v2, p2, r2 = fly(ad, [0, 0, 0], 3.0)
            k = AX[axis]
            up = analyse(t1, v1[:, k], 0.0, amp)
            down = analyse(t2, v2[:, k], up['final'], 0.0)
            other = [j for j in range(3) if j != k]
            cross = float(np.max(np.abs(v1[:, other]))) if len(other) else 0.0
            alt_dev = float(np.max(np.abs(p1[:, 2] - 3.0))) if axis != 'z' else None
            tilt = float(np.degrees(np.max(np.abs(r1[:, :2]))))
            res['steps'].append({'axis': axis, 'amp': amp, 'up': up, 'down': down,
                                 'max_cross_axis_speed': cross, 'max_altitude_deviation_m': alt_dev,
                                 'max_tilt_deg': tilt})
            curves.append((axis, amp, t1 - t1[0], v1[:, k], t2 - t1[0], v2[:, k]))
            print(f"{axis} step {amp:4.2f} m/s: delay {up['fit_delay_s']*1000:5.1f} ms, tau {up['fit_tau_s']*1000:5.1f} ms, "
                  f"t63 {up['t63']}, overshoot {up['overshoot_pct']:5.1f}%, settle {up['settle_5pct_s']:.2f}s, "
                  f"final {up['final']:.3f} | down: delay {down['fit_delay_s']*1000:5.1f} ms tau {down['fit_tau_s']*1000:5.1f} ms "
                  f"overshoot {down['overshoot_pct']:5.1f}% | tilt {tilt:4.1f} deg, alt dev {alt_dev}")
    # hover jitter
    ad.reset_world({'drone_0': [0, 0, 3.0]}, {'drone_0': [20, 0, 3.0]})
    fly(ad, [0, 0, 0], 3.0)
    t, v, p, r = fly(ad, [0, 0, 0], 8.0)
    sp = np.linalg.norm(v, axis=1)
    per_ctrl = sp[::int(round(ad.control_dt / ad.dt))]
    res['hover'] = {'speed_mean': float(sp.mean()), 'speed_p95': float(np.percentile(sp, 95)), 'speed_max': float(sp.max()),
                    'speed_max_at_control_steps': float(per_ctrl.max()),
                    'position_std_m': [float(x) for x in p.std(axis=0)],
                    'position_drift_m': float(np.linalg.norm(p[-1] - p[0])),
                    'below_settle_threshold_share': float(np.mean(per_ctrl < 0.15))}
    print("hover:", {k: (np.round(v_, 5).tolist() if isinstance(v_, list) else round(v_, 5)) for k, v_ in res['hover'].items()})
    ad.close()
    # summary + suggested lag model for PyBullet (32 ms control step)
    for axis in ('x', 'y', 'z'):
        s = [st for st in res['steps'] if st['axis'] == axis]
        L = float(np.median([st['up']['fit_delay_s'] for st in s]))
        tau = float(np.median([st['up']['fit_tau_s'] for st in s]))
        res.setdefault('summary', {})[axis] = {
            'delay_s': L, 'tau_s': tau,
            'delay_control_steps': L / ad.control_dt,
            'alpha_per_control_step': float(1 - np.exp(-ad.control_dt / tau)),
            'overshoot_pct_max': float(max(st['up']['overshoot_pct'] for st in s)),
            'overshoot_pct_down_max': float(max(st['down']['overshoot_pct'] for st in s))}
    json.dump(res, open(os.path.join(a.out, 'l2_sysid.json'), 'w'), indent=2)
    print('summary:', json.dumps(res['summary'], indent=1))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, axis in zip(axs, ('x', 'y', 'z')):
            for (a_, amp, ta, va, tb, vb) in curves:
                if a_ != axis:
                    continue
                line, = ax.plot(np.r_[ta, tb], np.r_[va, vb], lw=1.4, label=f'{amp} m/s')
                ax.plot([0, 4, 4, 7], [amp, amp, 0, 0], '--', lw=0.8, color=line.get_color())
            ax.set_title(f'v{axis} step response (Webots L2)'); ax.set_xlabel('time since command (s)')
            ax.set_ylabel('velocity (m/s)'); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, 'l2_sysid_steps.png'), dpi=120)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(t - t[0], sp, lw=1); ax.axhline(0.15, color='r', ls='--', lw=1, label='settle threshold 0.15 m/s')
        ax.set_xlabel('time (s)'); ax.set_ylabel('speed (m/s)'); ax.set_title('hover speed jitter (Webots L2)')
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(a.out, 'l2_sysid_hover.png'), dpi=120)
    except Exception as e:     # plots are optional
        print('plot failed:', e)


if __name__ == '__main__':
    main()
