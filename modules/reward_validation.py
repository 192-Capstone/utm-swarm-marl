"""
reward_validation.py
--------------------
Validates the actual RewardEngine (reward_engine.py) using synthetic trajectories.
No Webots, no trained policy needed.

Run with:
    python reward_validation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os

# Add parent directory to path so we can import reward_engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reward_engine import RewardEngine

# ── Minimal config matching default_config.yaml ───────────────────────────────
CONFIG = {
    "reward": {
        "goal_progress_scale":        1.0,
        "goal_reached_bonus":         100.0,
        "obstacle_collision_penalty": -50.0,
        "drone_collision_penalty":    -50.0,
        "step_penalty":               -0.01,
        "lambda_smooth":               0.1,
        "lambda_prox":                 0.5,
        "r_danger":                    5.0,
    },
    "environment": {
        "goal_threshold":   0.5,
        "collision_radius": 0.3,
    }
}

# ── Helper: build single-drone data dict ──────────────────────────────────────
def make_data(prev_dist, curr_dist, collision, drone_distances, action, prev_action):
    return {
        "prev_dist":       prev_dist,
        "curr_dist":       curr_dist,
        "collision":       collision,
        "drone_distances": drone_distances,
        "action":          np.array(action),
        "prev_action":     np.array(prev_action),
    }

# ── Scenario Runners ──────────────────────────────────────────────────────────

def run_scenario_1(n_steps=200):
    """Scenario 1: Drone navigating directly toward goal."""
    engine = RewardEngine(CONFIG)

    pos  = np.array([0.0, 0.0, 5.0])
    goal = np.array([20.0, 0.0, 5.0])
    speed = 0.2
    prev_action = np.zeros(4)
    prev_dist   = float(np.linalg.norm(pos - goal))

    history = {k: [] for k in ['goal_progress', 'step_penalty', 'total']}
    running_totals = {k: 0.0 for k in history.keys()}

    for _ in range(n_steps):
        direction   = (goal - pos)
        dist        = float(np.linalg.norm(direction))
        direction   = direction / (dist + 1e-6)
        curr_action = np.append(direction[:3] * speed, 0.0)
        pos         = pos + direction * speed
        curr_dist   = float(np.linalg.norm(pos - goal))

        data    = make_data(prev_dist, curr_dist, False, [], curr_action, prev_action)
        rewards = engine.compute_rewards_single(data)

        # Accumulate rewards for plotting
        for k in history:
            running_totals[k] += rewards[k]
            history[k].append(running_totals[k])

        prev_action = curr_action.copy()
        prev_dist   = curr_dist

        if curr_dist < CONFIG['environment']['goal_threshold']:
            # Add final goal bonus to the last point
            history['total'][-1] += CONFIG['reward']['goal_reached_bonus']
            break

    return history


def run_scenario_2(n_steps=300):
    """Scenario 2: Two drones on direct collision course."""
    engine = RewardEngine(CONFIG)

    # Started them 10 meters apart instead of 20
    pos_a  = np.array([-5.0, 0.0, 5.0])
    pos_b  = np.array([ 5.0, 0.0, 5.0])
    goal_a = np.array([ 20.0, 0.0, 5.0])
    
    # SLOW MOTION: Slowed down from 0.15 to 0.02 to capture more data points
    speed  = 0.02

    prev_action_a = np.zeros(4)
    prev_dist_a   = float(np.linalg.norm(pos_a - goal_a))

    history = {k: [] for k in ['proximity_penalty', 'drone_collision', 'total']}
    drone_separation = []
    running_totals = {k: 0.0 for k in history.keys()}

    for _ in range(n_steps):
        dir_a = (goal_a - pos_a) / (np.linalg.norm(goal_a - pos_a) + 1e-6)
        dir_b = -dir_a

        curr_action_a = np.append(dir_a[:3] * speed, 0.0)
        pos_a = pos_a + dir_a * speed
        pos_b = pos_b + dir_b * speed

        separation  = float(np.linalg.norm(pos_a - pos_b))
        curr_dist_a = float(np.linalg.norm(pos_a - goal_a))

        data = make_data(prev_dist_a, curr_dist_a, False, [separation], curr_action_a, prev_action_a)
        rewards = engine.compute_rewards_single(data)

        # Accumulate rewards
        for k in history:
            running_totals[k] += rewards[k]
            history[k].append(running_totals[k])
        
        # Track physical distance directly
        drone_separation.append(separation)

        prev_action_a = curr_action_a.copy()
        prev_dist_a   = curr_dist_a

        if rewards['drone_collision'] < 0 and len(history['total']) > 20:
            remaining = n_steps - len(history['total'])
            for k in history:
                history[k].extend([history[k][-1]] * remaining)
            drone_separation.extend([drone_separation[-1]] * remaining)
            break

    return history, drone_separation

def run_scenario_3(n_steps=150):
    """Scenario 3: Smooth vs jerky actions."""
    engine_s = RewardEngine(CONFIG)
    engine_j = RewardEngine(CONFIG)

    pos   = np.array([0.0, 0.0, 5.0])
    goal  = np.array([20.0, 0.0, 5.0])
    speed = 0.15

    smooth_penalties = []
    jerky_penalties  = []
    sum_s = 0.0
    sum_j = 0.0

    prev_s    = np.zeros(4)
    prev_j    = np.zeros(4)
    dist      = float(np.linalg.norm(pos - goal))
    pos_s     = pos.copy()

    for step in range(n_steps):
        direction = (goal - pos_s) / (np.linalg.norm(goal - pos_s) + 1e-6)
        pos_s     = pos_s + direction * speed
        curr_dist = float(np.linalg.norm(pos_s - goal))

        smooth_action = np.append(direction[:3] * speed, 0.0)

        if (step // 5) % 2 == 0:
            jerky_action = np.append(direction[:3] * speed * 2.5, 0.0)
        else:
            jerky_action = np.append(-direction[:3] * speed * 1.5, 0.0)

        rs = engine_s.compute_rewards_single(make_data(dist, curr_dist, False, [], smooth_action, prev_s))
        rj = engine_j.compute_rewards_single(make_data(dist, curr_dist, False, [], jerky_action, prev_j))

        # Accumulate penalties
        sum_s += rs['smoothness_penalty']
        sum_j += rj['smoothness_penalty']
        
        smooth_penalties.append(sum_s)
        jerky_penalties.append(sum_j)

        prev_s = smooth_action.copy()
        prev_j = jerky_action.copy()
        dist   = curr_dist

    return smooth_penalties, jerky_penalties


def run_scenario_4(n_steps=300):
    """Scenario 4: 3 drones with different behaviors."""
    engine = RewardEngine(CONFIG)

    pos1  = np.array([0.0,  0.0, 5.0]);  goal1 = np.array([20.0,  0.0, 5.0])
    pos2  = np.array([0.0, 10.0, 5.0]);  goal2 = np.array([20.0, 10.0, 5.0])
    pos3  = np.array([0.0, 20.0, 5.0]);  goal3 = np.array([20.0, 20.0, 5.0])
    speed = 0.15

    cum   = {'drone_1': [], 'drone_2': [], 'drone_3': []}
    c1 = c2 = c3 = 0.0

    pa1 = pa2 = pa3 = np.zeros(4)
    d1 = float(np.linalg.norm(pos1 - goal1))
    d2 = float(np.linalg.norm(pos2 - goal2))
    d3 = float(np.linalg.norm(pos3 - goal3))

    for _ in range(n_steps):
        dir1 = (goal1 - pos1) / (np.linalg.norm(goal1 - pos1) + 1e-6)
        a1   = np.append(dir1[:3] * speed, 0.0)
        pos1 = pos1 + dir1 * speed
        cd1  = float(np.linalg.norm(pos1 - goal1))

        a2   = np.zeros(4)
        cd2  = d2

        dir3 = (goal3 - pos3) / (np.linalg.norm(goal3 - pos3) + 1e-6)
        a3   = np.append(-dir3[:3] * speed, 0.0)
        pos3 = pos3 - dir3 * speed
        cd3  = float(np.linalg.norm(pos3 - goal3))

        r1 = engine.compute_rewards_single(make_data(d1, cd1, False, [], a1, pa1))
        r2 = engine.compute_rewards_single(make_data(d2, cd2, False, [], a2, pa2))
        r3 = engine.compute_rewards_single(make_data(d3, cd3, False, [], a3, pa3))

        c1 += r1['total'];  cum['drone_1'].append(c1)
        c2 += r2['total'];  cum['drone_2'].append(c2)
        c3 += r3['total'];  cum['drone_3'].append(c3)

        pa1 = a1.copy(); pa2 = a2.copy(); pa3 = a3.copy()
        d1 = cd1; d3 = cd3

        if r1['goal_reached'] > 0:
            remaining = n_steps - len(cum['drone_1'])
            for k in cum:
                cum[k].extend([cum[k][-1]] * remaining)
            break

    return cum

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all():
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Reward Engine Validation (Cumulative Over Time)\n'
        'Synthetic trajectory tests',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Goal-directed navigation ──
    ax1 = fig.add_subplot(gs[0, 0])
    h1  = run_scenario_1()
    s1  = range(len(h1['total']))

    ax1.plot(s1, h1['goal_progress'],   color='#2ecc71', label='Cumul. Goal Progress',  lw=2)
    ax1.plot(s1, h1['step_penalty'],    color='#e74c3c', label='Cumul. Step Penalty',   lw=1.5, ls='--')
    ax1.plot(s1, h1['total'],           color='#2c3e50', label='Total Cumulative',      lw=2.5)
    ax1.axhline(0, color='gray', lw=0.8, ls=':')
    ax1.set_title('Scenario 1: Goal-Directed Navigation\n'
                  'Steady climb confirms goal progression is mathematically stable', fontsize=10)
    ax1.set_xlabel('Timesteps'); ax1.set_ylabel('Cumulative Reward')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # ── Plot 2: Collision course ──
    ax2     = fig.add_subplot(gs[0, 1])
    h2, sep = run_scenario_2()
    s2      = range(len(h2['total']))
    ax2_r   = ax2.twinx()

    ax2.plot(s2, h2['proximity_penalty'], color='#9b59b6', label='Cumul. Proximity Penalty', lw=2)
    ax2.plot(s2, h2['drone_collision'],   color='#c0392b', label='Drone Collision (Terminal)', lw=2)
    ax2.plot(s2, h2['total'],             color='#2c3e50', label='Total Cumulative',      lw=2.5)
    ax2_r.plot(s2, sep, color='#3498db', label='Separation Distance (m)', lw=1.5, ls='-.')
    
    ax2_r.axhline(1.5, color='#3498db', lw=0.8, ls=':', alpha=0.6)
    ax2_r.text(5, 1.6, 'Danger Zone = 1.5m', color='#3498db', fontsize=9)
    ax2_r.set_ylabel('Separation (m)', color='#3498db', fontsize=10)
    ax2.axhline(0, color='gray', lw=0.8, ls=':')
    ax2.set_title('Scenario 2: Head-On Collision\n'
                  'Penalty smoothly accumulates as drones enter the 1.5m danger zone', fontsize=10)
    ax2.set_xlabel('Timesteps'); ax2.set_ylabel('Cumulative Penalty')
    lines = ax2.get_legend_handles_labels()
    lines2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines[0]+lines2[0], lines[1]+lines2[1], loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Plot 3: Smooth vs Jerky ──
    ax3 = fig.add_subplot(gs[1, 0])
    sp, jp = run_scenario_3()
    s3 = range(len(sp))

    ax3.plot(s3, sp, color='#27ae60', label='Cumul. Smooth Penalty', lw=2)
    ax3.plot(s3, jp, color='#e74c3c', label='Cumul. Jerky Penalty',  lw=2)
    ax3.fill_between(s3, sp, jp, alpha=0.15, color='#e74c3c')
    ax3.axhline(0, color='gray', lw=0.8, ls=':')
    ax3.set_title('Scenario 3: Motor Wear (Smooth vs Jerky Actions)\n'
                  'Jerky drone heavily penalized over time to protect physical rotors', fontsize=10)
    ax3.set_xlabel('Timesteps'); ax3.set_ylabel('Cumulative Smoothness Penalty')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # ── Plot 4: Cumulative reward by behavior ──
    ax4 = fig.add_subplot(gs[1, 1])
    cum = run_scenario_4()
    s4  = range(len(cum['drone_1']))

    ax4.plot(s4, cum['drone_1'], color='#27ae60', label='Drone 1: Moving toward goal', lw=2.5)
    ax4.plot(s4, cum['drone_2'], color='#f39c12', label='Drone 2: Hovering (step penalty drains score)', lw=2)
    ax4.plot(s4, cum['drone_3'], color='#e74c3c', label='Drone 3: Moving away from goal', lw=2)
    ax4.axhline(0, color='gray', lw=0.8, ls=':')
    ax4.set_title('Scenario 4: Swarm Discrimination\n'
                  'Reward mathematically isolates and punishes bad drone behavior', fontsize=10)
    ax4.set_xlabel('Timesteps'); ax4.set_ylabel('Total Cumulative Reward')
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    plt.savefig('reward_validation_cumulative.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: reward_validation_cumulative.png")
    plt.show()

if __name__ == "__main__":
    print("Generating Cumulative Reward Plots...")
    plot_all()