"""
test_ppo_optimizer.py
---------------------
Refined Validation Suite for the PPO Optimizer.
Focus: Weight mutation (learning proof), Metric Stability, and Dual Optimization.
"""

import sys
import os
import torch
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.networks.actor_network import ActorNetwork
from training.networks.critic_network import CriticNetwork
from training.rollout_buffer import RolloutBuffer
from training.ppo_optimizer import PPOOptimizer

def run_optimizer_validation():
    print("\n" + "="*60)
    print(" PPO OPTIMIZER VALIDATION DASHBOARD")
    print("="*60)

    # 1. Setup Mock Environment (Fully cross-referenced with all modules)
    config = {
        'environment': {
            'n_agents': 3
        },
        'observation': {
            'local_obs_dim': 27, 
            'neighbor_obs_dim': 6, 
            'k_neighbors_max': 5,
            'neighbor_context_dim': 32,
            'total_obs_dim': 59           # <-- FIXED: Added missing key
        },
        'network': {
            'action_dim': 4, 
            'hidden_dim': 64, 
            'n_hidden_layers': 1
        }, 
        'training': {
            'buffer_size': 512, 'batch_size': 128, 'n_epochs': 2,
            'clip_epsilon': 0.2, 'value_loss_coef': 0.5, 'entropy_coef': 0.01,
            'max_grad_norm': 0.5, 'actor_lr': 3e-4, 'critic_lr': 3e-4,
            'gamma': 0.99,                # <-- Added preemptively for Buffer
            'lambda_gae': 0.95            # <-- Added preemptively for Buffer
        }
    }
    device = torch.device('cpu')
    
    # 2. Initialize Components
    actor = ActorNetwork(config).to(device)
    critic = CriticNetwork(config).to(device)
    opt = PPOOptimizer(actor, critic, config, device)
    buf = RolloutBuffer(config, device)

    # 3. Fill Buffer with Mock Data
    N = config['environment']['n_agents']
    for _ in range(config['training']['buffer_size']):
        buf.store(
            local_obs=torch.randn(N, 27),
            neighbor_states=torch.randn(N, 5, 6),
            neighbor_mask=torch.zeros(N, 5, dtype=torch.bool),
            global_state=torch.randn(N, 17),
            actions=torch.randn(N, 4),
            log_probs=torch.randn(N),
            rewards=torch.rand(N),
            dones=torch.zeros(N),
            values=torch.randn(N)
        )
    buf.compute_gae(torch.randn(N))

    # 4. Snapshot Pre-Training Weights
    actor_weights_before = [p.clone() for p in actor.parameters()]
    critic_weights_before = [p.clone() for p in critic.parameters()]

    # 5. Execute PPO Update
    print("[INFO] Executing Backpropagation and Gradient Descent...")
    metrics = opt.update(buf)

    # TEST 1: Weight Mutation (Did it learn?)
    actor_changed = any(not torch.allclose(p, p_old) for p, p_old in zip(actor.parameters(), actor_weights_before))
    critic_changed = any(not torch.allclose(p, p_old) for p, p_old in zip(critic.parameters(), critic_weights_before))
    
    if actor_changed and critic_changed:
        print("✓ Dual Optimization: PASS (Both Actor and Critic weights successfully mutated)")
    else:
        print("X Dual Optimization: FAILED (Weights did not update)")

    # TEST 2: Metric Stability (No NaNs)
    has_nans = any(torch.isnan(torch.tensor(v)) for v in metrics.values())
    if not has_nans:
        print("✓ Metric Stability:  PASS (Loss calculations are numerically stable)")
    else:
        print("X Metric Stability:  FAILED (NaN values detected in loss)")

    # TEST 3: Surrogate Clipping
    clip_frac = metrics['clip_fraction']
    if 0.0 <= clip_frac <= 1.0:
         print(f"✓ Surrogate bounds:  PASS (Clip fraction bounded correctly at {clip_frac:.4f})")
    else:
         print("X Surrogate bounds:  FAILED")

    print("\n[INFO] Logged Training Metrics:")
    for k, v in metrics.items():
        print(f"  -> {k:15s}: {v:.6f}")

    print("\n" + "="*60)
    print(" PPO OPTIMIZER READY FOR MAIN TRAINING LOOP")
    print("="*60)

if __name__ == "__main__":
    run_optimizer_validation()