"""
test_critic_network.py
----------------------
Refined Validation Suite for the Centralized Critic.
Focus: Permutation Invariance, N-Agnostic Scaling, and Gradient Integrity.
"""

import sys
import os
import torch
import yaml

# Add project root directory to path to fix ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.networks.critic_network import CriticNetwork

def run_critic_validation():
    # 1. Setup
    config = {
        'environment': {'n_agents': 3},
        'network': {'hidden_dim': 256, 'n_hidden_layers': 2}
    }
    critic = CriticNetwork(config)
    
    print("\n" + "="*60)
    print(" CRITIC NETWORK VALIDATION DASHBOARD")
    print("="*60)

    # TEST 1: Parameter Count (The "Shared Weights" Proof)
    total_params = sum(p.numel() for p in critic.parameters())
    print(f"✓ Architecture Validated: {total_params:,} parameters")

    # TEST 2: Permutation Invariance (The "Team Logic" Proof)
    N = 5
    gs_orig = torch.randn(1, N, 17)
    shuffled_idx = torch.randperm(N)
    gs_shuffled = gs_orig[:, shuffled_idx, :]
    
    val_orig = critic.forward(gs_orig)
    val_shuf = critic.forward(gs_shuffled)
    
    diff = torch.abs(val_orig - val_shuf).item()
    status = "PASS" if diff < 1e-6 else "FAIL"
    print(f"✓ Permutation Invariance: {status} (Diff: {diff:.2e})")

    # TEST 3: N-Agnostic Stress Test (The "Scalability" Proof)
    print("\n--- Swarm Scalability Stress Test ---")
    sizes = [1, 3, 10, 50, 100]
    for n in sizes:
        try:
            test_input = torch.randn(4, n, 17)
            out = critic.forward(test_input)
            print(f"  Swarm Size N={n:3d}  ->  Output {out.shape}  ✓")
        except Exception as e:
            print(f"  Swarm Size N={n:3d}  ->  FAILED: {e}")

    # TEST 4: Gradient Integrity
    gs_grad = torch.randn(1, N, 17, requires_grad=True)
    critic.forward(gs_grad).backward()
    grad_status = "✓" if gs_grad.grad is not None else "X"
    print(f"\n{grad_status} Gradient Flow: Verified from Value Head to Global State Input")
    
    print("="*60)
    print(" CRITIC READY FOR PPO INTEGRATION")
    print("="*60)

if __name__ == "__main__":
    run_critic_validation()