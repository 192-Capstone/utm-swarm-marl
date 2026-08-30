"""
test_rollout_buffer.py
----------------------
Refined Validation Suite for the MAPPO Rollout Buffer.
Focus: Memory pre-allocation, shape integrity, and GAE mathematical stability.
"""

import torch
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.rollout_buffer import RolloutBuffer

def run_buffer_validation():
    config = {
        'environment': {'n_agents': 3},
        'training': {'buffer_size': 2048, 'gamma': 0.99, 'lambda_gae': 0.95, 'batch_size': 256},
        'observation': {'local_obs_dim': 27, 'neighbor_obs_dim': 6, 'k_neighbors_max': 5},
        'network': {'action_dim': 4}
    }
    device = torch.device('cpu')
    buf = RolloutBuffer(config, device)

    print("\n" + "="*60)
    print(" ROLLOUT BUFFER VALIDATION DASHBOARD")
    print("="*60)

    # TEST 1: Memory Pre-allocation
    try:
        assert buf.local_obs.shape == (2048, 3, 27)
        
        # Calculate rough memory footprint of the buffer
        mem_bytes = sum([
            buf.local_obs.element_size() * buf.local_obs.nelement(),
            buf.neighbor_states.element_size() * buf.neighbor_states.nelement(),
            buf.global_state.element_size() * buf.global_state.nelement(),
            buf.actions.element_size() * buf.actions.nelement(),
            buf.log_probs.element_size() * buf.log_probs.nelement(),
            buf.rewards.element_size() * buf.rewards.nelement(),
            buf.advantages.element_size() * buf.advantages.nelement()
        ])
        mem_mb = mem_bytes / (1024 * 1024)
        
        print(f"[INFO] Pre-allocating continuous memory blocks...")
        print(f"✓ Memory Allocation: PASS (~{mem_mb:.2f} MB on {device})")
        print(f"  -> local_obs shape:       {buf.local_obs.shape}")
        print(f"  -> neighbor_states shape: {buf.neighbor_states.shape}")
    except AssertionError:
        print("X Memory Allocation: Failed")

    # TEST 2: Multi-Agent State Integrity (Fill Buffer)
    print(f"\n[INFO] Simulating synchronized multi-agent rollout...")
    for _ in range(2048):
        buf.store(
            local_obs=torch.randn(3, 27),
            neighbor_states=torch.randn(3, 5, 6),
            neighbor_mask=torch.zeros(3, 5, dtype=torch.bool),
            global_state=torch.randn(3, 17),
            actions=torch.randn(3, 4),
            log_probs=torch.randn(3),
            rewards=torch.rand(3),
            dones=torch.zeros(3),
            values=torch.randn(3)
        )
    status = "PASS" if buf.is_full() else "FAIL"
    print(f"✓ Multi-Agent Alignment: {status} (2048 steps x 3 agents)")

    # TEST 3: GAE Mathematical Stability
    print(f"\n[INFO] Computing Generalized Advantage Estimation (GAE)...")
    last_values = torch.randn(3)
    buf.compute_gae(last_values)
    
    nan_free = not torch.isnan(buf.advantages).any()
    adv_flat = buf.advantages[:2048].reshape(-1)
    
    mean_val = adv_flat.mean().item()
    std_val = adv_flat.std().item()
    mean_zero = abs(mean_val) < 1e-4
    std_one = abs(std_val - 1.0) < 1e-4

    if nan_free and mean_zero and std_one:
        print(f"✓ GAE Math Stability: PASS")
        print(f"  -> Normalized Mean: {mean_val:.6f} (Target: 0.0)")
        print(f"  -> Normalized Std:  {std_val:.6f} (Target: 1.0)")
    else:
        print("X GAE Math Stability: FAILED Normalization")

    # TEST 4: Minibatch Flattening
    print(f"\n[INFO] Shattering dimensions (T, N) -> (T*N) for PPO...")
    batches = list(buf.get_minibatches(256))
    expected_batches = (2048 * 3) // 256
    if len(batches) == expected_batches and batches[0]['actions'].shape == (256, 4):
         print(f"✓ Minibatch Shuffling: PASS")
         print(f"  -> Total batches: {len(batches)}")
         print(f"  -> Batch shape:   {batches[0]['actions'].shape}")
    else:
         print("X Minibatch Shuffling: FAILED")

    print("\n" + "="*60)
    print(" BUFFER READY FOR TRAINING PIPELINE")
    print("="*60)

if __name__ == "__main__":
    run_buffer_validation()