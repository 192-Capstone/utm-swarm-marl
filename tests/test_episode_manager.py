"""
test_episode_manager.py
------------------------
Unit tests for the Episode Manager.

Purpose:
    - Verify episode termination conditions (collision, success, timeout)
    - Verify success rate tracking
    - Verify curriculum advancement logic (overall + recent performance)

Run with:
    python tests/test_episode_manager.py
"""

import sys
import os
import numpy as np
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.episode_manager import EpisodeManager


# ============================================================================
# Helper Functions
# ============================================================================

def load_config():
    """Load the real configuration file."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'default_config.yaml'
    )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_test_header(test_num, test_name, purpose):
    """Print formatted test header."""
    print("\n" + "=" * 60)
    print(f"TEST {test_num}: {test_name}")
    print("=" * 60)
    print(f"Purpose: {purpose}")


# ============================================================================
# TEST 1: Episode Manager Creation
# ============================================================================

def test_episode_manager_creation():
    """Test that Episode Manager can be instantiated with real config."""
    print_test_header(1, "Episode Manager Creation", 
                     "Verify Episode Manager can be instantiated with real config")
    
    config = load_config()
    ep_manager = EpisodeManager(config)
    
    print(f"\n✓ Episode Manager created successfully")
    print(f"  max_steps: {ep_manager.max_steps}")
    print(f"  goal_threshold: {ep_manager.goal_threshold}")
    print(f"  progression_window: {ep_manager.window}")
    print(f"  progression_threshold: {ep_manager.threshold}")
    
    assert ep_manager.max_steps == config["environment"]["max_steps"]
    assert ep_manager.goal_threshold == config["environment"]["goal_threshold"]
    assert ep_manager.window == config["curriculum"]["progression_window"]
    assert ep_manager.threshold == config["curriculum"]["progression_threshold"]
    
    print(f"\n✓ All config values loaded correctly")
    
    return ep_manager, config


# ============================================================================
# TEST 2: Episode Continues (No Termination)
# ============================================================================

def test_episode_continues(ep_manager, config):
    """Test that episode continues when no termination condition is met."""
    print_test_header(2, "Episode Continues (No Termination)", 
                     "Verify episode continues normally")
    
    ep_manager.reset()
    
    drone_positions = [(10.0, 20.0, 5.0), (15.0, 25.0, 5.0)]
    goal_positions = [(50.0, 50.0, 10.0), (50.0, 50.0, 10.0)]
    collisions = [False, False]
    ep_manager.current_step = 50
    
    done, reason = ep_manager.check_done(drone_positions, goal_positions, collisions)
    
    print(f"\n  Step {ep_manager.current_step}: done={done}, reason={reason}")
    print(f"\n  ✓ Episode continues as expected")
    
    assert done == False
    assert reason == None


# ============================================================================
# TEST 3: Collision Termination
# ============================================================================

def test_collision_termination(ep_manager, config):
    """Test that episode ends when a drone collides."""
    print_test_header(3, "Collision Termination", 
                     "Verify episode ends on collision")
    
    ep_manager.reset()
    
    drone_positions = [(10.0, 20.0, 5.0), (15.0, 25.0, 5.0)]
    goal_positions = [(50.0, 50.0, 10.0), (50.0, 50.0, 10.0)]
    collisions = [False, True]
    ep_manager.current_step = 30
    
    done, reason = ep_manager.check_done(drone_positions, goal_positions, collisions)
    
    print(f"\n  Collision detected: done={done}, reason={reason}")
    print(f"\n  ✓ Episode terminates with 'collision'")
    
    assert done == True
    assert reason == "collision"


# ============================================================================
# TEST 4: Goal Reached Termination
# ============================================================================

def test_goal_reached_termination(ep_manager, config):
    """Test that episode ends when all drones reach their goals."""
    print_test_header(4, "Goal Reached Termination", 
                     "Verify episode ends when all drones reach goals")
    
    ep_manager.reset()
    
    goal_threshold = config["environment"]["goal_threshold"]
    
    drone_positions = [(49.8, 49.9, 9.8), (50.0, 50.1, 9.9)]
    goal_positions = [(50.0, 50.0, 10.0), (50.0, 50.0, 10.0)]
    collisions = [False, False]
    ep_manager.current_step = 30
    
    done, reason = ep_manager.check_done(drone_positions, goal_positions, collisions)
    
    print(f"\n  Goal reached (threshold={goal_threshold}m): done={done}, reason={reason}")
    print(f"\n  ✓ Episode terminates with 'success'")
    
    assert done == True
    assert reason == "success"


# ============================================================================
# TEST 5: Timeout Termination
# ============================================================================

def test_timeout_termination(ep_manager, config):
    """Test that episode ends when max steps is exceeded."""
    print_test_header(5, "Timeout Termination", 
                     "Verify episode ends on timeout")
    
    ep_manager.reset()
    
    drone_positions = [(10.0, 20.0, 5.0), (15.0, 25.0, 5.0)]
    goal_positions = [(50.0, 50.0, 10.0), (50.0, 50.0, 10.0)]
    collisions = [False, False]
    ep_manager.current_step = ep_manager.max_steps + 1
    
    done, reason = ep_manager.check_done(drone_positions, goal_positions, collisions)
    
    print(f"\n  Step {ep_manager.current_step} (max={ep_manager.max_steps}): done={done}, reason={reason}")
    print(f"\n  ✓ Episode terminates with 'timeout'")
    
    assert done == True
    assert reason == "timeout"


# ============================================================================
# TEST 6: Success Rate Tracking
# ============================================================================

def test_success_rate_tracking():
    """Test that success rate is calculated correctly."""
    print_test_header(6, "Success Rate Tracking", 
                     "Verify success rate calculation")
    
    test_config = {
        "environment": {"max_steps": 100, "goal_threshold": 0.5},
        "curriculum": {"progression_window": 5, "progression_threshold": 0.8}
    }
    ep_manager = EpisodeManager(test_config)
    
    results = ["success", "collision", "success", "success", "timeout"]
    expected_rates = [1.0, 0.5, 0.667, 0.75, 0.6]
    
    print("\n  Recording episodes:")
    for i, result in enumerate(results):
        ep_manager.record_episode_result(result)
        rate = ep_manager.get_success_rate()
        status = "✓" if abs(rate - expected_rates[i]) < 0.01 else "✗"
        print(f"    Episode {i+1}: {result:10s} → success_rate = {rate:.3f} (expected {expected_rates[i]:.3f}) {status}")
        assert abs(rate - expected_rates[i]) < 0.01
    
    print(f"\n  ✓ Success rate tracking works correctly")


# ============================================================================
# TEST 7: Curriculum Advancement Logic
# ============================================================================

def test_curriculum_advancement():
    """Test that curriculum advancement requires sustained performance."""
    print_test_header(7, "Curriculum Advancement Logic", 
                     "Verify advancement requires BOTH overall AND recent performance")
    
    test_config = {
        "environment": {"max_steps": 100, "goal_threshold": 0.5},
        "curriculum": {"progression_window": 20, "progression_threshold": 0.8}
    }
    
    print(f"\n  Config: window=20, threshold=0.8")
    print(f"  recent_window = max(10, 20//5) = {max(10, 20//5)}")
    
    # Scenario A: Consistent good performance
    print("\n  " + "-" * 50)
    print("  Scenario A: Consistent Good Performance")
    print("  " + "-" * 50)
    
    ep_manager = EpisodeManager(test_config)
    for i in range(20):
        if i % 10 < 9:
            ep_manager.record_episode_result("success")
        else:
            ep_manager.record_episode_result("collision")
    
    overall = ep_manager.get_success_rate()
    recent = np.mean(ep_manager.history[-10:])
    print(f"    Overall: {overall:.1%}, Recent: {recent:.1%} → Advance: {ep_manager.should_advance_curriculum()}")
    assert ep_manager.should_advance_curriculum() == True
    
    # Scenario B: Good overall, poor recent
    print("\n  " + "-" * 50)
    print("  Scenario B: Good Overall, Poor Recent")
    print("  " + "-" * 50)
    
    ep_manager = EpisodeManager(test_config)
    for i in range(10):
        ep_manager.record_episode_result("success")
    for i in range(10):
        ep_manager.record_episode_result("collision")
    
    overall = ep_manager.get_success_rate()
    recent = np.mean(ep_manager.history[-10:])
    print(f"    Overall: {overall:.1%}, Recent: {recent:.1%} → Advance: {ep_manager.should_advance_curriculum()}")
    assert ep_manager.should_advance_curriculum() == False
    
    # Scenario C: Barely meeting threshold, stable recent
    print("\n  " + "-" * 50)
    print("  Scenario C: Barely Meeting Threshold, Stable Recent")
    print("  " + "-" * 50)
    
    ep_manager = EpisodeManager(test_config)
    for i in range(20):
        if i % 5 < 4:
            ep_manager.record_episode_result("success")
        else:
            ep_manager.record_episode_result("collision")
    
    overall = ep_manager.get_success_rate()
    recent = np.mean(ep_manager.history[-10:])
    print(f"    Overall: {overall:.1%}, Recent: {recent:.1%} → Advance: {ep_manager.should_advance_curriculum()}")
    assert ep_manager.should_advance_curriculum() == True
    
    # Scenario D: Below threshold overall
    print("\n  " + "-" * 50)
    print("  Scenario D: Below Threshold Overall")
    print("  " + "-" * 50)
    
    ep_manager = EpisodeManager(test_config)
    for i in range(20):
        if i < 15:
            ep_manager.record_episode_result("success")
        else:
            ep_manager.record_episode_result("collision")
    
    overall = ep_manager.get_success_rate()
    recent = np.mean(ep_manager.history[-10:])
    print(f"    Overall: {overall:.1%}, Recent: {recent:.1%} → Advance: {ep_manager.should_advance_curriculum()}")
    assert ep_manager.should_advance_curriculum() == False
    
    print(f"\n  ✓ Curriculum advancement logic works correctly")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "=" * 70)
    print(" EPISODE MANAGER - UNIT TEST SUITE")
    print("=" * 70)
    print("\nThis test suite verifies all aspects of the Episode Manager:")
    print("  • Episode creation and configuration")
    print("  • Episode continuation (no termination)")
    print("  • Collision termination")
    print("  • Goal reached termination")
    print("  • Timeout termination")
    print("  • Success rate tracking")
    print("  • Curriculum advancement (overall + recent performance)")
    print("\n" + "=" * 70)
    
    try:
        ep_manager, config = test_episode_manager_creation()
        test_episode_continues(ep_manager, config)
        test_collision_termination(ep_manager, config)
        test_goal_reached_termination(ep_manager, config)
        test_timeout_termination(ep_manager, config)
        test_success_rate_tracking()
        test_curriculum_advancement()
        
        print("\n" + "=" * 70)
        print(" ALL 7 TESTS PASSED! Episode Manager is ready.")
        print("=" * 70)
        
        print("\nSummary:")
        print("  ✓ Episode Manager created with real config")
        print("  ✓ Episode continuation (no termination)")
        print("  ✓ Collision termination")
        print("  ✓ Goal reached termination")
        print("  ✓ Timeout termination")
        print("  ✓ Success rate tracking")
        print("  ✓ Curriculum advancement (overall + recent performance)")
        
        print("\nEpisode Manager is ready for integration with training pipeline!")
        
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()