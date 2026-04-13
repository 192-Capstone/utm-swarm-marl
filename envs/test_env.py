import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from webots_adapter import WebotsAdapter

def run_test_flight():
    adapter = WebotsAdapter(drone_def="MAVIC")
    target_pos = np.array([3.0, 0.0, 2.0])
    
    print("--- Test Flight Initiated ---")
    
    while adapter.step_simulation():
        state = adapter.get_raw_state()
        curr_pos = state["position"]
        roll, pitch, yaw = state["orientation"] # Read the tilt!
        
        if np.isnan(curr_pos).any():
            continue
            
        error = target_pos - curr_pos
        distance = np.linalg.norm(error)
        
        # 1. LINEAR VELOCITY (Navigation + Smart Gravity)
        vx = error[0] * 0.5
        vy = error[1] * 0.5
        
        # Increase the pull multiplier (from 0.5 to 1.0) so it fights harder for the exact altitude
        vz = error[2] * 1.0 
        
        # Only apply the anti-gravity boost if we are BELOW the target altitude
        if curr_pos[2] < target_pos[2]:
            vz += 0.25 
        
        # Limit max speed so it doesn't overshoot wildly
        vx = np.clip(vx, -1.0, 1.0)
        vy = np.clip(vy, -1.0, 1.0)
        vz = np.clip(vz, -1.0, 1.0)

        # 2. ANGULAR VELOCITY (Active Stabilization)
        # If the drone tilts forward (pitch), we apply negative pitch velocity to snap it back.
        wx = -roll * 4.0   # Fight Roll
        wy = -pitch * 4.0  # Fight Pitch
        wz = 0.0           # Keep Yaw locked
        
        # Combine into our 6D action vector
        action = [vx, vy, vz, wx, wy, wz]
        adapter.apply_actions(action)
        
        print(f"Pos: {curr_pos.round(2)} | Dist to Target: {distance:.2f}m", end='\r')
        
        if distance < 0.2:
            print("\nMISSION ACCOMPLISHED: Target reached.")
            break

    adapter.reset_world()
    print("Environment reset. Test concluded.")

if __name__ == "__main__":
    run_test_flight()