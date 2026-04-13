from controller import Supervisor
import numpy as np

class WebotsAdapter:
    def __init__(self, drone_def="MAVIC", timestep=32):
        """
        Initializes the connection to the Webots Supervisor.
        """
        self.supervisor = Supervisor()
        
        # basicTimeStep is 8ms (from your .wbt). We control at 32ms.
        self.physics_step = int(self.supervisor.getBasicTimeStep()) 
        self.control_step = timestep 
        
        # 1. Access the Drone Node via the DEF name you set in the Scene Tree
        self.drone_node = self.supervisor.getFromDef(drone_def)
        if self.drone_node is None:
            print(f"CRITICAL ERROR: DEF '{drone_def}' not found in Webots!")
            return

        # 2. Initialize Mavic-specific Sensors
        # These strings MUST match the names in the Mavic PROTO file
        self.gps = self.supervisor.getDevice("gps")
        self.imu = self.supervisor.getDevice("inertial unit")
        
        # Enable sensors at the physics rate
        self.gps.enable(self.physics_step)
        self.imu.enable(self.physics_step)

    def get_raw_state(self):
        """
        Reads GPS and IMU data.
        Returns: {position: [x,y,z], velocity: [vx,vy,vz], orientation: [r,p,y]}
        """
        # FIX: Changed .values() to .getValues()
        pos = self.gps.getValues() 
        rot = self.imu.getRollPitchYaw() 
        
        vel = self.drone_node.getVelocity() 
        
        return {
            "position": np.array(pos),
            "velocity": np.array(vel[:3]),
            "orientation": np.array(rot)
        }

    def apply_actions(self, action):
        """
        Applies linear and angular velocity.
        action format: [vx, vy, vz, wx, wy, wz]
        """
        self.drone_node.setVelocity(action)

        
    def step_simulation(self):
        """
        Steps the simulation forward by 32ms.
        Returns False if the simulator window is closed.
        """
        return self.supervisor.step(self.control_step) != -1

    def reset_world(self, spawn_pos=[0, 0, 0.1]):
        """
        Restores the simulation to time zero and teleports the drone.
        """
        self.supervisor.simulationReset()
        self.drone_node.resetPhysics()
        
        trans_field = self.drone_node.getField("translation")
        trans_field.setSFVec3f(spawn_pos)
        
        # Step once to apply the teleportation
        self.supervisor.step(self.physics_step)