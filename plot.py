import numpy as np
import matplotlib.pyplot as plt

def plot_quadrant_lidar():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    quadrants = {
        'Front (Front + NE)': {'color': 'red', 'rays': [(0, 0), (45, -45), (-45, -45)]},
        'Right (Right + SE)': {'color': 'green', 'rays': [(0, -90), (45, -135), (-45, -135)]},
        'Back (Back + SW)':   {'color': 'blue', 'rays': [(0, 180), (45, 135), (-45, 135)]},
        'Left (Left + NW)':   {'color': 'orange', 'rays': [(0, 90), (45, 45), (-45, 45)]},
        'Z-Axis (Up/Down)':   {'color': 'gray', 'rays': [(90, 0), (-90, 0)]} # The remaining 2 rays
    }

    ray_length = 5.0

    # drone center
    ax.scatter([0], [0], [0], color='black', s=100, label='Drone Center')

    for quad_name, data in quadrants.items():
        color = data['color']
        for el, az in data['rays']:
            el_rad = np.radians(el)
            az_rad = np.radians(az)
            
            x = ray_length * np.cos(el_rad) * np.cos(az_rad)
            y = ray_length * np.cos(el_rad) * np.sin(az_rad)
            z = ray_length * np.sin(el_rad)
            
            # the ray line and the endpoint
            ax.plot([0, x], [0, y], [0, z], color=color, linestyle='-', linewidth=2, alpha=0.7)
            ax.scatter([x], [y], [z], color=color, s=40)
        
        ax.plot([0], [0], [0], color=color, label=quad_name)

    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])
    ax.set_zlim([-6, 6])
    ax.set_xlabel('X (Front / Back)')
    ax.set_ylabel('Y (Left / Right)')
    ax.set_zlabel('Z (Altitude)')
    ax.set_title('14-Ray Swarm MARL Observation Space', fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    ax.view_init(elev=25, azim=-35) 
    plt.show()

if __name__ == "__main__":
    plot_quadrant_lidar()