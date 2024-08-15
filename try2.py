import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridEnv:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = obstacles

    def animate(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
        ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.grid(True)

        # Plot obstacles
        for obs in self.obstacles:
            plot_x = obs[0] - 0.5
            plot_y = obs[1] - 0.5
            print(f"Obstacle (grid coord): {obs}, Plot coord: ({plot_x}, {plot_y})")
            rect = patches.Rectangle((plot_x, plot_y), 1, 1, linewidth=1, edgecolor='black', facecolor='red')
            ax.add_patch(rect)

        #plt.gca().invert_yaxis()
        plt.title("Grid Environment Animation")
        plt.show()

# Example usage
if __name__ == "__main__":
    grid_size = (5, 5)
    obstacles = [(4, 2), (1, 2)]
    env = GridEnv(grid_size, obstacles)
    env.animate()
