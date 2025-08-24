# animation_utils.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
def animate_results(env, actions_sequence):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-0.5, env.grid_size[1] - 0.5)
    ax.set_ylim(-0.5, env.grid_size[0] - 0.5)
    ax.set_xticks(np.arange(env.grid_size[1]))
    ax.set_yticks(np.arange(env.grid_size[0]))
    ax.grid(True)

    for obs in env.obstacles:
        rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, linewidth=1,
                                 edgecolor='black', facecolor='red')
        ax.add_patch(rect)

    for goal in env.goals:
        rect = patches.Rectangle((goal[0] - 0.5, goal[1] - 0.5), 1, 1, linewidth=1,
                                 edgecolor='black', facecolor='green')
        ax.add_patch(rect)

    agents_patches = [patches.Circle((pos[0], pos[1]), 0.3, color='blue') for pos in env.agents]
    for patch in agents_patches:
        ax.add_patch(patch)

    def update(frame):
        if frame == 0:
            env.reset()
            for i, agent in enumerate(env.agents):
                agents_patches[i].center = (agent[0], agent[1])
        else:
            if frame-1 < len(actions_sequence):
                actions = actions_sequence[frame - 1]
                env.agents, _, _, _ = env.step(actions)
                for i, agent in enumerate(env.agents):
                    agents_patches[i].center = (agent[0], agent[1])

        return agents_patches

    anim = animation.FuncAnimation(fig, update, frames=len(actions_sequence) + 1, blit=True, repeat=False, interval=750)
    plt.title("Grid Environment Animation")
    plt.show()
