import numpy as np
import matplotlib.pyplot as plt


class VacuumCleaningEnv2D:
    def __init__(self,
                 grid_size=(50, 50),
                 Lx=10.0,
                 Ly=10.0,
                 dt=0.1,
                 alpha=0.5,
                 beta=0.1,
                 r_const=0.05,
                 noise_std=0.01,
                 sigma=1.0,
                 cleaning_threshold=0.5,
                 lambda_cost=1.0,
                 max_time=50.0):
        """
        2D grid simulation for a vacuum cleaning process.

        Parameters:
          grid_size (tuple): (nx, ny) dimensions of the 2D grid.
          Lx (float): Physical length in the x-direction.
          Ly (float): Physical length in the y-direction.
          dt (float): Time step for updates.
          alpha (float): Cleaning efficiency parameter.
          beta (float): Scaling constant for defect probability.
          r_const (float): Base dust generation rate.
          noise_std (float): Standard deviation of noise in dust generation.
          sigma (float): Standard deviation for the Gaussian cleaning kernel.
          cleaning_threshold (float): Threshold above which cleaning resets the timer.
          lambda_cost (float): Weighting factor for the cost in the reward.
          max_time (float): Maximum simulation time.
        """
        self.nx, self.ny = grid_size
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.r_const = r_const
        self.noise_std = noise_std
        self.sigma = sigma
        self.cleaning_threshold = cleaning_threshold
        self.lambda_cost = lambda_cost
        self.max_time = max_time

        # Create a 2D spatial grid.
        self.xs = np.linspace(0, Lx, self.nx)
        self.ys = np.linspace(0, Ly, self.ny)
        self.dx = self.xs[1] - self.xs[0]
        self.dy = self.ys[1] - self.ys[0]
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing='ij')

        self.reset()

    def reset(self):
        """Resets the 2D environment to its initial state."""
        self.t = 0.0
        self.D = np.zeros((self.nx, self.ny))  # Dust accumulation in each cell.
        self.tau = np.zeros((self.nx, self.ny))  # Time since last proper cleaning.
        self.vacuum_position = np.array([self.Lx / 2.0, self.Ly / 2.0])  # Start at center.
        self.vacuum_velocity = np.array([0.0, 0.0])
        return self._get_obs()

    def _kernel(self, distances):
        """
        Gaussian kernel for cleaning effectiveness.
        K(distance) = exp( - (distance^2) / (2*sigma^2) )
        """
        return np.exp(- (distances ** 2) / (2 * self.sigma ** 2))

    def _get_obs(self):
        """
        Current observation that includes:
          - Dust levels (2D array).
          - Timer values (2D array).
          - Vacuum's position (2D coordinate).
        """
        return {
            'dust': self.D.copy(),
            'timer': self.tau.copy(),
            'vacuum_position': self.vacuum_position.copy()
        }

    def step(self, action):
        """
        Advances the simulation by one time step.

        Action: A tuple (velocity_command, cleaning_intensity)
          - velocity_command: A 2D vector [vx, vy] for the vacuum's motion.
          - cleaning_intensity: Scalar in [0, 1] (1 means full cleaning power).
        """
        velocity_command, cleaning_intensity = action

        # Update vacuum position.
        self.vacuum_velocity = np.array(velocity_command)
        self.vacuum_position += self.vacuum_velocity * self.dt
        # Ensure the vacuum stays within bounds.
        self.vacuum_position[0] = np.clip(self.vacuum_position[0], 0, self.Lx)
        self.vacuum_position[1] = np.clip(self.vacuum_position[1], 0, self.Ly)

        # Dust generation: constant with added noise.
        noise = np.random.normal(0, self.noise_std, size=(self.nx, self.ny))
        r = self.r_const + noise

        # Compute distances from every grid cell to the vacuum's current position.
        distances = np.sqrt((self.X - self.vacuum_position[0]) ** 2 +
                            (self.Y - self.vacuum_position[1]) ** 2)
        # Compute the cleaning effectiveness at each cell.
        K = self._kernel(distances)

        # Update dust using Euler's method.
        # D(x, t+dt) = D(x, t) + dt * [ r(x, t) - α * m(t)*K(distance)*D(x, t) ]
        cleaning_effect = self.alpha * cleaning_intensity * K * self.D
        self.D = self.D + self.dt * (r - cleaning_effect)
        # Ensure dust remains non-negative.
        self.D = np.clip(self.D, 0, None)

        # Update the cleaning timer τ.
        # Reset τ to zero where cleaning intensity is sufficient; increment otherwise.
        cleaning_mask = (cleaning_intensity * K >= self.cleaning_threshold)
        self.tau[cleaning_mask] = 0
        self.tau[~cleaning_mask] += self.dt

        # Compute the defect probability for each cell.
        # P_defect(x,t) = 1 - exp(-β * D(x,t) * τ(x,t))
        P_defect = 1 - np.exp(-self.beta * self.D * self.tau)
        # Total risk is the integration of defect probability.
        risk = np.sum(P_defect) * self.dx * self.dy

        # Define an action cost (for example, movement cost + squared cleaning intensity).
        action_cost = np.linalg.norm(self.vacuum_velocity) + cleaning_intensity ** 2

        # Reward: the agent is penalized by both risk and action cost.
        reward = - (risk + self.lambda_cost * action_cost)

        # Advance simulation time.
        self.t += self.dt
        done = self.t >= self.max_time

        info = {'risk': risk, 'action_cost': action_cost}
        return self._get_obs(), reward, done, info

    def render(self):
        """
        Renders the current state of the 2D grid:
         - Displays a heatmap of the dust accumulation.
         - Overlays the current vacuum position.
        """
        plt.figure(figsize=(6, 6))
        # Use imshow to display the dust levels (transposed for correct orientation).
        plt.imshow(self.D.T, origin='lower', extent=[0, self.Lx, 0, self.Ly], cmap='inferno')
        plt.colorbar(label='Dust Level')
        # Overlay the vacuum's position.
        plt.scatter(self.vacuum_position[0], self.vacuum_position[1],
                    color='cyan', marker='o', s=100, label='Vacuum')
        plt.title(f"Time: {self.t:.2f}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.show()


# Simulation loop to test the 2D grid environment.
if __name__ == "__main__":
    env = VacuumCleaningEnv2D(grid_size=(50, 50),
                              Lx=10.0,
                              Ly=10.0,
                              dt=0.1,
                              alpha=0.5,
                              beta=0.1,
                              r_const=0.05,
                              noise_std=0.01,
                              sigma=1.0,
                              cleaning_threshold=0.5,
                              lambda_cost=1.0,
                              max_time=20.0)

    obs = env.reset()
    done = False
    total_reward = 0.0

    # Run simulation until max time is reached.
    while not done:
        # Random actions for demonstration.
        velocity_command = np.random.uniform(-1, 1, size=2)  # [vx, vy]
        cleaning_intensity = np.random.uniform(0, 1)
        action = (velocity_command, cleaning_intensity)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Time: {env.t:.2f}, Vacuum pos: {obs['vacuum_position']}, "
              f"Reward: {reward:.2f}, Risk: {info['risk']:.2f}")

    print("Total Reward:", total_reward)
    env.render()