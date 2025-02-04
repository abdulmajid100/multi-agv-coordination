import numpy as np
import matplotlib.pyplot as plt


class VacuumCleaningEnv:
    def __init__(self,
                 grid_points=100,
                 L=10.0,
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
        Initialize the simulation environment.

        Parameters:
            grid_points (int): Number of discrete spatial cells.
            L (float): Length of the spatial domain.
            dt (float): Time step for simulation updates.
            alpha (float): Cleaning efficiency parameter.
            beta (float): Scaling constant for the defect probability.
            r_const (float): Base dust generation rate.
            noise_std (float): Standard deviation of noise in dust generation.
            sigma (float): Standard deviation for the Gaussian cleaning decay kernel.
            cleaning_threshold (float): Threshold to decide whether cleaning resets the
                                        elapsed timer τ at a cell.
            lambda_cost (float): Weighting factor for the action cost.
            max_time (float): Duration at which simulation terminates.
        """
        self.N = grid_points
        self.L = L
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.r_const = r_const
        self.noise_std = noise_std
        self.sigma = sigma
        self.cleaning_threshold = cleaning_threshold
        self.lambda_cost = lambda_cost
        self.max_time = max_time

        # Create a 1D spatial grid from 0 to L
        self.xs = np.linspace(0, L, grid_points)
        self.dx = self.xs[1] - self.xs[0]

        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.t = 0.0
        self.D = np.zeros(self.N)  # Dust accumulation at each cell
        self.tau = np.zeros(self.N)  # Elapsed time since last cleaning per cell
        self.x_vacuum = self.L / 2.0  # Start vacuum in the middle of the domain
        self.vacuum_velocity = 0.0  # Initial vacuum velocity
        return self._get_obs()

    def _kernel(self, distance):
        """
        Gaussian spatial decay kernel.

        K(distance) = exp( - distance² / (2*sigma²) )
        """
        return np.exp(- (distance ** 2) / (2 * self.sigma ** 2))

    def _get_obs(self):
        """
        Constructs the current observation.

        Returns a dictionary containing:
           - 'dust': Dust levels across the domain.
           - 'timer': Time since last cleaning for each cell.
           - 'vacuum_position': Current vacuum position.
        """
        return {
            'dust': self.D.copy(),
            'timer': self.tau.copy(),
            'vacuum_position': self.x_vacuum
        }

    def step(self, action):
        """
        Progresses the simulation by one time step.

        Parameters:
            action (tuple): A tuple (velocity_command, cleaning_intensity).
                            - velocity_command: The desired vacuum speed (can be negative).
                            - cleaning_intensity: A scalar in [0, 1], with 1 meaning full cleaning.

        Returns:
            obs (dict): The current observation/state.
            reward (float): Reward from the action.
            done (bool): Whether the simulation has reached its maximum time.
            info (dict): Additional information, including the current risk and cost.
        """
        # Unpack the action
        velocity_command, cleaning_intensity = action

        # Update the vacuum's position using the velocity command.
        self.vacuum_velocity = velocity_command
        self.x_vacuum += self.vacuum_velocity * self.dt
        # Ensure the vacuum stays within the domain.
        self.x_vacuum = np.clip(self.x_vacuum, 0, self.L)

        # Compute the dust generation rate with added noise.
        noise = np.random.normal(0, self.noise_std, size=self.N)
        r = self.r_const + noise

        # Calculate the cleaning effect using a Gaussian decay from the vacuum's position.
        distances = np.abs(self.xs - self.x_vacuum)
        K = self._kernel(distances)

        # Update dust using Euler's method:
        # D(x, t+dt) = D(x, t) + dt * [ r(x, t) - α * m(t)*K(distance)*D(x, t) ]
        cleaning_effect = self.alpha * cleaning_intensity * K * self.D
        self.D = self.D + self.dt * (r - cleaning_effect)

        # Update the elapsed cleaning timer τ for each location.
        # If the effective cleaning (cleaning_intensity * K) is above a threshold, we assume cleaning resets τ.
        cleaning_mask = (cleaning_intensity * K >= self.cleaning_threshold)
        self.tau[cleaning_mask] = 0  # Reset timer where cleaning is effective.
        self.tau[~cleaning_mask] += self.dt  # Else, increment the timer.

        # Calculate the defect probability for each cell.
        # P_defect(x,t) = 1 - exp(-β * D(x,t) * τ(x,t) )
        P_defect = 1 - np.exp(-self.beta * self.D * self.tau)
        # Estimate the risk as the integrated defect probability over the domain.
        risk = np.sum(P_defect) * self.dx

        # Define an action cost.
        # In this example, cost is a combination of the absolute velocity and the square of the cleaning intensity.
        action_cost = np.abs(velocity_command) + cleaning_intensity ** 2

        # The reward is penalized by accumulated risk and the action cost.
        reward = - (risk + self.lambda_cost * action_cost)

        # Increment simulation time.
        self.t += self.dt
        done = self.t >= self.max_time

        info = {'risk': risk, 'action_cost': action_cost}
        return self._get_obs(), reward, done, info

    def render(self):
        """
        Renders the current state of the simulation in a simple plot.
        It displays the dust accumulation, the timer, and the vacuum position.
        """
        plt.figure(figsize=(10, 4))
        plt.plot(self.xs, self.D, label='Dust level')
        plt.plot(self.xs, self.tau, label='Elapsed time (τ)')
        plt.axvline(self.x_vacuum, color='r', linestyle='--', label='Vacuum Position')
        plt.xlabel('Position (x)')
        plt.title(f"Time: {self.t:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# A simple simulation loop to demonstrate how the environment works.
if __name__ == "__main__":
    # Create the simulation environment.
    env = VacuumCleaningEnv()
    obs = env.reset()
    done = False
    total_reward = 0.0

    # Run the simulation until the maximum time is reached.
    while not done:
        # For this example, we use a simple random policy.
        # Velocity is chosen uniformly between -1 and 1, and cleaning intensity between 0 and 1.
        velocity_command = np.random.uniform(-1, 1)
        cleaning_intensity = np.random.uniform(0, 1)
        action = (velocity_command, cleaning_intensity)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(f"Time {env.t:.2f}: Vacuum at {obs['vacuum_position']:.2f}, "
              f"Reward={reward:.2f}, Risk={info['risk']:.2f}")

    print("Total Reward:", total_reward)
    # Render the final state.
    env.render()