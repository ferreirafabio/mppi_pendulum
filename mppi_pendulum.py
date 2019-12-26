import gym
import numpy as np


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env, K, T, U, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, noise_gaussian=True, downward_start=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.cost_total = np.zeros(shape=(self.K))

        self.env = env
        self.env.reset()
        if downward_start:
            self.env.env.state = [np.pi, 1]
        self.x_init = self.env.env.state

        if noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)

    def _compute_total_cost(self, k):
        self.env.env.state = self.x_init
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            _, reward, _, _ = self.env.step([perturbed_action_t])
            self.cost_total[k] += -reward

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))


    def control(self, iter=1000):
        for _ in range(iter):
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

            self.env.env.state = self.x_init
            s, r, _, _ = self.env.step([self.U[0]])
            print("action taken: {:.2f} cost received: {:.2f}".format(self.U[0], -r))
            self.env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.cost_total[:] = 0
            self.x_init = self.env.env.state


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 10  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 1

    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    env = gym.make(ENV_NAME)
    mppi_gym = MPPI(env=env, K=N_SAMPLES, T=TIMESTEPS, U=U, lambda_=lambda_, noise_mu=noise_mu, noise_sigma=noise_sigma, u_init=0, noise_gaussian=True)
    mppi_gym.control(iter=1000)
