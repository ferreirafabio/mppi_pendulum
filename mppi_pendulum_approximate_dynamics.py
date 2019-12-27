import gym
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')


class MPPI():
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, dynamics, K, T, U, running_cost, terminal_state_cost=None, lambda_=1.0, noise_mu=0,
                 noise_sigma=1, u_init=1, noise_gaussian=True):
        self.K = K  # N_SAMPLES
        self.T = T  # TIMESTEPS
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.U = U
        self.u_init = u_init
        self.noise_gaussian = noise_gaussian
        # dimensions of state and control
        self.nx = 2
        self.nu = 1

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.state = None

    def _start_action_consideration(self):
        # reseample noise each time we take an action; these can be done at the start
        self.cost_total = np.zeros(shape=(self.K))
        if self.noise_gaussian:
            self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.K, self.T))
        else:
            self.noise = np.full(shape=(self.K, self.T), fill_value=0.9)
        # cache action cost
        self.action_cost = self.lambda_ * (1 / self.noise_sigma) * self.noise

    def _compute_total_cost(self, k):
        state = torch.from_numpy(self.state)
        for t in range(self.T):
            perturbed_action_t = self.U[t] + self.noise[k, t]
            state = self.F(state, perturbed_action_t)
            self.cost_total[k] += self.running_cost(state, perturbed_action_t)
            # add action perturbation cost
            self.cost_total[k] += perturbed_action_t * self.action_cost[k, t]
        # this is the additional terminal cost (running state cost at T already accounted for)
        if self.terminal_state_cost:
            self.cost_total[k] += self.terminal_state_cost(state)

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, env, retrain_dynamics, retrain_after_iter=50, iter=1000):
        self.state = np.array(env.env.state)
        dataset = np.zeros((retrain_after_iter, self.nx + self.nu))
        for i in range(iter):
            self._start_action_consideration()
            for k in range(self.K):
                self._compute_total_cost(k)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1 / self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1 / eta * cost_total_non_zero

            self.U += [np.sum(omega * self.noise[:, t]) for t in range(self.T)]

            pre_action_state = self.state.copy()
            action = self.U[0]
            # env.env.state = self.state
            s, r, _, _ = env.step([self.U[0]])
            print("action taken: {:.2f} cost received: {:.2f}".format(self.U[0], -r))
            env.render()

            self.U = np.roll(self.U, -1)  # shift all elements to the left
            self.U[-1] = self.u_init  #
            self.state = np.array(env.env.state)
            self.state[0] = angle_normalize(self.state[0])
            logger.debug(self.state)

            di = i % retrain_after_iter
            if di == 0 and i > 0:
                retrain_dynamics(dataset)
                # don't have to clear dataset since it'll be overridden, but useful for debugging
                dataset = np.zeros((retrain_after_iter, self.nx + self.nu))
            dataset[di, :nx] = pre_action_state
            dataset[di, nx:] = action


if __name__ == "__main__":
    ENV_NAME = "Pendulum-v0"
    TIMESTEPS = 15  # T
    N_SAMPLES = 100  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    noise_mu = 0
    noise_sigma = 10
    lambda_ = 1

    # random seed
    import random

    randseed = 5
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 16
    TRAIN_EPOCH = 150
    BOOT_STRAP_ITER = 100

    nx = 2
    nu = 1
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double()


    def dynamics(state, perturbed_action):
        # true dynamics
        # th, thdot = state
        #
        # g = 10
        # m = 1
        # l = 1
        # dt = 0.05
        #
        # u = perturbed_action
        # u = np.clip(u, -2, 2)
        #
        # newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        # newth = th + newthdot * dt
        # newthdot = np.clip(newthdot, -8, 8)
        #
        # state = np.array([newth, newthdot])
        # return state

        u = torch.tensor([perturbed_action], dtype=torch.double)
        u = torch.clamp(u, ACTION_LOW, ACTION_HIGH)
        xu = torch.cat((state, u))
        state_residual = network(xu)
        next_state = state + state_residual
        next_state[0] = angle_normalize(next_state[0])
        return next_state


    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)


    import math


    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d


    def running_cost(state, action):
        theta = state[0]
        theta_dt = state[1]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * action ** 2
        return cost


    dataset = None


    def train(new_data):
        global dataset
        # not normalized inside the simulator
        new_data[:, 0] = angle_normalize(new_data[:, 0])
        new_data = torch.from_numpy(new_data)
        # clamp actions
        new_data[:, -1] = torch.clamp(new_data[:, -1], ACTION_LOW, ACTION_HIGH)
        # append data to whole dataset
        if dataset is None:
            dataset = new_data
        else:
            dataset = torch.cat((dataset, new_data), dim=0)

        # train on the whole dataset (assume small enough we can train on all together)
        XU = dataset
        dtheta = angular_diff_batch(XU[1:, 0], XU[:-1, 0])
        dtheta_dt = XU[1:, 1] - XU[:-1, 1]
        Y = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)), dim=1)  # x' - x residual
        XU = XU[:-1]  # make same size as Y

        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        for epoch in range(TRAIN_EPOCH):
            optimizer.zero_grad()
            # MSE loss
            Yhat = network(XU)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            loss.mean().backward()
            optimizer.step()
            logger.info("ds %d epoch %d loss %f", dataset.shape[0], epoch, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False


    U = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH, size=TIMESTEPS)  # pendulum joint effort in (-2, +2)

    downward_start = True
    env = gym.make(ENV_NAME).env # bypass the default TimeLimit wrapper
    env.reset()
    if downward_start:
        env.state = [np.pi, 1]

    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info("bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        new_data = np.zeros((BOOT_STRAP_ITER, nx + nu))
        for i in range(BOOT_STRAP_ITER):
            pre_action_state = env.state
            action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
            env.step([action])
            # env.render()
            new_data[i, :nx] = pre_action_state
            new_data[i, nx:] = action

        train(new_data)
        logger.info("bootstrapping finished")

    from gym import wrappers
    import time

    env = wrappers.Monitor(env, '/tmp/mppi/{}/'.format(time.time()), force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, 1]

    mppi_gym = MPPI(dynamics, K=N_SAMPLES, T=TIMESTEPS, U=U, running_cost=running_cost, lambda_=lambda_,
                    noise_mu=noise_mu, noise_sigma=noise_sigma,
                    u_init=0, noise_gaussian=True)
    mppi_gym.control(env, train, retrain_after_iter=50, iter=1000)
