# mppi_pendulum
This repository implements Model Predictive Path Integral (MPPI) as introduced by the paper 
[Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/document/7989202/) by 
(Williams et al., 2017) and takes as forward model the pendulum OpenAI Gym environment.

The paper derives an optimal control law as a (noise-) weighted average over sampled trajectory. In particular, 
the optimization problem is posed to compute the control input such that the controlled distribution Q is pushed as close as possible to the optimal distribution Q*. This corresponds to minimizing the KL divergence between Q and Q*.

The gists:
- the noise assumption v<sub>t</sub> &#820; N(u<sub>t</sub>, &sum;) stems from noise in low-level controllers
- given the optimal control input distribution Q*, it is derived u*<sub>t</sub> = &#8747;q*(V)v<sub>t</sub>dV
- computing the integral is not possible since q* is unknown
  - -> use importance sampling to sample from proposal distribution: 
  <img src="https://latex.codecogs.com/svg.latex?\int&space;q(v)&space;\underbrace{&space;\frac{q^{*}(V)}{p(V)}&space;\frac{p(V)}{q(V)}}_\text{w(V)}&space;v_t&space;dV&space;=&space;\mathop{\mathbb{E}_Q}&space;[w(V)v_t]" title="\int q(v) \underbrace{ \frac{q^{*}(V)}{p(V)} \frac{p(V)}{q(V)}}_\text{w(V)} v_t dV = \mathop{\mathbb{E}_Q} [w(V)v_t]" />
  where 
