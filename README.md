# mppi_pendulum
This repository implements Model Predictive Path Integral (MPPI) as introduced by the paper 
[Information Theoretic MPC for Model-Based Reinforcement Learning](https://ieeexplore.ieee.org/document/7989202/) by 
(Williams et al., 2017) and takes as forward model the pendulum OpenAI Gym environment.

The paper derives an optimal control law as a (noise-) weighted average over sampled trajectory. In particular, 
the optimization problem is posed to compute the control input such that the controlled distribution Q is pushed as close as possible to the optimal distribution Q*. This corresponds to minimizing the KL divergence between Q and Q*.

The gists:
- the noise assumption v<sub>t</sub> &#820; N(u<sub>t</sub>, &sum;) stems from noise in low-level controllers
- given the optimal control input distribution Q*, it is derived u*<sub>t</sub> = &#8747;q*(V)v<sub>t</sub>dV
- computing the integral is not possible since q* is unknown, instead use importance sampling to sample from proposal distribution: 
  
  <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\int(v)&space;\underbrace{&space;\frac{q^{*}(V)}{p(V)}&space;\frac{p(V)}{q(V)}}_{\mathrm{\omega}(V)}v_t&space;dV&space;=&space;\mathop{\mathbb{E}_Q}&space;[\omega(V)v_t]" title="\int(v) \underbrace{ \frac{q^{*}(V)}{p(V)} \frac{p(V)}{q(V)}}_{\mathrm{\omega}(V)}v_t dV = \mathop{\mathbb{E}_Q} [\omega(V)v_t]" />
  </p>
  
  where <img src="https://latex.codecogs.com/svg.latex?\frac{q^{*}(V)}{p(V)}" title="\frac{q^{*}(V)}{p(V)}" /> can be approximated by the Monte-Carlo estimate given in algorithm 2 by &eta;, yielding:
    <p align="center">
  <img src="https://latex.codecogs.com/svg.latex?u_t^{i&plus;1}&space;=&space;u_t^i&space;&plus;&space;\sum_{n=1}^N&space;\omega&space;(\mathcal{E}_n)&space;\epsilon_t^n" title="u_t^{i+1} = u_t^i + \sum_{n=1}^N \omega (\mathcal{E}_n) \epsilon_t^n" />
    </p>
  which resembles an iterative procedure to improve the MC estimate by using a more accurate importance sampler
  
