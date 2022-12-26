# Stochastic Optimization Homework
Stochastic optimization homework for the Numerical Optimization exam

Text of the problem: A company has $N = 6$ clerks. In the city where the company is located a light disease is circulating. Each day, every working clerk has a constant probability $p_i = 1/4$ of getting infected. The infection causes mild symptoms that can be immediately detected, but infected individuals are forced not to work until the are completely recovered. Every day each infected person has a probability $p_r = 1/5$ of recovering. By denoting The evolution of the system is thus governed by the following equation
```math
W_{n+1} = W_n - I_{n+1} + R_n
```
where
* $n$ stands for the $n$-th day
* $W_n$ is the number of effectively working clerks (state variable)
* $I_{n+1}$ (number of infected individuals) is a binomial random variable with parameters $n = W_n$ and $p = p_i$
* $R_{n+1}$ (number of recovered individuals) is a binomial random variable with parameters $n = N - W_n$ and $p = p_r$

For every working day the usual income of the company is given by the following formula, depending on the number $i$ of clerks that are at work, no matter how many infections or recoveries there will be in that day
```math
r(i,1,j) = 50 \cdot (exp(i/7)-1)
```
At any time the company is capable of setting up an "emergency protocol" with stronger safety measures that decrease the probability of new infections to $1/20$. The measures in the emergency protocol have the side effect of reducing the daily income to the following values
```math
r(i,2,j) = 0 \quad \text{if} \; i=0, \qquad 35 \cdot (\text{exp}(i-1)/7-1) \quad \text{otherwise}
```
The goal of the project is to find in which states it is optimal to adopt the standard safety protocol and in which states it is better to adopt the emergency protocol. Optimal in this context means that it maximizes the total discounted reward, using $\lambda = 0.995$ as discount factor. To this aim we ask you to use both dynamic programming and reinforcement learning (RL) to find the optimal policy and to compare the findings from the two methods.

## User guide
- **dynamic_programming.m**: solution for the stated problem through a **dynamic programming** approach (Q-factor value iteration algorithm)
- **reinforcement_learning.m**: solution for the stated problem through a **reinforcement learning** approach (Q-learning algorithm)
- **SO_report.pdf**: report of the homework \w results
