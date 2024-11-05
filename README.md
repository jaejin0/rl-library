# Collection of Reinforcement Learning algorithms

> This repository is a set of reusable implementations of various Reinforcement Learning algorithms and Marokv Decision Process environments.

The RL, DRL, and MARL algorithms are based on the pseudocode of [marl-book.com](https://www.marl-book.com/).

## TODOs:

- create a POMDP environment
- finish implementing minimax-Q algorithm

## resources used
[marl-book.com](https://www.marl-book.com/)<br>
[VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning](https://arxiv.org/abs/2207.03530)<br>
[Gymnasium](https://gymnasium.farama.org/index.html)<br>
[PettingZoo](https://pettingzoo.farama.org/)<br>
[Level-based Foraging](https://github.com/semitable/lb-foraging)<br>
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)<br>

## multiagent environments
|Environment|Observability|Observations|Actions|Rewards|
|-----------|----------|--------|----|-----|
|Environments:|||||
|LBF|full, part|dis|dis|spa|
|MPE|full, part|con|dis, con|den|
|SMAC|part|mix|dis|den|
|RWARE|part|dis|dis|spa|
|GRF|full, part|mix|dis|den, spa|
|Hanabi|part|dis|dis|spa|
|Overcooked|full|mix|dis|spa|
|Environment collections:|||||
|Melting Pot|part|con|dis|den, spa|
|OpenSpiel|full, part|dis|dis|den, spa|
|Petting Zoo|full, part|mix|dis, con|den, spa|

