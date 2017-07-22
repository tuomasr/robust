# robust
Robust optimization for power markets

Implements a simplified version of http://ieeexplore.ieee.org.libproxy.aalto.fi/document/7944676/. An upper-level agent makes
generation and transmission line investment decisions, while the market is cleared in the lower-level. The lower-level
problem is a robust optimization problem in which some parameters are stochastic.

Solved using Gurobi 7.5.
