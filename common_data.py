import numpy as np

np.random.seed(13)

# indices and sets
num_scenarios = 10
scenarios = range(num_scenarios)

num_nodes = 4
nodes = range(num_nodes)

existing_units = [1, 3]
existing_lines = [0, 2]

candidate_units = [0, 2]
candidate_lines = [1, 3]

units = candidate_units + existing_units
lines = candidate_lines + existing_lines

num_lines = len(lines)

# NOTE: Make sure all parameters are floats to avoid int*float -> int type issues if using Python 2
# costs
C_g = np.array([1., 5., 5., 5.])

# generation and flow limit
G_max = np.array([[10., 10., 10., 10.]])
G_max = np.tile(G_max.transpose(), (1, num_scenarios))
G_max += np.random.uniform(-2., 2., (num_nodes, num_scenarios))

F_max = np.array([[5., 5., 5., 5.]])
F_max = np.tile(F_max.transpose(), (1, num_scenarios))
F_max += np.random.uniform(-2., 2., (num_lines, num_scenarios))

F_min = -F_max

# lines x nodes
incidence = np.array([[-1., 1., 0., 0.],
					  [0., -1., 1., 0.],
					  [0., 0., -1., 1.],
					  [1., 0., 0., -1.]])

# equal scenario weights
weights = np.ones(num_scenarios)
