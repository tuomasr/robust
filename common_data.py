import numpy as np

# indices and sets
scenarios = [0, 1]
nodes = [0, 1, 2]

existing_units = [1, 2]
existing_lines = [1, 2]

candidate_units = [0]
candidate_lines = [0]

units = candidate_units + existing_units
lines = candidate_lines + existing_lines

# NOTE: Make sure all parameters are floats to avoid int*float -> int type issues if using Python 2
# costs
C_g = np.array([2., 2., 1.])

# generation and flow limit
G_max = np.array([[10., 10.],
				  [10., 10.],
				  [10., 10.]])

F_max = np.array([[1., 1.],
				  [10., 10.],
				  [1., 1.]])
F_min = -F_max

# lines x nodes
incidence = np.array([[-1., 1., 0.],
					  [0., -1., 1.],
					  [-1., 0., 1.]])

# scenario weights
weights = np.array([0.5, 0.5])
