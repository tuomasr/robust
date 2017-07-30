import numpy as np

# indices and sets
nodes = range(6)

existing_units = range(6)
existing_lines = [0, 1, 2]

candidate_lines = range(3, 9)

units = existing_units
lines = candidate_lines + existing_lines

# note: there is no generation unit at node 3. It is modelled as generator with zero capacity
C_g = np.array([18, 25, 16, 0, 32, 35], dtype=np.float32)

# generation and flow limits
G_min = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
G_max = np.array([300, 250, 400, 0, 300, 150], dtype=np.float32)

F_max = np.array([150, 150, 150, 150, 200, 200, 200, 150, 150], dtype=np.float32)
F_min = np.copy(F_max)

# transmission line susceptance
B = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=np.float32)

# demand parameters
D = np.array([200, 0, 0, 150, 100, 200], dtype=np.float32)
D_min = np.array([180, 0, 0, 135, 90, 180], dtype=np.float32)
D_max = np.array([220, 0, 0, 165, 110, 220], dtype=np.float32)

C_ls = np.array([40, 0, 0, 52, 55, 65])

# lines x nodes
incidence = np.array([[-1, 1, 0, 0, 0, 0], 	# node 1 -> node 2
					  [-1, 0, 1, 0, 0, 0], 	# 1 -> 3
					  [0, 0, 0, -1, 1, 0], 	# 4 -> 5
					  [0, -1, 1, 0, 0, 0],
					  [0, -1, 0, 1, 0, 0],
					  [0, 0, -1, 1, 0, 0],
					  [0, 0, -1, 0, 0, 1],
					  [0, 0, 0, -1, 0, 1],
					  [0, 0, 0, 0, -1, 1]],
					  dtype=np.float32)

# investment costs and budget
C_x = np.array([0, 0, 0, 700000, 1400000, 1800000, 1600000, 800000, 700000], dtype=np.float32)/8760.
investment_budget = 3000000./8760.

# objective scaling factor
sigma = 1.

# uncertainty budgets
generation_uncertainty_budget = 0.1
demand_uncertainty_budget = 0.1

# checks
assert D.shape == (6,)
assert incidence.shape == (9, 6)
