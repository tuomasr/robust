# Define data used by both the master problem and subproblem.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(13)

# Indices and sets.
num_years = 30
num_hours_per_year = 24
num_hours = num_years*num_hours_per_year

years = range(num_years)
hours = range(num_hours)

num_scenarios = 5
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

# Generation costs.
C_g = np.array([[1., 5., 5., 5.]])
C_g = np.tile(C_g, (num_hours, 1))

# Generation and flow limit
G_max = np.array([[10., 10., 10., 10.]])
G_max = np.tile(G_max, (num_scenarios, num_hours, 1))
G_max += np.random.uniform(-5.0, 5.0, (num_scenarios, num_hours, num_nodes))

F_max = np.array([[5., 5., 5., 5.]])
F_max = np.tile(F_max, (num_scenarios, num_hours, 1))
F_max += np.random.uniform(-3.0, 3.0, (num_scenarios, num_hours, num_lines))

F_min = -F_max

# lines x nodes indicence matrix connects the nodes through lines.
incidence = np.array([[-1., 1., 0., 0.],
					  [0., -1., 1., 0.],
					  [0., 0., -1., 1.],
					  [1., 0., 0., -1.]])


# Round data to avoid numerical precision issues.
F_max = np.round(F_max, 3)
F_min = np.round(F_min, 3)
G_max = np.round(G_max, 3)

# Equal scenario weights.
weights = np.ones(num_scenarios)
