from gurobipy import *
import numpy as np

from common_data import scenarios, nodes, units, lines, existing_units, existing_lines, \
	candidate_units, candidate_lines, G_max, F_max, F_min, incidence, weights, C_g


# problem-specific data: generation and investment costs
C_x = 1.
C_y = 1.


def augment_input_data(data, iterations):
	# Create a copy of NxM 2D input data for all K iterations, returning a 3D NxMxK array.
	# If K = 0, return a NxM array.
	new_data = np.copy(data)

	if iterations > 0:
		new_data = new_data[:, :, np.newaxis]
		new_data = np.tile(new_data, (1, 1, iterations))

	return new_data


def get_investment_cost(x, y):
	return C_x*x + C_y*y


def generate_master_problem(current_iteration, d):
	# generate master problem for K:th iteration
	iterations = range(current_iteration)

	# create a model
	m = Model("master_problem")

	# replicate generation and transmission capacities for each iteration
	G_maxs = augment_input_data(G_max, current_iteration)

	F_maxs = augment_input_data(F_max, current_iteration)
	F_mins = augment_input_data(F_min, current_iteration)

	# add generation variables for existing and candidate units
	g = m.addVars(units, scenarios, iterations, name='generation', lb=0., ub=G_maxs)

	# flow variables for existing and candidate lines
	f = m.addVars(lines, scenarios, iterations, name='flow', lb=F_mins, ub=F_maxs)

	# investment to a generation unit and transmission line
	x = m.addVar(vtype=GRB.BINARY, name='x')
	y = m.addVar(vtype=GRB.BINARY, name='y')

	# subproblem objective value
	theta = m.addVar(name='theta')

	# set objective
	m.setObjective(get_investment_cost(x, y) + theta, GRB.MINIMIZE)

	# at zeroth iteration, no constraints are added and the optimal solution is no investment
	if iterations > 0:
		# minimum value for the subproblem objective function
		m.addConstrs((theta - sum(sum(C_g[u]*g[u, o, v] for u in units) * weights[o]
		 			 for o in scenarios) >= 0. for v in iterations),
		 			 name='minimum_subproblem_objective')

		# balance equation. Note that d[n, v] is input data from the subproblem
		m.addConstrs((g[n, o, v] + sum(incidence[l, n]*f[l, o, v] for l in lines) == d[n, v]
					 for n in nodes for o in scenarios for v in iterations), name='balance')

		# generation constraint for the candidate units
		m.addConstrs((g[u, o, v] <= G_maxs[u, o, v]*x for u in candidate_units for o in scenarios
					  for v in iterations), name='maximum_candidate_generation')

		# flow constraint for the candidate lines
		m.addConstrs((f[l, o, v] <= F_maxs[l, o, v]*y for l in candidate_lines for o in scenarios
					 for v in iterations), name='maximum_candidate_flow')

		m.addConstrs((f[l, o, v] >= F_mins[l, o, v]*y for l in candidate_lines for o in scenarios
					  for v in iterations), name='minimum_candidate_flow')

	return m
