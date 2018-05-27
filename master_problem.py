# Master problem formulation.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import *
import numpy as np

from common_data import years, hours, num_hours_per_year, scenarios, nodes, units, lines, \
 	existing_units, existing_lines, candidate_units, candidate_lines, G_max, F_max, F_min, \
 	incidence, weights, C_g
from helpers import to_year, unit_built, line_built


# Problem-specific data: generation and investment costs for each year.
C_x = {(year, unit): 1. for year in years for unit in candidate_units}
C_y = {(year, line): 1. for year in years for line in candidate_lines}


def get_investment_cost(xhat, yhat):
	# Compute total investment cost for fixed generation and transmission investment decisions.
	return sum(sum(C_x[t, u]*xhat[t, u] for u in candidate_units) +
			   sum(C_y[t, l]*yhat[t, l] for l in candidate_lines) for t in years)


def add_primal_variables(iteration):
	# Generation variables for existing and candidate units. Upper bounds are set as constraints.
	g = m.addVars(scenarios, hours, units, [iteration], name='generation', lb=0., ub=GRB.INFINITY)

	# Flow variables for existing and candidate lines. Upper and lower bound are set as constraints.
	f = m.addVars(scenarios, hours, lines, [iteration], name='flow',
	 			  lb=-GRB.INFINITY, ub=GRB.INFINITY)

	return g, f


# Create the initial model - no constraints are added
m = Model("master_problem")

g, f = add_primal_variables(0) 	# Primal variables for the initial model

# Variables representing investment to generation units and transmission lines.
xhat = m.addVars(years, candidate_units, vtype=GRB.BINARY, name='unit_investment')
yhat = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name='line_investment')

# Variables indicating whether candidate generation units and transmission lines can be operated.
x = m.addVars(years, candidate_units, vtype=GRB.BINARY, name='unit_available')
y = m.addVars(years, candidate_lines, vtype=GRB.BINARY, name='line_available')

# Constraints defining that candidate units and transmission lines can be operated if investment
# has been made.
m.addConstrs((x[t, u] - sum(xhat[tt, u] for tt in range(t + 1)) <= 0.
 			  for t in years for u in candidate_units), name='unit_operational')
m.addConstrs((y[t, l] - sum(yhat[tt, l] for tt in range(t + 1)) <= 0.
 			  for t in years for l in candidate_lines), name='line_operational')

# Variable representing the subproblem objective value.
theta = m.addVar(name='theta', lb=-GRB.INFINITY, ub=GRB.INFINITY)

# Set master problem objective function. The optimal solution is no investment initially.
m.setObjective(get_investment_cost(xhat, yhat) + theta, GRB.MINIMIZE)


def augment_master_problem(current_iteration, d):
	# Augment the master problem for the current iteration.
	v = current_iteration

	# Create additional primal variables indexed with the current iteration.
	g, f = add_primal_variables(v)

	# Minimum value for the subproblem objective function.
	m.addConstr(theta - sum(sum(sum(C_g[t, u]*g[o, t, u, v] for u in units) for t in hours) *
	 			weights[o] for o in scenarios) >= 0., name='minimum_subproblem_objective')

	# Balance equation. Note that d[n, v] is input data from the subproblem.
	# TODO: Relax the assumption that there is one unit at each node.
	m.addConstrs((g[o, t, n, v] + sum(incidence[l, n]*f[o, t, l, v] for l in lines) -
	 			 d[t, n, v] == 0. for o in scenarios for t in hours for n in nodes), name='balance')

	# Generation constraint for the units.
	m.addConstrs((g[o, t, u, v] - G_max[o, t, u]*unit_built(x, t, u) <= 0.
	 			  for o in scenarios for t in hours for u in units), name='maximum_generation')

	# Flow constraint for the lines.
	m.addConstrs((f[o, t, l, v] - F_max[o, t, l]*line_built(y, t, l) <= 0.
	 			  for o in scenarios for t in hours for l in lines), name='maximum_flow')

	m.addConstrs((F_min[o, t, l]*line_built(y, t, l) - f[o, t, l, v] <= 0.
	 			  for o in scenarios for t in hours for l in lines), name='minimum_flow')


def get_investment_and_availability_decisions():
	# Read current investments to generation and transmission and whether the units and lines are
	# operational at some time point. Round to avoid numerical issues.
	def var_to_dict(var):
		return {key: np.round(value.x) for key, value in var.items()}

	current_xhat = var_to_dict(xhat)
	current_yhat = var_to_dict(yhat)

	current_x = var_to_dict(x)
	current_y = var_to_dict(y)

	return current_xhat, current_yhat, current_x, current_y


# Assign the master problem to a variable that can be imported elsewhere.
master_problem = m
