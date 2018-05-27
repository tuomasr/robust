# Subproblem formulation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gurobipy import *
import numpy as np

from common_data import hours, scenarios, nodes, units, lines, existing_units, existing_lines, \
	candidate_units, candidate_lines, G_max, F_max, F_min, G_ramp_max, G_ramp_min, incidence, \
	weights, C_g
from helpers import to_year, unit_built, line_built, node_of_unit, is_year_fist_hour, \
	is_year_last_hour


# Problem-specific data: Demand at each node in each time hour and uncertainty in it.
num_hours = len(hours)

nominal_demand = np.array([3., 3., 3., 3.])
nominal_demand = np.tile(nominal_demand, (num_hours, 1))

demand_increase = np.array([1., 1., 1., 1.])
demand_increase = np.tile(demand_increase, (num_hours, 1))

uncertainty_budget = 2.

# Create the model for the subproblem.
m = Model("subproblem")

K = 100. 	# Bad values (e.g. 10k) will lead to numerical issues

# Add hour- and nodewise uncertain demand variables and nodewise binary variables for deviating from
# the nominal demand values.
d = m.addVars(hours, nodes, name='uncertain_demand', lb=0., ub=GRB.INFINITY)
w = m.addVars(nodes, name='demand_deviation', vtype=GRB.BINARY)

# Variables for linearizing bilinear terms lambda_[o, t, n] * w[n]
z = m.addVars(scenarios, hours, nodes, name='linearization_z', lb=-K, ub=K)
lambda_tilde = m.addVars(scenarios, hours, nodes, name='linearization_lambda_tilde', lb=-K, ub=K)

# Maximum and minimum generation dual variables.
beta_bar = m.addVars(scenarios, hours, units, name='dual_maximum_generation', lb=0., ub=K)
beta_underline = m.addVars(scenarios, hours, units, name='dual_minimum_generation', lb=0., ub=K)

# Maximum up- and downramp dual variables.
beta_ramp_bar = m.addVars(scenarios, hours, units, name='dual_maximum_ramp_upwards', lb=0., ub=K)
beta_ramp_underline = m.addVars(scenarios, hours, units, name='dual_maximum_ramp_downwards',
 								lb=0., ub=K)

# Maximum and minimum transmission flow dual variables.
mu_bar = m.addVars(scenarios, hours, lines, name='dual_maximum_flow', lb=0., ub=K)
mu_underline = m.addVars(scenarios, hours, lines, name='dual_minimum_flow', lb=0., ub=K)

# Balance equation dual (i.e. price).
max_lambda_ = K 	
min_lambda_ = -K
lambda_ = m.addVars(scenarios, hours, nodes, name='dual_balance', lb=min_lambda_, ub=max_lambda_)


def get_objective(x, y):
	# Define subproblem objective function for fixed x and y (unit and line investments).
	obj = sum(sum(z[o, t, n]*demand_increase[t, n] + lambda_[o, t, n]*nominal_demand[t, n]
		 		  for n in nodes) -
			  sum(beta_bar[o, t, u]*G_max[o, t, u]*unit_built(x, t, u) for u in units) -
			  sum((mu_bar[o, t, l]*F_max[o, t, l] - mu_underline[o, t, l]*F_min[o, t, l]) *
			 	  line_built(y, t, l) for l in lines) -
			  sum(beta_ramp_bar[o, t, u]*G_ramp_max[o, t,  u]*unit_built(x, t, u) for u in units) +
			  sum(beta_ramp_underline[o, t, u]*G_ramp_min[o, t, u]*unit_built(x, t, u) for u in units)
			  for t in hours for o in scenarios)

	return obj


def set_subproblem_objective(x, y):
	# Set objective function for the subproblem for fixed x and y (unit and line investments).
	obj = get_objective(x, y)

	m.setObjective(obj, GRB.MAXIMIZE)


# Set initial subproblem objective with dummy investment decisions (i.e. no investment).
set_subproblem_objective(x=np.zeros((999, 999)), y=np.zeros((999, 999)))


# Dual constraints.
m.addConstrs((lambda_[o, t, node_of_unit(u)] - beta_bar[o, t, u] + beta_underline[o, t, u] -
			  (beta_ramp_bar[o, t - 1, u] if not is_year_fist_hour(t) else 0.) +
			  (beta_ramp_bar[o, t, u] if not is_year_last_hour(t) else 0.) +
			  (beta_ramp_underline[o, t - 1, u] if not is_year_fist_hour(t) else 0.) -
			  (beta_ramp_underline[o, t, u] if not is_year_last_hour(t) else 0.) -
 			  C_g[o, t, u]*weights[o] == 0. for o in scenarios for t in hours for u in units),
 			 name="generation_dual_constraint")

m.addConstrs((sum(incidence[l, n]*lambda_[o, t, n] for n in nodes) - mu_bar[o, t, l] +
 			  mu_underline[o, t, l] == 0. for o in scenarios for t in hours for l in lines),
 			  name='flow_dual_constraint')

# Constraints defining the uncertainty set.
m.addConstrs((d[t, n] - nominal_demand[t, n] - w[n]*demand_increase[t, n] == 0.
 			  for t in hours for n in nodes), name='uncertainty_set_demand_increase')

m.addConstr(sum(w[n] for n in nodes) - uncertainty_budget <= 0., name="uncertainty_set_budget")

# Constraints for linearizing lambda_[n, o] * d[n].
m.addConstrs((z[o, t, n] - lambda_[o, t, n] + lambda_tilde[o, t, n] == 0.
 			  for o in scenarios for t in hours for n in nodes), name='linearization_z_definition')

m.addConstrs((w[n]*min_lambda_ - z[o, t, n] <= 0. for o in scenarios for t in hours for n in nodes),
			 name='linearization_z_lb')

m.addConstrs((z[o, t, n] - w[n]*max_lambda_ <= 0. for o in scenarios for t in hours for n in nodes),
			 name='linearization_z_ub')

m.addConstrs(((1. - w[n])*min_lambda_ - lambda_tilde[o, t, n] <= 0.
			  for o in scenarios for t in hours for n in nodes), name='lambda_tilde_lb')

m.addConstrs((lambda_tilde[o, t, n] - (1. - w[n])*max_lambda_ <= 0.
 			  for o in scenarios for t in hours for n in nodes), name='lambda_tilde_ub')


# Set the subproblem model to a variable that can be imported elsewhere.
subproblem = m


def get_uncertain_variables():
	# Get the indices and values of uncertain variables of the subproblem.
	current_d = {key: np.round(value.x) for key, value in d.items()}

	return current_d
