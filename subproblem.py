from gurobipy import *
import numpy as np

from common_data import scenarios, nodes, units, lines, existing_units, existing_lines, \
	candidate_units, candidate_lines, G_max, F_max, F_min, incidence, weights, C_g


# problem-specific data
nominal_demand = np.array([1., 1., 1.])
demand_increase = np.array([1., 1., 1.])
uncertainty_budget = 2.

# create a model
m = Model("subproblem")

# add nodewise uncertain demand variables and binary variables for deviating from the
# nominal demand value
d = m.addVars(nodes, name='uncertain_demand', lb=0.)
u = m.addVars(units, name='demand_deviation', vtype=GRB.BINARY)

# variables for linearizing bilinear terms lambda_[n, o] * u[n]
z = m.addVars(nodes, scenarios, name='linearization_lambda_d')
lambda_tilde = m.addVars(nodes, scenarios, name='linearization_auxiliary')

# maximum and minimum generation dual variables
beta_bar = m.addVars(units, scenarios, name='dual_maximum_generation', lb=0.)
beta_underline = m.addVars(units, scenarios, name='dual_minimum_generation', lb=0.)

# maximum and minimum transmission flow dual variables
mu_bar = m.addVars(lines, scenarios, name='dual_maximum_flow', lb=0.)
mu_underline = m.addVars(lines, scenarios, name='dual_minimum_flow', lb=0.)

# balance equation dual (i.e. price)
max_lambda_ = 100. 	# bad values (e.g. 10k) will lead to numerical issues
min_lambda_ = -100.
lambda_ = m.addVars(nodes, scenarios, name='dual_balance', lb=min_lambda_, ub=max_lambda_)

# set objective
def get_objective(x, y):
	# get objective function for fixed values of x and y
	obj = \
		sum(sum(z[n, o]*demand_increase[n] + lambda_[n, o]*nominal_demand[n] for n in nodes) -
			sum(beta_bar[u, o]*G_max[u, o] for u in existing_units) -
			sum(mu_bar[l, o]*F_max[l, o] - mu_underline[l, o]*F_min[l, o] for l in existing_lines) -
			sum(beta_bar[u, o]*G_max[u, o]*x for u in candidate_units) -
			sum(mu_bar[l, o]*F_max[l, o]*y - mu_underline[l, o]*F_min[l, o]*y for l in candidate_lines)
			for o in scenarios)

	return obj

def set_subproblem_objective(x=0., y=0.):
	# set objective function for the subproblem
	obj = get_objective(x, y)

	m.setObjective(obj, GRB.MAXIMIZE)

set_subproblem_objective()

# dual constraints
m.addConstrs((lambda_[n, o] - beta_bar[n, o] + beta_underline[n, o] == C_g[n]*weights[o]
 			 for n in nodes for o in scenarios), name="generation_dual_constraint")

m.addConstrs((sum(incidence[l, n]*lambda_[n, o] for n in nodes) - mu_bar[l, o] +
 			  mu_underline[l, o] == 0. for l in lines for o in scenarios),
 			  name='flow_dual_constraint')

# constraints defining the uncertainty set
m.addConstrs((d[n] - nominal_demand[n] - u[n]*demand_increase[n] == 0. for n in nodes),
			 name='uncertainty_set_demand_increase')

m.addConstr(sum(u[n] for n in nodes) <= uncertainty_budget, name="uncertainty_set_budget")

# constraints for linearizing lambda_[n, o] * d[n]
m.addConstrs((z[n, o] - lambda_[n, o] + lambda_tilde[n, o] == 0. for n in nodes for o in scenarios),
			 name='linearization_z_definition')

m.addConstrs((u[n]*min_lambda_ <= z[n, o] <= u[n]*max_lambda_ for n in nodes for o in scenarios),
			 name='linearization_z_bounds')

m.addConstrs(((1. - u[n])*min_lambda_ <= lambda_tilde[n, o] <= (1. - u[n])*max_lambda_ for n in nodes
			 for o in scenarios), name='lambda_tilde_bounds')

subproblem = m


def get_uncertain_variables():
	# get the names and values of uncertain variables of the subproblem
	uncertain_variables = [v for v in subproblem.getVars() if 'uncertain_demand' in v.varName]

	names = np.array([v.varName for v in uncertain_variables])
	values = np.array([v.x for v in uncertain_variables])

	return names, values
