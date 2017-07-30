from gurobipy import *
import numpy as np

from data import *

def get_investment_cost(x):
	assert len(x) == len(candidate_lines)

	return sum(C_x[l]*x[l] for l in candidate_lines)


def add_primal_variables(iteration, d, g_max):
	# add generation variables
	g_max = [g_max[n].x for n in nodes]
	g = m.addVars(nodes, [iteration], name='generation', lb=0., ub=g_max)

	# load shedding
	ls_max = [d[n].x for n in nodes]
	ls = m.addVars(nodes, [iteration], name='load_shedding', lb=0., ub=ls_max)

	# flow variables for existing and candidate lines
	f = m.addVars(lines, [iteration], name='flow', lb=-F_min, ub=F_max)

	return g, ls, f


# create the initial model - no constraints are added
m = Model("master_problem")

# investment to transmission lines
x = m.addVars(candidate_lines, vtype=GRB.BINARY, name='x')

# subproblem objective value
eta = m.addVar(name='eta')

# set objective. The optimal solution is no investment
m.setObjective(get_investment_cost(x) + eta, GRB.MINIMIZE)

m.addConstr((get_investment_cost(x) <= investment_budget), name='investment_budget')


def augment_master_problem(current_iteration, d, g_max):
	# augment the master problem for the current iteration
	v = current_iteration

	# create additional primal variables indexed with the current iteration
	g, ls, f = add_primal_variables(v, d, g_max)

	# minimum value for the subproblem objective function
	m.addConstr(eta - sigma * sum(C_g[n]*g[n, v] + C_ls[n]*ls[n, v] for n in nodes) >= 0.,
	 			name='minimum_subproblem_objective')

	m.addConstrs((g[n, v] + sum(incidence[l, n]*f[l, v] for l in lines) - d[n].x + ls[n, v] == 0.
	 			 for n in nodes), name='balance_constraint')

	# flow constraint for the candidate lines
	m.addConstrs((-F_min[l]*x[l] <= f[l, v] <= F_max[l]*x[l] for l in candidate_lines),
	 			 name='candidate_flow_bounds')


master_problem = m
