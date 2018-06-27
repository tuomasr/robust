from gurobipy import *
import numpy as np

from common_data import scenarios, nodes, units, lines, existing_units, existing_lines, \
	candidate_units, candidate_lines, G_max, F_max, F_min, incidence, weights, C_g


# problem-specific data
nominal_demand = np.array([3., 3., 3., 3.])
demand_increase = np.array([1., 1., 1., 1.])
uncertainty_budget = 2.

K = 100. 	# bad values (e.g. 10k) will lead to numerical issues
max_lambda_ = K 	
min_lambda_ = -K

# Master problem definition
mp = Model("subproblem_master")

delta = mp.addVar(name='theta', lb=-GRB.INFINITY, ub=GRB.INFINITY)

w = mp.addVars(nodes, name='demand_deviation', vtype=GRB.BINARY)

mp.setObjective(delta, GRB.MAXIMIZE)

mp.addConstr(sum(w[n] for n in nodes) - uncertainty_budget <= 0., name="uncertainty_set_budget")

def augment_master(sigma, rho_bar, rho_underline, omega_bar, omega_underline):
	if sigma is None or rho_bar is None or rho_underline is None or omega_bar is None or omega_underline is None:
		return

	mp.addConstr((delta - sum((max_lambda_*sum((rho_bar[n, o] - omega_bar[n, o])*w[n] for n in nodes) +
				 min_lambda_*sum((rho_underline[n, o] - omega_underline[n, o])*w[n] for n in nodes) +
				 sum(C_g[u]*weights[o]*sigma[u, o] for u in units) +
				 max_lambda_*sum(omega_bar[n, o] for n in nodes) -
				 min_lambda_*sum(omega_underline[n, o] for n in nodes)) for o in scenarios) <= 0.),
				 name='theta_constraint')


def get_uncertain_demand_decisions():
	uncertain = {u: w[u].x for u in w.keys()}

	return uncertain

# Slave problem definition
def create_slave(x, y, ww):
	sp = Model("subproblem_slave")

	# variables for linearizing bilinear terms lambda_[n, o] * u[n]
	z = sp.addVars(nodes, scenarios, name='linearization_lambda_d', lb=-K, ub=K)
	lambda_tilde = sp.addVars(nodes, scenarios, name='linearization_auxiliary', lb=-K, ub=K)

	# maximum and minimum generation dual variables
	beta_bar = sp.addVars(units, scenarios, name='dual_maximum_generation', lb=0., ub=K)
	beta_underline = sp.addVars(units, scenarios, name='dual_minimum_generation', lb=0., ub=K)

	# maximum and minimum transmission flow dual variables
	mu_bar = sp.addVars(lines, scenarios, name='dual_maximum_flow', lb=0., ub=K)
	mu_underline = sp.addVars(lines, scenarios, name='dual_minimum_flow', lb=0., ub=K)

	obj = \
		sum(sum(z[n, o]*demand_increase[n] + (z[n, o] + lambda_tilde[n, o])*nominal_demand[n] for n in nodes) -
			sum(beta_bar[u, o]*G_max[u, o] for u in existing_units) -
			sum(mu_bar[l, o]*F_max[l, o] - mu_underline[l, o]*F_min[l, o] for l in existing_lines) -
			sum(beta_bar[u, o]*G_max[u, o]*x[u] for u in candidate_units) -
			sum(mu_bar[l, o]*F_max[l, o]*y[l] - mu_underline[l, o]*F_min[l, o]*y[l] for l in candidate_lines)
			for o in scenarios)

	sp.setObjective(obj, GRB.MAXIMIZE)

	# dual constraints
	sigma_constrs = \
		sp.addConstrs((z[n, o] + lambda_tilde[n, o] - beta_bar[n, o] + beta_underline[n, o] - C_g[n]*weights[o] == 0.
				  	  for n in nodes for o in scenarios), name="generation_dual_constraint")

	sp.addConstrs((sum(incidence[l, n]*(z[n, o] + lambda_tilde[n,o]) for n in nodes) - mu_bar[l, o] +
	 			  mu_underline[l, o] == 0. for l in lines for o in scenarios),
	 			  name='flow_dual_constraint')

	rho_underline_constrs = \
		sp.addConstrs((ww[n]*min_lambda_ - z[n, o] <= 0. for n in nodes for o in scenarios),
				      name='linearization_z_bounds1')

	rho_bar_constrs = \
		sp.addConstrs((z[n, o] - ww[n]*max_lambda_ <= 0. for n in nodes for o in scenarios),
				  	  name='linearization_z_bounds2')

	omega_underline_constrs = \
		sp.addConstrs(((1. - ww[n])*min_lambda_ - lambda_tilde[n, o] <= 0. for n in nodes
				  	  for o in scenarios), name='lambda_tilde_bounds1')

	omega_bar_constrs = \
		sp.addConstrs((lambda_tilde[n, o] - (1. - ww[n])*max_lambda_ <= 0. for n in nodes
				  	  for o in scenarios), name='lambda_tilde_bounds2')

	return sp, sigma_constrs, rho_bar_constrs, rho_underline_constrs, omega_bar_constrs, \
		omega_underline_constrs


def get_dual_variables(constrs):
	dual_values = dict()

	for u in units:
		for o in scenarios:
			dual_values[u, o] = max(0., constrs[u, o].getAttr('Pi'))

	return dual_values


def get_uncertain_variables(ww):
	# get the names and values of uncertain variables of the subproblem
	uncertain_variables = [v for v in sp.getVars() if 'uncertain_demand' in v.varName]

	names = np.array([v.varName for v in uncertain_variables])
	values = np.array([v.x for v in uncertain_variables])

	return names, values


def solve_subproblem(x, y):
	max_iterations = 10
	threshold = 1e-6

	sigma = rho_bar = rho_underline = omega_bar = omega_underline = None

	for i in range(max_iterations):
		augment_master(sigma, rho_bar, rho_underline, omega_bar, omega_underline)
		mp.optimize()

		if i > 0 and mp.Status != GRB.OPTIMAL:
			raise RuntimeError("Master problem not optimal.")

		lb = mp.objVal

		ww = get_uncertain_demand_decisions()

		sp, sigma_constrs, rho_bar_constrs, rho_underline_constrs, omega_bar_constrs, \
			omega_underline_constrs = create_slave(x, y, ww)
		sp.optimize()
		ub = sp.objVal

		if sp.Status != GRB.OPTIMAL:
			raise RuntimeError("Subproblem not optimal.")

		if i > 0: 
			if abs(ub - lb) / ub < threshold:
				return

		sigma = get_dual_variables(sigma_constrs)
		rho_bar = get_dual_variables(rho_bar_constrs)
		rho_underline = get_dual_variables(rho_underline_constrs)
		omega_bar = get_dual_variables(omega_bar_constrs)
		omega_underline = get_dual_variables(omega_underline_constrs)

		import pdb; pdb.set_trace()
		asd=1

