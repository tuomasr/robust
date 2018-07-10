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
min_lambda_ = 0.

# Master problem definition
mp = Model("subproblem_master")

delta = mp.addVar(name='delta')

w = mp.addVars(nodes, name='demand_deviation', vtype=GRB.BINARY)

mp.setObjective(delta, GRB.MAXIMIZE)

mp.addConstr(sum(w[n] for n in nodes) - uncertainty_budget <= 0., name="uncertainty_set_budget")


def augment_master(dual_values):
	# Unpack the dual values. Be careful with the order.
	sigma, rho_underline, rho_bar, omega_underline, omega_bar, beta_underline, beta_bar, \
		mu_underline, mu_bar = dual_values

	mp.addConstr((delta -
				 sum((max_lambda_*sum((rho_bar[n, o] - omega_bar[n, o])*w[n] for n in nodes) +
				 min_lambda_*sum((rho_underline[n, o] - omega_underline[n, o])*w[n] for n in nodes) +
				 sum(C_g[u]*weights[o]*sigma[u, o] for u in units) +
				 sum(beta_bar[u, o] for u in units) +
				 sum(beta_underline[u, o] for u in units) +
				 sum(mu_bar[l, o] for l in lines) +
				 sum(mu_underline[l, o] for l in lines) +
				 max_lambda_*sum(omega_bar[n, o] for n in nodes) -
				 min_lambda_*sum(omega_underline[n, o] for n in nodes)) for o in scenarios) <= 0.),
				 name='delta_constraint')


def get_uncertain_demand_decisions(initial):
	# Construct an initial solution for the first iteration, otherwise read the optimal solution.
	if initial:
		# The initial solution just assigns '1' in order until the uncertainty budget is met.
		uncertain = dict()
		total = 0

		for u in units:
			if total < uncertainty_budget:
				uncertain[u] = 1.
				total += 1
			else:
				uncertain[u] = 0.
	else:
		uncertain = {u: np.round(w[u].x, 3) for u in w.keys()}

	return uncertain


# Slave problem definition
def create_slave(x, y, ww):
	sp = Model("subproblem_slave")
	sp.Params.Method = 1 	# Dual simplex.

	# variables for linearizing bilinear terms lambda_[n, o] * u[n]
	z = sp.addVars(nodes, scenarios, name='linearization_lambda_d')
	lambda_tilde = sp.addVars(nodes, scenarios, name='linearization_auxiliary')

	# minimum and maximum generation dual variables
	beta_underline = sp.addVars(units, scenarios, name='dual_minimum_generation')
	beta_bar = sp.addVars(units, scenarios, name='dual_maximum_generation')
	
	# minimum and maximum transmission flow dual variables
	mu_underline = sp.addVars(lines, scenarios, name='dual_minimum_flow')
	mu_bar = sp.addVars(lines, scenarios, name='dual_maximum_flow')

	obj = \
		sum(sum(z[n, o]*demand_increase[n] + (z[n, o] + lambda_tilde[n, o])*nominal_demand[n] for n in nodes) -
			sum(beta_bar[u, o]*G_max[u, o] for u in existing_units) -
			sum(mu_bar[l, o]*F_max[l, o] - mu_underline[l, o]*F_min[l, o] for l in existing_lines) -
			sum(beta_bar[u, o]*G_max[u, o]*x[u] for u in candidate_units) -
			sum(mu_bar[l, o]*F_max[l, o]*y[l] - mu_underline[l, o]*F_min[l, o]*y[l] for l in candidate_lines)
			for o in scenarios)

	sp.setObjective(obj, GRB.MAXIMIZE)

	sp.addConstrs((0. <= beta_underline[u, o] for u in units for o in scenarios))
	sp.addConstrs((0. <= beta_bar[u, o] for u in units for o in scenarios))
	
	sp.addConstrs((0. <= mu_underline[u, o] for u in units for o in scenarios))
	sp.addConstrs((0. <= mu_bar[u, o] for u in units for o in scenarios))

	beta_underline_constrs = sp.addConstrs((beta_underline[u, o] <= K for u in units for o in scenarios))
	beta_bar_constrs = sp.addConstrs((beta_bar[u, o] <= K for u in units for o in scenarios))
	
	mu_underline_constrs = sp.addConstrs((mu_underline[u, o] <= K for u in units for o in scenarios))
	mu_bar_constrs = sp.addConstrs((mu_bar[u, o] <= K for u in units for o in scenarios))

	# dual constraints
	sigma_constrs = \
		sp.addConstrs((z[n, o] + lambda_tilde[n, o] - beta_bar[n, o] + beta_underline[n, o] - C_g[n]*weights[o] == 0.
				  	  for n in nodes for o in scenarios), name="generation_dual_constraint")

	sp.addConstrs((sum(incidence[l, n]*(z[n, o] + lambda_tilde[n,o]) for n in nodes) - mu_bar[l, o] +
	 			  mu_underline[l, o] == 0. for l in lines for o in scenarios),
	 			  name='flow_dual_constraint')

	rho_underline_constrs = \
		sp.addConstrs((ww[n]*min_lambda_ - z[n, o] <= 0.  for n in nodes for o in scenarios),
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

	all_constrs = (sigma_constrs,
	 		       rho_underline_constrs, rho_bar_constrs,
	 		       omega_underline_constrs, omega_bar_constrs,
	 		       beta_underline_constrs, beta_bar_constrs,
	 		       mu_underline_constrs, mu_bar_constrs)

	updatable_constrs = (rho_underline_constrs, rho_bar_constrs,
						 omega_underline_constrs, omega_bar_constrs)

	return sp, all_constrs, updatable_constrs


def update_slave(updatable_constrs, ww):
	# Update constraints involving the value of w. Be careful with the order when unpacking.
	rho_underline_constrs, rho_bar_constrs, omega_underline_constrs, omega_bar_constrs = \
		updatable_constrs

	for n in nodes:
		for o in scenarios:
			rho_underline_constrs[n, o].RHS = -ww[n]*min_lambda_
			rho_bar_constrs[n, o].RHS = ww[n]*max_lambda_

			omega_underline_constrs[n, o].RHS = -(1. - ww[n])*min_lambda_
			omega_bar_constrs[n, o].RHS = (1. - ww[n])*max_lambda_


def get_dual_variables(all_constrs):
	# Get dual variable values of all (relevant) constraints of the slave problem.
	def _get_dual_values(constrs):
		dual_values = dict()

		for u in units:
			for o in scenarios:
				dual_values[u, o] = constrs[u, o].getAttr('Pi')

		return dual_values

	return [_get_dual_values(constrs) for constrs in all_constrs]


def get_uncertain_variables():
	# Get the names and values of uncertain variables of the subproblem.
	names = np.array([w[u].varName for u in units])
	values = np.array([np.round(w[u].x*demand_increase[u] + nominal_demand[u], 3) for u in units])

	return names, values


def solve_subproblem(x, y):
	max_iterations = 10
	threshold = 1e-6
	separator = '-' * 50

	lb = -np.inf
	ub = np.inf

	for iteration in range(max_iterations):
		initial = iteration == 0

		if not initial:
			augment_master(dual_values)

			print(separator)
			print('Solving Benders master problem.')
			mp.optimize()
			print(separator)

			if mp.Status != GRB.OPTIMAL:
				raise RuntimeError("Benders master problem not optimal.")

			ub = mp.objVal
		
		ww = get_uncertain_demand_decisions(initial)

		# Create a slave problem during the first iteration. Otherwise, only update the values of w.
		if initial:
			sp, all_constrs, updatable_constrs = create_slave(x, y, ww)
		else:
			update_slave(updatable_constrs, ww)
			sp.Params.Method = 1 	# Dual simplex.

		print(separator)
		print('Solving Benders slave problem.')
		sp.optimize()
		print(separator)
		
		lb = sp.objVal

		if sp.Status != GRB.OPTIMAL:
			raise RuntimeError("Benders slave problem not optimal.")

		if lb > ub + threshold:
			raise RuntimeError("lb > ub in Benders")

		if abs(ub - lb) / ub < threshold:
			return (lb + ub)/2

		dual_values = get_dual_variables(all_constrs)

	raise RuntimeError("Max iterations hit in Benders.")
