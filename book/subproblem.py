from gurobipy import *
import numpy as np

from data import *

# create a model
m = Model("subproblem")

# primal variables
d = m.addVars(nodes, name='uncertain_demand', lb=D_min, ub=D_max)
g_max = m.addVars(nodes, name='uncertain_maximum_generation', lb=G_min, ub=G_max)

g = m.addVars(nodes, name='generation') 	# bounds set separately
ls = m.addVars(nodes, name='load_shedding')
f = m.addVars(lines, name='transmission_flow')

# investment decisions provided by the master problem (initially, no investment to candidate lines)
x = np.array([1. for line in existing_lines] + [0. for line in candidate_lines])

# dual variables
lambda_ = m.addVars(nodes, name='lambda')

beta_min = m.addVars(nodes, name='dual_min_generation', lb=0.)
beta_max = m.addVars(nodes, name='dual_max_generation', lb=0.)

phi_min = m.addVars(nodes, name='dual_min_load_shedding', lb=0.)
phi_max = m.addVars(nodes, name='dual_max_load_shedding', lb=0.)

mu_min = m.addVars(lines, name='dual_min_flow', lb=0.)
mu_max = m.addVars(lines, name='dual_max_flow', lb=0.)

m.setObjective(sigma*sum(C_g[n]*g[n] + C_ls[n]*ls[n] for n in nodes), GRB.MAXIMIZE)

######################
# uncertainty sets
# maximum generation uncertainty
m.addConstr(0. <= (sum(G_max[n] - g_max[n] for n in nodes) / sum(G_max[n] for n in nodes)) <=
			generation_uncertainty_budget, name='generation_uncertainty_budget')

# demand uncertainty
m.addConstr(0. <= (sum(d[n] - D_min[n] for n in nodes) / sum(D_max[n] - D_min[n] for n in nodes)) <=
			demand_uncertainty_budget, name='demand_uncertainty_budget')

######################
# primal constraints
# balance equation
m.addConstrs((g[n] + sum(incidence[l, n]*f[l] for l in lines) - d[n] + ls[n] == 0. for n in nodes),
			 name='balance_constraint')

# generation bounds
m.addConstrs((G_min[n] <= g[n] <= g_max[n] for n in nodes), name='generation_bounds')

# load shed bounds
m.addConstrs((0. <= ls[n] <= d[n] for n in nodes), name='load_shed_bounds')

# flow bounds
def add_flow_constraints(z):
	m.addConstrs((-f[l] <= F_min[l]*z[l]  for l in lines), name='minimum_flow')
	m.addConstrs((f[l] <= F_max[l]*z[l] for l in lines), name='maximum_flow')

######################
# KKT constraints
M = 10000

# w.r.t. generation variables g[n]
r1 = m.addVars(nodes, name='r1')
m.addConstrs((sigma*C_g[n] - lambda_[n] + beta_max[n] - beta_min[n] >= 0. for n in nodes),
 			 name='kkt_generation1')
m.addConstrs((sigma*C_g[n] - lambda_[n] + beta_max[n] - beta_min[n] <= M*r1[n] for n in nodes),
 			 name='kkt_generation2')
m.addConstrs((g[n] <= M*(1 - r1[n]) for n in nodes), name='kkt_generation3')

# w.r.t. load shedding ls[n]
r2 = m.addVars(nodes, name='r2')
m.addConstrs((sigma*C_ls[n] - lambda_[n] + phi_max[n] - phi_min[n] >= 0. for n in nodes),
			 name='kkt_load_shedding1')
m.addConstrs((sigma*C_ls[n] - lambda_[n] + phi_max[n] - phi_min[n] <= M*r2[n] for n in nodes),
			 name='kkt_load_shedding2')
m.addConstrs((ls[n] <= M*(1 - r2[n]) for n in nodes), name='kkt_load_shedding3')

# w.r.t flow f[l]
m.addConstrs((sum(incidence[l, n]*lambda_[n] for n in nodes) + mu_max[l] - mu_min[l] == 0.
 			 for l in lines), name='kkt_flow')

#######################
# complementary conditions
def generate_complementary_conditions(dual_variables, primal_variables, sign, constants, M=10000,
									  name='asd'):
	num_variables = len(dual_variables)
	indices = range(num_variables)

	u = m.addVars(num_variables, vtype=GRB.BINARY, name=name + '_disj_var')

	m.addConstrs((sign*primal_variables[i] + constants[i] >= 0. for i in indices),
				 name=name + '1')

	m.addConstrs((sign*primal_variables[i] + constants[i] <= M * u[i] for i in indices),
				 name=name + '2')

	m.addConstrs((dual_variables[i] <= M*(1 - u[i]) for i in indices), name=name + '3')


generate_complementary_conditions(beta_min, g, 1., np.zeros(len(g)), name='beta_min')
generate_complementary_conditions(beta_max, g, -1., G_max, name='beta_max')

generate_complementary_conditions(phi_min, ls, 1., np.zeros(len(ls)), name='phi_min')
generate_complementary_conditions(phi_max, ls, -1., D_max, name='phi_max')


def fix_investment_decisions(decisions):
	# remove constraints and add new ones
	def get_constr_by_name(name):
		return [constr for constr in m.getConstrs() if name in constr.ConstrName]

	def get_var_by_name(name):
		return [var for var in m.getVars() if name in var.varName]

	names = ['minimum_flow', 'maximum_flow', 'mu_min', 'mu_max']
	constrs = [get_constr_by_name(name) for name in names]
	unused_vars = [get_var_by_name(name) for name in ('mu_min', 'mu_max')]

	m.remove(constrs)
	m.remove(unused_vars)

	m.update()

	decisions = np.array([1. for line in existing_lines] + decisions)

	add_flow_constraints(decisions)

	generate_complementary_conditions(mu_min, f, 1., F_min*x, name='mu_min')
	generate_complementary_conditions(mu_max, f, -1., F_max*x, name='mu_max')

	m.update()


def get_uncertain_variables():
	return d, g_max


subproblem = m
