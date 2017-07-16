from gurobipy import *
import numpy as np

# model data
scenarios = [0, 1]
nodes = [0, 1, 2]

existing_units = [1, 2]
existing_lines = [1, 2]

candidate_units = [0]
candidate_lines = [0]

generation_investment_cost = 10
line_investment_cost = 10

units = candidate_units + existing_units
lines = candidate_lines + existing_lines

generation_costs = [1, 2, 3]

generation_capacities = np.array([[20, 20, 20], [20, 20, 20]])
transmission_capacities = np.array([[5, 5, 5], [5, 5, 5]])

demand = np.array([[5, 10],
				   [5, 10],
				   [5, 10]])

incidence = np.array([[-1, 1, 0],
					  [0, -1, 1],
					  [-1, 0, 1]])

# create a model
m = Model("3node")

# add generation variables for existing and candidate units
g = m.addVars(units, scenarios, name='generation', lb=0, ub=generation_capacities)

# flow variables for existing and candidate lines
f = m.addVars(lines, scenarios, name='flow', lb=-transmission_capacities,
			  ub=transmission_capacities)

# investment to a generation unit and transmission line
x = m.addVar(vtype=GRB.BINARY, name='x')
y = m.addVar(vtype=GRB.BINARY, name='y')

# set objective
m.setObjective(generation_investment_cost*x + line_investment_cost*y + \
 			   sum(generation_costs[u]*g[u, s] for u in units for s in scenarios), GRB.MINIMIZE)

# balance equation
m.addConstrs((g[n, s] + sum(incidence[l, n]*f[l, s] for l in lines) == demand[n, s]
			 for n in nodes for s in scenarios), name='balance')

# generation constraint for the candidate units
m.addConstrs((g[u, s] <= generation_capacities[s, u]*x for u in candidate_units for s in scenarios),
			name='maximum_candidate_generation')

# flow constraint for the candidate lines
m.addConstrs((f[l, s] <= transmission_capacities[s, l]*y for l in candidate_lines for s in scenarios),
		    name='maximum_candidate_flow')

m.addConstrs((f[l, s] >= -transmission_capacities[s, l]*y for l in candidate_lines for s in scenarios),
		    name='minimum_candidate_flow')


print m

# optimize and print variable values
m.optimize()

for v in m.getVars():
	print(v.varName, v.x)

for c in m.getConstrs():
	print(c.rhs)

print('Obj:', m.objVal)
