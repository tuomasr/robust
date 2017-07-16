from gurobipy import *
import numpy as np

# model data
nodes = [0, 1, 2]
units = [0, 1, 2]
lines = [0, 1, 2]
generation_costs = [1, 2, 3]
generation_capacities = [20, 20, 20]
transmission_capacities = np.array([5, 5, 5])
demand = [5, 5, 5]

incidence = np.array([[-1, 1, 0],
					  [0, -1, 1],
					  [-1, 0, 1]])

# create a model
m = Model("3node")

# add generation variables
g = m.addVars(units, name='generation', lb=0, ub=generation_capacities)

# flow variables
f = m.addVars(lines, name='flow', lb=-transmission_capacities, ub=transmission_capacities)

# set objective
m.setObjective(sum(generation_costs[u]*g[u] for u in units), GRB.MINIMIZE)

# balance equation
m.addConstrs((g[n] + sum(incidence[l, n]*f[l] for l in lines) == demand[n] for n in nodes),
			 name='balance')

"""
# maximum generation constraints
m.addConstrs((g[n] <= generation_capacities[n] for n in nodes), name='maximum_generation')

# maximum transmission flows (positive direction)
m.addConstrs((f[l] <= transmission_capacities[l] for l in lines), name='maximum_flow')

# minimum transmission flows (negative direction)
m.addConstrs((f[l] >= -transmission_capacities[l] for l in lines), name='minimum_flow')
"""

# optimize and print variable values
m.optimize()

for v in m.getVars():
	print(v.varName, v.x)

for c in m.getConstrs():
	print(c.rhs)

print('Obj:', m.objVal)
