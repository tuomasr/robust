import numpy as np

from subproblem import subproblem, set_subproblem_objective, get_uncertain_variables
from master_problem import master_problem, augment_master_problem, get_investment_cost
from common_data import nodes


MAX_ITERATIONS = 10
EPSILON = 1e-6 	# From Minguez (2016)

UB = np.inf
LB = -np.inf


def compute_objective_gap(LB, UB):
	return (UB - LB) / UB


d = np.zeros((len(nodes), 1)) 	# no uncertain variables for the first iteration
converged = False

separator = '-' * 50


def print_iteration_counter(iteration):
	print separator
	print 'ITERATION', iteration
	print separator


for iteration in range(MAX_ITERATIONS):
	print_iteration_counter(iteration)

	# augment the master problem for the current iteration
	if iteration > 0:
		augment_master_problem(iteration, d)

	master_problem.optimize()

	# update lower bound
	LB = master_problem.objVal

	# obtain investment decisions
	master_problem_x = master_problem.getVarByName('x')
	master_problem_y = master_problem.getVarByName('y')

	# round and convert to integer as there may be floating point errors
	x = int(np.round(master_problem_x.x))
	y = int(np.round(master_problem_y.x))

	if iteration == 0:
		assert x == y == 0, 'Initial master problem solution suboptimal.'

	# update subproblem objective function with the investment decisions
	set_subproblem_objective(x, y)

	subproblem.optimize()

	# update upper bound and compute new gap
	UB = get_investment_cost(x, y) + subproblem.objVal

	GAP = compute_objective_gap(LB, UB)
	assert GAP >= 0., 'Upper bound below lower bound :/'

	# exit if the algorithm converged
	if GAP < EPSILON:
		converged = True
		break

	# read the values of the uncertain variables
	_, uncertain_variable_vals = get_uncertain_variables()

	# add new column to d
	uncertain_variable_vals = uncertain_variable_vals[:, np.newaxis]
	d = np.concatenate((d, uncertain_variable_vals), axis=1)

print separator

if converged:
	print 'Converged at iteration %d! Gap: %s' % (iteration, GAP)
else:
	print 'Did not converge. Gap: ', GAP

print separator
print 'Objective value:', master_problem.objVal
print 'Investment cost %s, operation cost %s ' % (get_investment_cost(x, y), subproblem.objVal)
print separator
print 'Primal variables:'
print separator
for v in master_problem.getVars():
	print v.varName, v.x

print separator
print 'Uncertain variables:'
print separator
names, values = get_uncertain_variables()
for name, value in zip(names, values):
	print name, value
