import numpy as np

from subproblem import subproblem, fix_investment_decisions, get_uncertain_variables
from master_problem import master_problem, augment_master_problem, get_investment_cost
from data import *


MAX_ITERATIONS = 10
EPSILON = 1e-6 	# From Minguez (2016)

UB = np.inf
LB = -np.inf


def compute_objective_gap(LB, UB):
	return (UB - LB) / UB


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
		augment_master_problem(iteration, d, g_max)

	master_problem.optimize()

	# update lower bound
	LB = master_problem.objVal

	# obtain investment decisions
	master_problem_x = [master_problem.getVarByName('x[%d]' % i) for i in candidate_lines]

	# round and convert to integer as there may be floating point errors
	x = [int(np.round(var.x)) for var in master_problem_x]

	if iteration == 0:
		assert np.allclose(x, 0)

	# update subproblem objective function with the investment decisions
	fix_investment_decisions(x)

	subproblem.optimize()

	# update upper bound and compute new gap
	decisions = {l: decision for l, decision in zip(candidate_lines, x)}
	UB = get_investment_cost(decisions) + subproblem.objVal

	GAP = compute_objective_gap(LB, UB)
	assert GAP >= 0., 'Upper bound below lower bound :/'

	# exit if the algorithm converged
	if GAP < EPSILON:
		converged = True
		break

	# read the values of the uncertain variables
	d, g_max = get_uncertain_variables()

print separator

if converged:
	print 'Converged at iteration %d! Gap: %s' % (iteration, GAP)
else:
	print 'Did not converge. Gap: ', GAP

print separator
print 'Objective value:', master_problem.objVal
print 'Investment cost %s, operation cost %s ' % (get_investment_cost(decisions), subproblem.objVal)
print separator
print 'Master problem primal variables:'
print separator
for v in master_problem.getVars():
	print v.varName, v.x

print separator
print 'Subproblem primal variables:'
print separator
for v in subproblem.getVars():
	print v.varName, v.x

print separator
print 'Uncertain variables:'
print separator
names, values = get_uncertain_variables()
for name, value in zip(names, values):
	print name, value
