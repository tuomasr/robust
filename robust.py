import numpy as np

from subproblem import subproblem, set_subproblem_objective, get_uncertain_variables, \
	get_subproblem_objective_value
from master_problem import master_problem, augment_master_problem, get_investment_cost
from common_data import nodes, candidate_units, candidate_lines

# Configure Gurobi.
enable_custom_configuration = False

GRB_PARAMS = [('MIPGap', 0.),
			  ('FeasibilityTol', 1e-9),
			  ('IntFeasTol', 1e-9),
			  ('MarkowitzTol', 1e-4),
			  ('OptimalityTol', 1e-9),
			  ('MIPFocus', 2),
			  ('MIPGap', 0.),
			  ('Presolve', 0),
			  ('Cuts', 0),
			  ('Aggregate', 0)]

if enable_custom_configuration:
	for parameter, value in GRB_PARAMS:
		master_problem.setParam(parameter, value)
		subproblem.setParam(parameter, value)

MAX_ITERATIONS = 10
EPSILON = 1e-6 	# From Minguez (2016)
# A bug or numerical issues cause LB to become higher than UB in some cases.
# This allows some slack.
BAD_GAP_THRESHOLD = -1e-6

UB = np.inf
LB = -np.inf


def compute_objective_gap(LB, UB):
	return (UB - LB) / float(UB)


d = np.zeros((len(nodes), 1)) 	# no uncertain variables for the first iteration
converged = False

separator = '-' * 50

gaps = []


def print_iteration_counter(iteration):
	print(separator)
	print('ITERATION', iteration)
	print(separator)


def print_solution_quality(problem, problem_name):
	print(separator)
	print('%s quality and stats:' % problem_name)
	problem.printQuality()
	problem.printStats()
	print(separator)


for iteration in range(MAX_ITERATIONS):
	print_iteration_counter(iteration)

	# augment the master problem for the current iteration
	if iteration > 0:
		augment_master_problem(iteration, d)

	master_problem.optimize()

	print_solution_quality(master_problem, "Master problem")

	# update lower bound
	LB = master_problem.objVal

	# obtain investment decisions
	master_problem_x = [v for v in master_problem.getVars() if 'unit_investment' in v.varName]
	master_problem_y = [v for v in master_problem.getVars() if 'line_investment' in v.varName]

	# round and convert to integer as there may be floating point errors
	x = [int(v.x) for v in master_problem_x]
	y = [int(v.x) for v in master_problem_y]

	if iteration == 0:
		assert np.allclose(x, 0) and np.allclose(y, 0), 'Initial master problem solution suboptimal.'
		prev_x = x
		prev_y = y

	x = {u: v for u, v in zip(candidate_units, x)}
	y = {l: v for l, v in zip(candidate_lines, y)}

	# update subproblem objective function with the investment decisions
	set_subproblem_objective(x, y)

	subproblem.optimize()

	print_solution_quality(subproblem, "Subproblem")

	# update upper bound and compute new gap
	UB = get_investment_cost(x, y) + get_subproblem_objective_value(x, y)

	GAP = compute_objective_gap(LB, UB)

	gaps.append(GAP)

	# read the values of the uncertain variables
	_, uncertain_variable_vals = get_uncertain_variables()

	# add new column to d
	uncertain_variable_vals = uncertain_variable_vals[:, np.newaxis]
	d = np.concatenate((d, uncertain_variable_vals), axis=1)

	# exit if the algorithm converged. 1) LB and UB needs to be close to each other
	# 2) solutions to master problem and subproblem must stay constant.
	prev_x = x
	prev_y = y

	unchanged_decisions = (prev_x == x) and (prev_y == y) and all(d[:, -1] == d[:, -2])

	if GAP < EPSILON and unchanged_decisions:
		converged = True
		assert GAP >= BAD_GAP_THRESHOLD, 'Upper bound %f, lower bound %f.' % (UB, LB)
		break


print_primal_variables = True

if print_primal_variables:
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

print_dual_variables = False

if print_dual_variables:
	print separator
	print 'Dual variables:'
	print separator
	for v in subproblem.getVars():
		print v.varName, v.x

print separator

if converged:
	print 'Converged at iteration %d! Gap: %s' % (iteration, GAP)
else:
	print 'Did not converge. Gap: ', GAP

print separator
print 'Objective value:', master_problem.objVal
print 'Investment cost %s, operation cost %s ' % (get_investment_cost(x, y), subproblem.objVal)
print separator

plot_gap = False

if plot_gap:
	from matplotlib import pyplot as plt

	plt.plot(gaps)
	plt.ylim(np.amin(gaps), 1e-2)
	plt.ylabel('GAP')
	plt.xlabel('iteration')
	plt.show()
