# Solve robust optimization problem for power markets.
# Master problem and subproblem as well as input data are defined in their respective files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from common_data import hours, nodes, candidate_units, candidate_lines
from helpers import concatenate_to_uncertain_variables_array, is_solution_unchanged, Timer
from master_problem import master_problem, augment_master_problem, get_investment_cost, \
	get_investment_decisions
from subproblem import subproblem, set_subproblem_objective, get_uncertain_variables


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


# Configure the algorithm for solving the robust optimization problem.
MAX_ITERATIONS = 10

# Compile solution times for the master problem and subproblem to these objects.
master_problem_timer = Timer()
subproblem_timer = Timer()


# Threshold for algorithm convergence.
EPSILON = 1e-6 	# From Minguez et al. (2016)

# Numerical precision issues can cause LB to become higher than UB in some cases.
# This threshold allows some slack but an error is raised if the threshold is exceeded.
BAD_GAP_THRESHOLD = -1e-6

# Initial upper and lower bounds for the algorithm.
UB = np.inf
LB = -np.inf

# Used for collecting gaps when the algorithm runs.
gaps = []


def compute_objective_gap(LB, UB):
	# Compute the relative gap between upper and lower bounds of the algorithm.
	return (UB - LB) / float(UB)


# Initial uncertain variables for the first master problem iteration.
d = np.zeros((len(hours), len(nodes), 1))

# A line separating stuff in standard output.
separator = '-' * 50


def print_iteration_counter(iteration):
	# Print the current iteration number.
	print('ITERATION', iteration)
	print(separator)


def print_solution_quality(problem, problem_name):
	# Print information about the solution quality and problem statistics.
	print(separator)
	print('%s quality and stats:' % problem_name)
	problem.printQuality()
	problem.printStats()
	print(separator)


# Has the algorithm converged?
converged = False


# The main loop of the algorithm starts here.
for iteration in range(MAX_ITERATIONS):
	print_iteration_counter(iteration)

	# Augment the master problem for the current iteration. Skip the first iteration.
	if iteration > 0:
		augment_master_problem(iteration, d)

	# Solve the master problem. The context manager measures solution time.
	with master_problem_timer as t:
		master_problem.optimize()

	print_solution_quality(master_problem, "Master problem")

	# Update lower bound to the master problem objective value.
	LB = master_problem.objVal

	# Obtain investment decisions from the master problem solution.
	x, y = get_investment_decisions()

	# Sanity check for the initial master problem solution (no investment).
	if iteration == 0:
		no_generation_investment = all([np.isclose(v, 0.) for v in x.values()])
		no_transmission_investment = all([np.isclose(v, 0.) for v in y.values()])

		assert no_generation_investment and no_transmission_investment, \
			'Initial master problem solution suboptimal.'
		prev_x = x
		prev_y = y

	# Update subproblem objective function with the newly updated investment decisions.
	set_subproblem_objective(x, y)

	# Solve the subproblem.
	with subproblem_timer as t:
		subproblem.optimize()

	print_solution_quality(subproblem, "Subproblem")

	# Update the algorithm upper bound and compute new a gap.
	UB = get_investment_cost(x, y) + subproblem.objVal

	GAP = compute_objective_gap(LB, UB)
	gaps.append(GAP)

	# Read the values of the uncertain variables
	uncertain_variables = get_uncertain_variables()

	# Fill the next column in the uncertain variables array.
	d = concatenate_to_uncertain_variables_array(d, uncertain_variables)

	# Exit if the algorithm converged. Conditions are: 1) LB and UB needs to be close to each other,
	# 2) solutions to master problem and subproblem must stay constant for one iteration.
	prev_x = x
	prev_y = y

	unchanged_solution = is_solution_unchanged(prev_x, prev_y, x, y, d)

	if GAP < EPSILON and unchanged_solution:
		converged = True
		assert GAP >= BAD_GAP_THRESHOLD, 'Upper bound %f, lower bound %f.' % (UB, LB)
		break


# Report solution.
print_primal_variables = False

if print_primal_variables:
	print('Primal variables:')
	print(separator)
	for v in master_problem.getVars():
		print(v.varName, v.x)

	print(separator)
	print('Uncertain variables:')
	print(separator)
	v = get_uncertain_variables()
	for k in sorted(v.keys()):
		print('d', k, v[k])


print_dual_variables = False

if print_dual_variables:
	print(separator)
	print('Dual variables:')
	print(separator)
	for v in subproblem.getVars():
		print(v.varName, v.x)


# Report if the algorithm converged.
print(separator)

if converged:
	print('Converged at iteration %d! Gap: %s' % (iteration, GAP))
else:
	print('Did not converge. Gap: ', GAP)

print(separator)
print('Objective value:', master_problem.objVal)
print('Investment cost %s, operation cost %s ' % (get_investment_cost(x, y), subproblem.objVal))
print(separator)

# Report solution times.
print(separator)
print('Master problem solution times (seconds):', master_problem_timer.solution_times)
print('Subproblem solution times (seconds):', subproblem_timer.solution_times)


# Plot gaps when solving the problem.
plot_gap = False

if plot_gap:
	from matplotlib import pyplot as plt

	plt.plot(gaps)
	plt.ylim(np.amin(gaps), 1e-2)
	plt.ylabel('GAP')
	plt.xlabel('iteration')
	plt.show()
