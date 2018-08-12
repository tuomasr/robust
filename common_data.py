import numpy as np

np.random.seed(13)

# Data taken from http://orbit.dtu.dk/files/120568114/An_Updated_Version_of_the_IEEE_RTS_24Bus_System_for_Electricty_Market_an....pdf

# Scenarios.
num_scenarios = 1
scenarios = range(num_scenarios)

# Nodes.
num_nodes = 24
nodes = range(num_nodes)

# Load.
load_by_hour = [1775.835, 1669.815, 1590.3, 1563.795, 1563.795, 1590.3, 1961.37, 2279.43,
                2517.975, 2544.48, 2544.48, 2517.975, 2517.975, 2517.975, 2464.965, 2464.965,
                2623.995, 2650.5, 2650.5, 2544.48, 2411.955, 2199.915, 1934.865, 1669.815]

mean_load = np.mean(load_by_hour)

load_share_by_node = [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6., 6.1, 6.8, 0., 0., 9.3,
                      6.8, 11.1, 3.5, 0., 11.7, 6.4, 4.5, 0., 0., 0., 0.0]

assert len(load_by_hour) == 24
assert len(load_share_by_node) == num_nodes
assert sum(load_share_by_node) == 100.

load = mean_load * np.array(load_share_by_node) / 100.  # Convert shares to percentages.

# Units.
existing_units = range(12)

# Generator to node mapping.
existing_unit_to_node = {0: 0, 1: 1, 2: 6, 3: 12, 4: 14, 5: 14, 6: 15,
                         7: 17, 8: 20, 9: 21, 10: 22, 11: 22}


# Generation costs.
C_g = np.array([13.32, 13.32, 20.7, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0., 10.52, 10.89],
                dtype=np.float32)

# Generation limits.
G_max = np.array([[152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350]],
                 dtype=np.float32)

# Generate a candidate unit for each node.
candidate_units = []
generate_candidate_units = True

if generate_candidate_units:
    candidate_unit_to_node = dict()
    unit_idx = len(existing_units)
    
    for node_idx, node in enumerate(nodes):
        candidate_units.append(unit_idx)
        candidate_unit_to_node[unit_idx] = node_idx
        C_g = np.append(C_g, [5.]) 
        G_max = np.concatenate((G_max, [[200.]]), axis=1)
        unit_idx += 1

unit_to_node = existing_unit_to_node.copy()
unit_to_node.update(candidate_unit_to_node)

units = existing_units + candidate_units

node_to_unit = {node: [] for node in nodes}

for unit, node in unit_to_node.iteritems():
    node_to_unit[node].append(unit)

# Total number of units.
num_units = len(units)

G_max = np.tile(G_max.transpose(), (1, num_scenarios))
G_max += np.random.uniform(0.0, 0.0, (num_units, num_scenarios))

assert len(G_max) == len(C_g) == num_units

# Build lines x nodes incidence matrix for existing lines.
# List pairs of nodes that are connected. Note: these are in 1-based indexing.
lines = [(1, 2), (1, 3), (1, 5), (2, 4), (2, 6), (3, 9), (3, 24),
         (4, 9), (5, 10), (6, 10), (7, 8), (8, 9), (8, 10), (9, 11),
         (9, 12), (10, 11), (10, 12), (11, 13), (11, 14), (12, 13),
         (12, 23), (13, 23), (14, 16), (15, 16), (15, 21), (15, 24),
         (16, 17), (16, 19), (17, 18), (17, 22), (18, 21), (19, 20),
         (20, 23), (21, 22)]

num_existing_lines = len(lines)
existing_lines = range(len(lines))

incidence = np.zeros((num_existing_lines, num_nodes))

for line_idx, line in enumerate(lines):
    start, end = line

    # Correct for the 1-based indexing in node numbering.
    incidence[line_idx, start - 1] = -1
    incidence[line_idx, end - 1] = 1

F_max = np.array([[175, 175, 350, 175, 175, 175, 400, 175, 350, 175, 350, 175, 175, 400, 400, 400,
                   400, 500, 500, 500, 500, 500, 500, 500, 1000, 500, 500, 500, 500, 500,
                   1000, 1000, 1000, 500]], dtype=np.float32)

assert num_existing_lines == len(incidence) == len(F_max[0])

line_idx = len(existing_lines)

candidate_lines = []

# Generate candidate lines between each pair of nodes.
generate_candidate_lines = True

if generate_candidate_lines:
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            row = np.zeros((1, num_nodes))
            row[0, i] = -1.
            row[0, j] = 1.

            incidence = np.concatenate((incidence, row), axis=0)
            F_max = np.concatenate((F_max, [[200.]]), axis=1)

            candidate_lines.append(line_idx)
            line_idx += 1

lines = existing_lines + candidate_lines
num_lines = len(lines)

F_max = np.tile(F_max.transpose(), (1, num_scenarios))
F_max += np.random.uniform(0.0, 0.0, (num_lines, num_scenarios))

F_min = -F_max  # Same transmission capacity to both directions.

assert len(incidence) == len(F_max) == num_lines

# Smoothen the parameter values.
F_max = np.round(F_max, 3)
F_min = np.round(F_min, 3)
G_max = np.round(G_max, 3)

# equal scenario weights
weights = np.ones(num_scenarios)
