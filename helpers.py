# Helper functions.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from common_data import years, num_hours_per_year, candidate_units, candidate_lines


def to_year(h):
    # Map hour to the year it belongs to. Assume each year has the same amount of hours.
    return int(h / num_hours_per_year)


def unit_built(x, h, u):
    # Check if a generation unit is built at hour h.
    year = to_year(h)

    return x[year, u] if u in candidate_units else 1


def line_built(y, h, l):
    # Check if a transmission line is built at hour.
    year = to_year(h)

    return y[year, l] if l in candidate_lines else 1


def node_of_unit(u):
    # Return the node where the unit is located at.
    # TODO: Allow multiple units in one node.
    return u


def get_ramp_hours():
    # Return the hours for which ramp constraints are defined. Exclude the last hour in each year
    # because the constraint is defined as lb <= g_{t+1} - g_t <= ub. By excluding the last hour in
    # each year we also avoid linking subsequent years.
    ramp_hours_in_year = range(num_hours_per_year - 1)

    ramp_hours = []

    for year in years:
        offset = year * num_hours_per_year

        for i in ramp_hours_in_year:
            hour = offset + i
            ramp_hours.append(hour)

    return ramp_hours


def is_year_fist_hour(h):
    # Check if the given hour is the first hour of the year.
    remainder = h % num_hours_per_year

    return remainder == 0


def is_year_last_hour(h):
    # Check if the given hour is the last hour of the year.
    remainder = (h + 1) % num_hours_per_year

    return remainder == 0


def concatenate_to_uncertain_variables_array(current_d, new_d):
    # Add a new column to the uncertain variables array.
    new_column = np.zeros(current_d.shape[:-1] + (1, ))

    for k, v in new_d.items():
        new_column[k + (0, )] = v

    current_d = np.concatenate((current_d, new_column), axis=-1)

    return current_d


def is_solution_unchanged(prev_x, prev_y, x, y, d):
    # Check if master problem and subproblem solutions remained unchanged for one iteration.
    same_x = np.allclose(prev_x.values(), x.values())
    same_y = np.allclose(prev_y.values(), y.values())
    same_d = np.allclose(d[..., -2], d[..., -1])

    return same_x and same_y and same_d


class Timer:
    """Context manager for timing a function."""

    def __init__(self):
        self._collection = list()

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self._elapsed = self._end - self._start
        self._collection.append(round(self._elapsed, 2))

    @property
    def solution_times(self):
        return self._collection

