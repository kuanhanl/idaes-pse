# -*- coding: utf-8 -*-
#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pyomo.environ as pyo

from pyomo.core.base.componentuid import ComponentUID
from pyomo.common.collections import ComponentMap
import pandas as pd
from collections import OrderedDict
import copy

__author__ = "Robert Parker and Kuan-Han Lin"


def _generate_cuids(components):
    for comp in components:
        if comp.is_reference():
            yield ComponentUID(comp.referent)
        else:
            yield ComponentUID(comp)


def _filter_duplicates(items):
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item


def find_and_merge_variables(block, varlist1, varlist2):
    '''
    This function converts two lists of variable to CUIDs, combines them,
    filtering duplicates, and finds the corresponding variables on the
    block. This functionality is repeated when we want to initialize
    a data manager that stores some default states (differential variables)
    and any states the user specified.

    Parameters
    ----------
    block : Pyomo Block
        The block where the variables are searched.
    varlist1 : list
        First list of variables to merge
    varlist2 : list
        Second list of variables to merge

    Returns
    -------
    merged_list : list

    '''
    cuids1 = list(_generate_cuids(varlist1))
    cuids2 = list(_generate_cuids(varlist2))
    merged_cuids = list(_filter_duplicates(cuids1 + cuids2))
    merged_list = [
        cuid.find_component_on(block) for cuid in merged_cuids
    ]
    return merged_list


def empty_dataframe_from_variables(variables, rename_map=None):
    '''
    This function creates a pandas dataframe with variables in columns.

    Parameters
    ----------
    variables : list of variables of interest
    rename_map : dictionary or componentmap that maps the variable to a
                    specific name string.

    Returns
    -------
    dataframe : pandas dataframe

    '''
    if rename_map is None:
        # Sometimes we construct a variable with a programatically
        # generated name that is not that meaningful to the user, e.g.
        # ACTUALMEASUREMENTBLOCK[1].var. This argument lets us rename
        # to something more meaningful, such as temperature_actual.
        rename_map = ComponentMap()
    var_dict = OrderedDict()
    var_dict["iteration"] = []
    for var in variables:
        if var in rename_map:
            key = rename_map[var]
        else:
            if var.is_reference():
                # NOTE: This will fail if var is a reference-to-list
                # or reference-to-dict.
                key = str(ComponentUID(var.referent))
            else:
                key = str(ComponentUID(var))
        var_dict[key] = []

    dataframe = pd.DataFrame(var_dict)
    # Change the type of iteration column from float to integer
    dataframe["iteration"] = dataframe["iteration"].astype(int)
    return dataframe


def add_variable_values_to_dataframe(
        dataframe,
        variables,
        iteration,
        time_subset=None,
        rename_map=None,
        time_map=None,
        ):
    '''
    This function appends data to the dataframe.

    Parameters
    ----------
    dataframe : pandas dataframe that we want to append the data to.
    variables : list of variables of interest
    iteration : current iteration.
    time_subset : time indices of interset in the variables.
    rename_map : dictionary or componentmap that maps the variable to a
                    specific name string.
    time_map : map the time in time_subset to real time points
    '''

    if rename_map is None:
        rename_map = ComponentMap()

    if len(dataframe.index) == 0:
        initial_time = 0.0
    else:
        initial_time = dataframe.index[-1]

    if (len(dataframe.index) != 0 and time_subset is not None
            and time_subset[0] == 0 and time_map is None):
        # The data frame has data in it that we would override with
        # the provided time subset.
        raise RuntimeError()

    df_map = OrderedDict()
    hash_set = set()
    for var in variables:
        idx_set = var.index_set()
        set_hash = hash(tuple(idx_set))
        if len(hash_set) == 0:
            hash_set.add(set_hash)

            if time_subset is None:
                time_subset = list(idx_set)
                if (len(dataframe.index) != 0 and time_subset[0] == 0
                    and time_map is None):
                    time_subset = time_subset[1:]

            df_map["iteration"] = len(time_subset)*[iteration]

        elif set_hash not in hash_set:
            raise ValueError()

        if var in rename_map:
            key = rename_map[var]
        else:
            if var.is_reference():
                # NOTE: This will fail if var is a reference-to-list
                # or reference-to-dict.
                key = str(ComponentUID(var.referent))
            else:
                key = str(ComponentUID(var))

        # append only the desired values
        df_map[key] = [var[t].value for t in time_subset] # Values to append for each variable

    if time_map:
        time_points = [time_map[t] for t in time_subset]
    else:
        time_points = [initial_time + t for t in time_subset]
        # If we aren't using initial_time, we should avoid doing floating
        # point arithmetic.
        #time_points = [t for t in time_subset]

    df = pd.DataFrame(df_map, index=time_points)

    return dataframe.append(df)

def add_variable_setpoints_to_dataframe(dataframe,
                                        variables,
                                        iteration,
                                        time_subset,
                                        map_for_user_given_vars = None):
    '''
    Save the setpoints for states of interest in nmpc's dataframe.

    TODO: replace this function with 'add_variable_values_to_dataframe' when we
    don't save setpoints in attributes'
    '''

    df_map = OrderedDict()
    df_map["iteration"] = len(time_subset)*[iteration]

    for var in variables:
        column_name = str(ComponentUID(var.referent))
        if var in map_for_user_given_vars:
            # User given variables are not nmpc_var, so they don't have setpoint attribute.
            # Use a componentmap to get corresponding nmpc_var.
            df_map[column_name] = len(time_subset)* \
                                    [map_for_user_given_vars[var].setpoint]
        else:
            df_map[column_name] = len(time_subset)*[var.setpoint]

    if len(dataframe.index) == 0:
        initial_time = 0.0
    else:
        initial_time = dataframe.index[-1]
    time_points = [initial_time + t for t in time_subset]
    df = pd.DataFrame(df_map, index=time_points)

    return dataframe.append(df)

class PlantDataManager(object):
    def __init__(self,
                 plantblock,
                 user_interested_states = None,):

        self.plantblock = plantblock

        if user_interested_states is None:
            user_interested_states = []
        self.plant_user_interested_states = user_interested_states
        self.plant_states_of_interest = find_and_merge_variables(
                                                    plantblock,
                                                    self.plantblock.differential_vars,
                                                    user_interested_states,
                                                    )

        self.plant_vars_of_interest = self.plant_states_of_interest + \
                                        self.plantblock.input_vars

        self.plant_df = empty_dataframe_from_variables(self.plant_vars_of_interest)

    def get_plant_dataframe(self):
        return self.plant_df

    def save_initial_plant_data(self):
        self.plant_df = add_variable_values_to_dataframe(self.plant_df,
                                                         self.plant_states_of_interest, #no inputs
                                                         iteration = 0,
                                                         time_subset = [self.plantblock.time.first()])

    def save_plant_data(self, iteration):
        #skip time.first()
        time_subset = self.plantblock.time.ordered_data()[1:]
        self.plant_df = add_variable_values_to_dataframe(self.plant_df,
                                                         self.plant_vars_of_interest,
                                                         iteration,
                                                         time_subset = time_subset,)


class ControllerDataManager(object):
    def __init__(self,
                 controllerblock,
                 user_interested_states = None,):

        self.controllerblock = controllerblock

        if user_interested_states is None:
            user_interested_states = []
        self.controller_states_of_interest = \
            find_and_merge_variables(
                controllerblock,
                self.controllerblock.differential_vars,
                user_interested_states,
            )
        self.extra_vars_user_interested = self.controller_states_of_interest

        self.controller_df = empty_dataframe_from_variables(self.controllerblock.input_vars)

        self.setpoint_df = empty_dataframe_from_variables(self.controller_states_of_interest)

        # HACK:
        # User given variables are not nmpc_var, so they don't have setpoint
        # attribute. Create this componentmap to map the given vars to
        # corresponding nmpc_vars.
        # TODO: We should stop using the setpoint attribute, and custom vars
        # altogether. Setpoint should instead be a dict mapping cuids to values.
        vardata_map = self.controllerblock.vardata_map
        t0 = self.controllerblock.time.first()
        self.user_given_vars_map_nmpcvar = ComponentMap((var, vardata_map[var[t0]])
                                                        for var in self.extra_vars_user_interested)

    def get_controller_dataframe(self):
        return self.controller_df

    def get_setpoint_dataframe(self):
        return self.setpoint_df

    def save_controller_data(self, iteration):
        #skip time.first()
        time = self.controllerblock.time
        #Save values in the first sample time
        time_subset = [t for t in time if t <= self.controllerblock.sample_points[1]
                                           and t != time.first()]
        self.controller_df = add_variable_values_to_dataframe(self.controller_df,
                                                              self.controllerblock.input_vars,
                                                              iteration,
                                                              time_subset = time_subset,)

        self.setpoint_df = add_variable_setpoints_to_dataframe(
            self.setpoint_df,
            self.controller_states_of_interest,
            iteration,
            time_subset,
            self.user_given_vars_map_nmpcvar,
        )

class EstimatorDataManager(object):
    def __init__(self,
                 estimatorblock,
                 user_interested_states = None,):

        self.estimatorblock = estimatorblock
        # Convert vars in plant to vars in estimator
        if user_interested_states is None:
            user_interested_states = []
        self.estimator_vars_of_interest = \
            find_and_merge_variables(
                estimatorblock,
                estimatorblock.differential_vars,
                user_interested_states,
            )

        self.estimator_df = empty_dataframe_from_variables(self.estimator_vars_of_interest)

    def get_estimator_dataframe(self):
        return self.estimator_df

    def save_estimator_data(self, iteration):
        time = self.estimatorblock.time
        t_last = time.last()
        # Estimation starts from the very first sample time.
        time_map = OrderedDict([(t_last, (iteration+1)*self.estimatorblock.sample_time),])
        self.estimator_df = add_variable_values_to_dataframe(self.estimator_df,
                                                             self.estimator_vars_of_interest,
                                                             iteration,
                                                             time_subset = [t_last],
                                                             time_map = time_map,)
