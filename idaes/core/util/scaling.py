##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
This module contains utilities to provide variable and expression scaling
factors by providing an expression to calculate them via a suffix.

The main purpose of this code is to use the calculate_scaling_factors function
to calculate scaling factors to be used with the Pyomo scaling transformation or
with solvers. A user can provide a scaling_expression suffix to calculate scale
factors from existing variable scaling factors. This allows scaling factors from
a small set of fundamental variables to be propagated to the rest of the model.

The scaling_expression suffix contains Pyomo expressions with model variables.
The expressions can be evaluated with variable scaling factors in place of
variables to calculate additional scaling factors.
"""

__author__ = "John Eslick, Tim Bartholomew"

import pyomo.environ as pyo
from pyomo.core.base.constraint import _ConstraintData
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.common.modeling import unique_component_name
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import number_activated_objectives
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


def __none_mult(x, y):
    """PRIVATE FUNCTION, Multiply x by y and if x is None return None"""
    if x is not None:
        x *= y
    return x


def set_scaling_factor(c, v):
    """Set a scaling factor for a model component. This function creates the
    scaling_factor suffix if needed.

    Args:
        c: component to supply scaling factor for
        v: scaling factor
    Returns:
        None
    """
    try:
        c.parent_block().scaling_factor[c] = v
    except AttributeError:
        c.parent_block().scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        c.parent_block().scaling_factor[c] = v


def get_scaling_factor(c, default=None):
    """Get a component scale factor.

    Args:
        c: compoent
        default: value to return if no scale factor exists (default=None)
    """
    try:
        sf = c.parent_block().scaling_factor.get(c, default)
    except AttributeError:
        sf = default
    return sf


def unset_scaling_factor(c):
    """Delete a component scaling factor.

    Args:
        c: component

    Returns:
        None
    """
    try:
        del c.parent_block().scaling_factor[c]
    except AttributeError:
        pass # no scaling factor suffix, is fine
    except KeyError:
        pass # no scaling factor is fine


def unscaled_variables_generator(blk, descend_into=True, include_fixed=False):
    """Generator for unscaled variables

    Args:
        block

    Yields:
        variables with no scale factor
    """

    for v in blk.component_data_objects(pyo.Var, descend_into=descend_into):
        if v.fixed and not include_fixed:
            continue
        if get_scaling_factor(v) is None:
            yield v


def unscaled_constraints_generator(blk, descend_into=True):
    """Generator for unscaled constraints

    Args:
        block

    Yields:
        constraints with no scale factor
    """
    for c in blk.component_data_objects(
        pyo.Constraint, active=True, descend_into=descend_into):
        if get_scaling_factor(c) is None:
            yield c


def badly_scaled_var_generator(
    blk, large=1e4, small=1e-3, zero=1e-10, descend_into=True, include_fixed=False):
    """This provides a rough check for variables with poor scaling based on
    their current scale factors and values. For each potentially poorly scaled
    variable it returns the var and its current scaled value.

    Args:
        blk: pyomo block
        large: Magnitude that is considered to be too large
        small: Magnitude that is considered to be too small
        zero: Magnitude that is considered to be zero, variables with a value of
            zero are okay, and not reported.

    Yields:
        variable data object, current absolute value of scaled value
    """
    for v in blk.component_data_objects(pyo.Var, descend_into=descend_into):
        if v.fixed and not include_fixed:
            continue
        sf = get_scaling_factor(v, default=1)
        sv = abs(pyo.value(v) * sf)  # scaled value
        if sv > large:
            yield v, sv
        elif sv < zero:
            continue
        elif sv < small:
            yield v, sv


def scale_single_constraint(c):
    """This transforms a constraint with its scaling factor. If there is no
    scaling factor for the constraint, the constraint is not scaled and a
    message is logged. After transforming the constraint the scaling factor,
    scaling expression, and nomical value are all unset to ensure the constraint
    isn't scaled twice.

    Args:
        c: Pyomo constraint

    Returns:
        None
    """
    if not isinstance(c, _ConstraintData):
        raise TypeError(
            "{} is not a constraint and cannot be the input to "
            "scale_single_constraint".format(c.name))

    v = get_scaling_factor(c)
    if v is None:
        _log.warning(
            f"{c.name} constraint has no scaling factor, so it was not scaled.")
        return
    c.set_value(
        (__none_mult(c.lower, v), __none_mult(c.body, v), __none_mult(c.upper, v)))
    unset_scaling_factor(c)


def __scale_block_constraints(b):
    """PRIVATE FUNCTION
    Scales all of the constraints in a block. Does not descend into other
    blocks.

    Args:
        b: Pyomo block

    Returns:
        None
    """
    for c in b.component_data_objects(pyo.Constraint, descend_into=False):
        scale_single_constraint(c)


def scale_constraints(blk, descend_into=True):
    """This scales all constraints with their scaling factor suffix for a model
    or block. After scaling the constraints, the scaling factor and expression
    for each constraint is set to 1 to avoid double scaling the constraints.

    Args:
        blk: Pyomo block
        descend_into: indicates whether to descend into the other blocks on blk.
            (default = True)

    Returns:
        None
    """
    __scale_block_constraints(blk)
    if descend_into:
        for b in blk.component_data_objects(pyo.Block, descend_into=True):
            __scale_block_constraints(b)


def constraint_autoscale_large_jac(
    m,
    ignore_constraint_scaling=False,
    ignore_variable_scaling=False,
    max_grad=100,
    min_scale=1e-6,
    no_scale = False
):
    """Automatically scale constraints based on the Jacobian.  This function
    immitates Ipopt's default constraint scaling.  This scales constraints down
    to avoid extremely large values in the Jacobian

    Args:
        m: model to scale
        ignore_constraint_scaling: ignore existing constraint scaling
        ignore_variable_scaling: ignore existing variable scaling
        max_grad: maximum value in Jacobian after scaling, subject to minimum
            scaling factor restriction.
        min_scale: minimum scaling factor allowed, keeps constraints from being
            scaled too much.
        no_scale: just calculate the Jacobian and scaled Jacobian, don't scale
            anything
    """
    # Pynumero requires an objective, but I don't, so let's see if we have one
    n_obj = number_activated_objectives(m)
    # Add an objective if there isn't one
    if n_obj == 0:
        dummy_objective_name = unique_component_name(m, "objective")
        setattr(m, dummy_objective_name, pyo.Objective(expr=0))
    # Create NLP and calculate the objective
    nlp = PyomoNLP(m)
    jac = nlp.evaluate_jacobian().tocsr()
    # Get lists of varibles and constraints to translate Jacobian indexes
    clist = nlp.get_pyomo_constraints()
    vlist = nlp.get_pyomo_variables()
    # Create a scaled Jacobian to account for variable scaling, for now ignore
    # constraint scaling
    jac_scaled = jac.copy()
    for i in range(len(clist)):
        for j in jac_scaled[i].indices:
            v = vlist[j]
            if ignore_variable_scaling:
                sv = 1
            else:
                sv = get_scaling_factor(v, default=1)
            jac_scaled[i,j] = jac_scaled[i,j]/sv
    # calculate constraint scale factors
    for i in range(len(clist)):
        c = clist[i]
        sc = get_scaling_factor(c, default=1)
        if not no_scale:
            if (ignore_constraint_scaling or get_scaling_factor(c) is None):
                row = jac_scaled[i]
                for d in row.indices:
                    row[0,d] = abs(row[0,d])
                mg = row.max()
                if mg > max_grad:
                    sc = max(min_scale, max_grad/mg)
                set_scaling_factor(c, sc)
        for j in jac_scaled[i].indices:
            # update the scaled jacobian
            jac_scaled[i,j] = jac_scaled[i,j]*sc
    # delete dummy objective
    if n_obj == 0:
        delattr(m, dummy_objective_name)
    return jac, jac_scaled, nlp
