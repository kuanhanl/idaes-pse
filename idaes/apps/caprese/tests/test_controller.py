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
""" Tests for the controller model subclass of block
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.tests.test_simple_model import (
        make_model,
        make_small_model,
        initialize_t0,
        copy_values_forward,
        )
from idaes.apps.caprese.controller import (
        _ControllerBlockData,
        ControllerBlock,
        SimpleControllerBlock,
        IndexedControllerBlock,
        )
from idaes.apps.caprese.dynamic_var import (
        DiffVar,
        AlgVar,
        InputVar,
        FixedVar,
        DerivVar,
        )
from idaes.apps.caprese.common.config import (
        ControlPenaltyType,
        )
from idaes.core.util.model_statistics import degrees_of_freedom

solver_available = pyo.SolverFactory('ipopt').available()
if solver_available:
    solver = pyo.SolverFactory('ipopt')
else:
    solver = None

class TestControllerBlock(object):

    @pytest.mark.unit
    def test_construct(self):
        model = make_small_model()
        time = model.time
        t0 = time.first()
        inputs = [model.flow_in[t0]]
        controller = ControllerBlock(
                model=model,
                time=time,
                inputs=inputs,
                )
        assert type(controller) is SimpleControllerBlock
        assert isinstance(controller, ControllerBlock)
        assert isinstance(controller, _ControllerBlockData)

        controller.construct()
        assert controller[None] is controller
        assert controller.mod is model
        assert controller.time is time
        assert all(i1 is i2 for i1, i2 in zip(controller._inputs, inputs))

    @pytest.mark.unit
    def test_construct_indexed(self):
        controller_set = pyo.Set(initialize=[0,1,2])
        controller_set.construct()
        horizon_map = {0: 1., 1: 3., 2: 5.}
        nfe_map = {0: 2, 1: 6, 2: 10}
        model_map = {i: make_model(horizon_map[i], nfe_map[i])
                for i in controller_set}
        time_map = {i: model_map[i].time for i in controller_set}
        inputs_map = {i: [model_map[i].flow_in[0]] for i in controller_set}

        controller = ControllerBlock(
                controller_set,
                model=model_map,
                time=time_map,
                inputs=inputs_map,
                )

        assert type(controller) is IndexedControllerBlock
        assert isinstance(controller, IndexedControllerBlock)
        controller.construct()
        assert all(b.parent_component() is controller for b in controller.values())

        for i in controller_set:
            assert i in controller
        for i, c in controller.items():
            assert c.mod is model_map[i]
            assert c.time is time_map[i]
            assert all(i1 is i2 for i1, i2 in zip(c._inputs, inputs_map[i]))

    @pytest.mark.unit
    def make_controller(self, sample_time=0.5, horizon=1., nfe=2):
        model = make_model(horizon=horizon, nfe=nfe)
        time = model.time
        t0 = time.first()
        inputs = [model.flow_in[t0]]
        controller = ControllerBlock(
                model=model,
                time=time,
                inputs=inputs,
                )
        controller.construct()
        controller.set_sample_time(sample_time)
        controller.estimator_is_existing = False
        return controller

    @pytest.mark.unit
    def test_add_setpoint_objective(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()
        setpoint = [
                (controller.mod.flow_in[t0], 3.0),
                ]
        weights = [
                (controller.mod.flow_in[t0], 2.0),
                ]
        controller.mod.flow_in[:].set_value(3.0)
        initialize_t0(controller.mod)

        controller.add_setpoint_objective(setpoint, weights)

        assert hasattr(controller, 'setpoint_objective')

        pred_obj_expr = (2.0*(controller.mod.flow_in[t0] - 3.0)**2)
        obj_expr = controller.setpoint_objective.expr
        assert pyo.value(pred_obj_expr) == pyo.value(obj_expr)
        assert pred_obj_expr.to_string() == obj_expr.to_string()

        # Because now can change the setpoint, test should pass without this line.
        # controller.del_component(controller.setpoint_objective)

        setpoint = [
                (controller.mod.flow_in[t0], 3.0),
                (controller.mod.conc[t0,'A'], 1.5),
                ]
        weights = [
                (controller.mod.flow_in[t0], 1.0),
                (controller.mod.conc[t0,'A'], 5.0),
                ]
        controller.add_setpoint_objective(setpoint, weights)
        pred_obj_expr = (
                1.0*(controller.mod.flow_in[t0] - 3.0)**2 + 
                5.0*(controller.mod.conc[t0,'A'] - 1.5)**2
                )
        obj_expr = controller.setpoint_objective.expr
        assert pyo.value(pred_obj_expr) == pyo.value(obj_expr)
        assert pred_obj_expr.to_string() == obj_expr.to_string()

    @pytest.mark.component
    @pytest.mark.skipif(not solver_available, reason='IPOPT is not available')
    def test_solve_setpoint_steady(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()
        setpoint = [
                (controller.mod.flow_in[t0], 3.0),
                ]
        weights = [
                (controller.mod.flow_in[t0], 1.0),
                ]
        controller.add_setpoint_objective(setpoint, weights)
        controller.mod.flow_in[:].set_value(3.0)
        initialize_t0(controller.mod)
        
        # It should work even if the tracking objective exists and is active.
        controller.tracking_objective = pyo.Objective(expr = 0.0)

        dof_prior = degrees_of_freedom(controller)
        controller.solve_setpoint(solver, require_steady=True)
        dof_post = degrees_of_freedom(controller)

        assert dof_prior == dof_post
        assert controller.tracking_objective.active

        assert controller.differential_vars[0].setpoint == \
                pytest.approx(3.75, abs=1e-3)
        assert controller.differential_vars[1].setpoint == \
                pytest.approx(1.25, abs=1e-3)
        assert controller.algebraic_vars[0].setpoint == \
                pytest.approx(3.0, abs=1e-3)
        assert controller.algebraic_vars[1].setpoint == \
                pytest.approx(-3.75, abs=1e-3)
        assert controller.algebraic_vars[2].setpoint == \
                pytest.approx(3.75, abs=1e-3)
        assert controller.input_vars[0].setpoint == \
                pytest.approx(3.0, abs=1e-3)

    @pytest.mark.component
    @pytest.mark.skipif(not solver_available, reason='IPOPT is not available')
    def test_solve_setpoint_unsteady(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()
        setpoint = [
                (controller.mod.flow_in[t0], 3.0),
                (controller.mod.conc[t0,'A'], 5.0),
                ]
        weights = [
                (controller.mod.flow_in[t0], 1.0),
                (controller.mod.conc[t0,'A'], 1.0),
                ]
        controller.add_setpoint_objective(setpoint, weights)
        controller.mod.flow_in[:].set_value(3.0)
        initialize_t0(controller.mod)

        dof_prior = degrees_of_freedom(controller)
        controller.solve_setpoint(solver, require_steady=False)
        dof_post = degrees_of_freedom(controller)

        assert dof_prior == dof_post

        assert controller.differential_vars[0].setpoint == \
                pytest.approx(5.0, abs=1e-3)
        assert controller.differential_vars[1].setpoint == \
                pytest.approx(7.11, abs=1e-3)
        assert controller.algebraic_vars[0].setpoint == \
                pytest.approx(3.0, abs=1e-3)
        assert controller.algebraic_vars[1].setpoint == \
                pytest.approx(-5.0, abs=1e-3)
        assert controller.algebraic_vars[2].setpoint == \
                pytest.approx(5.0, abs=1e-3)
        assert controller.input_vars[0].setpoint == \
                pytest.approx(3.0, abs=1e-3)

    @pytest.mark.component
    @pytest.mark.skipif(not solver_available, reason='IPOPT is not available')
    def test_add_tracking_objective(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()
        setpoint = [
                (controller.mod.flow_in[t0], 3.0),
                ]
        weights = [
                (controller.mod.flow_in[t0], 1.0),
                ]
        controller.mod.flow_in[:].set_value(3.0)
        initialize_t0(controller.mod)
        copy_values_forward(controller.mod)
        controller.add_setpoint_objective(setpoint, weights)
        controller.solve_setpoint(solver)

        # Re-initialize inputs so they are not at the setpoint
        # for objective evaluation.
        controller.mod.flow_in[:].set_value(2.5)
        
        weights = [
                (controller.mod.conc[t0,'A'], 1),
                (controller.mod.conc[t0,'B'], 1),
                (controller.mod.rate[t0,'A'], 1),
                (controller.mod.rate[t0,'B'], 1),
                (controller.mod.flow_out[t0], 1),
                (controller.mod.flow_in[t0], 1),
                ]

        # Construct predicted objective functions:
        pred_obj = {i: 0. for i in range(1, 4)}
        sample_points = controller.sample_points[1:]
        sample_time = controller.sample_time
        for v in controller.component_objects(DiffVar):
            for t in sample_points:
                pred_obj[1] += (v[t] - v.setpoint)**2
                pred_obj[2] += (v[t] - v.setpoint)**2
                pred_obj[3] += (v[t] - v.setpoint)**2
        for v in controller.component_objects(AlgVar):
            for t in sample_points:
                pred_obj[3] += (v[t] - v.setpoint)**2
        for v in controller.component_objects(InputVar):
            for t in sample_points:
                i_prev = time.find_nearest_index(t-sample_time, tolerance=1e-8)
                t_prev = time[i_prev]
                pred_obj[1] += (v[t] - v.setpoint)**2
                pred_obj[2] += (v[t] - v[t_prev])**2
                pred_obj[3] += (v[t] - v[t_prev])**2

        controller.add_tracking_objective(
                weights,
                control_penalty_type=ControlPenaltyType.ERROR,
                state_ctypes=DiffVar,
                )
        assert pyo.value(controller.tracking_objective.expr == pred_obj[1])
        assert pyo.value(controller.tracking_objective.expr) > 0
        controller.del_component(controller.tracking_objective)

        controller.add_tracking_objective(
                weights,
                control_penalty_type=ControlPenaltyType.ACTION,
                state_ctypes=DiffVar,
                )
        assert pyo.value(controller.tracking_objective.expr == pred_obj[2])
        assert pyo.value(controller.tracking_objective.expr) > 0
        controller.del_component(controller.tracking_objective)

        controller.add_tracking_objective(
                weights,
                control_penalty_type=ControlPenaltyType.ACTION,
                state_ctypes=(DiffVar, AlgVar),
                )
        assert pyo.value(controller.tracking_objective.expr == pred_obj[3])
        assert pyo.value(controller.tracking_objective.expr) > 0
        
        # This shold work even if the controller block has alreay had the 
        # tracking objective.
        controller.add_setpoint_objective(setpoint, weights)

    @pytest.mark.unit
    def test_constrain_control_inputs_piecewise_constant(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()

        controller.constrain_control_inputs_piecewise_constant()
        assert hasattr(controller, 'pwc_constraint')
        sample_points = controller.sample_points
        sample_point_set = set(sample_points)

        for i, t in controller.INPUT_SET*controller.time:
            if t in sample_point_set:
                assert (i, t) not in controller.pwc_constraint
            else:
                assert (i, t) in controller.pwc_constraint
        for i, t in controller.pwc_constraint:
            pwc_expr = controller.pwc_constraint[i,t].expr
            tn = time.next(t)
            inputs = controller.vectors.input
            pred_expr = (inputs[i, tn] - inputs[i, t] == 0.)
            assert pwc_expr.to_string() == pred_expr.to_string()
            
    @pytest.mark.unit
    def test_load_estimates(self):
        controller = self.make_controller()
        time = controller.time
        t0 = time.first()
        
        estimates = [100., 200.]
        controller.load_estimates(estimates)
        
        assert controller.vectors.differential[0, t0].value == 100.
        assert controller.vectors.differential[1, t0].value == 200.
        
    @pytest.mark.unit
    def test_load_initial_conditions(self):
        model = make_model(horizon=1., nfe=2)
        sample_time = 0.5
        time = model.time
        t0 = time.first()
        inputs = [model.flow_in[t0]]
        measurements = [model.flow_out[t0], model.conc[t0, "A"]]
        controller = ControllerBlock(
                model=model,
                time=time,
                inputs=inputs,
                measurements=measurements,
                )
        controller.construct()
        controller.set_sample_time(sample_time)

        #case 1: estimator doesn't exist
        controller.estimator_is_existing = False
        ics = [10., 20.]
        controller.load_initial_conditions(ics)

        assert controller.vectors.measurement[0, t0].value == 10.
        assert controller.vectors.measurement[1, t0].value == 20.
        
        #case 2: estimator does exist
        controller.estimator_is_existing = True
        ics = [11., 21.]
        controller.load_initial_conditions(ics)

        assert controller.vectors.differential[0, t0].value == 11.
        assert controller.vectors.differential[1, t0].value == 21.
        
        #case 3: type of ics is declared => "measurement"
        ics = [12., 22.]
        controller.load_initial_conditions(ics, "measurement")

        assert controller.vectors.measurement[0, t0].value == 12.
        assert controller.vectors.measurement[1, t0].value == 22.
        
        #case 3: type of ics is declared => "estimates"
        ics = [13., 23.]
        controller.load_initial_conditions(ics, "estimate")

        assert controller.vectors.differential[0, t0].value == 13.
        assert controller.vectors.differential[1, t0].value == 23.
        
abc = TestControllerBlock()
abc.test_solve_setpoint_steady()