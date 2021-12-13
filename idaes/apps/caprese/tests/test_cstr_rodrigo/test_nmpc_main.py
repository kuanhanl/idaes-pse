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
""" Tests for the estimator model subclass of block
"""

import pytest
import pyomo.environ as pyo

from idaes.apps.caprese.examples.cstr_rodrigo.nmpc_main import (
            setup_controller
            )
from idaes.apps.caprese.dynamic_builder import (
            DynamicSim
            )
from idaes.apps.caprese.data_manager import(
            PlantDataManager,
            ControllerDataManager,
            )

# solver_available = pyo.SolverFactory('ipopt').available()
# if solver_available:
#     solver = pyo.SolverFactory('ipopt')
# else:
#     solver = None

class TestNMPCMain(object):

    @pytest.mark.unit
    def test_setup_controller(self,
                              nmpc_horizon=10,
                              nmpc_ntfe=5,
                              nmpc_ntcp=2,
                              plant_ntfe=2,
                              plant_ntcp=2,
                              sample_time=2.0):

        nmpc, plant_data, controller_data = setup_controller(nmpc_horizon=10,
                                                             nmpc_ntfe=5,
                                                             nmpc_ntcp=2,
                                                             plant_ntfe=2,
                                                             plant_ntcp=2,
                                                             sample_time=2.0)

        cmod = nmpc.controller.mod
        var_value_dict = {"Ca[*]": 0.0192,
                           "k[*]": 1818.53008,
                           "Tjinb[*]": 250.0,
                           "Cadot[*]": 0.0,
                           "Tall[*,T]": 384.0,
                           "Tall[*,Tj]": 371.0,
                           "Talldot[*,T]": 0.0,
                           "Talldot[*,Tj]": 0.0,
                           }
        for varstr_slice in var_value_dict.keys():
            cuid = pyo.ComponentUID(varstr_slice)
            for comp in cuid.find_component_on(cmod).values():
                assert comp.value == pytest.approx(
                    var_value_dict[varstr_slice],
                    1e-3,
                    )

        assert hasattr(nmpc.controller, 'tracking_objective')
        assert hasattr(nmpc.controller, 'pwc_constraint')

        assert type(nmpc) is DynamicSim
        assert type(plant_data) is PlantDataManager
        assert type(controller_data) is ControllerDataManager
