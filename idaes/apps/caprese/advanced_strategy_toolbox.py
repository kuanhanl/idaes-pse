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
""" Advanced step strategy with sensitivity calculation for MHE/NMPC
"""

from pyomo.contrib.sensitivity_toolbox.sens import SensitivityInterface as SensInt
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface#, InTempDir
from pyomo.opt import SolverFactory, SolverStatus


def advanced_strategy_setup(blockitem, 
                            paramlist,
                            method = None,
                            k_aug_solver = None,
                            dot_sens_solver = None,
                            ipopt_sens_solver = None,
                            ):
    
    blockitem.sens = SensInt(blockitem, clone_model = False)
    blockitem.sens.setup_sensitivity(paramlist)
    
    if method not in {"k_aug", "sipopt"}:
        raise ValueError("Only methods 'k_aug' and 'sipopt' are supported'")
    else:
        blockitem.sens_method = method
    
    if blockitem.sens_method == "k_aug":
        if k_aug_solver is None:
            blockitem.k_aug = SolverFactory('k_aug', solver_io='nl')
            blockitem.k_aug.options['dsdp_mode'] = ""  #: sensitivity mode!
        else:
            blockitem.k_aug = k_aug_solver
            if "dsdp_mode" not in blockitem.k_aug.options:
                blockitem.k_aug.options['dsdp_mode'] = ""  #: sensitivity mode!
        
        if dot_sens_solver is None:
            blockitem.dot_sens = SolverFactory('dot_sens', solver_io='nl')
            blockitem.dot_sens.options["dsdp_mode"] = ""
        else:
            blockitem.dot_sens = dot_sens_solver
            if "dsdp_mode" not in blockitem.dot_sens.options:
                blockitem.dotsens.options['dsdp_mode'] = ""
                
        blockitem.k_aug_interface = K_augInterface(k_aug=blockitem.k_aug, dot_sens=blockitem.dot_sens)                
                
    elif blockitem.sens_method == "sipopt":
        if ipopt_sens_solver is None:
            blockitem.ipopt_sens = SolverFactory('ipopt_sens', solver_io='nl')              
            blockitem.ipopt_sens.options['run_sens'] = 'yes'
        else:
            blockitem.ipopt_sens = ipopt_sens_solver
            if "run_sens" not in blockitem.ipopt_sens.options:
                blockitem.ipopt_sens.options['run_sens'] = 'yes'
           
    
def calculate_sensitivity(blockitem, tee = True):
    if blockitem.sens_method == "k_aug":
        blockitem.ipopt_zL_in.update(blockitem.ipopt_zL_out)  #: important!
        blockitem.ipopt_zU_in.update(blockitem.ipopt_zU_out)  #: important! 
        
        blockitem.k_aug_interface.k_aug(blockitem, tee=tee)
        
        
def sensitivity_update(blockitem, 
                       perturblist,
                       tee = True):
    
    if blockitem.sens_method == "k_aug":
        calculate_sensitivity(blockitem, tee)
    
    blockitem.sens.perturb_parameters(perturblist)
    
    if blockitem.sens_method == "k_aug":
        blockitem.k_aug_interface.dot_sens(blockitem, tee = tee)
        
    elif blockitem.sens_method == "sipopt":
        blockitem.ipopt_sens.solve(blockitem, keepfiles = False, tee = tee)