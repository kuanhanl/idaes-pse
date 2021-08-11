##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
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
Example for Caprese's module for MHE with advanced step strategy.
Sensitivity solver: k_aug.
"""
import random
from idaes.apps.caprese.dynamic_builder import DynamicSim
from idaes.apps.caprese.util import apply_noise_with_bounds
from pyomo.environ import SolverFactory, Reference
from pyomo.dae.initialization import solve_consistent_initial_conditions
# import idaes.logger as idaeslog
from idaes.apps.caprese.examples.cstr_rodrigo.cstr_rodrigo_model import make_model
from idaes.apps.caprese.data_manager import EstimatorDataManager

__author__ = "Kuan-Han Lin"


# See if ipopt is available and set up solver
if SolverFactory('ipopt').available():
    solver = SolverFactory('ipopt')
    solver.options = {
            'tol': 1e-6,
            'bound_push': 1e-8,
            'halt_on_ampl_error': 'yes',
            'linear_solver': 'ma57',
            }
else:
    solver = None
    
# Specify sensitivity solver, which can be either 'sipopt' or 'k_aug'
if SolverFactory("k_aug").available():
    k_aug = SolverFactory("k_aug", solver_io = "nl")
    k_aug.options['dsdp_mode'] = ""
else:
    k_aug = None
    
if SolverFactory("dot_sens").available():
    dot_sens = SolverFactory("dot_sens", solver_io = "nl")
    dot_sens.options["dsdp_mode"]=""
else:
    dot_sens = None


def main():
    m_estimator = make_model(horizon=10., ntfe=10, ntcp=2, bounds=True)
    sample_time = 2.
    m_plant = make_model(horizon=sample_time, ntfe=4, ntcp=2, bounds = True)
    m_predictor = make_model(horizon=sample_time, ntfe=4, ntcp=2, bounds=True)
    time_plant = m_plant.t

    simulation_horizon = 20
    n_samples_to_simulate = round(simulation_horizon/sample_time)

    samples_to_simulate = [time_plant.first() + i*sample_time
                           for i in range(1, n_samples_to_simulate)]

    # We must identify for the estimator which variables are our
    # inputs and measurements.
    inputs = [
            m_plant.Tjinb[0],
            ]
    measurements = [
            m_plant.Tall[0, "T"],
            m_plant.Ca[0],
            ]
    
    # Construct the "MHE simulator" object
    mhe = DynamicSim(
            plant_model=m_plant,
            plant_time_set=m_plant.t,
            predictor_model = m_predictor,
            predictor_time_set = m_predictor.t,            
            estimator_model=m_estimator, 
            estimator_time_set=m_estimator.t,
            inputs_at_t0=inputs,
            measurements_at_t0=measurements,
            sample_time=sample_time,
            as_strategy = True,
            )

    plant = mhe.plant
    predictor = mhe.predictor
    estimator = mhe.estimator
    
    p_t0 = mhe.plant.time.first()
    pre_t0 = mhe.predictor.time.first()
    e_t0 = mhe.estimator.time.first()
    p_ts = mhe.plant.sample_points[1]
    pre_ts = mhe.predictor.sample_points[1]
    e_ts = mhe.estimator.sample_points[1]
    
    #--------------------------------------------------------------------------
    # Declare variables of interest for plotting.
    # It's ok not declaring anything. The data manager will still save some 
    # important data, but the user should use the default string of CUID for plotting afterward.
    states_of_interest = [Reference(mhe.plant.mod.Ca[:]),
                          Reference(mhe.plant.mod.Tall[:, "T"])]

    # Set up data manager to save estimation data
    data_manager = EstimatorDataManager(plant, 
                                        estimator,
                                        states_of_interest,)
    #--------------------------------------------------------------------------
    solve_consistent_initial_conditions(plant, plant.time, solver)
    solve_consistent_initial_conditions(predictor, predictor.time, solver)
    
    # Here we solve for a steady state and use it to fill in past measurements
    desired_ss = [(estimator.mod.Ca[0], 0.021)]
    ss_weights = [(estimator.mod.Ca[0], 1.)]
    mhe.estimator.initialize_past_info_with_steady_state(desired_ss, ss_weights, solver)
        
    # Now we are ready to construct the objective function for MHE
    model_disturbance_weights = [
            (estimator.mod.Ca[0], 1.),
            (estimator.mod.Tall[0, "T"], 1.),
            (estimator.mod.Tall[0, "Tj"], 1.),
            ]

    measurement_noise_weights = [
            (estimator.mod.Ca[0], 100.),
            (estimator.mod.Tall[0, "T"], 20.),
            ]   
    
    mhe.estimator.add_noise_minimize_objective(model_disturbance_weights,
                                               measurement_noise_weights)
    
    mhe.estimator.MHE_advanced_strategy_setup(method = "k_aug",
                                              k_aug_solver = k_aug,
                                              dot_sens_solver = dot_sens,)
    
    #-------------------------------------------------------------------------
    # Set up measurement noises that will be applied to measurements
    variance = [
        (mhe.estimator.mod.Tall[0, "T"], 0.05),
        (mhe.estimator.mod.Ca[0], 1.0E-2),
        ]
    mhe.estimator.set_variance(variance)
    measurement_variance = [v.variance for v in estimator.measurement_vars]
    measurement_noise_bounds = [
            (var[e_t0].lb, var[e_t0].ub) for var in estimator.measurement_vars
            ]
    #-------------------------------------------------------------------------
    
    cinput = {ind: 250.+ind*5 if ind<=5 else 260.-ind*5 for ind in range(0, 12)}
    
    # [OFFLINE] Predict next measurement with predictor
    inputs = [cinput[0]]
    mhe.predictor.inject_inputs(inputs)
    mhe.predictor.initialize_by_solving_elements(solver)
    mhe.predictor.vectors.input[...].fix() #Fix the input to solve the predictor
    solver.solve(mhe.predictor, tee = True)
    
    pre_measurements = mhe.predictor.generate_measurements_at_time(pre_ts)
    
    # [OFFLINE] Load predicted measurements and control inputs to MHE
    mhe.estimator.load_measurements(pre_measurements,
                                    target = "actualmeasurement",
                                    timepoint = estimator.time.last())
    mhe.estimator.load_inputs_for_MHE(inputs)

    # [OFFLINE] Solve MHE based on predicted measurements
    mhe.estimator.check_var_con_dof(skip_dof_check = False)
    solver.solve(mhe.estimator, tee=True)
    
    data_manager.save_initial_plant_data()
    
    for i in range(0,11):
        print('\nENTERING MHE LOOP ITERATION %s\n' % i)
        
        # [ONLINE] ############################################################
        if i > 0:
            mhe.plant.advance_one_sample()
            mhe.plant.initialize_to_initial_conditions()
        
        inputs = [cinput[i]]
        mhe.plant.inject_inputs(inputs)
        
        mhe.plant.initialize_by_solving_elements(solver)
        mhe.plant.vectors.input[...].fix() #Fix the input to solve the plant
        solver.solve(mhe.plant, tee = True)
        data_manager.save_plant_data(iteration = i)
        
        unnoised_measurements = mhe.plant.generate_measurements_at_time(p_ts)
        real_measurements = apply_noise_with_bounds(
                    unnoised_measurements,
                    measurement_variance,
                    random.gauss,
                    measurement_noise_bounds,
                    )
        mhe.estimator.MHE_sensitivity_update(real_measurements, tee = True)
        data_manager.save_estimator_data(iteration = i)

        # [OFFLINE] ###########################################################        
        mhe.predictor.advance_one_sample()
        mhe.predictor.initialize_to_initial_conditions()
        
        inputs = [cinput[i+1]]
        mhe.predictor.inject_inputs(inputs)
        
        # Predict next measurement with predictor
        mhe.predictor.initialize_by_solving_elements(solver)
        mhe.predictor.vectors.input[...].fix() #Fix the input to solve the predictor
        solver.solve(mhe.predictor, tee = True)
        
        pre_measurements = mhe.predictor.generate_measurements_at_time(pre_ts)
        
        # Load predicted measurements and control inputs to MHE
        mhe.estimator.advance_one_sample()
        mhe.estimator.load_measurements(pre_measurements,
                                        target = "actualmeasurement",
                                        timepoint = estimator.time.last())
        mhe.estimator.load_inputs_for_MHE(inputs)
    
        # Solve MHE based on predicted measurements
        mhe.estimator.check_var_con_dof(skip_dof_check = False)
        solver.solve(mhe.estimator, tee=True)
        
        
    data_manager.plot_estimation_results(states_of_interest)
    return mhe, data_manager

if __name__ == '__main__':
    mhe, data_manager = main()