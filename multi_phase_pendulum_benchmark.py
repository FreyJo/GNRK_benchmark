# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# authors: Katrin Baumgaertner, Jonathan Frey
from typing import List, Union
import numpy as np
import pickle, os

from setup_acados_integrator import setup_acados_integrator
from mpc_parameters import MpcPendulumParameters

from setup_acados_ocp_solver import (
    setup_acados_ocp_solver,
    augment_model_with_cost_state,
    setup_acados_ocp_solver_clc,
)

from acados_template import AcadosMultiphaseOcp, AcadosOcp, AcadosOcpSolver

from models import setup_linearized_model, setup_pendulum_model, modify_model_to_use_cost_state
from utils import plot_simulation_result, get_label_from_setting, get_results_filename, plot_open_loop_trajectory_pwpol_u,  plot_open_loop_trajectories_pwpol_u, get_subdict, get_relevant_keys, KEY_TO_TEX, TEX_FOLDER, plot_open_loop_trajectories_pwpol_ux,plot_simplest_pareto
from simulate import simulate, initialize_controller


dummy_model, dummy_model_params = setup_pendulum_model()
dummy_mpc_params = MpcPendulumParameters(xs=dummy_model_params.xs, us=dummy_model_params.us)
N_REF = int(dummy_mpc_params.T / dummy_mpc_params.dt)
DT_PLANT = dummy_mpc_params.dt

# benchmark stuff
REFERENCE_SETTING = {
    "cost_hess_variant": "GNRK",
    "N": N_REF,
    "sim_method_num_stages": 4,
    "use_rti": False,
    "time_grid": "nonuniform",
    "degree_u_polynom": 0,
    "u_polynom_constraints": 1,
    "squash_u": False,
    "time_horizon": dummy_mpc_params.T,
}

SETTING_CLC = {
    "cost_hess_variant": "GNRK",
    "N": 10,
    "sim_method_num_stages": 4,
    "use_rti": False,
    "time_grid": "nonuniform",
    "degree_u_polynom": 0,
    "u_polynom_constraints": 1,
    "clc": True,
    "time_horizon": dummy_mpc_params.T,
    "squash_u": False,
    "barrier_type": "None",
}

T_CLC = 3.7
N_CLC = 5

SETTING_NO_CLC = SETTING_CLC.copy()
SETTING_NO_CLC['clc'] = False
SETTING_NO_CLC['time_grid'] = 'clc_grid'

if SETTING_NO_CLC['time_grid'] != 'clc_grid':
    SETTING_NO_CLC['N'] = int(SETTING_CLC['N'] - N_CLC)
    SETTING_NO_CLC['time_horizon'] = SETTING_CLC['time_horizon'] - T_CLC

PW_POL_SETTING_A = SETTING_NO_CLC.copy()
PW_POL_SETTING_A['degree_u_polynom'] = 1
PW_POL_SETTING_A['u_polynom_constraints'] = 2

PW_POL_SETTING_B = PW_POL_SETTING_A.copy()
PW_POL_SETTING_B['time_grid'] = 'nonuniform'

PW_POL_SETTING_2 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_2['degree_u_polynom'] = 2
PW_POL_SETTING_2['u_polynom_constraints'] = 4

PW_POL_SETTING_QUAD_2 = PW_POL_SETTING_2.copy()
PW_POL_SETTING_QUAD_2['u_polynom_constraints'] = 4

PW_POL_SETTING_QUAD_3 = PW_POL_SETTING_2.copy()
PW_POL_SETTING_QUAD_3['u_polynom_constraints'] = 10

PW_POL_SETTING_3 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_3['degree_u_polynom'] = 3
PW_POL_SETTING_3['u_polynom_constraints'] = 4

PW_POL_SETTING_4 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_4['degree_u_polynom'] = 3
PW_POL_SETTING_4['u_polynom_constraints'] = 5

PW_POL_SETTING_5 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_5['degree_u_polynom'] = 3
PW_POL_SETTING_5['u_polynom_constraints'] = 6

PW_POL_SETTING_6 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_6['degree_u_polynom'] = 3
PW_POL_SETTING_6['u_polynom_constraints'] = 7

PW_POL_SETTING_7 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_7['degree_u_polynom'] = 3
PW_POL_SETTING_7['u_polynom_constraints'] = 10

PW_POL_SETTING_8 = PW_POL_SETTING_A.copy()
PW_POL_SETTING_8['degree_u_polynom'] = 3
PW_POL_SETTING_8['u_polynom_constraints'] = 10
PW_POL_SETTING_8['time_grid'] = 'nonuniform'

SETTING_NO_CLC_RTI = SETTING_NO_CLC.copy()
SETTING_NO_CLC_RTI['use_rti'] = True

SETTING_CLC_SQUASHED = SETTING_CLC.copy()
SETTING_CLC_SQUASHED['squash_u'] = True
SETTING_CLC_SQUASHED['barrier_type'] = "None"

SETTING_CLC_SQUASHED_CONST_BARRIER = SETTING_CLC.copy()
SETTING_CLC_SQUASHED_CONST_BARRIER['squash_u'] = True
SETTING_CLC_SQUASHED_CONST_BARRIER['barrier_type'] = "constant"

SETTING_CLC_SQUASHED_PROGRESSIVE_BARRIER = SETTING_CLC.copy()
SETTING_CLC_SQUASHED_PROGRESSIVE_BARRIER['squash_u'] = True
SETTING_CLC_SQUASHED_PROGRESSIVE_BARRIER['barrier_type'] = "progressive"


REF_GNRK_SETTING = {
        "cost_hess_variant": "GNRK",
        "N": 20,
        "sim_method_num_stages": 4,
        "use_rti": False,
        "time_grid": "nonuniform",
        "degree_u_polynom": 0,
        "u_polynom_constraints": 1,
        "time_horizon": dummy_mpc_params.T,
    }

REF_GNRK_SETTING_10 = REF_GNRK_SETTING.copy()
REF_GNRK_SETTING_10['N'] = 10

REF_GNRK_SETTING_RTI = REF_GNRK_SETTING.copy()
REF_GNRK_SETTING_RTI['use_rti'] = True


def compute_open_loop_solution(setting=None, plot_results=False):
    if setting is None:
        setting = {
        "cost_hess_variant": "GNRK",
        "N": 10,
        "sim_method_num_stages": 4,
        "use_rti": False,
        "time_grid": "nonuniform",
        "degree_u_polynom": 2,
        "u_polynom_constraints": 8,
    }

    model, model_params = setup_pendulum_model()
    x0 = np.array([0.0, .2*np.pi, 0.0, 0.0])

    mpc_params = MpcPendulumParameters(xs=model_params.xs, us=model_params.us)
    levenberg_marquardt=0.0
    exact_hess_dyn = True

    if setting['cost_hess_variant'] == 'GNRK':
        mpc_params.cost_integration = 1
        hessian_approx = 'GAUSS_NEWTON'
    elif setting['cost_hess_variant'] == 'GNSN':
        hessian_approx = 'GAUSS_NEWTON'
    elif setting['cost_hess_variant'] == 'EHRK':
        model = augment_model_with_cost_state(model, model_params, mpc_params)
        hessian_approx = 'EXACT'
        levenberg_marquardt=1e-2
        model = modify_model_to_use_cost_state(model, model_params)
    elif setting['cost_hess_variant'] == 'EHSN':
        hessian_approx = 'EXACT'

    mpc_params.N = setting['N']
    mpc_params.sim_method_num_stages = setting['sim_method_num_stages']
    use_rti = setting['use_rti']
    time_grid = setting['time_grid']
    degree_u_polynom = setting['degree_u_polynom']

    ocp_solver: AcadosOcpSolver
    ocp_solver, eval_u_funs = setup_acados_ocp_solver(model,
            model_params, mpc_params, use_rti=use_rti, time_grid=time_grid, levenberg_marquardt=levenberg_marquardt,
            hessian_approx=hessian_approx,
            exact_hess_dyn=exact_hess_dyn,
            degree_u_polynom=degree_u_polynom,
            u_polynom_constraints=setting['u_polynom_constraints'],
            N_clc = N_CLC,
            T_clc=T_CLC
            )

    print(f"{setting=}, {mpc_params.T}")
    u0 = ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False)
    ocp_solver.print_statistics()

    u_traj = []
    N_horizon = mpc_params.N
    x_traj = np.zeros((N_horizon+1, len(x0)))
    for i in range(N_horizon):
        x_traj[i, :] = ocp_solver.get(i, "x")
        u_traj.append(ocp_solver.get(i, "u"))
    x_traj[N_horizon, :] = ocp_solver.get(N_horizon, "x")

    u_fine_traj = []

    # compute U fine:
    n_pol_eval = 40
    u_fine_traj = []
    i_ocp = 0
    ocp: Union[AcadosMultiphaseOcp, AcadosOcp] = ocp_solver.acados_ocp
    nu_original = 1
    if isinstance(ocp, AcadosMultiphaseOcp):
        N_list = ocp.N_list
    else:
        N_list = [ocp.dims.N]
    for i_phase in range(len(N_list)):
        for _ in range(N_list[i_phase]):
            dt = ocp.solver_options.time_steps[i_ocp]
            u_coeff = u_traj[i_ocp]
            U_interval = np.zeros((n_pol_eval+1, nu_original))
            for j in range(n_pol_eval+1):
                t = (j/n_pol_eval) * dt
                U_interval[j, :] = eval_u_funs[i_phase](u_coeff, t)
            u_fine_traj.append(U_interval)
            i_ocp += 1
    if degree_u_polynom > 0:
        print(f"{(ocp.constraints[1].D @ u_traj[1])[-1]=} should match {u_fine_traj[1][-1]}")

    if plot_results:
        plot_open_loop_trajectory_pwpol_u(ocp_solver.acados_ocp.solver_options.shooting_nodes, x_traj, u_fine_traj, plt_show=True, lbu=mpc_params.umin, ubu=mpc_params.umax, x_labels=dummy_model.x_labels, u_labels=dummy_model.u_labels, idxpx=[0, 1])
    return ocp_solver.acados_ocp.solver_options.shooting_nodes, x_traj, u_fine_traj


def compute_open_loop_solution_clc(setting, plot_results=False, N_clc=N_CLC):
    Tsim = 4.0

    plant_model, plant_model_params = setup_pendulum_model()
    mpc_params = MpcPendulumParameters(xs=plant_model_params.xs, us=plant_model_params.us)

    # to compute LQR matrix P
    # linearized_model = setup_linearized_model(plant_model, plant_model_params, dummy_mpc_params)

    # plant_model = augment_model_with_cost_state(plant_model, plant_model_params, mpc_params=mpc_params)

    Nsim = int(Tsim / DT_PLANT)
    if not (Tsim / DT_PLANT).is_integer():
        print("WARNING: Tsim / DT_PLANT should be an integer!")

    integrator = setup_acados_integrator(plant_model, DT_PLANT, mpc_params,
                    num_steps=mpc_params.sim_method_num_steps,
                    num_stages=mpc_params.sim_method_num_stages, integrator_type="IRK")

    label = get_label_from_setting(setting)

    model, model_params = setup_pendulum_model()
    x0 = np.array([0.0, .2*np.pi, 0.0, 0.0])
    nx = len(x0)

    mpc_params = MpcPendulumParameters(xs=model_params.xs, us=model_params.us)
    levenberg_marquardt=0.0
    exact_hess_dyn = True
    if setting['cost_hess_variant'] == 'GNRK':
        mpc_params.cost_integration = 1
        hessian_approx = 'GAUSS_NEWTON'
    elif setting['cost_hess_variant'] == 'GNSN':
        hessian_approx = 'GAUSS_NEWTON'
    elif setting['cost_hess_variant'] == 'EHRK':
        model = augment_model_with_cost_state(model, model_params, mpc_params)
        hessian_approx = 'EXACT'
        levenberg_marquardt=1e-2
        model = modify_model_to_use_cost_state(model, model_params)
    elif setting['cost_hess_variant'] == 'EHSN':
        hessian_approx = 'EXACT'

    mpc_params.N = setting['N']
    mpc_params.sim_method_num_stages = setting['sim_method_num_stages']
    use_rti = setting['use_rti']
    time_grid = setting['time_grid']
    mpc_params.T = setting['time_horizon']

    clc = setting.get('clc', False)

    ocp_solver: AcadosOcpSolver

    if not clc:
        raise ValueError("This function is only for CLC")

    ocp_solver, eval_u_funs = setup_acados_ocp_solver_clc(model,
            model_params, mpc_params, use_rti=use_rti, time_grid=time_grid,
            levenberg_marquardt=levenberg_marquardt,
            hessian_approx=hessian_approx,
            exact_hess_dyn=exact_hess_dyn,
            T_clc=T_CLC,
            squash_u=setting['squash_u'],
            barrier_type=setting['barrier_type'],
            N_clc=N_clc,
            )


    print(f"{setting=}, {mpc_params.T}")
    ocp_solver.options_set("max_iter", 3)
    # initialize_controller(ocp_solver, model_params, x0/2)
    u0 = ocp_solver.solve_for_x0(x0, fail_on_nonzero_status=False)
    ocp_solver.print_statistics()

    u_traj = []
    N_horizon = mpc_params.N
    x_traj = np.zeros((N_horizon+1, len(x0)))
    for i in range(N_horizon):
        x_traj[i, :] = ocp_solver.get(i, "x")
        u_traj.append(ocp_solver.get(i, "u"))
    x_traj[N_horizon, :] = ocp_solver.get(N_horizon, "x")

    u_fine_traj = []

    # compute U fine:
    integrator = setup_acados_integrator(plant_model, DT_PLANT, mpc_params,
                    num_steps=mpc_params.sim_method_num_steps,
                    num_stages=mpc_params.sim_method_num_stages, integrator_type="IRK")

    n_pol_eval = 40
    u_fine_traj = []
    i_ocp = 0
    ocp: Union[AcadosMultiphaseOcp, AcadosOcp] = ocp_solver.acados_ocp
    nu_original = 1
    x_fine_traj = []
    if isinstance(ocp, AcadosMultiphaseOcp):
        N_list = ocp.N_list
    else:
        N_list = [ocp.dims.N]
    for i_phase in range(len(N_list)):
        for _ in range(N_list[i_phase]):
            X_interval = np.zeros((n_pol_eval+1, nx))
            U_interval = np.zeros((n_pol_eval+1, nu_original))

            dt = ocp.solver_options.time_steps[i_ocp]
            x_current = x_traj[i_ocp]
            integrator.set("T", dt/n_pol_eval)
            for j in range(n_pol_eval+1):
                # t = (j/n_pol_eval) * dt
                X_interval[j, :] = x_current
                if i_phase == 0:
                    U_interval[j, :] = u_traj[i_ocp]
                else: # CLC phase
                    U_interval[j, :] = eval_u_funs[i_phase](x_current)
                x_current = integrator.simulate(x=x_current, u=U_interval[j, :])

            u_fine_traj.append(U_interval)
            x_fine_traj.append(X_interval)
            i_ocp += 1

    if plot_results:
        plot_open_loop_trajectory_pwpol_u(ocp_solver.acados_ocp.solver_options.shooting_nodes, x_traj, u_fine_traj, plt_show=True, lbu=mpc_params.umin, ubu=mpc_params.umax, x_labels=dummy_model.x_labels, u_labels=dummy_model.u_labels, idxpx=[0, 1])
    return ocp_solver.acados_ocp.solver_options.shooting_nodes, x_fine_traj, u_fine_traj

def compute_and_plot_open_loop_solutions_clc(settings, variant_ids=None):
    shooting_nodes_list = []
    x_trajs = []
    u_fine_trajs = []
    labels = []
    if variant_ids is None:
        variant_ids = len(settings) * [""]
    for s, id in zip(settings, variant_ids):
        label = id
        labels.append(label)
        shooting_nodes, x_traj, u_fine_traj = compute_open_loop_solution_clc(s, plot_results=False)
        shooting_nodes_list.append(shooting_nodes)
        x_trajs.append(x_traj)
        u_fine_trajs.append(u_fine_traj)
    plot_open_loop_trajectories_pwpol_ux(shooting_nodes_list, x_trajs, u_fine_trajs, lbu=dummy_mpc_params.umin, ubu=dummy_mpc_params.umax,
                                         tmax=1.2,
                                        x_labels=dummy_model.x_labels, u_labels=dummy_model.u_labels, idxpx=[0], labels=labels, fig_filename="clc_multiple_shooting_trajs.pdf",
                                        bbox_to_anchor=(0.8, 0.42), figsize=(9, 4.5))



def compute_and_plot_open_loop_solutions(settings, variant_ids=None):
    shooting_nodes_list = []
    x_trajs = []
    u_fine_trajs = []
    labels = []
    if variant_ids is None:
        variant_ids = len(settings) * [""]
    for s, id in zip(settings, variant_ids):
        label = f"Grid {'A' if s['time_grid'] == 'clc_grid' else 'B'}, "
        if s['degree_u_polynom'] == 0:
            label += "pw. constant"
        else:
            label += r"pw. polynomial $ n_{\mathrm{deg}} =" + f"{s['degree_u_polynom']}, " + r"n_{\mathrm{pc}} = " + f"{s['u_polynom_constraints']} $"
        label = id + ": " + label
        labels.append(label)
        shooting_nodes, x_traj, u_fine_traj = compute_open_loop_solution(s)
        shooting_nodes_list.append(shooting_nodes)
        x_trajs.append(x_traj)
        u_fine_trajs.append(u_fine_traj)
    plot_open_loop_trajectories_pwpol_u(shooting_nodes_list, x_trajs, u_fine_trajs, lbu=dummy_mpc_params.umin, ubu=dummy_mpc_params.umax,
                                        x_labels=dummy_model.x_labels, u_labels=dummy_model.u_labels, idxpx=[], labels=labels, fig_filename="pw_pol_open_loop_trajs.pdf",
                                        bbox_to_anchor=(0.37, 0.37), figsize=(9, 4.5))



def run_pendulum_benchmark_closed_loop(settings: List[dict]):
    Tsim = 4.0

    plant_model, plant_model_params = setup_pendulum_model()
    mpc_params = MpcPendulumParameters(xs=plant_model_params.xs, us=plant_model_params.us)

    # to compute LQR matrix P
    # linearized_model = setup_linearized_model(plant_model, plant_model_params, dummy_mpc_params)

    plant_model = augment_model_with_cost_state(plant_model, plant_model_params, mpc_params=mpc_params)

    Nsim = int(Tsim / DT_PLANT)
    if not (Tsim / DT_PLANT).is_integer():
        print("WARNING: Tsim / DT_PLANT should be an integer!")

    integrator = setup_acados_integrator(plant_model, DT_PLANT, mpc_params,
                    num_steps=mpc_params.sim_method_num_steps,
                    num_stages=mpc_params.sim_method_num_stages, integrator_type="IRK")

    labels_all = []

    for setting in settings:
        label = get_label_from_setting(setting)

        model, model_params = setup_pendulum_model()
        x0 = np.array([0.0, .2*np.pi, 0.0, 0.0])

        mpc_params = MpcPendulumParameters(xs=model_params.xs, us=model_params.us)
        levenberg_marquardt=0.0
        exact_hess_dyn = True
        if setting['cost_hess_variant'] == 'GNRK':
            mpc_params.cost_integration = 1
            hessian_approx = 'GAUSS_NEWTON'
        elif setting['cost_hess_variant'] == 'GNSN':
            hessian_approx = 'GAUSS_NEWTON'
        elif setting['cost_hess_variant'] == 'EHRK':
            model = augment_model_with_cost_state(model, model_params, mpc_params)
            hessian_approx = 'EXACT'
            levenberg_marquardt=1e-2
            model = modify_model_to_use_cost_state(model, model_params)
        elif setting['cost_hess_variant'] == 'EHSN':
            hessian_approx = 'EXACT'

        mpc_params.N = setting['N']
        mpc_params.sim_method_num_stages = setting['sim_method_num_stages']
        use_rti = setting['use_rti']
        time_grid = setting['time_grid']
        degree_u_polynom = setting['degree_u_polynom']
        mpc_params.T = setting['time_horizon']

        clc = setting.get('clc', False)

        ocp_solver: AcadosOcpSolver

        if clc:
            ocp_solver, eval_u_funs = setup_acados_ocp_solver_clc(model,
                model_params, mpc_params, use_rti=use_rti, time_grid=time_grid,
                levenberg_marquardt=levenberg_marquardt,
                hessian_approx=hessian_approx,
                exact_hess_dyn=exact_hess_dyn,
                T_clc=T_CLC,
                squash_u=setting['squash_u'],
                barrier_type=setting['barrier_type'],
                N_clc=N_CLC,
                )
        else:
            ocp_solver, eval_u_funs = setup_acados_ocp_solver(model,
                model_params, mpc_params, use_rti=use_rti, time_grid=time_grid, levenberg_marquardt=levenberg_marquardt,
                hessian_approx=hessian_approx,
                exact_hess_dyn=exact_hess_dyn,
                degree_u_polynom=degree_u_polynom,
                u_polynom_constraints=setting['u_polynom_constraints'],
                N_clc = N_CLC,
                T_clc=T_CLC
                )

        print(f"{setting=}, {mpc_params.T}")

        print(f"\n\nRunning CLOSED loop simulation with {label}\n\n")
        results = simulate(ocp_solver, integrator, model_params, x0, Nsim, n_runs=50)
        results = add_total_cost_to_results(results)

        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        results['label'] = label
        results['mpc_params'] = mpc_params
        labels_all.append(label)
        pickle.dump(results, open(results_filename, "wb"))
        print(f"saved result as {results_filename}")

        print("nlp_iter:", results['nlp_iter'])

        ocp_solver = None

    print(f"ran all experiments with {len(labels_all)} different settings")

def add_total_cost_to_results(results):
    x_end = results['x_traj'][-1]
    terminal_cost_state = x_end[-1]
    x_og_vec = np.expand_dims(x_end[:dummy_model_params.nx_original], axis=1)
    terminal_cost_term = (x_og_vec.T @ dummy_mpc_params.P @ x_og_vec)[0][0]
    results['cost_total'] = terminal_cost_state + terminal_cost_term
    return results

def get_latex_label_from_setting(setting: dict):
    label = ''
    for key, value in setting.items():
        if key == 'use_rti':
            if value == False:
                label += 'SQP '
            else:
                label += 'RTI '
        elif key == 'squash_u':
            if value:
                label += 'squashed '
        elif key == 'barrier_type':
            if value == 'constant':
                label += 'const barrier '
            elif value == 'progressive':
                label += 'prog barrier '
        elif key == 'clc':
            if value:
                label += 'CLC '
        else:
            label += f"{KEY_TO_TEX[key]} {value} "
    return label

def plot_trajectories(settings, labels=None, ncol_legend=1, title=None, bbox_to_anchor=None, fig_filename=None, figsize=None):
    X_all = []
    U_all = []
    labels_all = []

    relevant_keys, constant_keys = get_relevant_keys(settings)
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    # load
    for i, setting in enumerate(settings):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)

        descriptive_setting = get_subdict(setting, relevant_keys)
        if labels is not None:
            latex_label = labels[i]
        else:
            latex_label = get_latex_label_from_setting(descriptive_setting)
        # check if file exists
        if not os.path.exists(results_filename):
            print(f"File {results_filename} corresponding to {label} does not exist.")
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if results['status'] == 0:
            X_all.append(results['x_traj'])
            U_all.append(results['u_traj'])
            # labels_all.append(results['label'])
            labels_all.append(latex_label)

            # doesnt matter when we unpack this
            # X_ref = results['x_ref']
            # U_ref = results['u_ref']
            # print
            closed_loop_cost = results['cost_total']
            cpu_min = np.min(results['timings']) * 1e3
            cpu_max = np.max(results['timings']) * 1e3
            time_per_iter = 1e3 * np.sum(results['timings']) / np.sum(results['nlp_iter'])
            print(f"Simulation {latex_label}\n\tCPU time: {time_per_iter:.2} ms/iter min {cpu_min:.3} ms, max {cpu_max:.3} ms , closed loop cost: {closed_loop_cost:.3e}")
        else:
            print(f"Simulation failed with {label}")

    x_lables_list = ["$p$ [m]", r"$\theta$ [rad/s]", "$s$ [m/s]", r"$\omega$", r"cost state"]
    u_lables_list = [r"$\nu$ [N]"]

    if title is None:
        title = common_description

    plot_simulation_result(
        DT_PLANT,
        X_all,
        U_all,
        dummy_mpc_params.umin,
        dummy_mpc_params.umax,
        x_lables_list,
        u_lables_list,
        labels_all,
        # title='closed loop' if CLOSED_LOOP else 'open loop',
        idxpx=[0, 1], # 4
        title=title,
        # X_ref=X_ref,
        # U_ref=U_ref,
        linestyle_list=['--', ':', '--', ':', '--', '-.', '-.', ':'],
        single_column=True,
        xlabel='$t$ [s]',
        idx_xlogy= [4],
        figsize=figsize,
        ncol_legend = ncol_legend,
        # color_list=['C0', 'C0', 'C1', 'C1']
        fig_filename=fig_filename,
        bbox_to_anchor=bbox_to_anchor,
    )



def pareto_plot_comparison_custom_labels(settings, labels, fig_filename='pendulum_pareto_wip.pdf'):

    cost_all = []
    timings = []

    n_points = len(settings)
    assert n_points == len(labels)
    colors = 2* ["C0"] + 2*["C1"] + 2*["C2"] + 2*["C3"] + 2*["C4"] + 3*["C5"]
    # markers = 5 * ['o', 'v', 's', 'd', '^', '>', '<', 'P', 'D']
    markers = ['o', 'v'] + ['o', 'v'] + ['o', 'v'] + ['o', 'v'] + ['o', 'v'] + ['o', 'v', 's']

    for setting in settings:
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        cost_all.append(results['cost_total'])
        # labels_all.append(latex_label)
        time_per_iter = 1e3 * np.sum(results['timings']) / np.sum(results['nlp_iter'])
        timings.append(time_per_iter)

    with_reference = True
    if with_reference:
        label = get_label_from_setting(REFERENCE_SETTING)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        ref_cost = results['cost_total']

    best_cost = min(cost_all + [ref_cost])
    rel_subopt = [((c - best_cost) / best_cost) * 100 for c in cost_all]
    points = list(zip(rel_subopt, timings))

    label='relative suboptimality [\%]'
    ylabel='Average comp. time per NLP iter. [ms]'

    plot_simplest_pareto(points, labels, colors, markers, xlabel=label, ylabel=ylabel, fig_filename=fig_filename,
                         bbox_to_anchor=[1.02, 1.0])



def create_table_ocp_costing_paper(settings, labels=None, use_rti=False):
    labels_all = []

    relevant_keys, constant_keys = get_relevant_keys(settings)
    common_description = get_latex_label_from_setting(get_subdict(settings[0], constant_keys))

    res_list = []
    costs_all = []

    # load
    for i, setting in enumerate(settings):
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)

        descriptive_setting = get_subdict(setting, relevant_keys)
        if labels is not None:
            latex_label = labels[i]
        else:
            latex_label = get_latex_label_from_setting(descriptive_setting)
        # check if file exists
        if not os.path.exists(results_filename):
            print(f"File {results_filename} corresponding to {label} does not exist.")
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if results['status'] == 0:
            labels_all.append(latex_label)

            # doesnt matter when we unpack this
            X_ref = results['x_ref']
            U_ref = results['u_ref']
            # print
            costs_all.append(results['cost_total'])
            res_list.append(results)
        else:
            print(f"Simulation failed with {label}")

    min_cost = min(costs_all)
    min_cpu = min([1e3 * np.sum(r['timings']) / np.sum(r['nlp_iter']) for r in res_list])

    # write table
    table_filename = os.path.join(TEX_FOLDER, "table_clc_benchmark.tex")
    with open(table_filename, 'w') as f:
        f.write(r"\begin{table*}" + "\n")
        f.write(r"\centering" + "\n")
        # f.write(r"\caption{Closed-loop comparison of different controller variants on the pendulum test problem. The variants vary the number of shooting intervals $N$, the discretization grid, the control parametrization, with closed-loop-costing on the latter part of Grid A, and piecewise polynomial controls on all but the first shooting interval.}" + "\n")
        f.write(r"\caption{Closed-loop comparison of different controller variants on the pendulum test problem with different discretization grids and the control parametrizations, including closed-loop-costing and piecewise polynomial controls.}" + "\n")
        f.write(r"\label{tab:pendulum_closed_loop_costing}" + "\n")
        f.write(r"\begin{tabular}{lrrcrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"variant ID & $N$ & Grid & control parametrization & comp. time / iter [ms] & relative suboptimality [\%] \\ \midrule" + "\n")

        for i, (setting, label, results) in enumerate(zip(settings, labels_all, res_list)):
            line_string = f"{label} &"
            line_string += f"{setting['N']} &"
            clc = setting.get('clc', False)
            if clc or setting['time_grid'] == 'clc_grid':
                line_string += r" A &"
            else:
                line_string += r" B &"

            if not clc:
                if setting['degree_u_polynom'] == 0:
                    line_string += "pw. constant &"
                else:
                    line_string += r"pw. polynomials with $n\ind{deg} = " + f" {setting['degree_u_polynom']}, " + r"n\ind{pc} = " + f"{setting['u_polynom_constraints']} $ &"
            elif setting['squash_u']:
                line_string += " Squashed "
                if setting['barrier_type'] == 'progressive':
                    line_string += r" + prog. barrier"
                elif setting['barrier_type'] == 'constant':
                    line_string += r" + const. barrier"
                line_string += " CLC &"
            else:
                line_string += " unconstrained LQR CLC &"

            time_per_iter = 1e3 * np.sum(results['timings']) / np.sum(results['nlp_iter'])
            # time_per_iter = 1e3 * np.min(results['timings'] / results['nlp_iter'])
            line_string += f"{time_per_iter:.2f} &"
            # if time_per_iter > 1.2 * min_cpu:
            #     line_string += f"{time_per_iter:.2f} &"
            # else:
            #     line_string += r"\textbf{" + f"{time_per_iter:.2f}" + r"} &"

            if results['cost_total'] == min_cost:
                rel_subopt = 0.0
            else:
                rel_subopt = ((results['cost_total'] - min_cost) / min_cost) * 100
            # if rel_subopt < 1.0:
            #     line_string += r"\textbf{" + f"{rel_subopt:.2f}" + r"} \\"
            # else:
            #     line_string += f"{rel_subopt:.2f} \\\\"

            line_string += f"{rel_subopt:.2f}"
            # mean_nlp_iter = np.median(results['nlp_iter'])
            # line_string += f" & {mean_nlp_iter}"
            line_string += "\\\\\n"
            if i in [4, 7, 10]:
                line_string += "\\midrule\n"
            f.write(line_string)

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table*}" + "\n")
    # print table to terminal
    with open(table_filename, 'r') as f:
        print(f.read())
    print(f"saved table as {table_filename}")


if __name__ == "__main__":

    CLC_PAPER_SETTINGS_W_LABELS = \
                    [(REFERENCE_SETTING, 'IDEAL',),
                      (REF_GNRK_SETTING, 'REF',),
                      (REF_GNRK_SETTING_10, 'REF-N10',),
                      (PW_POL_SETTING_B, 'PW-LIN-B',),
                      (PW_POL_SETTING_8, 'PW-CUBIC-B',),
                      (SETTING_CLC, 'CLC-LQR',),
                      (SETTING_CLC_SQUASHED_PROGRESSIVE_BARRIER, 'CLC-SQB',),
                      (SETTING_NO_CLC, 'PW-CONST-A',),
                      (PW_POL_SETTING_A, 'PW-LIN-A',),
                      (PW_POL_SETTING_2, 'PW-QUAD-1',),
                      (PW_POL_SETTING_QUAD_3, 'PW-QUAD-2',),
                      (PW_POL_SETTING_3, 'PW-CUBIC-1',),
                      (PW_POL_SETTING_5, 'PW-CUBIC-2',),
                      (PW_POL_SETTING_7, 'PW-CUBIC-3',),
                    ]
    CLC_PAPER_LIST = [setting for setting, label in CLC_PAPER_SETTINGS_W_LABELS]
    CLC_LABEL_LIST = [label for setting, label in CLC_PAPER_SETTINGS_W_LABELS]

    ### Multiple shooting CLC plot
    compute_and_plot_open_loop_solutions_clc([SETTING_CLC, SETTING_CLC_SQUASHED_PROGRESSIVE_BARRIER], variant_ids=['CLC-LQR', 'CLC-SQB'])

    ### Open-loop plot
    compute_and_plot_open_loop_solutions([REF_GNRK_SETTING_10, PW_POL_SETTING_8, PW_POL_SETTING_A, PW_POL_SETTING_5],
                                         variant_ids=['REF-N10', 'PW-CUBIC-B', 'PW-LIN-A', 'PW-CUBIC-2'])

    ### Multi-phase paper benchmark
    run_pendulum_benchmark_closed_loop(CLC_PAPER_LIST)


    create_table_ocp_costing_paper(CLC_PAPER_LIST,
                                   labels=CLC_LABEL_LIST)

    pareto_plot_comparison_custom_labels(CLC_PAPER_LIST[1:], CLC_LABEL_LIST[1:],
                           fig_filename='pendulum_pareto_multiphase.pdf')

    # closed-loop trajectories
    idx_cl_plot = [0, 1, 2, 3, 6, len(CLC_PAPER_LIST)-1]
    PLOT_SETTING_LIST = [CLC_PAPER_LIST[i] for i in idx_cl_plot]
    PLOT_LABEL_LIST = [CLC_LABEL_LIST[i] for i in idx_cl_plot]
    plot_trajectories(PLOT_SETTING_LIST,
                        labels = PLOT_LABEL_LIST,
                        ncol_legend=3,
                        title="",
                        bbox_to_anchor=(0.45, -0.8),
                        fig_filename='pendulum_trajectories_clc.pdf',
                        figsize=(7, 6),
                    )

