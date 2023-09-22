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

import os
import numpy as np
import itertools
import pickle

from setup_acados_integrator import setup_acados_integrator
from setup_acados_ocp_solver import (
    MpcPendulumParameters,
    setup_acados_ocp_solver,
    AcadosOcpSolver,
)
from models import setup_linearized_model, augment_model_with_cost_state, setup_pendulum_model, modify_model_to_use_cost_state
from utils import plot_simulation_result, get_label_from_setting, get_results_filename, get_latex_label_from_setting, plot_pareto, plot_simple_pareto, TEX_FOLDER
from simulate import simulate

dummy_model, dummy_model_params = setup_pendulum_model()
dummy_mpc_params = MpcPendulumParameters(xs=dummy_model_params.xs, us=dummy_model_params.us)
N_REF = int(dummy_mpc_params.T / dummy_mpc_params.dt)
DT_PLANT = dummy_mpc_params.dt

# benchmark stuff
# COST_DISCRETIZATION_VARIANTS = ['expl Euler']
COST_DISCRETIZATION_VARIANTS = ['GNRK']
# COST_DISCRETIZATION_VARIANTS = ['EHRK']
COST_DISCRETIZATION_VARIANTS = ['GNRK', 'GNSN', 'EHRK', 'EHSN']
# COST_DISCRETIZATION_VARIANTS = ['EHRK']
# COST_DISCRETIZATION_VARIANTS = ['EHSN']

TIME_GRID_VALUES = ["nonuniform"]
TIME_GRID_VALUES = ["uniform_long", "nonuniform", "uniform_short"]
USE_RTI_VALUES = [True, False]
# USE_RTI_VALUES = [False]
NUM_STAGE_VALUES = [4]
N_VALUES = [20]
N_VALUES = [20, N_REF]

ALL_SETTINGS = [s for s in itertools.product(COST_DISCRETIZATION_VARIANTS, N_VALUES, NUM_STAGE_VALUES, USE_RTI_VALUES, TIME_GRID_VALUES)]

SETTINGS = ALL_SETTINGS
SETTINGS = [
    ["EHSN", 20, 4, False, 'nonuniform'],
    ["EHRK", 20, 4, False, 'nonuniform'],
    ["GNRK", 20, 4, False, 'nonuniform'],
    ]


SETTINGS_TABLE = []
SETTINGS_TABLE += [s for s in itertools.product(['GNRK'], [N_REF], [NUM_STAGE_VALUES[-1]], [False], ['uniform_long'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNSN'], [N_REF], [NUM_STAGE_VALUES[-1]], [False], ['uniform_long'])]
# SETTINGS_TABLE += [s for s in itertools.product(['GNSN'], [N_REF], [NUM_STAGE_VALUES[-1]], [True], ['uniform_long'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNRK'], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['nonuniform'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNSN'], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['nonuniform'])]
# SETTINGS_TABLE += [s for s in itertools.product(['EHSN',], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['nonuniform'])]
SETTINGS_TABLE += [s for s in itertools.product(['EHSN', 'EHRK'], [20], [NUM_STAGE_VALUES[-1]], [False], ['nonuniform'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNRK'], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['uniform_long'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNSN'], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['uniform_long'])]
SETTINGS_TABLE += [s for s in itertools.product(['GNRK'], [20], [NUM_STAGE_VALUES[-1]], [True, False], ['uniform_short'])]

SETTINGS_PAPER_PLOT = []
SETTINGS_PAPER_PLOT += [s for s in itertools.product(['GNRK'], [N_REF], [NUM_STAGE_VALUES[-1]], [False], ['uniform_long'])]
SETTINGS_PAPER_PLOT += [s for s in itertools.product(['GNRK'], [20], [NUM_STAGE_VALUES[-1]], [True], ['nonuniform'])]
SETTINGS_PAPER_PLOT += [s for s in itertools.product(['GNSN'], [20], [NUM_STAGE_VALUES[-1]], [True], ['nonuniform'])]
SETTINGS_PAPER_PLOT += [s for s in itertools.product(['GNRK'], [20], [NUM_STAGE_VALUES[-1]], [True], ['uniform_long'])]


SETTINGS_PARETO = []
SETTINGS_PARETO = ALL_SETTINGS


def run_pendulum_benchmark_closed_loop(settings):

    Tsim = 4.0

    x0 = np.array([0.0, .2*np.pi, 0.0, 0.0])

    plant_model, plant_model_params = setup_pendulum_model()
    mpc_params = MpcPendulumParameters(xs=plant_model_params.xs, us=plant_model_params.us)

    # to compute LQR matrix P
    # liniearized_model = setup_linearized_model(plant_model, plant_model_params, dummy_mpc_params)

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
        if setting[0] == 'GNRK':
            mpc_params.cost_integration = 1
            hessian_approx = 'GAUSS_NEWTON'
        elif setting[0] == 'GNSN':
            hessian_approx = 'GAUSS_NEWTON'
        elif setting[0] == 'EHRK':
            model = augment_model_with_cost_state(model, model_params, mpc_params)
            hessian_approx = 'EXACT'
            levenberg_marquardt=1e-2
            model = modify_model_to_use_cost_state(model, model_params)
        elif setting[0] == 'EHSN':
            hessian_approx = 'EXACT'

        mpc_params.N = setting[1]
        mpc_params.sim_method_num_stages = setting[2]
        use_rti = setting[3]
        time_grid = setting[4]

        ocp_solver: AcadosOcpSolver = setup_acados_ocp_solver(model,
                model_params, mpc_params, use_rti=use_rti, time_grid=time_grid, levenberg_marquardt=levenberg_marquardt,
                hessian_approx=hessian_approx,
                exact_hess_dyn=exact_hess_dyn)

        print(f"{setting=}, {mpc_params.T}")

        print(f"\n\nRunning CLOSED loop simulation with {label}\n\n")
        results = simulate(ocp_solver, integrator, model_params, x0, Nsim, n_runs=5)
        results = add_total_cost_to_results(results)

        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        results['label'] = label
        results['mpc_params'] = mpc_params
        labels_all.append(label)
        pickle.dump(results, open(results_filename, "wb"))

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


def plot_trajectories():

    X_all = []
    U_all = []
    labels_all = []

    # load
    for setting in SETTINGS:
        label = get_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if results['status'] == 0:
            X_all.append(results['x_traj'])
            U_all.append(results['u_traj'])
            labels_all.append(results['label'])

            # doesnt matter when we unpack this
            X_ref = results['x_ref']
            U_ref = results['u_ref']
            # print
            closed_loop_cost = results['cost_total']
            cpu_min = np.min(results['timings']) * 1e3
            print(f"Simulation {label} min CPU time {cpu_min:.3} s, closed loop cost: {closed_loop_cost:.3e}")
        else:
            print(f"Simulation failed with {label}")

    # compare trajectories
    x_lables_list = ["$p$ [m]", r"$\theta$ [rad/s]", "$v$ [m/s]", r"$\dot{\theta}$", r"cost state [k\$]"]
    u_lables_list = ["$u$ [N]"]
    plot_simulation_result(
        DT_PLANT,
        X_all,
        U_all,
        dummy_mpc_params.umin,
        dummy_mpc_params.umax,
        x_lables_list,
        u_lables_list,
        labels_all,
        title='closed loop',
        X_ref=X_ref,
        U_ref=U_ref,
        linestyle_list=['--', ':', '--', ':'],
        color_list=['C0', 'C0', 'C1', 'C1']
    )


def get_ith_letter(i: int) -> str:
    return chr(ord('A')+i)

def plot_trajectories_paper():

    X_all = []
    U_all = []
    labels_all = []

    color_list = []
    linestyle_list = []
    alpha_list = []
    settings = []

    ## load
    settings = SETTINGS_PAPER_PLOT

    # reference
    linestyle_list += ['-']
    color_list += ['k']
    alpha_list += [0.8]

    # others
    color_list += ['C0']
    linestyle_list += ['--']
    alpha_list += [0.8]

    color_list += ['C1']
    linestyle_list += ['-.']
    alpha_list += [0.8, 1.0]

    linestyle_list += [':']
    color_list += ['C2']
    alpha_list += [0.8]

    for i, setting in enumerate(settings):
        label = get_label_from_setting(setting)
        latex_label = get_latex_label_from_setting(setting).replace(', n_s = 4', '')
        # latex_label = r"\textcircled{" + f"{get_ith_letter(i)}" r"} :" + latex_label
        latex_label = f"{get_ith_letter(i)}: {latex_label}"
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if True: # results['status'] == 0:
            X_all.append(results['x_traj'])
            U_all.append(results['u_traj'])
            labels_all.append(latex_label)

            # doesnt matter when we unpack this
            X_ref = results['x_ref']
            U_ref = results['u_ref']
            print(f"cost state at end is {results['x_traj'][-1][-1]} for {label}")
        else:
            print(f"Simulation failed with {label}")

    # compare trajectories
    x_lables_list = ["$p$ [m]", r"$\theta$ [rad/s]", "$s$ [m/s]", r"$\dot{\theta}$", r"cost state"]
    u_lables_list = ["$u$ [N]"]

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
        idxpx=[4, 0, 1],
        linestyle_list=linestyle_list,
        color_list=color_list,
        alpha_list=alpha_list,
        single_column=True,
        xlabel='$t$ [s]',
        idx_xlogy= [4],
        fig_filename='pendulum_trajectories_paper.pdf')


def pareto_plot_paper():
    # settings = SETTINGS_PARETO + SETTINGS_PAPER_PLOT
    settings = SETTINGS_TABLE
    for setting in SETTINGS_PAPER_PLOT:
        # remove setting from settings
        settings.remove(setting)

    labels_all = []
    cost_all = []
    timings = []
    settings_plot = []
    time_grid_values_plot = ['nonuniform', 'uniform']
    T_horizon_values = [.4, 4.]

    for setting in settings:
        label = get_label_from_setting(setting)
        latex_label = get_latex_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if True: # results['status'] == 0:
            # doesnt matter when we unpack this
            X_ref = results['x_ref']
            U_ref = results['u_ref']
            status = results['status']
            # print(f"cost state at end is {results['x_traj'][-1][-1]} for {label}, status {status}")
            if status != 4:
                # append results
                cost_all.append(results['cost_total'])
                labels_all.append(label)
                timings.append(np.max(1e3*results['timings']))
                # make setting that contains T and time_grid \in [nonuinform, uniform] instead uniform_x
                T_horizon = results['mpc_params'].T
                setting += (T_horizon,)
                t_grid = setting[4]
                if t_grid.startswith('uniform'):
                    t_grid = 'uniform'
                cost_hess_variant = setting[0]
                N_horizon = setting[1]
                setting_plot = (cost_hess_variant, N_horizon, t_grid, T_horizon)

                settings_plot.append(setting_plot)
        else:
            print(f"Simulation failed with {label}")

    best_cost = min(cost_all)
    rel_subopt = [((c - best_cost) / best_cost) * 100 for c in cost_all]
    points = list(zip(rel_subopt, timings))

    # add special points
    special_points = []
    special_labels = []
    for i, setting in enumerate(SETTINGS_PAPER_PLOT):
        label = get_label_from_setting(setting)
        latex_label = get_latex_label_from_setting(setting).replace(', n_s = 4', '')
        # latex_label = r"\textcircled{" + f"{get_ith_letter(i)}" r"} :" + latex_label
        latex_label = f"{get_ith_letter(i)}: {latex_label}"
        special_labels.append(latex_label)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        c = results['cost_total']
        timing = np.max(1e3*results['timings'])
        rel_subopt = ((c - best_cost) / best_cost) * 100
        special_points.append((rel_subopt, timing))

    # plot_pareto(points, settings_plot, [COST_DISCRETIZATION_VARIANTS, N_VALUES, time_grid_values_plot, T_horizon_values], [0, 2, 1, 3], labels_all, fig_filename='pendulum_gnrk_pareto.pdf', special_points=special_points, special_labels=special_labels, xlabel='relative suboptimality [\%]', ylabel='max computation time [ms]')

    plot_simple_pareto(points, fig_filename='pendulum_gnrk_pareto.pdf', special_points=special_points, special_labels=special_labels, xlabel='relative suboptimality [\%]', ylabel='max computation time [ms]')



def fstring_formater(double, it_decimal, max_char:int =4):
    st_float = f"{double:.{it_decimal}f}"
    # if len(st_float) > max_char:
    #     st_float = f"{double:.0f}"
    return st_float

def closed_loop_cost_table_paper():
    labels_all = []
    cost_all = []
    N_all = []
    T_all = []
    ns_all = []
    rti_all = []
    uniform_all = []
    cost_disc_all = []
    iter_all = []
    cpu1_all = []
    cpu2_all = []
    in_plot = []
    settings = SETTINGS_TABLE

    n_variants = len(settings)
    for setting in settings:
        label = get_label_from_setting(setting)
        latex_label = get_latex_label_from_setting(setting)
        results_filename = get_results_filename(label, dummy_model.name, DT_PLANT, closed_loop=True)
        with open(results_filename, 'rb') as f:
            results = pickle.load(f)
        if results['status'] == 0:
            cost_all.append(results['cost_total'])
            cpu1_all.append(np.min(results['timings']) * 1e3)
            cpu2_all.append(np.max(results['timings']) * 1e3)
            iter_all.append(results['nlp_iter'])

            labels_all.append(latex_label)
            cost_disc_all.append(setting[0])
            N_all.append(setting[1])
            ns_all.append(setting[2])
            rti_all.append(setting[3])
            uniform_all.append(setting[4])
            T_all.append(results['mpc_params'].T)

            if setting in SETTINGS_PAPER_PLOT:
                i_plot = SETTINGS_PAPER_PLOT.index(setting)
                in_plot.append(get_ith_letter(i_plot))
            else:
                in_plot.append(" ")

    best_cost = min(cost_all)
    best_cpu1 = min(cpu1_all)
    best_cpu2 = min(cpu2_all)
    # print(r"cost disc. & $N$ & RTI & unif. & rel. subopt. & $t_{\mathrm{cpu}}$ [ms] \\ \midrule")

    table_string = ""
    table_string += r"\begin{tabular}{cccccrrrrrc}" + "\n"
    table_string += r"\toprule" + "\n"
    table_string += r"\begin{tabular}{@{}c@{}}Hessian approximation \\ and cost discretization \end{tabular}"
    table_string += r"& $N$ & $T [\mathrm{s}]$ & RTI & uniform & rel. subopt. & max $n\ind{iter}$ & median $n\ind{iter}$ & $t_{\min} [\mathrm{ms}]$ & $t_{\max} [\mathrm{ms}]$ & in Fig.~\ref{pendulum_GNRK_trajectories}\\ \midrule"
    for i in range(n_variants):
        # print(f"{label:50} {cost_total:.2f}")
        cost_total = cost_all[i]
        rel_subopt = ((cost_total - best_cost) / best_cost) * 100

        line_str = f"{cost_disc_all[i]}"
        line_str += f" & {N_all[i]} & {fstring_formater(T_all[i], 1)} & {'x' if rti_all[i] else ''}"
        line_str += f" & {'x' if uniform_all[i].startswith('uniform') else ''} & "
        if rel_subopt < 10:
            line_str += r"\textbf{"
            line_str += f"{fstring_formater(rel_subopt,1)}"
            line_str += "} \% & "
        else:
            line_str += f"{fstring_formater(rel_subopt,1)} \% & "

        line_str += f"{int(max(iter_all[i])):d} &"
        line_str += f"{np.mean(iter_all[i]):.2f} &"

        rel_cpu1 = ((cpu1_all[i] - best_cpu1) / best_cpu1) * 100
        if rel_cpu1 < 10.:
            line_str += r"\textbf{"
            line_str += f"{fstring_formater(cpu1_all[i], 1)}"
            line_str += "}"
        else:
            line_str += f"{fstring_formater(cpu1_all[i], 1)}"

        rel_cpu2 = ((cpu2_all[i] - best_cpu2) / best_cpu2) * 100
        if rel_cpu2 < 10.:
            line_str += r"& \textbf{"
            line_str += f"{fstring_formater(cpu2_all[i], 1)}"
            line_str += "}"
        else:
            line_str += f"& {fstring_formater(cpu2_all[i], 1)}"

        line_str += f'& {in_plot[i]}'

        line_str += f"\\\\"
        if i > 0 and i < n_variants-1 and (N_all[i] != N_all[i-1]):
            line_str = r'\midrule ' + line_str

        # print(line_str)
        table_string += f"\n{line_str}"
    table_string += "\n\\bottomrule \n"
    table_string += r"\end{tabular}"
    print(table_string)

    table_filename = "table_pendulum.tex"
    table_filename = os.path.join(os.getcwd(), TEX_FOLDER, table_filename)

    with open(table_filename, "w") as file:
        file.write(table_string)

    # make table for slides:
    table_string_slides = table_string
    table_string_slides = table_string_slides.replace("Hessian approximation", "Hess. approx.")
    table_string_slides = table_string_slides.replace("and cost discretization", "\& cost discr.")
    table_string_slides = table_string_slides.replace("uniform", "unif.")
    table_string_slides = table_string_slides.replace(r"in Fig.~\ref{pendulum_GNRK_trajectories}", "ID")
    table_string_slides = table_string_slides.replace(r"median $n\ind{iter}$", r"\begin{tabular}{@{}c@{}}median \\ $n\ind{iter}$ \end{tabular}")

    table_slides_filename = "table_pendulum_slides.tex"
    table_slides_filename = os.path.join(os.getcwd(), TEX_FOLDER, table_slides_filename)

    with open(table_slides_filename, "w") as file:
        file.write(table_string_slides)

if __name__ == "__main__":
    PRODUCE_RESULTS = True
    PRODUCE_RESULTS = False
    if PRODUCE_RESULTS:
        run_pendulum_benchmark_closed_loop(SETTINGS_TABLE)
        # run_pendulum_benchmark_closed_loop(ALL_SETTINGS)
    plot_trajectories_paper()
    closed_loop_cost_table_paper()
    pareto_plot_paper()

    # plot_trajectories()
