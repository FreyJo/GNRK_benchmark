import numpy as np
import itertools, os
import json
import pickle

from setup_acados_ocp_solver import (
    MpcPendulumParameters,
    setup_acados_ocp_solver,
    AcadosOcpSolver,
)
from models import setup_pendulum_model, augment_model_with_cost_state, modify_model_to_use_cost_state
from utils import get_label_from_setting, get_results_filename, get_latex_label_from_setting, RESULTS_FOLDER
from acados_template import latexify_plot


import matplotlib.pyplot as plt

dummy_model, dummy_model_params = setup_pendulum_model()
dummy_mpc_params = MpcPendulumParameters(xs=dummy_model_params.xs, us=dummy_model_params.us)
N_ref = int(dummy_mpc_params.T / dummy_mpc_params.dt)
DT_PLANT = dummy_mpc_params.dt

# benchmark stuff
COST_DISCRETIZATION_VARIANTS = ['GNRK', 'GNSN'] #, 'EHSN', 'EHRK']
# COST_DISCRETIZATION_VARIANTS = ['EHRK'] #'EHRK',
# COST_DISCRETIZATION_VARIANTS = ['EHSN'] #'EHRK',
TIME_GRID_VALUES = ["uniform_long"]
NUM_STAGE_VALUES = [4]
N_VALUES = [20]
USE_RTI_VALUES = [False]

THETA_0_VALUES = [np.pi / 4, np.pi/ 8]
THETA_0_STRINGS = [r'\frac{\pi}{4}', r'\frac{\pi}{8}']
# THETA_0_VALUES = [np.pi / 100]
# THETA_0_STRINGS = [r'\frac{\pi}{100}']


def experiment():

    tol = 1e-9

    plant_model, plant_model_params = setup_pendulum_model()

    mpc_params = MpcPendulumParameters(xs=plant_model_params.xs, us=plant_model_params.us)

    labels_all = []
    settings = [s for s in itertools.product(COST_DISCRETIZATION_VARIANTS, N_VALUES, NUM_STAGE_VALUES, USE_RTI_VALUES, TIME_GRID_VALUES)]

    for theta_0 in THETA_0_VALUES:
        for setting in settings:
            label = get_label_from_setting(setting)

            model, model_params = setup_pendulum_model()
            x0 = np.array([0.0, theta_0, 0.0, 0.0])

            mpc_params = MpcPendulumParameters(xs=model_params.xs, us=model_params.us)
            levenberg_marquardt=0.0
            if setting[0] == 'GNRK':
                mpc_params.cost_integration = 1
                hessian_approx = 'GAUSS_NEWTON'
            elif setting[0] == 'GNSN':
                hessian_approx = 'GAUSS_NEWTON'
            elif setting[0] == 'EHRK':
                model = augment_model_with_cost_state(model, model_params, mpc_params)
                hessian_approx = 'EXACT'
                # levenberg_marquardt=1e-2
                model = modify_model_to_use_cost_state(model, model_params)
                x0 = np.concatenate((x0, np.array([0.0])))
            elif setting[0] == 'EHSN':
                hessian_approx = 'EXACT'

            mpc_params.N = setting[1]
            mpc_params.sim_method_num_stages = setting[2]
            _ = setting[3]
            time_grid = setting[4]

            ocp_solver: AcadosOcpSolver = setup_acados_ocp_solver(model, model_params, mpc_params, use_rti=True, time_grid=time_grid, nlp_tol=tol, levenberg_marquardt=levenberg_marquardt, hessian_approx=hessian_approx)
            ocp_solver.constraints_set(0, 'lbx', x0)
            ocp_solver.constraints_set(0, 'ubx', x0)

            iterate = None
            step = None
            step_norms = []
            residuals = []
            # call RTI solver in the loop
            max_iter = 100
            for i in range(max_iter):
                iterate_filename = os.path.join(RESULTS_FOLDER, f"iterate_{i}_{label.replace(' ', '_').replace('=', '_')}.json")
                print(f"{i=}")
                ocp_solver.solve()
                prev_iterate = iterate

                # get current iterate as dict
                ocp_solver.store_iterate(iterate_filename, overwrite=True)
                ocp_solver.print_statistics()
                with open(iterate_filename, 'r') as f:
                    iterate = json.load(f)

                # compute step
                if i > 0:
                    step = dict()
                    step_norm = 0.0
                    for k in iterate:
                        step[k] = np.array(iterate[k]) - np.array(prev_iterate[k])
                        step_norm = max(step_norm, np.max(np.abs(step[k])))

                    step_norms.append(step_norm)
                    residual = np.max(ocp_solver.get_residuals())
                    residuals.append(residual)
                    if residual < tol:
                        break

            results_filename = get_results_filename(label, f'convergence_{theta_0}', DT_PLANT, closed_loop=False)
            results = dict(label = label, residuals = residuals, step_norms = step_norms)
            pickle.dump(results, open(results_filename, "wb"))

            ocp_solver = None

        print(f"ran all experiments with {len(labels_all)} different settings")


def plot_convergence_rates(loglog=False):
    latexify_plot()
    settings = [s for s in itertools.product(COST_DISCRETIZATION_VARIANTS, N_VALUES, NUM_STAGE_VALUES, USE_RTI_VALUES, TIME_GRID_VALUES)]

    nsub = len(THETA_0_VALUES)
    fig, axes = plt.subplots(1, 2, figsize=(5., 2.24), sharex=True, sharey=True)

    n_iter_max = 0
    for isub in range(nsub):
        theta_0 = THETA_0_VALUES[isub]

        for setting in settings:
            label = get_label_from_setting(setting)
            latex_label = get_latex_label_from_setting(setting).replace('uniform', '').replace(', n_s = 4 ', '').replace('  ', ' ')
            results_filename = get_results_filename(label, f'convergence_{theta_0}', DT_PLANT, closed_loop=False)
            with open(results_filename, 'rb') as f:
                results = pickle.load(f)

            n_iter = len(results["step_norms"])
            print(f"{label=}, {n_iter=}")
            n_iter_max = max(n_iter_max, n_iter)
            rates = [(results["step_norms"][i])/results["step_norms"][i-1] for i in range(1, n_iter)]
            axes[isub].plot(range(n_iter-1), rates, label=latex_label)
            # print(results["step_norms"])

        axes[isub].set_title(r'$\theta_0 = ' + THETA_0_STRINGS[isub] + '$' )
        if isub == nsub -1:
            axes[isub].legend()
        # axes[isub].yscale('log')
        axes[isub].set_xlabel('iteration $k$')

        if not loglog:
            # axes[isub].set_xlim([0, n_iter_max])
            axes[isub].set_ylim([0., 1.2])

        if isub == 0:
            axes[isub].set_ylabel(r'emp. contraction rate $\hat{\kappa}_k$')
        axes[isub].grid()
        if loglog:
            axes[isub].set_yscale('log')
            axes[isub].set_xscale('log')

    for isub in range(nsub):
        if not loglog:
            axes[isub].set_xlim([0, n_iter_max])
    plt.tight_layout()


    fig_filename = 'figures/contraction_rate_pendulum.pdf'
    plt.savefig(fig_filename)
    print(f"stored figure as {fig_filename}")
    plt.show()

if __name__ == "__main__":
    # experiment()
    # plot_convergence_rates()
    plot_convergence_rates(loglog=False)
