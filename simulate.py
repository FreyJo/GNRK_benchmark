from typing import Optional
import numpy as np
from setup_acados_integrator import setup_acados_integrator
from acados_template import (
    AcadosOcpSolver, AcadosSimSolver, AcadosMultiphaseOcp, AcadosOcp
)
from models import ModelParameters


def initialize_controller(controller: AcadosOcpSolver, model_params: ModelParameters, x0: np.ndarray):
    time_steps = controller.acados_ocp.solver_options.time_steps
    if isinstance(controller.acados_ocp, AcadosOcp):
        nx_ocp = controller.acados_ocp.dims.nx
    elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
        nx_ocp = controller.acados_ocp.phases_dims[0].nx

    # append zeros to x0 for
    nx0 = x0.size
    x0_ocp = np.hstack((x0, np.zeros(nx_ocp-nx0)))
    cost_guess = 0.0

    t = 0.0
    T = sum(time_steps)
    controller.set(0, 'x', x0_ocp)
    for i, dt in enumerate(time_steps):
        u_guess = model_params.us
        # linspace in time between x0 and xs
        x_guess = t/T*model_params.xs + (T-t)/T * x0_ocp

        if model_params.cost_state_idx is not None:
            cost_guess += dt * model_params.cost_state_dyn_fun(x_guess, u_guess, model_params.xs[-1], model_params.us).full().item()
            x_guess[model_params.cost_state_idx] = cost_guess
            # print(f"{x_guess=}")

        x_init = x_guess

        controller.set(i+1, 'x', x_init)
        if i<controller.N:
            if isinstance(controller.acados_ocp, AcadosOcp):
                controller.set(i, 'u', u_guess)
            elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
                controller.set(i, 'u', 0*controller.get(i, 'u'))
            t += dt

    # last stage
    controller.set(controller.N, 'x', x_init)

    return


def simulate(
    controller: Optional[AcadosOcpSolver],
    plant: AcadosSimSolver,
    model_params: ModelParameters,
    x0: np.ndarray,
    Nsim: int,
    n_runs = 1
):

    nx = plant.acados_sim.dims.nx
    nu = plant.acados_sim.dims.nu

    if isinstance(controller.acados_ocp, AcadosOcp):
        nx_ocp = controller.acados_ocp.dims.nx
    elif isinstance(controller.acados_ocp, AcadosMultiphaseOcp):
        nx_ocp = controller.acados_ocp.phases_dims[0].nx

    X_sim = np.ndarray((Nsim + 1, nx))
    U_sim = np.ndarray((Nsim, nu))

    X_ref = np.tile(model_params.xs, (Nsim+1, 1))
    U_ref = np.tile(model_params.us, (Nsim, 1))
    timings_solver = np.zeros((Nsim))
    timings_integrator = np.zeros((Nsim))
    nlp_iter = np.zeros((Nsim))

    for irun in range(n_runs):
        if controller is not None:
            controller.reset()
            # initialize_controller(controller, model_params, x0)
            initialize_controller(controller, model_params, model_params.xs)

        # closed loop
        xcurrent = np.concatenate((x0.T, np.zeros((nx-model_params.nx_original))))
        X_sim[0, :] = xcurrent
        for i in range(Nsim):

            # call controller
            if controller is not None:

                x0_bar = xcurrent[:nx_ocp]

                controller.set(0, "lbx", x0_bar)
                controller.set(0, "ubx", x0_bar)

                # solve ocp
                status = controller.solve()

                if status not in [0, 2]:
                    controller.print_statistics()
                    msg = f"acados controller returned status {status} in simulation step {i}."
                    print("warning: " + msg + "\n\n\nEXITING with unfinished simulation.\n\n\n")

                    controller.dump_last_qp_to_json("failing_qp.json", overwrite=True)
                    controller.store_iterate("failing_iterate.json", overwrite=True)
                    return dict(x_traj=X_sim, u_traj=U_sim, timings=timings_solver, x_ref=X_ref, u_ref=U_ref, status=status, nlp_iter=nlp_iter)

                if status == 2:
                    print(f"controller got max iter in simulation step {i}. continuing..")
                    controller.print_statistics()

                U_sim[i, :] = controller.get(0, "u")
                if irun == 0:
                    timings_solver[i] = controller.get_stats("time_tot")
                else:
                    timings_solver[i] = min(controller.get_stats("time_tot"), timings_solver[i])

                nlp_iter[i] = controller.get_stats("sqp_iter")
            else:
                U_sim[i, :] = model_params.us

            # simulate system
            plant.set("x", xcurrent)
            plant.set("u", U_sim[i, :])

            if plant.acados_sim.solver_options.integrator_type == "IRK":
                plant.set("xdot", np.zeros((nx,)))

            status = plant.solve()
            if status != 0:
                raise Exception(
                    f"acados integrator returned status {status} in simulation step {i}. Exiting."
                )

            timings_integrator[i] = plant.get("time_tot")
            # update state
            xcurrent = plant.get("x")
            X_sim[i + 1, :] = xcurrent

    return dict(x_traj=X_sim, u_traj=U_sim, timings=timings_solver, x_ref=X_ref, u_ref=U_ref, status=status, nlp_iter=nlp_iter)
