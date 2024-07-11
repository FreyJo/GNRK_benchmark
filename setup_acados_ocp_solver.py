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

from acados_template import AcadosModel, AcadosOcp, AcadosMultiphaseOcp, AcadosOcpSolver, casadi_length, is_empty
from scipy.linalg import block_diag
import numpy as np
import casadi as ca
from casadi import vertcat

from typing import Optional

from models import ModelParameters, substitute_u_with_squashed_lqr, substitute_u_with_lqr, substitute_u_with_saturated_lqr
from utils import Reference, get_nbx_violation_expression, compute_lqr_gain_continuous_time
from mpc_parameters import MpcParameters
from copy import deepcopy

OCP_SOLVER_NUMBER = 0

def augment_model_with_clock_state(model: AcadosModel):

    t = ca.SX.sym('t')
    tdot = ca.SX.sym('tdot')

    model.x = ca.vertcat(model.x, t)
    model.xdot = ca.vertcat(model.xdot, tdot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, 1)
    model.f_impl_expr = model.f_expl_expr - model.xdot

    model.clock_state = t

    return model


def augment_model_with_cost_state(model: AcadosModel, params: ModelParameters, mpc_params: MpcParameters):

    cost_state = ca.SX.sym('cost_state')
    cost_state_dot = ca.SX.sym('cost_state_dot')

    x_ref = ca.SX.sym('x_ref', params.nx_original)
    u_ref = ca.SX.sym('u_ref', params.nu_original)
    xdiff = x_ref - model.x[:params.nx_original]
    udiff = u_ref - model.u[:params.nu_original]

    cost_state_dyn = .5 * (xdiff.T @ mpc_params.Q @xdiff + udiff.T @mpc_params.R @udiff)
    if mpc_params.lbx is not None:
        nbx = mpc_params.lbx.size
        # formulate as cost!
        x = model.x
        violation_expr = get_nbx_violation_expression(model, mpc_params)
        cost_state_dyn += 0.5 * (violation_expr.T @ mpc_params.gamma_penalty*np.eye(nbx) @ violation_expr)
    if hasattr(model, 'barrier'):
        cost_state_dyn += model.barrier

    model.x = ca.vertcat(model.x, cost_state)
    params.cost_state_idx = casadi_length(model.x) - 1
    params.cost_state_dyn_fun = ca.Function('cost_state_dyn_fun', [model.x, model.u, x_ref, u_ref], [cost_state_dyn])
    model.xdot = ca.vertcat(model.xdot, cost_state_dot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, cost_state_dyn)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.p = ca.vertcat(model.p, x_ref, u_ref)

    params.parameter_values = np.concatenate((params.parameter_values, np.zeros(params.nx_original + params.nu_original)))
    params.xs = np.append(params.xs, [0.0])

    params.xlabels = params.xlabels + ['cost_state']

    return model



def augment_model_with_picewise_linear_u(model: AcadosModel, model_params: ModelParameters, mpc_params: MpcParameters):

    model = augment_model_with_clock_state(model)
    nu = casadi_length(model.u)
    # new controls
    u_0 = ca.SX.sym('u_0', nu)
    u_1 = ca.SX.sym('u_1', nu)
    mpc_params.umin = np.concatenate((mpc_params.umin, mpc_params.umin))
    mpc_params.umax = np.concatenate((mpc_params.umax, mpc_params.umax))
    # parameters
    clock_state0 = ca.SX.sym('clock_state0', 1)
    delta_t_n = ca.SX.sym('delta_t_n', 1)
    model.p = ca.vertcat(model.p, clock_state0, delta_t_n)

    tau = (model.clock_state - clock_state0) / delta_t_n

    u_pwlin = u_0 * (1 - tau) + u_1 * tau
    model.f_expl_expr = ca.substitute(model.f_expl_expr, model.u, u_pwlin)
    model.f_impl_expr = ca.substitute(model.f_impl_expr, model.u, u_pwlin)
    model.cost_y_expr = ca.substitute(model.cost_y_expr, model.u, u_pwlin)

    model_params.xs = np.append(model_params.xs, [0.0])
    model_params.us = np.concatenate((model_params.us, model_params.us))
    model_params.parameter_values = np.concatenate((model_params.parameter_values, [0.0, 0.0]))

    model.u = ca.vertcat(u_0, u_1)
    return model, model_params, mpc_params


def setup_acados_ocp_without_options(model: AcadosModel, model_params: ModelParameters, mpc_params: MpcParameters, cost_type="CONL") -> AcadosOcp:

    ocp = AcadosOcp()

    # set model
    ocp.model = model
    x = model.x
    u = model.u
    nx = x.shape[0]
    nu = u.shape[0]

    # set cost
    ocp.cost.W_e = mpc_params.P
    ocp.cost.W = block_diag(mpc_params.Q, mpc_params.R)
    if model.cost_y_expr is not None:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.yref = np.zeros((casadi_length(ocp.model.cost_y_expr),))
        ocp.cost.yref = np.concatenate((model_params.xs, model_params.us))
    else:
        ocp.cost.cost_type = "EXTERNAL"

    if model.cost_y_expr_e is not None:
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.yref_e = np.zeros((casadi_length(ocp.model.cost_y_expr_e),))
        ocp.cost.yref_e = model_params.xs
    else:
        ocp.cost.cost_type_e = "EXTERNAL"

    nx = casadi_length(model.x)
    nxs = model_params.xs.size
    ocp.constraints.x0 = np.hstack((model_params.xs, np.zeros(nx-nxs)))

    # set constraints
    if mpc_params.umin is not None:
        ocp.constraints.lbu = mpc_params.umin
        ocp.constraints.ubu = mpc_params.umax
        ocp.constraints.idxbu = np.arange(nu)

    # formulate constraint violation as cost!
    if mpc_params.lbx is not None and ocp.cost.cost_type != "EXTERNAL":
        nbx = mpc_params.lbx.size
        # ocp.constraints.lbx = mpc_params.lbx
        # ocp.constraints.ubx = mpc_params.ubx
        # ocp.constraints.idxbx = mpc_params.idxbx

        violation_expression = get_nbx_violation_expression(model, mpc_params)
        ocp.model.cost_y_expr = vertcat(ocp.model.cost_y_expr, violation_expression)
        ocp.cost.yref = np.concatenate((ocp.cost.yref, np.zeros((nbx))))
        ocp.cost.W = block_diag(ocp.cost.W, mpc_params.gamma_penalty*np.eye(nbx))

        # ocp.model.cost_y_expr = vertcat(violation_expression, ocp.model.cost_y_expr)
        # ocp.cost.yref = np.concatenate((np.zeros((nbx)), ocp.cost.yref))
        # ocp.cost.W = block_diag(mpc_params.gamma_penalty*np.eye(nbx), ocp.cost.W)

        ocp.model.cost_y_expr_e = vertcat(ocp.model.cost_y_expr_e, violation_expression)
        ocp.cost.yref_e = np.concatenate((ocp.cost.yref_e, np.zeros((nbx))))
        ocp.cost.W_e = block_diag(ocp.cost.W_e, mpc_params.gamma_penalty*np.eye(nbx))

    if cost_type == "CONL":
        ocp.translate_nls_cost_to_conl()

    if isinstance(model.p, list):
        ocp.parameter_values = np.zeros(0)
    else:
        ocp.parameter_values = np.zeros(model.p.rows())

    return ocp

# def replace_u_bounds_with_penalty(ocp: AcadosOcp, penalty=1e6):
#     for i, idx in enumerate(ocp.constraints.idxbu.tolist()):
#         ocp.formulate_constraint_as_L2_penalty(ocp.model.u[idx], penalty, ocp.constraints.ubu[i], ocp.constraints.lbu[i], f'u_bound_{i}')
#     ocp.constraints.idxbu = np.array([])
#     ocp.constraints.ubu = np.array([])
#     ocp.constraints.lbu = np.array([])
#     return


def compute_linear_constraints_for_polynomial_control(dt, n_constraints, nu_original, d, nx, umin, umax):
    # NOTE: polynom is v(t) = sum_{i=0}^d u_i * (t)^i.
    nu_pol = nu_original * (d+1)
    points = np.linspace(0, 1, n_constraints) * dt
    new_C = np.zeros((n_constraints * nu_original, nx))
    new_D = np.zeros((n_constraints * nu_original, nu_pol))
    for ip, point in enumerate(points):
        coeff = np.array([point ** j for j in range(d + 1)])
        for j in range(nu_original):
            row = np.zeros(nu_original*(d+1))
            row[j*(d+1):(j+1)*(d+1)] = coeff
            new_D[ip*nu_original+j, :] = row
    new_lg = np.tile(umin, n_constraints)
    new_ug = np.tile(umax, n_constraints)
    return new_C, new_D, new_lg, new_ug


def setup_acados_ocp_solver(
    model: AcadosModel, model_params, mpc_params: MpcParameters, use_rti=False, reference:Optional[Reference]=None, time_grid:str="nonuniform",
    nlp_tol=1e-6,
    levenberg_marquardt=1e-4,
    nlp_solver_max_iter=100,
    hessian_approx='GAUSS_NEWTON',
    exact_hess_dyn=True,
    regularize_method = 'NO_REGULARIZE',
    degree_u_polynom=None,
    u_polynom_constraints: Optional[int] = None,
    N_clc: int = None,
    T_clc: float = None,
    cost_type="CONL",
):

    if time_grid == "nonuniform":
        time_steps = np.array([mpc_params.dt]+ (mpc_params.N-1)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N-1)])
        # time_steps = np.array([mpc_params.dt] +
        #                                          int(mpc_params.N/2-1)*[(0.5 - mpc_params.dt)/ int(mpc_params.N/2-1)] +
        #                                         int(mpc_params.N/2) * [(mpc_params.T - 0.5 - mpc_params.dt)/ int(mpc_params.N/2)])
    elif time_grid == "uniform_long":
        time_steps = np.array((mpc_params.N)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N)])
    elif time_grid == "uniform_short":
        mpc_params.T = mpc_params.dt * mpc_params.N
        time_steps = np.array((mpc_params.N)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N)])
    elif time_grid == "nonuniform_long_end":
        # (mpc_params.N-2)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N-1)]
        horizon_length_1N = (mpc_params.T - mpc_params.dt)
        last_dt = horizon_length_1N / 2
        middle_dt = last_dt / (mpc_params.N-2)
        time_steps = np.array([mpc_params.dt] + (mpc_params.N-2) * [middle_dt] + [last_dt])
    elif time_grid == "clc_grid":
        if N_clc is None or T_clc is None:
            raise Exception('T_clc and N_clc need to be provided for clc_grid!')
        N_1 = mpc_params.N - 1 - N_clc
        dt_1 = (mpc_params.T - mpc_params.dt - T_clc)/N_1
        time_steps = np.array([mpc_params.dt] + N_1 * [dt_1] + N_clc * [T_clc/N_clc])

    dummy_t = ca.SX.sym('t')
    eval_u_funs = [ca.Function('eval_u_0', [model.u, dummy_t], [model.u]), None]
    if degree_u_polynom == 0:
        ocp = setup_acados_ocp_without_options(model, model_params, mpc_params, cost_type=cost_type)
    else:
        N_pwconst = 1
        ocp = AcadosMultiphaseOcp(N_list=[N_pwconst, mpc_params.N-N_pwconst])
        # TODO: why is this necessary?
        global OCP_SOLVER_NUMBER
        ocp.name = f'mocp_{OCP_SOLVER_NUMBER}'
        OCP_SOLVER_NUMBER += 1

        for i, d in enumerate([0, degree_u_polynom]):
            phase_model = deepcopy(model)

            phase = setup_acados_ocp_without_options(phase_model, model_params, mpc_params)
            if i > 0:
                # replace_u_bounds_with_penalty(phase, penalty=1e6)
                eval_u_funs[i] = phase_model.reformulate_with_polynomial_control(d)
                # set general constraints:
                # NOTE: these constraints are only correct if phase time grid is uniform.
                dt = (mpc_params.T - mpc_params.dt) / (mpc_params.N-1)

                nu_original = phase_model.nu_original
                nx = casadi_length(phase_model.x)
                new_C, new_D, new_lg, new_ug = \
                    compute_linear_constraints_for_polynomial_control(dt,
                                    u_polynom_constraints, nu_original,
                                    d, nx, mpc_params.umin, mpc_params.umax)

                phase.add_linear_constraint(new_C, new_D,
                                            new_lg, new_ug)

                # remove u bounds
                phase.constraints.idxbu = np.array([])
                phase.constraints.lbu = np.array([])
                phase.constraints.ubu = np.array([])

            if not is_empty(phase_model.p):
                phase.parameter_values = np.zeros(casadi_length(phase_model.p))
            print(f"{phase.parameter_values=} {phase_model.p=}")
            ocp.set_phase(phase, i)

    # set options
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_DAQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = mpc_params.N  # for partial condensing

    # number of shooting intervals
    if isinstance(ocp, AcadosOcp):
        ocp.dims.N = mpc_params.N

    ocp.solver_options.time_steps = time_steps

    # set prediction horizon
    ocp.solver_options.tf = sum(ocp.solver_options.time_steps)

    ocp.solver_options.hessian_approx = hessian_approx
    if use_rti:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    else:
        ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP

    if mpc_params.linear_mpc:
        ocp.solver_options.integrator_type = "DISCRETE"
    else:
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = mpc_params.sim_method_num_stages
        ocp.solver_options.sim_method_num_steps = np.array([mpc_params.sim_method_num_steps_0] + (mpc_params.N-1)*[mpc_params.sim_method_num_steps])
        ocp.solver_options.sim_method_newton_iter = 3
        # ocp.solver_options.sim_method_newton_tol = 1e-6
        ocp.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        # ocp.solver_options.collocation_type = "EXPLICIT_RUNGE_KUTTA"

    if mpc_params.cost_integration:
        ocp.solver_options.cost_discretization = 'INTEGRATOR'

    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.levenberg_marquardt = levenberg_marquardt
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.qp_tol = 1e-1 * nlp_tol
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    # ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_ext_qp_res = 1

    ocp.solver_options.regularize_method = regularize_method
    ocp.solver_options.reg_epsilon = 1e-8
    if hessian_approx == 'EXACT':
        ocp.solver_options.regularize_method = 'PROJECT'

    # create
    ocp_solver = AcadosOcpSolver(ocp, json_file=f"acados_ocp_{ocp.solver_options.cost_discretization}_{time_grid}.json", verbose=True)

    # fix linear u constraints wrt changing dt
    if degree_u_polynom > 0 and time_grid != "nonuniform":
        for i, dt in enumerate(time_steps[N_pwconst:]):
            new_C, new_D, new_lg, new_ug = \
                compute_linear_constraints_for_polynomial_control(dt,
                            u_polynom_constraints, nu_original,
                            d, nx, mpc_params.umin, mpc_params.umax)
            ocp_solver.constraints_set(i+N_pwconst, 'lg', new_lg, api='new')
            ocp_solver.constraints_set(i+N_pwconst, 'ug', new_ug, api='new')
            ocp_solver.constraints_set(i+N_pwconst, 'C', new_C, api='new')
            ocp_solver.constraints_set(i+N_pwconst, 'D', new_D, api='new')

    return ocp_solver, eval_u_funs



def setup_acados_ocp_solver_clc(
    model: AcadosModel, model_params, mpc_params: MpcParameters, use_rti=False, reference:Optional[Reference]=None, time_grid:str="nonuniform",
    nlp_tol=1e-6,
    levenberg_marquardt=1e-4,
    nlp_solver_max_iter=400,
    hessian_approx='GAUSS_NEWTON',
    exact_hess_dyn=True,
    regularize_method = 'NO_REGULARIZE',
    T_clc=0.0,
    squash_u=True,
    barrier_type='progressive',
    N_clc=1,
    name = None,
):
    dummy_t = ca.SX.sym('t')

    ocp = AcadosMultiphaseOcp(N_list=[mpc_params.N-N_clc, N_clc])
    # TODO: why is this necessary?
    global OCP_SOLVER_NUMBER
    ocp.name = f'mocp_{OCP_SOLVER_NUMBER}'
    OCP_SOLVER_NUMBER += 1

    # set first phase
    i_phase = 0
    phase_model = deepcopy(model)
    phase = setup_acados_ocp_without_options(phase_model, model_params, mpc_params)
    ocp.set_phase(phase, i_phase)

    # set second phase (squashed LQR)
    i_phase = 1
    phase_model = deepcopy(model)

    P_mat, K_mat = compute_lqr_gain_continuous_time(phase_model, model_params, mpc_params)

    if squash_u:
        phase_model, u_fun = substitute_u_with_squashed_lqr(phase_model, mpc_params, K_mat)
        # phase_model, u_fun = substitute_u_with_saturated_lqr(phase_model, mpc_params, K_mat)
    else:
        phase_model, u_fun = substitute_u_with_lqr(phase_model, mpc_params, K_mat)

    # eval_u_funs = [ca.Function('eval_u_0', [model.x, model.u], [model.u]), u_fun]
    eval_u_funs = [None, u_fun]

    u_expr = u_fun(phase_model.x)

    # remove u bounds
    u_lqr_mpc_params = deepcopy(mpc_params)
    u_lqr_mpc_params.umin = None
    u_lqr_mpc_params.umax = None
    phase = setup_acados_ocp_without_options(phase_model, model_params, u_lqr_mpc_params)

    phase.translate_nls_cost_to_conl()

    # add barrier
    if barrier_type not in ['constant', 'progressive', 'None']:
        raise ValueError(f"barrier_type {barrier_type} not known")

    if barrier_type in ['progressive', 'constant']:
        nu = casadi_length(model.u)
        new_r_in_psi_expr = ca.SX.sym('new_r_in_psi_expr', nu)
        u_center = (mpc_params.umax + mpc_params.umin)/2
        u_width_half = (mpc_params.umax - mpc_params.umin)/2
        phase.model.cost_y_expr = vertcat(phase.model.cost_y_expr, (u_expr - u_center) / (u_width_half))
        phase.model.cost_r_in_psi_expr = vertcat(phase.model.cost_r_in_psi_expr, new_r_in_psi_expr)
        normalized_u = (1-1e-8) * (new_r_in_psi_expr)
        t = ca.SX.sym('t', 1)
        phase.model.t = t
        if barrier_type == 'progressive':
            weighting = 1e4 * ca.exp(ca.log(10.0) * t)
        elif barrier_type == 'constant':
            weighting = 1e4 * ca.exp(ca.log(10.0) * (mpc_params.T - T_clc))

        for iu in range(nu):
            phase.model.cost_psi_expr = phase.model.cost_psi_expr + \
                                    weighting * (-ca.log((1 - normalized_u[iu])) - ca.log((1 + normalized_u[iu])))
        phase.cost.yref = np.concatenate((phase.cost.yref, np.zeros((nu,))))

    elif barrier_type == "penalty":
        # TODO here we assume a lot of knowledge of the cost function!!
        u_expr = u_expr + 20*(ca.fmax(u_expr - mpc_params.umax, 0) + ca.fmax(mpc_params.umin - u_expr, 0))
        phase.model.cost_y_expr[phase.model.x.rows()] = u_expr
    ocp.set_phase(phase, i_phase)

    # set options
    # ocp.solver_options.qp_solver = "FULL_CONDENSING_DAQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = mpc_params.N  # for partial condensing

    # number of shooting intervals
    # ocp.dims.N = mpc_params.N
    if time_grid == "nonuniform":
        N_planning = mpc_params.N - N_clc
        dt_clc = T_clc/N_clc
        ocp.solver_options.time_steps = np.array([mpc_params.dt] +
                                                 (N_planning-1)*[(mpc_params.T - T_clc - mpc_params.dt)/ (N_planning-1)]
                                                 + N_clc * [dt_clc])
    else:
        breakpoint()
        raise NotImplementedError("if degree_u_polynom is not None, dt must be known to set constraints on u")

    # set prediction horizon
    ocp.solver_options.tf = sum(ocp.solver_options.time_steps)

    ocp.solver_options.hessian_approx = hessian_approx
    if use_rti:
        ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    else:
        ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP

    if mpc_params.linear_mpc:
        ocp.solver_options.integrator_type = "DISCRETE"
    else:
        ocp.solver_options.integrator_type = "IRK"
        ocp.solver_options.sim_method_num_stages = mpc_params.sim_method_num_stages
        # ocp.solver_options.sim_method_num_steps = [mpc_params.sim_method_num_steps] * (mpc_params.N-1) + [10]
        ocp.solver_options.sim_method_num_steps = np.array([mpc_params.sim_method_num_steps_0] + (mpc_params.N-1)*[mpc_params.sim_method_num_steps])
        ocp.solver_options.sim_method_newton_iter = 3
        # ocp.solver_options.sim_method_newton_tol = 1e-6
        ocp.solver_options.collocation_type = "GAUSS_RADAU_IIA"
        # ocp.solver_options.collocation_type = "EXPLICIT_RUNGE_KUTTA"

    if mpc_params.cost_integration:
        ocp.solver_options.cost_discretization = 'INTEGRATOR'

    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.levenberg_marquardt = levenberg_marquardt
    ocp.solver_options.tol = nlp_tol
    ocp.solver_options.qp_tol = 1e-1 * nlp_tol
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    # ocp.solver_options.print_level = 1
    # ocp.solver_options.nlp_solver_ext_qp_res = 1

    ocp.solver_options.regularize_method = regularize_method
    ocp.solver_options.reg_epsilon = 1e-8
    if hessian_approx == 'EXACT':
        ocp.solver_options.regularize_method = 'PROJECT'

    # create
    ocp_solver = AcadosOcpSolver(ocp, json_file=f"acados_ocp_{ocp.solver_options.cost_discretization}.json", verbose=True)

    return ocp_solver, eval_u_funs


if __name__ == "__main__":
    setup_acados_ocp_solver()
