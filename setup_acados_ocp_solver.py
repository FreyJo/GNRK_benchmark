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

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, casadi_length
from scipy.linalg import block_diag
import numpy as np
import casadi as ca
from dataclasses import dataclass
from casadi import vertcat
from utils import Reference, get_nbx_violation_expression
from typing import Optional


@dataclass
class MpcParameters:
    umin: np.ndarray  # lower bound on u
    umax: np.ndarray  # upper bound on u
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    us: np.ndarray
    xs: np.ndarray
    # Tf: float = 0.25 * 13  # horizon length
    N: int
    dt: float
    linear_mpc: bool = False
    cost_integration: bool = False
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    sim_method_num_steps_0: int = 1
    lbx: Optional[np.ndarray] = None
    ubx: Optional[np.ndarray] = None
    idxbx: Optional[np.ndarray] = None
    gamma_penalty: float = 1.0


@dataclass
class MpcPendulumParameters(MpcParameters):

    def __init__(self, xs, us, dt=0.25, linear_mpc=False, N=16, Tf=4):
        self.Q = 2*np.diag([1e2, 1e3, 1e-2, 1e-2])
        # self.R = 2*np.diag([1e-0])
        self.R = 2*np.diag([2e-1])
        self.us = us
        self.xs = xs
        self.umin = -np.array([40.0])
        self.umax = np.array([40.0])
        #
        # NOTE: computed with setup_linearized_model()
        self.P = np.array([[ 215.86822404, -294.53515325,  114.32904314,  -98.65929707],
                           [-294.53515325,  872.1412253 , -216.25863682,  216.67703862],
                           [ 114.32904314, -216.25863682,   92.23297629,  -81.47144968],
                           [ -98.65929707,  216.67703862,  -81.47144968,   75.74192616]]

            )
        self.N = 20
        self.dt = 0.02
        self.T = 4.0
        self.sim_method_num_steps = 1
        self.sim_method_num_steps_0 = 1
        # constraints
        self.lbx = np.array([-1.0])
        self.ubx = np.array([1.0])
        self.idxbx = np.array([0])
        self.gamma_penalty = 1e5


def setup_acados_ocp_solver(
    model: AcadosModel, model_params, mpc_params: MpcParameters, use_rti=False, reference:Optional[Reference]=None, time_grid:str="nonuniform",
    nlp_tol=1e-7,
    levenberg_marquardt=1e-4,
    nlp_solver_max_iter=400,
    hessian_approx='GAUSS_NEWTON',
    exact_hess_dyn=True,
    regularize_method = 'NO_REGULARIZE'
):

    ocp = AcadosOcp()

    # set model
    ocp.model = model
    x = model.x
    u = model.u
    nx = x.shape[0]
    nu = u.shape[0]

    # number of shooting intervals
    ocp.dims.N = mpc_params.N
    if time_grid == "nonuniform":
        ocp.solver_options.time_steps = np.array([mpc_params.dt]+ (mpc_params.N-1)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N-1)])
    elif time_grid == "uniform_long":
        ocp.solver_options.time_steps = np.array((mpc_params.N)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N)])
    elif time_grid == "uniform_short":
        mpc_params.T = mpc_params.dt * mpc_params.N
        ocp.solver_options.time_steps = np.array((mpc_params.N)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N)])
    elif time_grid == "nonuniform_long_end":
        # (mpc_params.N-2)*[(mpc_params.T - mpc_params.dt)/ (mpc_params.N-1)]
        horizon_length_1N = (mpc_params.T - mpc_params.dt)
        last_dt = horizon_length_1N / 2
        middle_dt = last_dt / (mpc_params.N-2)
        ocp.solver_options.time_steps = np.array([mpc_params.dt] + (mpc_params.N-2) * [middle_dt] + [last_dt])


    # set prediction horizon
    ocp.solver_options.tf = sum(ocp.solver_options.time_steps)

    # set cost
    ocp.cost.W_e = mpc_params.P
    ocp.cost.W = block_diag(mpc_params.Q, mpc_params.R)

    if model.cost_y_expr is not None:
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.yref = np.zeros((casadi_length(ocp.model.cost_y_expr),))
    else:
        ocp.cost.cost_type = "EXTERNAL"

    if model.cost_y_expr_e is not None:
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.yref_e = np.zeros((casadi_length(ocp.model.cost_y_expr_e),))
    else:
        ocp.cost.cost_type_e = "EXTERNAL"

    nx = casadi_length(model.x)
    nxs = model_params.xs.size
    ocp.constraints.x0 = np.hstack((model_params.xs, np.zeros(nx-nxs)))

    # set constraints
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

    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP
    # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.qp_solver_cond_N = mpc_params.N  # for partial condensing

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
        ocp.solver_options.sim_method_num_steps = mpc_params.sim_method_num_steps
        ocp.solver_options.sim_method_num_steps = np.array([mpc_params.sim_method_num_steps_0] + (mpc_params.N-1)*[mpc_params.sim_method_num_steps])
        ocp.solver_options.sim_method_newton_iter = 20
        ocp.solver_options.sim_method_newton_tol = 1e-10
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
        # ocp.solver_options.exact_hess_dyn = exact_hess_dyn


    if isinstance(model.p, list):
        ocp.parameter_values = np.zeros(0)
    else:
        ocp.parameter_values = np.zeros(model.p.rows())

    # create
    ocp_solver = AcadosOcpSolver(ocp, json_file=f"acados_ocp_{ocp.solver_options.cost_discretization}.json", verbose=True)

    return ocp_solver


if __name__ == "__main__":
    setup_acados_ocp_solver()
