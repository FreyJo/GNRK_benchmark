import casadi as ca
import numpy as np

from dataclasses import dataclass
from acados_template import AcadosModel, casadi_length

from utils import compute_lqr_gain, get_nbx_violation_expression
from setup_acados_ocp_solver import MpcParameters

from typing import Tuple

@dataclass
class ModelParameters:
    nx_original: int = None
    nu_original: int = None
    xs: np.ndarray = None
    us: np.ndarray = None
    parameter_values : np.ndarray = None
    cost_state_idx: int = None
    cost_state_dyn_fun = None


def augment_model_with_clock_state(model: AcadosModel):

    t = ca.SX.sym('t')
    tdot = ca.SX.sym('tdot')

    model.x = ca.vertcat(model.x, t)
    model.xdot = ca.vertcat(model.xdot, tdot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, 1)
    model.f_impl_expr = model.f_expl_expr - model.xdot

    model.clock_state = t

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

    model.x = ca.vertcat(model.x, cost_state)
    params.cost_state_idx = casadi_length(model.x) - 1
    params.cost_state_dyn_fun = ca.Function('cost_state_dyn_fun', [model.x, model.u, x_ref, u_ref], [cost_state_dyn])
    model.xdot = ca.vertcat(model.xdot, cost_state_dot)
    model.f_expl_expr = ca.vertcat(model.f_expl_expr, cost_state_dyn)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.p = ca.vertcat(model.p, x_ref, u_ref)

    params.parameter_values = np.concatenate((params.parameter_values, np.zeros(params.nx_original + params.nu_original)))
    params.xs = np.append(params.xs, [0.0])

    return model



def modify_model_to_use_cost_state(model: AcadosModel, model_params: ModelParameters) -> AcadosModel:
    model.cost_expr_ext_cost = 0 # model.x[model_params.cost_state_idx]
    model.cost_y_expr = None

    model.cost_expr_ext_cost_e = model.x[model_params.cost_state_idx]
    model.cost_y_expr_e = None

    return model


def setup_linearized_model(model, model_params, mpc_params):
    # linearized dynamics
    A, B, P = compute_lqr_gain(model, model_params, mpc_params)
    f_discrete = (
        model_params.xs + A @ (model.x - model_params.xs) + B @ (model.u - model_params.us)
    )

    model.disc_dyn_expr = f_discrete

    print(f"P_mat {P}")
    return model, model_params


def setup_pendulum_model() -> Tuple[AcadosModel, ModelParameters]:

    model_name = 'pendulum'

    # constants
    M = 1. # mass of the cart [kg] -> now estimated
    m = 0.1 # mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    l = 0.8 # length of the rod [m]

    # set up states & controls
    x1      = ca.SX.sym('x1')
    theta   = ca.SX.sym('theta')
    v1      = ca.SX.sym('v1')
    dtheta  = ca.SX.sym('dtheta')

    x = ca.vertcat(x1, theta, v1, dtheta)

    F = ca.SX.sym('F')
    u = ca.vertcat(F)

    # xdot
    x1_dot      = ca.SX.sym('x1_dot')
    theta_dot   = ca.SX.sym('theta_dot')
    v1_dot      = ca.SX.sym('v1_dot')
    dtheta_dot  = ca.SX.sym('dtheta_dot')

    xdot = ca.vertcat(x1_dot, theta_dot, v1_dot, dtheta_dot)

    # parameters
    p = []

    # dynamics
    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)
    denominator = M + m - m*cos_theta*cos_theta
    f_expl = ca.vertcat(v1,
                     dtheta,
                     (-m*l*sin_theta*dtheta*dtheta + m*g*cos_theta*sin_theta+F)/denominator,
                     (-m*l*cos_theta*sin_theta*dtheta*dtheta + F*cos_theta+(M+m)*g*sin_theta)/(l*denominator)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    # cost
    model.cost_y_expr = ca.vertcat(x, u)
    model.cost_y_expr_e = x

    # Model params
    model_params = ModelParameters(
        nx_original = 4,
        nu_original = 1,
        xs = np.array([0.0, 0.0, 0.0, 0.0]),
        us = np.array([0.0]),
        parameter_values= np.array([]),
    )

    return model, model_params
