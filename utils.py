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
import hashlib

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import casadi as ca

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from acados_template import latexify_plot, AcadosModel, casadi_length


RESULTS_FOLDER = "results"
FIGURE_FOLDER = "figures"
TEX_FOLDER = "tex"

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
if not os.path.exists(TEX_FOLDER):
    os.makedirs(TEX_FOLDER)
if not os.path.exists(FIGURE_FOLDER):
    os.makedirs(FIGURE_FOLDER)


KEY_TO_TEX = {
    "cost_hess_variant": "",
    "N": "$N = $",
    "sim_method_num_stages": "$n_s = $",
    "use_rti": "RTI",
    "time_grid": "",
    "degree_u_polynom": "$d = $",
    "u_polynom_constraints": "$n_{\mathrm{pc}} = $",
    "squash_u": "squashed",
    "time_horizon": "$T = $",
    "clc": "CLC",
    "barrier_type": "barrier",
    "cl_grid": "CLC grid"
}


@dataclass
class Reference:
    x_ref: List[np.ndarray] = None
    u_ref: List[np.ndarray] = None
    t_jump: List[float] = None

    def get_y_ref_expr(self, t_expr):
        y_ref_expr = ca.vertcat(self.x_ref[-1], self.u_ref[-1])

        for t_jump, x_ref, u_ref in reversed(
            list(zip(self.t_jump, self.x_ref[:-1], self.u_ref[:-1]))
        ):
            y_ref_expr = ca.if_else(
                t_expr < t_jump, ca.vertcat(x_ref, u_ref), y_ref_expr
            )

        return y_ref_expr

    def get_y_ref_fun(self):
        t_expr = ca.SX.sym("t")
        y_ref_expr = self.get_y_ref_expr(t_expr)

        y_ref_fun = ca.Function("y_ref", [t_expr], [y_ref_expr])
        return y_ref_fun


def plot_simulation_result(
    dt,
    X_list,
    U_list,
    u_min,
    u_max,
    x_labels_list,
    u_labels_list,
    labels_list,
    figsize=None,
    X_ref=None,
    U_ref=None,
    fig_filename=None,
    x_min=None,
    x_max=None,
    title=None,
    idxpx=None,
    idxpu=None,
    color_list=None,
    linestyle_list=None,
    single_column=False,
    alpha_list=None,
    xlabel=None,
    idx_xlogy=None,
    show_legend=True,
    ncol_legend=1,
    bbox_to_anchor=None,
):
    nx = X_list[0].shape[1]
    nu = U_list[0].shape[1]
    Ntraj = len(X_list)

    if idxpx is None:
        idxpx = list(range(nx))
    if idxpu is None:
        idxpu = list(range(nu))

    if color_list is None:
        color_list = [f"C{i}" for i in range(Ntraj)]
    if linestyle_list is None:
        linestyle_list = Ntraj * ["-"]
    if alpha_list is None:
        alpha_list = Ntraj * [0.8]

    if idx_xlogy is None:
        idx_xlogy = []

    if xlabel is None:
        xlabel = "$t$ [min]"

    nxpx = len(idxpx)
    nxpu = len(idxpu)

    nx_original = nx - 2

    Nsim = U_list[0].shape[0]

    ts = dt * np.arange(0, Nsim + 1)

    nrows = max(nxpx, nxpu)

    latexify_plot()
    if single_column:
        if figsize is None:
            figsize=(5.5, 1.65 * (nxpx + nxpu + 1))
        fig, axes = plt.subplots(
            ncols=1,
            nrows=nxpx + nxpu,
            figsize=figsize,
            sharex=True,
        )
    else:
        if figsize is None:
            figsize=(10, (nxpx + nxpu))
        fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=figsize,
            sharex=True)
        axes = np.ravel(axes, order="F")

    if title is not None:
        axes[0].set_title(title)

    for i in idxpx:
        isubplot = idxpx.index(i)
        for X, label, color, linestyle, alpha in zip(
            X_list, labels_list, color_list, linestyle_list, alpha_list
        ):
            axes[isubplot].plot(
                ts, X[:, i], label=label, alpha=alpha, color=color, linestyle=linestyle
            )

        if i < nx_original and X_ref is not None:
            axes[isubplot].step(
                ts,
                X_ref[:, i],
                alpha=0.8,
                where="post",
                label="reference",
                linestyle="dotted",
                color="k",
            )
        axes[isubplot].set_ylabel(x_labels_list[i])
        axes[isubplot].grid()
        axes[isubplot].set_xlim(ts[0], ts[-1])

        if i in idx_xlogy:
            axes[isubplot].set_yscale("log")

        if x_min is not None:
            axes[isubplot].set_ylim(bottom=x_min[i])

        if x_max is not None:
            axes[isubplot].set_ylim(top=x_max[i])

    for i in idxpu:
        for U, label, color, linestyle, alpha in zip(
            U_list, labels_list, color_list, linestyle_list, alpha_list
        ):
            axes[i + nrows].step(
                ts,
                np.append([U[0, i]], U[:, i]),
                label=label,
                alpha=alpha,
                color=color,
                linestyle=linestyle,
            )
        if U_ref is not None:
            axes[i + nrows].step(
                ts,
                np.append([U_ref[0, i]], U_ref[:, i]),
                alpha=0.8,
                label="reference",
                linestyle="dotted",
                color="k",
            )
        axes[i + nrows].set_ylabel(u_labels_list[i])
        axes[i + nrows].grid()

        axes[i + nrows].hlines(
            u_max[i], ts[0], ts[-1], linestyles="dashed", alpha=0.4, color="k"
        )
        axes[i + nrows].hlines(
            u_min[i], ts[0], ts[-1], linestyles="dashed", alpha=0.4, color="k"
        )
        axes[i + nrows].set_xlim(ts[0], ts[-1])
        bound_margin = 0.05
        u_lower = (
            (1 - bound_margin) * u_min[i]
            if u_min[i] > 0
            else (1 + bound_margin) * u_min[i]
        )
        axes[i + nrows].set_ylim(bottom=u_lower, top=(1 + bound_margin) * u_max[i])

    axes[nxpx + nxpu - 1].set_xlabel(xlabel)
    if not single_column:
        axes[nxpx - 1].set_xlabel(xlabel)

    if show_legend:
        if single_column:
            if bbox_to_anchor is None:
                bbox_to_anchor = (0.4, -0.7)
            axes[nxpx + nxpu - 1].legend(
                loc="lower center", ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor
            )  # bbox_to_anchor=(0.5, -1.5), , handlelength=1.)
        else:
            if bbox_to_anchor is None:
                bbox_to_anchor = (1.0, 0.)
            axes[nxpx + nxpu - 1].legend(loc="lower center", ncol=ncol_legend, bbox_to_anchor=bbox_to_anchor)

    # plt.subplots_adjust(
    #     left=None, bottom=None, right=None, top=None, hspace=0.3, wspace=0.4
    # )
    fig.align_ylabels(axes)
    # plt.tight_layout()

    if not single_column:
        for i in range(nxpu, nxpx):
            fig.delaxes(axes[i + nrows])

    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()


def plot_pareto(
    points,
    settings,
    settings_lists,
    idx_setting,
    labels,
    fig_filename: Optional[str],
    special_points=None,
    special_labels=None,
    xlabel=None,
    ylabel=None,
):
    latexify_plot()
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6.6, 6.0))
    cmap = plt.get_cmap("tab10")
    fill_vals = ["full", "none", "top", "left"]
    marker_vals = ["o", "v", "s", "d", "^", ">", "<", "P", "D"]
    n_colors = len(settings_lists[idx_setting[1]])
    color_vals = [cmap(i) for i in range(n_colors)]
    # breakpoint()
    alpha_vals = np.linspace(0.4, 1.0, len(settings_lists[idx_setting[3]]))

    fill_styles = {k: v for k, v in zip(settings_lists[0], fill_vals)}
    markers = {s: c for s, c in zip(settings_lists[1], marker_vals)}
    colors = {s: c for s, c in zip(settings_lists[2], color_vals)}
    alphas = {s: c for s, c in zip(settings_lists[3], alpha_vals)}
    for p, l, setting in zip(points, labels, settings):
        axes.plot(
            p[0],
            p[1],
            fillstyle=fill_styles[setting[idx_setting[0]]],
            color=colors[setting[idx_setting[1]]],
            marker=markers[setting[idx_setting[2]]],
            alpha=alphas[setting[idx_setting[3]]],
            label=l,
        )

    # breakpoint()
    legend_elements = (
        [
            Line2D([0], [0], color=colors[key], lw=4, label=f"{key}")
            for key in settings_lists[2]
        ]
        + [
            Line2D([0], [0], marker=markers[key], lw=0, color="k", label=f"$N = {key}$")
            for key in settings_lists[1]
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                fillstyle=fill_styles[key],
                lw=0,
                label=f"{key}",
            )
            for key in settings_lists[0]
        ]
        + [
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                alpha=alphas[key],
                lw=0,
                label=f"$T = {key}$",
            )
            for key in settings_lists[idx_setting[3]]
        ]
    )

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    axes.set_xscale("symlog")
    axes.set_yscale("log")

    if special_points is not None:
        special_markers = ["o", "v", "s", "d", "^", ">", "<", "P", "D"]
        for p, label, marker in zip(special_points, special_labels, special_markers):
            alpha = 1.0
            fillstyle = "none"
            color = f"C{n_colors}"
            axes.plot(
                p[0],
                p[1],
                color=color,
                marker=marker,
                alpha=alpha,
                markersize=11,
                markeredgewidth=2,
                fillstyle=fillstyle,
                label=l,
            )
            legend_elements += [
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    alpha=alpha,
                    markersize=11,
                    markeredgewidth=2,
                    fillstyle=fillstyle,
                    color=color,
                    lw=0,
                    label=label,
                )
            ]

    plt.grid()
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0)) #  loc='upper right'
    plt.legend(
        handles=legend_elements, bbox_to_anchor=(1.0, -0.15), ncol=3
    )  #  loc='upper right'
    plt.tight_layout()

    # breakpoint()
    plt.tight_layout()
    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()

def plot_simple_pareto(points, fig_filename: Optional[str], special_points=None, special_labels=None, xlabel=None, ylabel=None, figsize=None):
    latexify_plot()
    if figsize is None:
        figsize=(6.6, 4.4)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for p in points:
        axes.plot(p[0], p[1], fillstyle='full', color='gray', marker='o', alpha=1.0)

    legend_elements = []

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    axes.set_xscale('symlog', linthresh=1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # axes.set_xscale('symlog', linthresh=1, subs=[2, 4, 6, 8])
    # axes.xaxis.grid(True, which='minor')
    axes.set_yscale('log')

    if special_points is not None:
        special_markers = ['o', 'v', 's', 'd', '^', '>', '<', 'P', 'D']
        special_markers = ['o' for _ in special_points]
        special_annotations = ['A', 'B', 'C', 'D']
        for p, label, marker, annotation in zip(special_points, special_labels, special_markers, special_annotations):
            alpha = 1.0
            fillstyle='none'
            color = f'C1'
            axes.plot(p[0], p[1], color=color, marker=marker, alpha=alpha, markersize=11, markeredgewidth=2, fillstyle=fillstyle, label=label)
            # legend_elements += [Line2D([0], [0], marker=marker, alpha=alpha, markersize=11, markeredgewidth=2, fillstyle=fillstyle, color=color, lw=0, label=label)]
            axes.annotate(annotation, xy=(p[0], p[1]*1.3))


    legend_elements += [Line2D([0], [0], marker=marker, alpha=alpha, markersize=11, markeredgewidth=2, fillstyle=fillstyle, color=color, lw=0, label='Variants A-D')]

    legend_elements += [Line2D([0], [0], marker='o', alpha=1.0, fillstyle='full', color='gray', lw=0, label='Other variants from Table II')]

    plt.grid()
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0)) #  loc='upper right'
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, -.15), ncol=2) #  loc='upper right'
    # plt.legend(handles=legend_elements, bbox_to_anchor=(1.0, -.1), ncol=1) #  loc='upper right'
    plt.tight_layout()

    # breakpoint()
    plt.tight_layout()
    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()



def plot_simplest_pareto(
    points,
    labels,
    colors,
    markers,
    fig_filename: Optional[str] = None,
    xlabel=None,
    ylabel=None,
    title=None,
    bbox_to_anchor=None,
):
    latexify_plot()
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8.8, 4.))

    if xlabel is not None:
        axes.set_xlabel(xlabel)
    if ylabel is not None:
        axes.set_ylabel(ylabel)
    axes.set_xscale("symlog", linthresh=0.1, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    # axes.set_yscale("log")
    axes.set_xscale("log")

    plt.grid()
    plt.tight_layout()

    if title is not None:
        axes.set_title(title)

    assert len(points) == len(labels)
    assert len(colors) >= len(points)
    assert len(markers) >= len(points)

    legend_elements = []

    for p, label, marker, color in zip(points, labels, markers, colors):
        alpha = 1.0
        fillstyle = "full"
        color = color
        axes.plot(
            p[0],
            p[1],
            color=color,
            marker=marker,
            alpha=alpha,
            markersize=10,
            markeredgewidth=2,
            fillstyle=fillstyle,
        )
        legend_elements += [
            Line2D(
                [0],
                [0],
                marker=marker,
                alpha=alpha,
                markersize=10,
                markeredgewidth=2,
                fillstyle=fillstyle,
                color=color,
                lw=0,
                label=label,
            )
        ]

    plt.legend(handles=legend_elements, ncol=1, bbox_to_anchor=bbox_to_anchor)
    plt.tight_layout()
    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    plt.show()




def compute_lqr_gain(model, model_params, mpc_params):
    from scipy.linalg import solve_discrete_are

    # linearize dynamics
    from setup_acados_integrator import setup_acados_integrator

    integrator = setup_acados_integrator(
        model, mpc_params.dt, mpc_params=mpc_params, sensitivity_propagation=True
    )
    integrator.set("x", model_params.xs)
    integrator.set("u", model_params.us)

    integrator.solve()

    A_mat = integrator.get("Sx")
    B_mat = integrator.get("Su")

    Q_mat = mpc_params.dt * mpc_params.Q
    R_mat = mpc_params.dt * mpc_params.R
    P_mat = solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)

    return A_mat, B_mat, P_mat


def compute_lqr_gain_continuous_time(model, model_params, mpc_params):
    from scipy.linalg import solve_continuous_are

    # linearize dynamics
    A_sym_fun = ca.Function(
        "A_sym_fun", [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.x)]
    )
    B_sym_fun = ca.Function(
        "B_sym_fun", [model.x, model.u], [ca.jacobian(model.f_expl_expr, model.u)]
    )

    A_mat = A_sym_fun(model_params.xs, model_params.us).full()
    B_mat = B_sym_fun(model_params.xs, model_params.us).full()

    P_mat = solve_continuous_are(A_mat, B_mat, mpc_params.Q, mpc_params.R)
    K_mat = np.linalg.inv(mpc_params.R) @ B_mat.T @ P_mat
    print(f"P_mat {P_mat}")
    return P_mat, K_mat


def get_label_from_setting(setting: dict):
    label = ""
    for key, value in setting.items():
        label += f"{key} = {value}, "
    return label


def get_latex_label_from_setting(setting):
    label = f"{setting[0]} $ N = {setting[1]}, n_s = {setting[2]} $ {'RTI' if setting[3] else ''}"
    if setting[4] == "nonuniform":
        pass
    elif setting[4] == "uniform_long":
        label += " uniform"
    else:
        label += setting[4]

    if len(setting) > 5:
        label += f" $d = {setting[5]}$"

    return label


def get_subdict(d, keys):
    subdict = {}
    for k in keys:
        if k in d:
            subdict[k] = d[k]
    return subdict


def get_relevant_keys(settings: List[dict]):
    relevant_keys = []
    for k in settings[0].keys():
        for setting in settings:
            if k in setting and setting[k] != settings[0][k]:
                relevant_keys.append(k)
                break
    constant_keys = list(set(settings[0].keys()) - set(relevant_keys))
    return relevant_keys, constant_keys


def get_latex_label_from_setting_cost_to_go(setting):
    label = f"{setting[0]} $ N = {setting[1]}, n_s = {setting[2]} $ {'RTI' if setting[3] else ''}"
    if setting[4] == "nonuniform":
        pass
    elif setting[4] == "uniform_long":
        label += " uniform"
    else:
        label += setting[4]

    if len(setting) > 5:
        label += f" $T = {setting[5]}$"

    if len(setting) > 6:
        if setting[6] == 1:
            label += f", pw lin $u$"

    return label


def get_results_filename(
    label: str, model_name: str, dt_plant: float, closed_loop: bool
):
    loop = "CL" if closed_loop else "OL"
    label_str = f"{loop}_{model_name}_dtplant_{dt_plant}_{label.replace(' ', '_').replace('=', '_')}"
    # print(f"hasing {label_str}")
    hash_str = int(hashlib.md5(label_str.encode()).hexdigest(), 16)
    return os.path.join(RESULTS_FOLDER, f"results_{hash_str}.pkl")


def get_nbx_violation_expression(model, mpc_params):
    x = model.x
    violation_expression = ca.fmax(
        ca.fmax(mpc_params.lbx - x[mpc_params.idxbx], 0),
        x[mpc_params.idxbx] - mpc_params.ubx,
    )
    return violation_expression


def plot_open_loop_trajectory_pwpol_u(
    shooting_nodes,
    X_traj: np.ndarray,
    U_fine_traj: list,
    plt_show=True,
    lbu: Optional[np.ndarray] = None,
    ubu: Optional[np.ndarray] = None,
    title=None,
    x_labels=None,
    u_labels=None,
    idxpx=None,
    idxpu=None,
):
    nx = X_traj.shape[1]
    nu = U_fine_traj[0].shape[1]
    t = shooting_nodes

    if idxpx is None:
        idxpx = list(range(nx))
    if idxpu is None:
        idxpu = list(range(nu))
    if u_labels is None:
        u_labels = [f"$u_{i}$" for i in range(nu)]
    if x_labels is None:
        x_labels = [f"$x_{i}$" for i in range(nx)]

    nxpx = len(idxpx)
    nxpu = len(idxpu)

    nrows = nxpx + nxpu

    latexify_plot()
    fig, axes = plt.subplots(
        ncols=1, nrows=nxpx + nxpu, figsize=(5.5, 1.65 * (nxpx + nxpu + 1)), sharex=True
    )
    plt.subplot(nrows, 1, 1)

    # plot X
    for isub, ix in enumerate(idxpx):
        axes[isub].plot(
            t, X_traj[:, ix], "--", marker="o", markersize=4, color="C0", alpha=0.5
        )
        axes[isub].set_ylabel(x_labels[ix])
        axes[isub].grid()

    # plot U
    for i, u_interval in enumerate(U_fine_traj):
        n_pol_interval = u_interval.shape[0]
        t_grid_interval = np.linspace(
            shooting_nodes[i], shooting_nodes[i + 1], n_pol_interval
        )
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].plot(
                t_grid_interval,
                u_interval[:, iu],
                color="C0",
                # alpha=0.5 + 0.5 * (i % 2),
            )
    for isub, iu in enumerate(idxpu):
        axes[isub+nxpx].set_ylabel(u_labels[iu])
        axes[isub+nxpx].grid()

    if lbu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                lbu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )
    if ubu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                ubu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )

    axes[-1].set_xlabel("$t [s]$")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    axes[0].set_xlim([shooting_nodes[0], shooting_nodes[-1]])
    fig.align_ylabels()

    if title is not None:
        axes[0].set_title(title)

    if plt_show:
        plt.show()



def plot_open_loop_trajectories_pwpol_u(
    shooting_nodes: list[np.ndarray],
    X_traj: list[np.ndarray],
    U_fine_traj: list[list],
    plt_show=True,
    lbu: Optional[np.ndarray] = None,
    ubu: Optional[np.ndarray] = None,
    title=None,
    x_labels=None,
    u_labels=None,
    idxpx=None,
    idxpu=None,
    labels=None,
    fig_filename=None,
    bbox_to_anchor=None,
    figsize=None,
):
    nx = X_traj[0].shape[1]
    nu = U_fine_traj[0][0].shape[1]

    if idxpx is None:
        idxpx = list(range(nx))
    if idxpu is None:
        idxpu = list(range(nu))
    if u_labels is None:
        u_labels = [f"$u_{i}$" for i in range(nu)]
    if x_labels is None:
        x_labels = [f"$x_{i}$" for i in range(nx)]

    nxpx = len(idxpx)
    nxpu = len(idxpu)

    nrows = nxpx + nxpu

    latexify_plot()

    if figsize is None:
        figsize=(6.5, 1.65 * (nxpx + nxpu + 1))
    fig, axes = plt.subplots(
        ncols=1, nrows=nxpx + nxpu, figsize=figsize, sharex=True
    )
    plt.subplot(nrows, 1, 1)

    n_traj = len(shooting_nodes)
    linestyles = n_traj * ['-', '--', '-.', ':']
    # colors = ['k'] + [f"C{itraj}" for itraj in range(n_traj)]
    colors = [f"C{itraj}" for itraj in range(n_traj)]

    if nxpx + nxpu == 1:
        axes = [axes]

    # plot X
    for itraj, (t, x_traj) in enumerate(zip(shooting_nodes, X_traj)):
        for isub, ix in enumerate(idxpx):
            axes[isub].plot(
                t, x_traj[:, ix], "--", marker="o", markersize=4, color=colors[itraj], alpha=0.5, linestyle=linestyles[itraj],
            )
            axes[isub].set_ylabel(x_labels[ix])
            axes[isub].grid()

    # plot U
    for itraj, (t, U_fine) in enumerate(zip(shooting_nodes, U_fine_traj)):
        for i, u_interval in enumerate(U_fine):
            n_pol_interval = u_interval.shape[0]
            t_grid_interval = np.linspace(
                t[i], t[i + 1], n_pol_interval
            )
            for isub, iu in enumerate(idxpu):
                axes[isub+nxpx].plot(
                    t_grid_interval,
                    u_interval[:, iu],
                    color=colors[itraj], linestyle=linestyles[itraj],
                    # alpha=0.5 + 0.5 * (i % 2),
                )
    for isub, iu in enumerate(idxpu):
        axes[isub+nxpx].set_ylabel(u_labels[iu])
        axes[isub+nxpx].grid()

    if lbu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                lbu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )
    if ubu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                ubu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )

    axes[-1].set_xlabel("$t [s]$")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    t0 = min([min(t) for t in shooting_nodes])
    tmax = max([max(t) for t in shooting_nodes])
    axes[0].set_xlim([t0, tmax])
    fig.align_ylabels()

    if labels is not None:
        legend_elements = []
        for itraj, l in enumerate(labels):
            legend_elements += [Line2D([0], [0], color=colors[itraj], linestyle=linestyles[itraj], lw=2, label=l)]
        axes[-1].legend(handles=legend_elements, bbox_to_anchor=bbox_to_anchor)
    if title is not None:
        axes[0].set_title(title)

    plt.tight_layout()

    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    if plt_show:
        plt.show()



def plot_open_loop_trajectories_pwpol_ux(
    shooting_nodes: list[np.ndarray],
    X_fine_traj: list[np.ndarray],
    U_fine_traj: list[list],
    plt_show=True,
    lbu: Optional[np.ndarray] = None,
    ubu: Optional[np.ndarray] = None,
    tmax=None,
    title=None,
    x_labels=None,
    u_labels=None,
    idxpx=None,
    idxpu=None,
    labels=None,
    fig_filename=None,
    bbox_to_anchor=None,
    figsize=None,
    single_column=True,
):
    nx = X_fine_traj[0][0].shape[1]
    nu = U_fine_traj[0][0].shape[1]

    if idxpx is None:
        idxpx = list(range(nx))
    if idxpu is None:
        idxpu = list(range(nu))
    if u_labels is None:
        u_labels = [f"$u_{i}$" for i in range(nu)]
    if x_labels is None:
        x_labels = [f"$x_{i}$" for i in range(nx)]

    nxpx = len(idxpx)
    nxpu = len(idxpu)

    nrows = max(nxpx, nxpu)

    latexify_plot()
    if single_column:
        if figsize is None:
            figsize=(5.5, 1.65 * (nxpx + nxpu + 1))
        fig, axes = plt.subplots(
            ncols=1,
            nrows=nxpx + nxpu,
            figsize=figsize,
            sharex=True,
        )
    else:
        fig, axes = plt.subplots(ncols=2, nrows=nrows, figsize=(10, (nxpx + nxpu)))
        axes = np.ravel(axes, order="F")


    n_traj = len(shooting_nodes)
    linestyles = n_traj * ['-', '--', '-.', ':']
    # colors = ['k'] + [f"C{itraj}" for itraj in range(n_traj)]
    colors = [f"C{itraj}" for itraj in range(n_traj)]

    if nxpx + nxpu == 1:
        axes = [axes]

    # plot X
    for itraj, (t, X_fine) in enumerate(zip(shooting_nodes, X_fine_traj)):
        for i, x_interval in enumerate(X_fine):
            n_pol_interval = x_interval.shape[0]
            t_grid_interval = np.linspace(
                t[i], t[i + 1], n_pol_interval
            )
            for isub, ix in enumerate(idxpx):
                axes[isub].plot(
                    t_grid_interval,
                    x_interval[:, ix],
                    color=colors[itraj], linestyle=linestyles[itraj],
                    # alpha=0.5 + 0.5 * (i % 2),
                )
    for isub, ix in enumerate(idxpx):
        axes[isub].set_ylabel(x_labels[ix])
        axes[isub].grid()


    # plot U
    for itraj, (t, U_fine) in enumerate(zip(shooting_nodes, U_fine_traj)):
        for i, u_interval in enumerate(U_fine):
            n_pol_interval = u_interval.shape[0]
            t_grid_interval = np.linspace(
                t[i], t[i + 1], n_pol_interval
            )
            for isub, iu in enumerate(idxpu):
                axes[isub+nxpx].plot(
                    t_grid_interval,
                    u_interval[:, iu],
                    color=colors[itraj], linestyle=linestyles[itraj],
                    # alpha=0.5 + 0.5 * (i % 2),
                )

    for isub, iu in enumerate(idxpu):
        axes[isub+nxpx].set_ylabel(u_labels[iu])
        axes[isub+nxpx].grid()

    if lbu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                lbu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )
    if ubu is not None:
        for isub, iu in enumerate(idxpu):
            axes[isub+nxpx].hlines(
                ubu[iu], t[0], t[-1], linestyles="dashed", alpha=0.4, color="k"
            )

    axes[-1].set_xlabel("$t [s]$")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    t0 = min([min(t) for t in shooting_nodes])
    if tmax is None:
        tmax = max([max(t) for t in shooting_nodes])
    axes[0].set_xlim([t0, tmax])
    fig.align_ylabels()

    if labels is not None:
        legend_elements = []
        for itraj, l in enumerate(labels):
            legend_elements += [Line2D([0], [0], color=colors[itraj], linestyle=linestyles[itraj], lw=2, label=l)]
        axes[-1].legend(handles=legend_elements, bbox_to_anchor=bbox_to_anchor)
    if title is not None:
        axes[0].set_title(title)

    plt.tight_layout()

    if fig_filename is not None:
        fig_filename = os.path.join(os.getcwd(), FIGURE_FOLDER, fig_filename)
        plt.savefig(
            fig_filename, bbox_inches="tight", transparent=True, pad_inches=0.05
        )
        print(f"\nstored figure in {fig_filename}")

    if plt_show:
        plt.show()
