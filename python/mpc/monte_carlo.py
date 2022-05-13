import matplotlib.patches as m_patches
import matplotlib.pyplot as plt 
import numpy as np 
import crocoddyl 
from tqdm import tqdm
import os, sys, time
src_path = os.path.abspath('../')
sys.path.append(src_path)

import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 2  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30 #DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ['pdf', 'png']  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = './'


mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['font.family'] = DEFAULT_FONT_FAMILY
mpl.rcParams['font.size'] = DEFAULT_FONT_SIZE
mpl.rcParams['font.serif'] = DEFAULT_FONT_SERIF
mpl.rcParams['text.usetex'] = DEFAULT_TEXT_USE_TEX
mpl.rcParams['axes.labelsize'] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = DEFAULT_LEGEND_FONT_SIZE
mpl.rcParams['legend.framealpha'] = 1.
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
scale = 1.0
mpl.rcParams['figure.figsize'] = 30*scale, 10*scale #23, 18  # 12, 9
# line_styles = 10*['g-', 'r--', 'b-.', 'k:', '^c', 'vm', 'yo']
line_styles = 10*['k', 'r', 'm', 'b' , 'c', 'g', 'y']

from models.quadrotor_action import DifferentialActionModelQuadrotor
from utils.measurements import PositionMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
from solvers.partial_neutral import PartialNeutralSolver
from simulation import simulate

LINE_WIDTH = 100


def create_plots(P0, pm, mm, MU, N_simulation, label, show=False):
    nx = 6
    ny = 3
    nu = 2
     
    horizon = 60
    plan_dt = 0.05
    x0_hat = np.zeros(nx)

    quadrotor_diff_running =  DifferentialActionModelQuadrotor()
    quadrotor_diff_terminal = DifferentialActionModelQuadrotor(isTerminal=True)
    quadrotor_running = crocoddyl.IntegratedActionModelRK(quadrotor_diff_running, crocoddyl.RKType.four, stepTime=plan_dt)
    quadrotor_terminal = crocoddyl.IntegratedActionModelRK(quadrotor_diff_terminal, crocoddyl.RKType.four, stepTime=plan_dt)
    process_models = [quadrotor_running]*(horizon) + [quadrotor_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0_hat, process_models[:-1], process_models[-1])
    measurement_models = [PositionMeasurement(quadrotor_terminal, mm)]*horizon + [PositionMeasurement(quadrotor_terminal, mm)]
    measurement_trajectory = MeasurementTrajectory(measurement_models)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))

    
    x_init = [x0_hat]*(horizon+1)
    u_init = [np.zeros(nu)]*horizon
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    ddp_solver.solve(init_xs=x_init, init_us=u_init)
    state_ddp_plan = np.array(ddp_solver.xs)
    control_ddp_plan = np.array(ddp_solver.us)
    x_init = ddp_solver.xs.tolist().copy()
    u_init = ddp_solver.us.tolist().copy()


    dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
    dg_solver.solve(init_xs=x_init, init_us=u_init)
    state_dg_plan = np.array(dg_solver.xs)
    control_dg_plan = np.array(dg_solver.us)


    record_cost = []
    record_states = []
    record_controls = []
    record_cost_DDP = []
    record_states_DDP = []
    record_controls_DDP = []

    n_fail = 0
    for sim in tqdm(range(N_simulation)):
        try: 
            cost, states, controls, _, cost_DDP, states_DDP, controls_DDP, _ = simulate(ddp_problem, MU, x0_hat, pm, P0, mm, process_models, measurement_trajectory, horizon, x_init, u_init)
            record_cost.append(cost)
            record_states.append(np.array(states))
            record_controls.append(np.array(controls))
            record_cost_DDP.append(cost_DDP)
            record_states_DDP.append(np.array(states_DDP))
            record_controls_DDP.append(np.array(controls_DDP))
        except:
            print("Simulation failed")
            n_fail += 1
            continue
    print(n_fail, "simulation failed")
    record_cost = np.array(record_cost)
    record_states = np.array(record_states)
    record_controls = np.array(record_controls)
    record_cost_DDP = np.array(record_cost_DDP)
    record_states_DDP = np.array(record_states_DDP)
    record_controls_DDP = np.array(record_controls_DDP)
    
    label_dg = f'DG cost = {np.mean(record_cost):.3f}   \u00B1  {np.std(record_cost):.3f}'
    label_ddp = f'DDP cost = {np.mean(record_cost_DDP):.3f}   \u00B1  {np.std(record_cost_DDP):.3f}'
    print(label_dg)
    print(label_ddp)


    states_mean = np.mean(record_states, axis=0)
    states_std = np.std(record_states, axis=0)
    controls_mean = np.mean(record_controls, axis=0)
    controls_std = np.std(record_controls, axis=0)

    states_mean_DDP = np.mean(record_states_DDP, axis=0)
    states_std_DDP = np.std(record_states_DDP, axis=0)
    controls_mean_DDP = np.mean(record_controls_DDP, axis=0)
    controls_std_DDP = np.std(record_controls_DDP, axis=0)


    plt.figure(figsize=(20, 10))
    width = 0.1
    height = 2
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = plt.axes()
    ax.add_patch(m_patches.Ellipse((1., -0.1), width=0.55*width, height=0.55*height,fill=False,hatch='/',))
    plt.plot(states_mean_DDP[:, 0], states_mean_DDP[:, 1], color="b", linewidth=2, label="Neutral")
    plt.plot(states_mean[:, 0], states_mean[:, 1], color="g", linewidth=2, label="DG")
    # plt.plot(state_dg_plan[:, 0], state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG initial plan")
    # plt.plot(state_ddp_plan[:, 0], state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP initial plan")

    # Plot shaded std
    y_1 = states_mean_DDP[:, 1] - states_std_DDP[:, 1]
    y_2 = states_mean_DDP[:, 1] + states_std_DDP[:, 1]
    plt.fill_between(states_mean_DDP[:, 0], y_1, y_2, color="b", alpha=0.2)
    y_1 = states_mean[:, 1] - states_std[:, 1]
    y_2 = states_mean[:, 1] + states_std[:, 1]
    plt.fill_between(states_mean[:, 0], y_1, y_2, color="g", alpha=0.2)


    plt.xlabel(r"$p_x$ [m]") 
    plt.ylabel(r"$p_y$ [m]")
    plt.xlim([-0.05, 2.05])
    plt.ylim([-0.1, 0.7])
    plt.legend(loc='upper right')
    plt.savefig("quadrotor_mean_trajectory_" + label + ".pdf", bbox_inches='tight')


    time = np.linspace(0, plan_dt*(horizon+1), horizon+1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 20))

    ax1.plot(time, states_mean_DDP[:, 0], color="b", label="Neutral")
    ax2.plot(time, states_mean_DDP[:, 1], color="b", label="Neutral")
    ax3.plot(time, states_mean_DDP[:, 2], color="b", label="Neutral")

    ax1.plot(time, states_mean[:, 0], color="g", label="DG")
    ax2.plot(time, states_mean[:, 1], color="g", label="DG")
    ax3.plot(time, states_mean[:, 2], color="g", label="DG")

    # ax1.plot(time, state_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
    # ax2.plot(time, state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")
    # ax3.plot(time, state_dg_plan[:, 2], color=cycle[0], linestyle='dashed', label="DG plan")

    # ax1.plot(time, state_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
    # ax2.plot(time, state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")
    # ax3.plot(time, state_ddp_plan[:, 2], color=cycle[1], linestyle='dashed', label="DDP plan")

    ax1.fill_between(time, states_mean_DDP[:, 0] - states_std_DDP[:, 0], states_mean_DDP[:, 0] + states_std_DDP[:, 0], color="b", alpha=0.2)
    ax2.fill_between(time, states_mean_DDP[:, 1] - states_std_DDP[:, 1], states_mean_DDP[:, 1] + states_std_DDP[:, 1], color="b", alpha=0.2)
    ax3.fill_between(time, states_mean_DDP[:, 2] - states_std_DDP[:, 2], states_mean_DDP[:, 2] + states_std_DDP[:, 2], color="b", alpha=0.2)

    ax1.fill_between(time, states_mean[:, 0] - states_std[:, 0], states_mean[:, 0] + states_std[:, 0], color="g", alpha=0.2)
    ax2.fill_between(time, states_mean[:, 1] - states_std[:, 1], states_mean[:, 1] + states_std[:, 1], color="g", alpha=0.2)
    ax3.fill_between(time, states_mean[:, 2] - states_std[:, 2], states_mean[:, 2] + states_std[:, 2], color="g", alpha=0.2)


    ax3.set_xlabel("time [s]")
    ax1.set_ylabel(r"$p_x$ [m]")
    ax2.set_ylabel(r"$p_y$ [m]")
    ax3.set_ylabel(r"$\theta$ [rad]")
    
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(loc='lower right')

    plt.savefig("quadrotor_mean_state_" + label + ".pdf", bbox_inches='tight')


    time = np.linspace(0, plan_dt*horizon, horizon)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

    ax1.plot(time, controls_mean_DDP[:, 0], color="b", label="Neutral")
    ax2.plot(time, controls_mean_DDP[:, 1], color="b", label="Neutral")

    ax1.plot(time, controls_mean[:, 0], color="g", label="DG")
    ax2.plot(time, controls_mean[:, 1], color="g", label="DG")

    # ax1.plot(time, control_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
    # ax2.plot(time, control_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")

    # ax1.plot(time, control_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
    # ax2.plot(time, control_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")

    ax1.fill_between(time, controls_mean_DDP[:, 0] - controls_std_DDP[:, 0], controls_mean_DDP[:, 0] + controls_std_DDP[:, 0], color="b", alpha=0.2)
    ax2.fill_between(time, controls_mean_DDP[:, 1] - controls_std_DDP[:, 1], controls_mean_DDP[:, 1] + controls_std_DDP[:, 1], color="b", alpha=0.2)

    ax1.fill_between(time, controls_mean[:, 0] - controls_std[:, 0], controls_mean[:, 0] + controls_std[:, 0], color="g", alpha=0.2)
    ax2.fill_between(time, controls_mean[:, 1] - controls_std[:, 1], controls_mean[:, 1] + controls_std[:, 1], color="g", alpha=0.2)

    ax2.set_xlabel("time [s]")
    ax1.set_ylabel(r"$u_1$ [N]")
    ax2.set_ylabel(r"$u_2$ [N]")
    ax1.legend(loc='lower right')
    plt.savefig("quadrotor_mean_control_" + label + ".pdf", bbox_inches='tight')

    if show:
        plt.show()


if __name__ == "__main__":

    N_simulation = 1000
    MU = 8
    pm = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
    P0 = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])
    mm = 1e-5 * np.diag([10, 10, 0.1])  # measurement error weight matrix

    create_plots(P0, pm, mm, MU, N_simulation, "mc", show=True)
