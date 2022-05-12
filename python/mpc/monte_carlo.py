import matplotlib.patches as m_patches
import matplotlib.pyplot as plt 
import numpy as np 
import crocoddyl 
from tqdm import tqdm
import os, sys, time
src_path = os.path.abspath('../')
sys.path.append(src_path)


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


    plt.figure(figsize=(20, 20))
    width = 0.1
    height = 2
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ax = plt.axes()
    ax.add_patch(m_patches.Ellipse((1., -0.1), width=0.45*width, height=0.45*height,fill=False,hatch='/',))
    plt.plot(states_mean[:, 0], states_mean[:, 1], color=cycle[0], label=label_dg)
    plt.plot(states_mean_DDP[:, 0], states_mean_DDP[:, 1], color=cycle[1], label=label_ddp)
    plt.plot(state_dg_plan[:, 0], state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG initial plan")
    plt.plot(state_ddp_plan[:, 0], state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP initial plan")

    # Plot shaded std
    y_1 = states_mean[:, 1] - states_std[:, 1]
    y_2 = states_mean[:, 1] + states_std[:, 1]
    plt.fill_between(states_mean[:, 0], y_1, y_2, alpha=0.2)

    y_1 = states_mean_DDP[:, 1] - states_std_DDP[:, 1]
    y_2 = states_mean_DDP[:, 1] + states_std_DDP[:, 1]
    plt.fill_between(states_mean_DDP[:, 0], y_1, y_2, alpha=0.2)

    plt.xlabel(r"$p_x$ [m]") 
    plt.ylabel(r"$p_y$ [m]")
    plt.xlim([-0.1, 2.1])
    plt.ylim([-0.1, 0.6])
    plt.grid()
    plt.legend(loc='upper right')
    plt.savefig("quadrotor_mean_trajectory_" + label + ".pdf")


    time = np.linspace(0, plan_dt*(horizon+1), horizon+1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(24, 12))
    ax1.plot(time, states_mean[:, 0], label="DG")
    ax2.plot(time, states_mean[:, 1], label="DG")
    ax3.plot(time, states_mean[:, 2], label="DG")

    ax1.plot(time, state_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
    ax2.plot(time, state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")
    ax3.plot(time, state_dg_plan[:, 2], color=cycle[0], linestyle='dashed', label="DG plan")

    ax1.plot(time, states_mean_DDP[:, 0], label="DDP")
    ax2.plot(time, states_mean_DDP[:, 1], label="DDP")
    ax3.plot(time, states_mean_DDP[:, 2], label="DDP")

    ax1.plot(time, state_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
    ax2.plot(time, state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")
    ax3.plot(time, state_ddp_plan[:, 2], color=cycle[1], linestyle='dashed', label="DDP plan")

    ax1.fill_between(time, states_mean[:, 0] - states_std[:, 0], states_mean[:, 0] + states_std[:, 0], alpha=0.2)
    ax2.fill_between(time, states_mean[:, 1] - states_std[:, 1], states_mean[:, 1] + states_std[:, 1], alpha=0.2)
    ax3.fill_between(time, states_mean[:, 2] - states_std[:, 2], states_mean[:, 2] + states_std[:, 2], alpha=0.2)

    ax1.fill_between(time, states_mean_DDP[:, 0] - states_std_DDP[:, 0], states_mean_DDP[:, 0] + states_std_DDP[:, 0], alpha=0.2)
    ax2.fill_between(time, states_mean_DDP[:, 1] - states_std_DDP[:, 1], states_mean_DDP[:, 1] + states_std_DDP[:, 1], alpha=0.2)
    ax3.fill_between(time, states_mean_DDP[:, 2] - states_std_DDP[:, 2], states_mean_DDP[:, 2] + states_std_DDP[:, 2], alpha=0.2)

    ax3.set_xlabel("time [s]")
    ax1.set_ylabel(r"$p_x$ [m]")
    ax2.set_ylabel(r"$p_y$ [m]")
    ax3.set_ylabel(r"$\theta$ [rad/s]")

    ax1.grid()
    ax2.grid()
    ax3.grid()
    
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 12})

    plt.savefig("quadrotor_mean_state_" + label + ".pdf")


    time = np.linspace(0, plan_dt*horizon, horizon)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


    ax1.plot(time, controls_mean[:, 0], label="DG")
    ax2.plot(time, controls_mean[:, 1], label="DG")

    ax1.plot(time, control_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
    ax2.plot(time, control_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")


    ax1.plot(time, controls_mean_DDP[:, 0], label="DDP")
    ax2.plot(time, controls_mean_DDP[:, 1], label="DDP")

    ax1.plot(time, control_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
    ax2.plot(time, control_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")

    ax1.fill_between(time, controls_mean[:, 0] - controls_std[:, 0], controls_mean[:, 0] + controls_std[:, 0], alpha=0.2)
    ax2.fill_between(time, controls_mean[:, 1] - controls_std[:, 1], controls_mean[:, 1] + controls_std[:, 1], alpha=0.2)

    ax1.fill_between(time, controls_mean_DDP[:, 0] - controls_std_DDP[:, 0], controls_mean_DDP[:, 0] + controls_std_DDP[:, 0], alpha=0.2)
    ax2.fill_between(time, controls_mean_DDP[:, 1] - controls_std_DDP[:, 1], controls_mean_DDP[:, 1] + controls_std_DDP[:, 1], alpha=0.2)
    ax2.set_xlabel("time [s]")
    ax1.set_ylabel(r"$u_1$ [N]")
    ax2.set_ylabel(r"$u_2$ [N]")
    ax1.grid()
    ax2.grid()
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 6})
    plt.savefig("quadrotor_mean_control_" + label + ".pdf")

    plt.figure()
    plt.hist(record_cost, label="DG")
    plt.hist(record_cost_DDP, label="DDP")
    plt.legend(loc='upper right')

    plt.savefig("quadrotor_mean_control_" + label + ".pdf")
    if show:
        plt.show()


if __name__ == "__main__":

    N_simulation = 10
    MU = 8
    pm = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
    P0 = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])
    mm = 1e-5 * np.diag([10, 10, 0.1])  # measurement error weight matrix

    create_plots(P0, pm, mm, MU, N_simulation, "mc", show=True)
