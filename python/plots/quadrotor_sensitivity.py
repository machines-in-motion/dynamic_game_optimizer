import os, sys
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models.quadrotor_action import DifferentialActionModelQuadrotor
import matplotlib.pyplot as plt 
from utils.measurements import PositionMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 
import plotting_tools as plut 
LINE_WIDTH = 100 



dt = 0.05
horizon = 60


pm = 1e-4 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
P0 = 1e-2 * np.diag([1000, 1000, 1, 1, 1, 1])
mm = 1e-2 * np.diag([1000, 1000, 1])  # measurement error weight matrix


t_solve = 10

x0 = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])

MAX_ITER = 100

MUs = [-.04, -.01, .01, .02]


plut.SAVE_FIGURES = True

if __name__ == "__main__":
    print(" Testing Quadrotor with DDP ".center(LINE_WIDTH, "#"))
    quadrotor_diff_running = DifferentialActionModelQuadrotor()
    quadrotor_diff_terminal = DifferentialActionModelQuadrotor(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))

    quadrotor_running = crocoddyl.IntegratedActionModelRK(quadrotor_diff_running, crocoddyl.RKType.four, stepTime=dt)
    quadrotor_terminal = crocoddyl.IntegratedActionModelRK(quadrotor_diff_terminal, crocoddyl.RKType.four, stepTime=dt)
    process_models = [quadrotor_running]*(horizon) + [quadrotor_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [PositionMeasurement(quadrotor_running, mm)]*horizon + [PositionMeasurement(quadrotor_terminal, mm)]


    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    measurement_trajectory =  MeasurementTrajectory(measurement_models)

    xs = [x0]*(horizon+1)

    #________ Solve DDP to generate measurements ____________________# 
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    ddp_xs = [x0]*(horizon+1)
    ddp_us = [np.zeros(2)]*horizon
    ddp_converged = ddp_solver.solve(ddp_xs,ddp_us, MAX_ITER)
    x_init = ddp_solver.xs.tolist()
    u_init = ddp_solver.us.tolist()
    solvers = [ddp_solver]
    xnexts = []
    xnexts += [[d.xnext.copy() for d in solvers[-1].problem.runningDatas]]
    solver_names = ["DDP"] + ["$\mu=%s$"%mui for mui in MUs]
    print(solver_names)
    for MU in MUs:
        solvers += [PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)]
        print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))
        xs = [x0]*(horizon+1)
        ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve+1], ddp_solver.us[:t_solve])
        u_init[:t_solve] = ddp_solver.us[:t_solve]
        solvers[-1].solve(init_xs=x_init, init_us=u_init, init_ys=ys)
        xnexts += [[d.xnext.copy() for d in solvers[-1].problem.runningDatas]]


    plut.plot_2d_quad_trajectory_gaps(solvers, xnexts, solver_names, t_solve, "quadrotor_traj_sensitivity", r"$p_x$ [m]", r"$p_y$ [m]")
    state_names = [r"$p_x$ [m]", r"$p_y$ [m]", r"$\theta$ [rad]", r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$\dot{\theta}$ [rad/s]"] 
    control_names = [r"$\tau_1$ [N]", r"$\tau_2$ [N]"]
    plut.plot_states_controls(solvers, solver_names, dt, "quadrotor_states_controls_sensitivity", state_names, control_names, t_solve)