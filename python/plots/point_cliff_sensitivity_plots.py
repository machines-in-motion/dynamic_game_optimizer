""" a demo for the partially observable case with the point cliff example """

import os, sys, time
from cv2 import solve 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import point_cliff_action as point_cliff 
import matplotlib.pyplot as plt 
from utils.measurements import PositionMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 
import plotting_tools as plut 
LINE_WIDTH = 100 
horizon = 100 
plan_dt = 1.e-2 
x0 = np.zeros(4)

MAX_ITER = 100
PLOT_DDP = True 
pm = 1e-2 * np.eye(4) # process error weight matrix 
mm = 1e-2 * np.eye(2) # measurement error weight matrix 
P0  = 1e-2 * np.eye(4)
MUs = [-7., -5. , -1, .01, .05, .1] 

t_solve = 50 # solve problem for t = 50 


if __name__ == "__main__":
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt) 
    process_models = [cliff_running]*(horizon) + [cliff_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [PositionMeasurement(cliff_running, mm)]*horizon + [PositionMeasurement(cliff_terminal, mm)]


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
        u_init = [np.zeros(2)]*horizon
        u_init[:t_solve] = ddp_solver.us[:t_solve]
        solvers[-1].solve(init_xs=xs, init_us=u_init, init_ys=ys)
        xnexts += [[d.xnext.copy() for d in solvers[-1].problem.runningDatas]]


    plut.plot_2d_trajectory_gaps(solvers, xnexts, solver_names, 1.e-2, "point cliff trajectory", "x [m]", "y [m]")
    plut.plot_states(solvers, solver_names, 1.e-2, "states", ["x [m]", "y [m]", r"$v_x$ [m/s]", r"$v_y$ [m/s]"])
    plut.plot_controls(solvers, solver_names, 1.e-2, "controls", [r"$\tau_x$ [N]", r"$\tau_y$ [N]"])
    # print(" Plotting DDP and DG Solutions ".center(LINE_WIDTH, '-'))
    # time_array = plan_dt*np.arange(horizon+1)
    
    # plt.figure("Different Sensitivity Trajectory")
    # for i, ithsovlver in enumerate(solvers):
    #     xs = np.array(ithsovlver.xs)
    #     plt.plot(xs[:,0], xs[:,1], label=solver_names[i])
    # plt.legend()

    # plt.figure("Different Sensitivity Y Control")
    # for i, ithsovlver in enumerate(solvers):
    #     us = np.array(ithsovlver.us)
    #     # plt.plot(time_array[:-1], us[:,0], label=solver_names[i]+" ux")
    #     plt.plot(time_array[:-1], us[:,1], label=solver_names[i]+" uy")
    # plt.legend()


    # plt.figure("Different Sensitivity X Control")
    # for i, ithsovlver in enumerate(solvers):
    #     us = np.array(ithsovlver.us)
    #     plt.plot(time_array[:-1], us[:,0], label=solver_names[i]+" ux")
    #     # plt.plot(time_array[:-1], us[:,1], label=solver_names[i]+" uy")
    # plt.legend()

    plt.show()