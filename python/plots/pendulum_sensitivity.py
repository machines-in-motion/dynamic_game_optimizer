import os, sys, time
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import pendulum_action as pendulum 
import matplotlib.pyplot as plt 
from utils.measurements import PendulumCartesianMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 
import plotting_tools as plut 


LINE_WIDTH = 100 
horizon = 100
plan_dt = 1.e-2 
x0 = np.zeros(2)

MAX_ITER = 100
PLOT_DDP = True 
pm = 1e-2 * np.eye(2) # process error weight matrix 
mm = 1e-2 * np.eye(2) # measurement error weight matrix 
P0  = 1e-2 * np.eye(2)
MUs = [ -1., -0.01, .01, 0.02] 
t_solve = 30
plut.SAVE_FIGURES = True
if __name__ == "__main__":
    pendulum_diff_running =  pendulum.DifferentialActionModelPendulum()
    pendulum_diff_terminal = pendulum.DifferentialActionModelPendulum(isTerminal=True)
    pendulum_running = crocoddyl.IntegratedActionModelEuler(pendulum_diff_running, plan_dt) 
    pendulum_terminal = crocoddyl.IntegratedActionModelEuler(pendulum_diff_terminal, plan_dt) 
    process_models = [pendulum_running]*(horizon) + [pendulum_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [PendulumCartesianMeasurement(pendulum_running, mm)]*horizon + [PendulumCartesianMeasurement(pendulum_terminal, mm)]


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
    ddp_us = [np.zeros(1)]*horizon
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
        u_init = [np.zeros(1)]*horizon
        u_init[:t_solve] = ddp_solver.us[:t_solve]
        solvers[-1].solve(init_xs=xs, init_us=u_init, init_ys=ys)
        xnexts += [[d.xnext.copy() for d in solvers[-1].problem.runningDatas]]

    # plut.plot_2d_trajectory_gaps(solvers, xnexts, solver_names, t_solve, "pendulum trajectory", r"$\theta$ [rad]", r"$\dot{\theta}$ [rad/s]")
    plut.plot_states(solvers, solver_names, 1.e-2, "pendulum_states", [r"$\theta$ [rad]", r"$\dot{\theta}$ [rad/s]"], t_solve)
    plut.plot_controls(solvers, solver_names, 1.e-2, "pendulum_controls", [r"$\tau$ [N]"], t_solve)
    # plut.plot_pendulum_xy(solvers, solver_names)
    plt.show()