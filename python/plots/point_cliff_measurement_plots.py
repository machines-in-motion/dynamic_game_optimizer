import os, sys
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
mm = 1e-3 * np.eye(2) # measurement error weight matrix 
scales = [0.01, 10, 100, 1000]
P0  = 1e-2 * np.eye(4)
MU = .1 

t_solve = 20 # solve problem for t = 50 

plut.SAVE_FIGURES = True 
if __name__ == "__main__":
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt) 
    process_models = [cliff_running]*(horizon) + [cliff_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])


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
    solver_names = ["DDP"] + ["$\gamma=%s$"%s for s in scales]
    print(solver_names)
    for s in scales:
        measurement_models = [PositionMeasurement(cliff_running, s*mm)]*horizon \
                                + [PositionMeasurement(cliff_terminal, s*mm)]


        print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
        measurement_trajectory =  MeasurementTrajectory(measurement_models)
        solvers += [PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)]
        print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))
        xs = [x0]*(horizon+1)
        ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve+1], ddp_solver.us[:t_solve])
        u_init = [np.zeros(2)]*horizon
        u_init[:t_solve] = ddp_solver.us[:t_solve]
        solvers[-1].solve(init_xs=xs, init_us=u_init, init_ys=ys)
        xnexts += [[d.xnext.copy() for d in solvers[-1].problem.runningDatas]]

    plut.plot_2d_trajectory_gaps(solvers, xnexts, solver_names, t_solve, "pnt_cliff_traj_measurement", "x [m]", "y [m]")
    state_names = ["x [m]", "y [m]", r"$v_x$ [m/s]", r"$v_y$ [m/s]"]
    control_names = [r"$\tau_x$ [N]", r"$\tau_y$ [N]"]
    plut.plot_states_controls(solvers, solver_names, plan_dt, "pnt_cliff_states_controls_measurement", state_names, control_names, t_solve)