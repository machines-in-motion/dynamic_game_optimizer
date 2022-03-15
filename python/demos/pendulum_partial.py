import os, sys, time
from cv2 import solve 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import pendulum_action as pendulum
import matplotlib.pyplot as plt 
from utils.measurements import FullStateMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 

LINE_WIDTH = 100 
horizon = 60
plan_dt = 1.e-2 
x0 = np.zeros(2)

MAX_ITER = 100
PLOT_DDP = True 
pm = 1e-2 * np.eye(2) # process error weight matrix 
mm = 1e0 * np.eye(2) # measurement error weight matrix 
P0  = 1e-2 * np.eye(2)
MU = 0.02

t_solve = 30

if __name__ == "__main__":
    pendulum_diff_running =  pendulum.DifferentialActionModelPendulum()
    pendulum_diff_terminal = pendulum.DifferentialActionModelPendulum(isTerminal=True)
    pendulum_running = crocoddyl.IntegratedActionModelEuler(pendulum_diff_running, plan_dt) 
    pendulum_terminal = crocoddyl.IntegratedActionModelEuler(pendulum_diff_terminal, plan_dt) 
    process_models = [pendulum_running]*(horizon) + [pendulum_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [FullStateMeasurement(pendulum_running, mm)]*horizon + [FullStateMeasurement(pendulum_terminal, mm)]

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

    ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve+1])
    
    dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
    print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))

    u_init = [np.zeros(1)]*horizon
    u_init[:t_solve] = ddp_solver.us[:t_solve]
    dg_solver.solve(init_xs=xs, init_us=u_init, init_ys=ys)

    print(" Plotting DDP and DG Solutions ".center(LINE_WIDTH, '-'))
    time_array = plan_dt*np.arange(horizon+1)

    x = np.array(dg_solver.xs)
    x_next = [d.xnext.copy() for d in dg_solver.problem.runningDatas]

    plt.figure("trajectory plot")
    for t in range(len(np.array(dg_solver.xs[:t_solve-1]))):
        if t == 0:
            plt.plot(np.array([t, t+1]), np.array([x[t][0], x_next[t][0]]), 'green', label='DG estimation')
        else:
            plt.plot(np.array([t, t+1]), np.array([x[t][0], x_next[t][0]]), 'green')

    for t_ in range(len(np.array(dg_solver.xs[t_solve-1:-1]))):
        t = t_ + t_solve-1
        if t_ == 0:
            plt.plot(np.array([t, t+1]), np.array([x[t][0], x_next[t][0]]), 'red', label='DG control')
        else:
            plt.plot(np.array([t, t+1]), np.array([x[t][0], x_next[t][0]]), 'red')
    plt.xlabel("Time")    
    plt.ylabel("$\\theta$")
    plt.plot(np.array(ddp_solver.xs)[:,0], label="DDP Trajectory")
    plt.plot(np.array(ddp_solver.xs)[:t_solve,0] , 'black', label="Measurements")
    plt.legend()

    plt.figure()   
    for t in range(len(np.array(dg_solver.xs[:t_solve-1]))):
        if t == 0:
            plt.plot(np.array([x[t][0], x_next[t][0]]), np.array([x[t][1], x_next[t][1]]), 'green', label='DG estimation')
        else:
            plt.plot(np.array([x[t][0], x_next[t][0]]), np.array([x[t][1], x_next[t][1]]), 'green')

    for t_ in range(len(np.array(dg_solver.xs[t_solve-1:-1]))):
        t = t_ + t_solve -1 
        if t_ == 0:
            plt.plot(np.array([x[t][0], x_next[t][0]]), np.array([x[t][1], x_next[t][1]]), 'red', label='DG control')
        else:
            plt.plot(np.array([x[t][0], x_next[t][0]]), np.array([x[t][1], x_next[t][1]]), 'red')

    plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="DDP Trajectory")
    plt.plot(np.array(ddp_solver.xs)[:t_solve,0],np.array(ddp_solver.xs)[:t_solve,1], 'black', label="Measurements")
    plt.xlabel("$\\theta$")
    plt.ylabel("$\dot\\theta$")
    plt.title("Position trajectory")
    plt.legend()
    
    
    plt.figure()
    plt.plot(np.array(ddp_solver.us)[:,0], label="DDP")   
    plt.plot(np.array(dg_solver.us)[:,0], label="DG")   
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.title("Control inputs")
    plt.legend()
    plt.show()