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

LINE_WIDTH = 100 
horizon = 50
plan_dt = 1.e-2 
x0 = np.zeros(2)

MAX_ITER = 100
PLOT_DDP = True 
pm = 1e-2 * np.eye(2) # process error weight matrix 
mm = 1e0 * np.eye(2) # measurement error weight matrix 
P0  = 1e-2 * np.eye(2)
MU = 0.02

t_solve = 15

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


    ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve])
    
    dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
    print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))

    u_init = [np.zeros(1)]*horizon
    u_init[:t_solve-1] = ddp_solver.us[:t_solve-1]
    dg_solver.solve(init_xs=xs, init_us=u_init, init_ys=ys)
    print(" Plotting DDP and DG Solutions ".center(LINE_WIDTH, '-'))
    
    
    
    time_array = plan_dt*np.arange(horizon+1)
    x_next = [d.xnext.copy() for d in dg_solver.problem.runningDatas]



    y_dg = measurement_trajectory.calc(dg_solver.xs)
    x_next = [d.xnext.copy() for d in dg_solver.problem.runningDatas]
    y_next = measurement_trajectory.calc(x_next)
    ys = np.array(ys)

    plt.figure("Measurement plot")    
    for t in range(len(np.array(dg_solver.xs[:t_solve-1]))):
        if t == 0:
            plt.plot(np.array([y_dg[t][0], y_next[t][0]]), np.array([y_dg[t][1], y_next[t][1]]), 'green', label='DG estimation')
        else:
            plt.plot(np.array([y_dg[t][0], y_next[t][0]]), np.array([y_dg[t][1], y_next[t][1]]), 'green')

    for t_ in range(len(np.array(dg_solver.xs[t_solve-1:-1]))):
        t = t_ + t_solve -1 
        if t_ == 0:
            plt.plot(np.array([y_dg[t][0], y_next[t][0]]), np.array([y_dg[t][1], y_next[t][1]]), 'red', label='DG control')
        else:
            plt.plot(np.array([y_dg[t][0], y_next[t][0]]), np.array([y_dg[t][1], y_next[t][1]]), 'red')

    plt.plot(ys[:t_solve,0],ys[:t_solve,1], 'black', label="Measurements")
    plt.legend()
    plt.axis([-0.5, 1.1, -1.1, 1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    



    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Measurement')

    for t in range(len(np.array(dg_solver.xs[:t_solve-1]))):
        ax1.plot(np.array([t, t+1]), np.array([y_dg[t][0], y_next[t][0]]), 'green')
        ax2.plot(np.array([t, t+1]), np.array([y_dg[t][1], y_next[t][1]]), 'green')

    for t_ in range(len(np.array(dg_solver.xs[t_solve-1:-1]))):
        t = t_ + t_solve -1 
        ax1.plot(np.array([t, t+1]), np.array([y_dg[t][0], y_next[t][0]]), 'red')
        ax2.plot(np.array([t, t+1]), np.array([y_dg[t][1], y_next[t][1]]), 'red')

    ax1.plot(ys[:t_solve,0], 'black')
    ax2.plot(ys[:t_solve,1], 'black')

    ax2.set_xlabel('time')
    ax1.set_ylabel('y1')
    ax2.set_ylabel('y2')

    plt.show()
  
        
    # plt.figure("u plot")
    # plt.plot(np.array(ddp_solver.us)[:,0], label="DDP")   
    # plt.plot(np.array(dg_solver.us)[:,0], label="DG")   
    # plt.legend()
    # plt.show()