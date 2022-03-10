""" a demo for the partially observable case with the point cliff example """

import os, sys, time
from cv2 import solve 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import lin_quad_action as lin_quad 
import matplotlib.pyplot as plt 
from utils.measurements import FullStateMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 

LINE_WIDTH = 100 
horizon = 100 
plan_dt = 1.e-2 
x0 = np.zeros(4)

MAX_ITER = 100
PLOT_DDP = True 
pm = 1e-2 * np.eye(4) # process error weight matrix 
mm = 1e-2 * np.eye(4) # measurement error weight matrix 
P0  = 1e-2 * np.eye(4)
MU = 0.01

t_solve = 20 # solve problem for t = 50 

if __name__ == "__main__":
    lq_diff_running =  lin_quad.DifferentialActionModelLQ()
    lq_diff_terminal = lin_quad.DifferentialActionModelLQ(isTerminal=True)
    lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, plan_dt) 
    lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, plan_dt) 
    process_models = [lq_running]*(horizon) + [lq_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [FullStateMeasurement(lq_running, mm)]*horizon + [FullStateMeasurement(lq_terminal, mm)]


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


    
    ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve], ddp_solver.us[:t_solve])
    
    dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
    print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))

    u_init = [np.zeros(2)]*horizon
    u_init[:t_solve-1] = ddp_solver.us[:t_solve-1]
    dg_solver.solve(init_xs=xs, init_us=u_init, init_ys=ys)

    print(" Plotting DDP and DG Solutions ".center(LINE_WIDTH, '-'))
    time_array = plan_dt*np.arange(horizon+1)
    
    # plt.figure("trajectory plot")
    # plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="DDP Trajectory")
    # plt.plot(np.array(dg_solver.xs)[:,0],np.array(dg_solver.xs)[:,1], label="DG Trajectory")
    # plt.legend()

    # plt.show()



    x_n = [d.xnext.copy() for d in dg_solver.problem.runningDatas]
    plt.figure("trajectory plot")

    x = np.array(dg_solver.xs)

    for t in range(len(np.array(dg_solver.xs[:t_solve-1]))):
        if t == 0:
            plt.plot(np.array([x[t][0], x_n[t][0]]), np.array([x[t][1], x_n[t][1]]), 'green', label='DG estimation')
        else:
            plt.plot(np.array([x[t][0], x_n[t][0]]), np.array([x[t][1], x_n[t][1]]), 'green')

    for t_ in range(len(np.array(dg_solver.xs[t_solve-1:-1]))):
        t = t_ + t_solve-1
        if t_ == 0:
            plt.plot(np.array([x[t][0], x_n[t][0]]), np.array([x[t][1], x_n[t][1]]), 'red', label='DG control')
        else:
            plt.plot(np.array([x[t][0], x_n[t][0]]), np.array([x[t][1], x_n[t][1]]), 'red')


    plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="DDP Trajectory")

    plt.plot(np.array(ddp_solver.xs)[:t_solve,0],np.array(ddp_solver.xs)[:t_solve,1], 'black', label="Measurements")
    plt.legend()
    plt.show()
    
    
    # plt.figure("u plot")
    # plt.plot(np.array(ddp_solver.us)[:,0], label="DDP 0")   
    # plt.plot(np.array(ddp_solver.us)[:,1], label="DDP 1")   
    # plt.plot(np.array(dg_solver.us)[:,0], label="DG 0")   
    # plt.plot(np.array(dg_solver.us)[:,1], label="DG 1")   
    # plt.legend()
    # plt.show()