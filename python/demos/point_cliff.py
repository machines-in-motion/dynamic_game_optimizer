import os, sys, time 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import point_cliff_action as point_cliff 
import matplotlib.pyplot as plt 

from solvers import full 




LINE_WIDTH = 100 
horizon = 100 
plan_dt = 1.e-2 
x0 = np.zeros(4)

MAX_ITER = 1000
SOLVE_DDP = False 

if __name__ == "__main__":
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt) 
    models = [cliff_running]*(horizon) + [cliff_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    if SOLVE_DDP:
        ddp = crocoddyl.SolverFDDP(problem)
        print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
        ddp.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose()
        ])
        xs = [x0]*(horizon+1)
        us = [np.zeros(2)]*horizon
        converged = ddp.solve(xs,us, MAX_ITER)

    # print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
    # plt.figure("trajectory plot")
    # plt.plot(np.array(ddp.xs)[:,0],np.array(ddp.xs)[:,1], label="ddp")
    # plt.show()

    dg_solver = full.SaddlePointSolver(problem)
    print(" Constructing saddle point solver completed ".center(LINE_WIDTH, '-'))
