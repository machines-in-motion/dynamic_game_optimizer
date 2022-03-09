import os, sys, time 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import pendulum_action as pendulum 
import matplotlib.pyplot as plt 

from solvers import full 



color = 'black'
LINE_WIDTH = 100 
horizon = 30
plan_dt = 1.e-2
x0 = np.zeros(2)


Q = 1e-2 * np.eye(2)
mu = 0.2


MAX_ITER = 1000
SOLVE_DDP = False 


pendulum_diff_running =  pendulum.DifferentialActionModelPendulum()
pendulum_diff_terminal = pendulum.DifferentialActionModelPendulum(isTerminal=True)
pendulum_running = crocoddyl.IntegratedActionModelEuler(pendulum_diff_running, plan_dt) 
pendulum_terminal = crocoddyl.IntegratedActionModelEuler(pendulum_diff_terminal, plan_dt) 
models = [pendulum_running]*(horizon) + [pendulum_terminal]
print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))
problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
xs = [x0]*(horizon+1)
us = [np.zeros(1)]*horizon

if SOLVE_DDP:
    ddp = crocoddyl.SolverFDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])   
    converged = ddp.solve(xs,us, MAX_ITER)
    print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
    plt.figure("theta plot")
    plt.plot(np.array(ddp.xs)[:,0], label="ddp")
    plt.show()
    plt.figure("velocity plot")
    plt.plot(np.array(ddp.xs)[:,1], label="ddp")
    plt.show()



dg_solver = full.SaddlePointSolver(problem, mu, Q)
print(" Constructing saddle point solver completed ".center(LINE_WIDTH, '-'))
dg_solver.solve(xs, us)
print(dg_solver.isFeasible)
print(dg_solver.gap_norms)

xnom = [d.xnext.copy() for d in dg_solver.problem.runningDatas]


plt.figure("theta plot")
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = np.array(dg_solver.xs)[t]
    x_n = xnom[t]
    if t==0:
        plt.plot(np.array([t, t+1]), np.array([x[0], x_n[0]]), color, label=str(mu))
    else:
        plt.plot(np.array([t, t+1]), np.array([x[0], x_n[0]]), color)
plt.legend()
plt.show()




plt.figure("portrait phase")
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = np.array(dg_solver.xs)[t]
    x_n = xnom[t]
    if t==0:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color, label=str(mu))
    else:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color)
# plt.plot(np.array(dg_solver.xs)[:,0], np.array(dg_solver.xs)[:,1], label="disturbed")
# plt.plot(np.array(xnom)[:,0], np.array(xnom)[:,1], label="nominal")
plt.legend()
plt.show()
