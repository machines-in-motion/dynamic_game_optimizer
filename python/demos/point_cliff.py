import os, sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
from models import point_cliff_action as point_cliff
from solvers import full


horizon = 100
plan_dt = 1.0e-2
x0 = np.zeros(4)
Q = 1e-2 * np.eye(4)
mu = 0.1
mu_list = [-1, -0.1, 0.01, 0.1]
color_list = ["darkblue", "blue", "black", "red"]


MAX_ITER = 1000
LINE_WIDTH = 100

for mu, color in zip(mu_list, color_list):
    cliff_diff_running = point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt)
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt)
    models = [cliff_running] * (horizon) + [cliff_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    xs = [x0] * (horizon + 1)
    us = [np.zeros(2)] * horizon

    dg_solver = full.SaddlePointSolver(problem, mu, Q)
    print(" Constructing saddle point solver completed ".center(LINE_WIDTH, "-"))
    dg_solver.solve(xs, us)

    x_traj = np.array(dg_solver.xs)
    xnext = [d.xnext.copy() for d in dg_solver.problem.runningDatas]

    plt.figure("trajectory plot")
    for t in range(len(np.array(dg_solver.xs[:-1]))):
        x = x_traj[t]
        x_n = xnext[t]
        if t == 0:
            plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color, label=str(mu))
        else:
            plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color)
    plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
