import os, sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
from models import lin_quad_action as lin_quad
from solvers import full


horizon = 100
plan_dt = 1e-2
x0 = np.zeros(4)
Q = 1e-2 * np.eye(4)
mu = 0.01

LINE_WIDTH = 100

lq_diff_running = lin_quad.DifferentialActionModelLQ()
lq_diff_terminal = lin_quad.DifferentialActionModelLQ(isTerminal=True)
lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, plan_dt)
lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, plan_dt)
models = [lq_running] * (horizon) + [lq_terminal]
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
    plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), "black")
plt.xlabel('x')
plt.ylabel('y')
plt.title("$\mu = $ " + str(mu))
plt.show()
