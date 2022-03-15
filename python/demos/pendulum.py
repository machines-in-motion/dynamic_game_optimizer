import os, sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

import matplotlib.pyplot as plt
import numpy as np
import crocoddyl
from models import pendulum_action as pendulum
from solvers import full


horizon = 30
plan_dt = 1e-2
x0 = np.zeros(2)
Q = 1e-2 * np.eye(2)
mu = 0.01


MAX_ITER = 10000
LINE_WIDTH = 100

pendulum_diff_running = pendulum.DifferentialActionModelPendulum()
pendulum_diff_terminal = pendulum.DifferentialActionModelPendulum(isTerminal=True)
pendulum_running = crocoddyl.IntegratedActionModelEuler(pendulum_diff_running, plan_dt)
pendulum_terminal = crocoddyl.IntegratedActionModelEuler(pendulum_diff_terminal, plan_dt)
models = [pendulum_running] * (horizon) + [pendulum_terminal]
print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))
problem = crocoddyl.ShootingProblem(x0, models[:-1], models[-1])
print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))
xs = [x0] * (horizon + 1)
us = [np.zeros(1)] * horizon


dg_solver = full.SaddlePointSolver(problem, mu, Q)
print(" Constructing saddle point solver completed ".center(LINE_WIDTH, "-"))
dg_solver.solve(xs, us)


x_traj = np.array(dg_solver.xs)
xnext = [d.xnext.copy() for d in dg_solver.problem.runningDatas]

plt.figure("Theta plot")
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = x_traj[t]
    x_n = xnext[t]
    plt.plot(np.array([t, t + 1]), np.array([x[0], x_n[0]]), "black")
plt.xlabel("Time")
plt.ylabel("$\\theta$")
plt.title("$\mu = $ " + str(mu))

plt.figure("Phase portrait")
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = x_traj[t]
    x_n = xnext[t]
    plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), "black")
plt.xlabel("x")
plt.ylabel("y")
plt.title("$\mu = $ " + str(mu))
plt.show()
