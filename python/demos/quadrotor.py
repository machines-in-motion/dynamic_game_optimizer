import os, sys

src_path = os.path.abspath("../")
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
from models.quadrotor_action import DifferentialActionModelQuadrotor
from solvers import full

dt = 0.05
T = 80
Q = 1e-4 * np.diag([1, 1, 1, 1, 1, 1])
mu = 1
color = "black"

x0 = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])

MAX_ITER = 1000
LINE_WIDTH = 100

print(" Testing Quadrotor with DDP ".center(LINE_WIDTH, "#"))
quad_diff_running = DifferentialActionModelQuadrotor()
quad_diff_terminal = DifferentialActionModelQuadrotor(isTerminal=True)
print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))


quadrotor_running = crocoddyl.IntegratedActionModelRK(quad_diff_running, crocoddyl.RKType.four, stepTime=dt)
quadrotor_terminal = crocoddyl.IntegratedActionModelRK(quad_diff_terminal, crocoddyl.RKType.four, stepTime=dt)

print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))
problem = crocoddyl.ShootingProblem(x0, [quadrotor_running] * T, quadrotor_terminal)
print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))
ddp = crocoddyl.SolverDDP(problem)

print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
xs = [x0] * (T + 1)
us = [np.zeros(2)] * T
converged = ddp.solve(xs, us, maxiter=1000)
if converged:
    print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
else:
    print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))


xs = ddp.xs
us = ddp.us
dg_solver = full.SaddlePointSolver(problem, mu, Q)
print(" Constructing saddle point solver completed ".center(LINE_WIDTH, "-"))
dg_solver.solve(xs, us, maxiter=1000)
x_traj = np.array(dg_solver.xs)
xnext = [d.xnext.copy() for d in dg_solver.problem.runningDatas]
plt.figure("trajectory plot")
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = x_traj[t]
    x_n = xnext[t]
    if t == 0:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color, label="DG")
    else:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color)

plt.plot(np.array(ddp.xs)[:, 0], np.array(ddp.xs)[:, 1], label="ddp")
plt.xlabel("$p_x$")
plt.ylabel("$p_y$")
plt.legend()
plt.grid()
plt.title("Trajectory")


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(np.array(ddp.xs)[:, 0], label="DDP")
ax2.plot(np.array(ddp.xs)[:, 1])
ax3.plot(np.array(ddp.xs)[:, 2])
ax1.plot(np.array(dg_solver.xs)[:, 0], label="DG")
ax2.plot(np.array(dg_solver.xs)[:, 1])
ax3.plot(np.array(dg_solver.xs)[:, 2])
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_ylabel("$p_x$")
ax2.set_ylabel("$p_y$")
ax3.set_ylabel("$\\theta$")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", prop={"size": 16})


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(ddp.us)[:, 0], label="DDP")
ax2.plot(np.array(ddp.us)[:, 1])
ax1.plot(np.array(dg_solver.us)[:, 0], label="DG")
ax2.plot(np.array(dg_solver.us)[:, 1])
ax1.grid()
ax2.grid()
ax1.set_ylabel("F1")
ax2.set_ylabel("F2")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", prop={"size": 16})
plt.show()
