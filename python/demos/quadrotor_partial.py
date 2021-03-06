import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
import os, sys
src_path = os.path.abspath("../")
sys.path.append(src_path)
from models import quadrotor_action
from utils.measurements import FullStateMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver


dt = 0.05
horizon = 60
MU = 10
pm = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
P0 = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])
mm = 1e-5 * np.diag([10, 10, 0.1, 0.1, 0.1, 0.1])  # measurement error weight matrix
t_solve = 20

x0 = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])


LINE_WIDTH = 100

print(" Testing Quadrotor with DDP ".center(LINE_WIDTH, "#"))
quadrotor_diff_running = quadrotor_action.DifferentialActionModelQuadrotor()
quadrotor_diff_terminal = quadrotor_action.DifferentialActionModelQuadrotor(isTerminal=True)
quadrotor_running = quadrotor_action.IntegratedActionModelQuadrotor(quadrotor_diff_running, dt)
quadrotor_terminal = quadrotor_action.IntegratedActionModelQuadrotor(quadrotor_diff_terminal, dt)
print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))
ddp_problem = crocoddyl.ShootingProblem(x0, [quadrotor_running] * horizon, quadrotor_terminal)
print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))
ddp_solver = crocoddyl.SolverDDP(ddp_problem)
print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
ddp_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
x_init = [x0] * (horizon + 1)
u_init = [np.zeros(2)] * horizon
converged = ddp_solver.solve(x_init, u_init, maxiter=1000)
if converged:
    print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
else:
    print(" DDP solver DID NOT CONVERGED ".center(LINE_WIDTH, "-"))


measurement_models = [FullStateMeasurement(quadrotor_running, mm)] * horizon + [FullStateMeasurement(quadrotor_terminal, mm)]
print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))
measurement_trajectory = MeasurementTrajectory(measurement_models)

ys = measurement_trajectory.calc(ddp_solver.xs[: t_solve + 1], ddp_solver.us[:t_solve])

print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, "-"))
u_init =  ddp_solver.us

u_init[:t_solve] = ddp_solver.us[:t_solve]
x_init = ddp_solver.xs
dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
print(" Constructing saddle point solver completed ".center(LINE_WIDTH, "-"))
dg_solver.solve(init_xs=x_init, init_us=u_init, init_ys=ys, maxiter=1000)

x_traj = np.array(dg_solver.xs)
xnext = [d.xnext.copy() for d in dg_solver.problem.runningDatas]
plt.figure("trajectory plot")
color = "black"
for t in range(len(np.array(dg_solver.xs[:-1]))):
    x = x_traj[t]
    x_n = xnext[t]
    if t == 0:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color, label="DG")
    else:
        plt.plot(np.array([x[0], x_n[0]]), np.array([x[1], x_n[1]]), color)
plt.plot(np.array(ddp_solver.xs)[:, 0], np.array(ddp_solver.xs)[:, 1], label="DPP")
plt.scatter(np.array(ddp_solver.xs)[t_solve, 0], np.array(ddp_solver.xs)[t_solve, 1], s=150.0, alpha=1.0, zorder=2.0,)
plt.scatter(np.array(dg_solver.xs)[t_solve, 0], np.array(dg_solver.xs)[t_solve, 1], s=150.0, alpha=1.0, zorder=2.0,)
plt.legend()
plt.grid()
plt.xlabel(r"$p_x$ [m]")
plt.ylabel(r"$p_y$ [m]")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(np.array(ddp_solver.xs)[:, 0], label="DDP")
ax2.plot(np.array(ddp_solver.xs)[:, 1])
ax3.plot(np.array(ddp_solver.xs)[:, 2])
ax1.plot(np.array(dg_solver.xs)[:, 0], label="DG")
ax2.plot(np.array(dg_solver.xs)[:, 1])
ax3.plot(np.array(dg_solver.xs)[:, 2])
ax1.grid()
ax2.grid()
ax3.grid()
ax1.set_ylabel(r"$p_x$")
ax2.set_ylabel(r"$p_y$")
ax3.set_ylabel(r"$\theta$")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", prop={"size": 16})

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(np.array(ddp_solver.us)[:, 0], label="DDP")
ax2.plot(np.array(ddp_solver.us)[:, 1])
ax1.plot(np.array(dg_solver.us)[:, 0], label="DG")
ax2.plot(np.array(dg_solver.us)[:, 1])
ax1.grid()
ax2.grid()
ax1.set_ylabel(r"$u_1$")
ax2.set_ylabel(r"$u_2$")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", prop={"size": 16})
plt.show()