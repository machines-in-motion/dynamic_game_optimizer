import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
import os, sys, time
src_path = os.path.abspath('../')
sys.path.append(src_path)



from models import quadrotor_action as quadrotor 
from utils.measurements import PositionMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
from solvers.partial_neutral import PartialNeutralSolver
from simulation import simulate

LINE_WIDTH = 100 


plan_dt = 0.05
horizon = 60


pm = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
P0 = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])
mm = 1e-5 * np.diag([10, 10, 0.1])  # measurement error weight matrix
MU = 8


x0_hat = np.array([0, 0, 0.0, 0.0, 0.0, 0.0])


quadrotor_diff_running =  quadrotor.DifferentialActionModelQuadrotor()
quadrotor_diff_terminal = quadrotor.DifferentialActionModelQuadrotor(isTerminal=True)
quadrotor_running = quadrotor.IntegratedActionModelQuadrotor(quadrotor_diff_running, plan_dt)
quadrotor_terminal = quadrotor.IntegratedActionModelQuadrotor(quadrotor_diff_terminal, plan_dt)


process_models = [quadrotor_running]*(horizon) + [quadrotor_terminal]

print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

ddp_problem = crocoddyl.ShootingProblem(x0_hat, process_models[:-1], process_models[-1])
measurement_models = [PositionMeasurement(quadrotor_running, mm)]*horizon + [PositionMeasurement(quadrotor_terminal, mm)]
print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))

measurement_trajectory = MeasurementTrajectory(measurement_models)


x_init = [x0_hat]*(horizon+1)
u_init = [np.zeros(2)]*horizon

ddp_solver = crocoddyl.SolverDDP(ddp_problem)
ddp_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
ddp_solver.solve(init_xs=x_init, init_us=u_init)
state_ddp_plan = np.array(ddp_solver.xs)
control_ddp_plan = np.array(ddp_solver.us)
next_state_ddp_plan = [d.xnext.copy() for d in ddp_solver.problem.runningDatas]
x_init = ddp_solver.xs.tolist().copy()
u_init = ddp_solver.us.tolist().copy()

cost, states, controls, measurements, cost_DDP, states_DDP, controls_DDP, measurements_DDP = simulate(ddp_problem, MU, x0_hat, pm, P0, mm, process_models, measurement_trajectory, horizon, x_init, u_init)

print("DDP cost = " + str(cost_DDP))
print("DG cost = " + str(cost))

# Planning phase (t=0)
dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
dg_solver.solve(init_xs=x_init, init_us=u_init, init_ys=[0], maxiter=1000)
state_dg_plan = np.array(dg_solver.xs)
control_dg_plan = np.array(dg_solver.us)


x = np.array(states)
x_DDP = np.array(states_DDP)


plt.figure()
plt.plot(state_dg_plan[:, 0], state_dg_plan[:, 1], linestyle='-.', color='blue', label='DG plan')
plt.plot(state_ddp_plan[:, 0], state_ddp_plan[:, 1], linestyle='-.', color='k', label='DDP plan')
plt.plot(x[:, 0], x[:, 1], 'blue', label='DG')
plt.plot(x_DDP[:, 0], x_DDP[:, 1], 'black', label='DDP')
plt.xlabel(r"$p_x$ [m]")
plt.ylabel(r"$p_y$ [m]")
plt.grid()
plt.legend()


time_lin = np.linspace(0, plan_dt*len(x), len(x))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.plot(time_lin, x[:, 0], label="DG")
ax2.plot(time_lin, x[:, 1])
ax3.plot(time_lin, x[:, 2])
ax1.plot(time_lin, measurements[:, 0], 'x')
ax2.plot(time_lin, measurements[:, 1], 'x')
ax3.plot(time_lin, measurements[:, 2], 'x')
ax1.plot(time_lin, x_DDP[:, 0], '.-', label="DDP")
ax2.plot(time_lin, x_DDP[:, 1], '.-')
ax3.plot(time_lin, x_DDP[:, 2], '.-')
ax1.plot(time_lin, measurements_DDP[:, 0], 'x')
ax2.plot(time_lin, measurements_DDP[:, 1], 'x')
ax3.plot(time_lin, measurements_DDP[:, 2], 'x')
ax1.grid()
ax2.grid()
ax3.grid()
ax3.set_xlabel("time [s]")
ax1.set_ylabel(r"$p_x$ [m]")
ax2.set_ylabel(r"$p_y$ [m]")
ax3.set_ylabel(r"$\theta$ [rad/s]")

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(time_lin[:-1], np.array(controls)[:, 0], label="DG")
ax2.plot(time_lin[:-1], np.array(controls)[:, 1], label="DG")
ax1.plot(time_lin[:-1], np.array(controls_DDP)[:, 0], label="DDP")
ax2.plot(time_lin[:-1], np.array(controls_DDP)[:, 1], label="DDP")
ax1.grid()
ax2.grid()
ax2.set_xlabel("time [s]")
ax1.set_ylabel(r"$u_1$ [N]")
ax2.set_ylabel(r"$u_2$ [N]")
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', prop={'size': 16})
plt.show()