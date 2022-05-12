import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import os, sys
src_path = os.path.abspath('../')
sys.path.append(src_path)

from solvers.partial import PartialDGSolver
from solvers.partial_neutral import PartialNeutralSolver

def simulate(ddp_problem, MU, x0_hat, pm, P0, mm, process_models, measurement_trajectory, horizon, x_ws, u_ws):
    nx = len(x0_hat)
    ny = mm.shape[0]
    # Sample all the future noise.
    w_0 = np.random.multivariate_normal(np.zeros(nx), P0)
    process_noise = np.random.multivariate_normal(np.zeros(nx), pm, size=horizon)
    measurement_noise = np.random.multivariate_normal(np.zeros(ny), mm, size=horizon)


    # DG controller
    x_init = x_ws.copy()
    u_init = u_ws.copy()
    measurements = [np.zeros(ny)]
    states = [x0_hat + w_0]
    controls = []
    for t in range(horizon):
        # Solve dynamic game
        dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory, verbose=False)
        dg_solver.solve(init_xs=x_init, init_us=u_init, init_ys=measurements, maxiter=1000)

        x_init = dg_solver.xs.tolist()
        u_init = dg_solver.us.tolist()
        # Apply the first control input
        u = dg_solver.us[t]

        # Reccord control
        u_init[t] = dg_solver.us[t]
        controls.append(u)

        # Compute next state
        data = process_models[t].createData()
        process_models[t].calc(data, states[-1], u)
        x = data.xnext.copy()
        states.append(x + process_noise[t])

        # Compute next measurement
        data = measurement_trajectory.runningDatas[t+1]
        y = measurement_trajectory.runningModels[t+1].calc(data, x)
        measurements.append(y + measurement_noise[t])
    # Compute cost
    cost = dg_solver.problem.calc(states, controls)

    # Neutral controller
    x_init = x_ws.copy()
    u_init = u_ws.copy()
    measurements_DDP = [np.zeros(ny)]
    states_DDP = [x0_hat + w_0]
    controls_DDP = []
    for t in range(horizon):
        # Solve dynamic game
        dg_solver = PartialNeutralSolver(ddp_problem, pm, P0, measurement_trajectory, verbose=False)
        dg_solver.solve(init_xs=x_init, init_us=u_init, init_ys=measurements_DDP, maxiter=1000)

        x_init = dg_solver.xs.tolist()
        u_init = dg_solver.us.tolist()
        # Apply the first control input
        u = dg_solver.us[t]

        # Reccord control
        u_init[t] = u
        controls_DDP.append(u)

        # Compute next state
        data = process_models[t].createData()
        process_models[t].calc(data, states_DDP[-1], u)
        x = data.xnext.copy()
        states_DDP.append(x + process_noise[t])

        # Compute next measurement
        data = measurement_trajectory.runningDatas[t+1]
        y = measurement_trajectory.runningModels[t+1].calc(data, x)
        measurements_DDP.append(y + measurement_noise[t])
    
    # Compute cost
    cost_DDP = dg_solver.problem.calc(states_DDP, controls_DDP)

    measurements = np.array(measurements)
    measurements_DDP = np.array(measurements_DDP)

    return cost, states, controls, measurements, cost_DDP, states_DDP, controls_DDP, measurements_DDP