import matplotlib.patches as m_patches
import matplotlib.pyplot as plt 
import numpy as np 
import crocoddyl 
from tqdm import tqdm
import os, sys, time
src_path = os.path.abspath('../')
sys.path.append(src_path)


from models.quadrotor_action import DifferentialActionModelQuadrotor
from utils.measurements import PositionMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
from solvers.partial_neutral import PartialNeutralSolver
from simulation import simulate

LINE_WIDTH = 100


def create_plots(P0, pm, mm, MU, N_simulation):
    nx = 6
    ny = 3
    nu = 2
     
    horizon = 60
    plan_dt = 0.05
    x0_hat = np.zeros(nx)

    quadrotor_diff_running =  DifferentialActionModelQuadrotor()
    quadrotor_diff_terminal = DifferentialActionModelQuadrotor(isTerminal=True)
    quadrotor_running = crocoddyl.IntegratedActionModelRK(quadrotor_diff_running, crocoddyl.RKType.four, stepTime=plan_dt)
    quadrotor_terminal = crocoddyl.IntegratedActionModelRK(quadrotor_diff_terminal, crocoddyl.RKType.four, stepTime=plan_dt)
    process_models = [quadrotor_running]*(horizon) + [quadrotor_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0_hat, process_models[:-1], process_models[-1])
    measurement_models = [PositionMeasurement(quadrotor_terminal, mm)]*horizon + [PositionMeasurement(quadrotor_terminal, mm)]
    measurement_trajectory = MeasurementTrajectory(measurement_models)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))

    
    x_init = [x0_hat]*(horizon+1)
    u_init = [np.zeros(nu)]*horizon
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    ddp_solver.solve(init_xs=x_init, init_us=u_init)
    state_ddp_plan = np.array(ddp_solver.xs)
    control_ddp_plan = np.array(ddp_solver.us)
    x_init = ddp_solver.xs.tolist().copy()
    u_init = ddp_solver.us.tolist().copy()


    dg_solver = PartialDGSolver(ddp_problem, MU, pm, P0, measurement_trajectory)
    dg_solver.solve(init_xs=x_init, init_us=u_init)
    state_dg_plan = np.array(dg_solver.xs)
    control_dg_plan = np.array(dg_solver.us)


    record_cost = []
    record_states = []
    record_controls = []
    record_cost_DDP = []
    record_states_DDP = []
    record_controls_DDP = []

    n_fail = 0
    for sim in tqdm(range(N_simulation)):
        try: 
            cost, states, controls, _, cost_DDP, states_DDP, controls_DDP, _ = simulate(ddp_problem, MU, x0_hat, pm, P0, mm, process_models, measurement_trajectory, horizon, x_init, u_init)
            record_cost.append(cost)
            record_states.append(np.array(states))
            record_controls.append(np.array(controls))
            record_cost_DDP.append(cost_DDP)
            record_states_DDP.append(np.array(states_DDP))
            record_controls_DDP.append(np.array(controls_DDP))
        except:
            print("Simulation failed")
            n_fail += 1
            continue
    print(n_fail, "simulation failed")
    record_cost = np.array(record_cost)
    record_states = np.array(record_states)
    record_controls = np.array(record_controls)
    record_cost_DDP = np.array(record_cost_DDP)
    record_states_DDP = np.array(record_states_DDP)
    record_controls_DDP = np.array(record_controls_DDP)

    np.save('record_cost.npy', record_cost) 
    np.save('record_states.npy', record_states) 
    np.save('record_controls.npy', record_controls) 
    np.save('record_cost_DDP.npy', record_cost_DDP) 
    np.save('record_states_DDP.npy', record_states_DDP) 
    np.save('record_controls_DDP.npy', record_controls_DDP) 


if __name__ == "__main__":

    N_simulation = 1000
    MU = 8
    pm = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])  # process error weight matrix
    P0 = 1e-5 * np.diag([1, 1, 1, 1, 1, 1])
    mm = 1e-5 * np.diag([10, 10, 0.1])  # measurement error weight matrix

    create_plots(P0, pm, mm, MU, N_simulation)
