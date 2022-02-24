""" a demo for the partially observable case with the point cliff example """

import os, sys, time 
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import point_cliff_action as point_cliff 
import matplotlib.pyplot as plt 
from utils.measurements import FullStateMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver

LINE_WIDTH = 100 
horizon = 100 
plan_dt = 1.e-2 
x0 = np.zeros(4)

MAX_ITER = 1000
SOLVE_DDP = False 
pm = 1.e-3*np.eye(4) # process error weight matrix 
mm = np.eye(4) # measurement error weight matrix 
MU = -1 

if __name__ == "__main__":
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt) 
    process_models = [cliff_running]*(horizon) + [cliff_terminal]
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    ddp_problem = crocoddyl.ShootingProblem(x0, process_models[:-1], process_models[-1])

    measurement_models = [FullStateMeasurement(cliff_running, mm)]*horizon + [FullStateMeasurement(cliff_terminal, mm)]


    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    measurement_trajectory =  MeasurementTrajectory(measurement_models)

    xs = [x0]*(horizon+1)
    us = [np.zeros(2)]*horizon

    solver = PartialDGSolver(ddp_problem, MU, pm, measurement_trajectory)
    print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))