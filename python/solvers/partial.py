""" solve for full observable case 

"""
import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100 

VERBOSE = False    

def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class PartialDGSolver(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        self.mu = 1.
        self.inv_mu = 1./self.mu  
        self.merit = 0.
        self.merit_try = 0. 
        self.alphas = [2**(-n) for n in range(10)]
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop =  1.e-9 
        self.n_little_improvement = 0
        self.state_covariance = 1.e-3 
        self.inv_state_covariance = 1./self.state_covariance
        # 
        self.merit_runningDatas = [m.createData() for m in self.problem.runningModels]
        self.merit_terminalData = self.problem.terminalModel.createData()  
        self.gap_norms = 0. 
        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self):
        raise NotImplementedError("solve method not implemented yet") 

    def computeDirection(self, recalc=True):
        raise NotImplementedError("solve method not implemented yet")

    def computeUpdates(self): 
        raise NotImplementedError("solve method not implemented yet")

    def backwardPass(self): 
        raise NotImplementedError("solve method not implemented yet")

    def tryStep(self, alpha):
        raise NotImplementedError("solve method not implemented yet")

    def meritFunction(self, t, data_previous):
        raise NotImplementedError("solve method not implemented yet")


    def solve(self, init_xs=None, init_us=None, maxiter=10, isFeasible=False, regInit=None):
        raise NotImplementedError("solve method not implemented yet")



    def allocateData(self):
        self.ws = [np.zeros(m.state.nx) for m in self.models()] 
        self.Q = [self.state_covariance*np.eye(m.state.ndx) for m in self.models()]   
        self.invQ = [self.inv_state_covariance*np.eye(m.state.ndx) for m in self.models()]   
        # 
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        self.ws_try = [np.zeros(m.state.nx) for m in self.models()]
        # 
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dw = [np.zeros(m.state.ndx) for m in self.models()]
        # 
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        self.vx = [np.zeros(m.state.ndx) for m in self.models()]   
        self.dv = [0. for _ in self.models()]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # 
        self.Gammas = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]  
        #
        self.x_grad = [np.zeros(m.state.ndx) for m in self.models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]
