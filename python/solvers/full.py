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

class SaddlePointSolver(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)

        self.allocateData()


    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self):
        raise NotImplementedError("calc method not implemented yet!") 
    
    def _forward_pass(self): 
        raise NotImplementedError("_forward_pass method not implemented yet!")  

    def backwardPass(self): 
        raise NotImplementedError("backwardPass method not implemented yet!")  

    def backwardPass(self): 
        raise NotImplementedError("backwardPass method not implemented yet!")  
 
    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass() 
        # print("Backward Pass is Done")
        


    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        raise NotImplementedError("solve method not implemented yet!")  


    def allocateData(self):
        self.xs_try = [self.problem.x0] + [np.nan]*self.problem.T 
        self.us_try = [np.nan]*self.problem.T
        self.ws = [np.zeros(m.state.nx) for m in self.models()]
        self.ws_try = [np.zeros(m.state.nx) for m in self.models()]
        self.dx = [np.zeros(m.state.ndx) for m  in self.models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dw = [np.zeros(m.state.ndx) for m in self.models()]

        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        self.vx = [np.zeros(m.state.ndx) for m in self.models()]   
        self.dv = [0. for _ in self.models()]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
