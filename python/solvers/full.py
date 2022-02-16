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
        raise NotImplementedError("allocateData method not implemented yet!")  