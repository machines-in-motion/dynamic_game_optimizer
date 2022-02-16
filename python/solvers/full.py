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
    def __init__(self, shootingProblem, problemUncertainty, sensitivity):
        SolverAbstract.__init__(self, shootingProblem)
        


    def _forward_pass(self): 
        pass 

    def _backward_pass(self): 
        pass 