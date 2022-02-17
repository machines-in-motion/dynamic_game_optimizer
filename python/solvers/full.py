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
        self.mu = -1. 
        self.allocateData()


    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass()  

        if recalc:
            for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.du[t][:] = -self.K[t].dot(self.dx[t]) - self.k[t] 
                Lb = scl.cho_factor(self.Gammas[t], lower=True) 
                dx_right = self.mu*self.Q[t+1].dot(self.vx[t+1]) - self.ws[t+1]
                dx_right += data.Fx.dot(self.dx[t]) + data.Fx.dot(self.dx[t])    
                self.dx[t+1][:] = scl.cho_solve(Lb, dx_right)



    def backwardPass(self): 
        self.Vxx[-1][:,:] = self.problem.terminalData.Lxx
        self.vx[-1][:] = self.problem.terminalData.Lx 
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels,self.problem.runningDatas)):
            self.Gammas[t][:,:] = np.eye(model.state.ndx) - self.mu*self.Vxx[t+1].dot(self.Q[t+1]) 
            Lb = scl.cho_factor(self.Gammas[t], lower=True) 
            aux1 = scl.cho_solve(Lb, self.Vxx[t+1])
            aux2 = scl.cho_solve(Lb, self.vx[t+1])
            Quu = data.Luu + data.Fu.T.dot(aux1).dot(data.Fu) 
            Qux = data.Lxu.T + data.Fu.T.dot(aux1).dot(data.Fx)
            Qu = data.Lu + data.Fu.T.dot(aux2) + data.Fu.T.dot(aux1).dot(self.ws[t+1])

            Lb_uu = scl.cho_factor(Quu, lower=True)  
            self.K[t][:,:] = scl.cho_solve(Lb_uu, Qux)
            self.k[t][:] = scl.cho_solve(Lb_uu, Qu)
            self.Vxx[t][:,:] = data.Lxx + data.Fx.T.dot(aux1).dot(data.Fx) - Qux.T.dot(self.K[t])
            self.vx[t][:] =  data.Lx + data.Fx.T.dot(aux2 - aux1.dot(self.ws[t+1])) + Qux.T.dot(self.k[t])



    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        raise NotImplementedError("solve method not implemented yet!")  

    def allocateData(self):
        self.ws = [np.zeros(m.state.nx) for m in self.models()] 
        self.Q = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]   
        # 
        self.xs_try = [self.problem.x0] + [np.nan]*self.problem.T 
        self.us_try = [np.nan]*self.problem.T
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