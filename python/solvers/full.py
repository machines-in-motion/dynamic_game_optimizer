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
    def __init__(self, shootingProblem, mu, Q):
        SolverAbstract.__init__(self, shootingProblem)
        self.mu = mu
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
        self.state_covariance = Q 
        self.inv_state_covariance = np.linalg.inv(self.state_covariance) 
        self.gap_norms = 0. 
        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
        
    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.ws[0][:] = np.zeros(self.problem.runningModels[0].state.ndx)
        self.gap_norms = 0. 
        for t, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
            self.ws[t + 1] = m.state.diff(d.xnext, x)
            self.gap_norms += np.linalg.norm(self.ws[t+1])
        if self.gap_norms > 1.e-8:
            self.isFeasible = False 
        else:
            self.isFeasible = True  

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass()  
        self.computeUpdates()

    def computeUpdates(self): 
        """ computes step updates dx and du """
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # here we compute the direction 
                self.du[t][:] = -self.K[t].dot(self.dx[t]) - self.k[t] 
                Gamma = np.eye(model.state.ndx) - self.mu*self.Q[t+1].dot(self.Vxx[t+1])
                Lb = scl.cho_factor(Gamma, lower=True) 
                dx_right = self.mu*self.Q[t+1].dot(self.vx[t+1]) - self.ws[t+1]
                dx_right += data.Fx.dot(self.dx[t]) + data.Fu.dot(self.du[t])    
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
            Qu = data.Lu + data.Fu.T.dot(aux2) - data.Fu.T.dot(aux1).dot(self.ws[t+1])
            #
            Lb_uu = scl.cho_factor(Quu, lower=True)  
            self.K[t][:,:] = scl.cho_solve(Lb_uu, Qux)
            self.k[t][:] = scl.cho_solve(Lb_uu, Qu)
            self.Vxx[t][:,:] = data.Lxx + data.Fx.T.dot(aux1).dot(data.Fx) - Qux.T.dot(self.K[t])
            self.vx[t][:] =  data.Lx + data.Fx.T.dot(aux2 - aux1.dot(self.ws[t+1])) - Qux.T.dot(self.k[t])

    def tryStep(self, alpha):
        self.merit_try = 0. 
        data_prev = None
        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.merit_runningDatas)):
            self.xs_try[t][:] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t][:] = self.us[t] + alpha*self.du[t]
            model.calc(data, self.xs_try[t], self.us_try[t])
            model.calcDiff(data, self.xs_try[t], self.us_try[t]) 
            if t == 0:
                data_prev = self.merit_runningDatas[t] 
                continue
            self.ws_try[t][:] = model.state.diff(data_prev.xnext, self.xs_try[t]) 
            self.merit_try += self.meritFunction(t, data_prev)
            data_prev = self.merit_runningDatas[t] 

        self.xs_try[-1][:] = self.problem.terminalModel.state.integrate(self.xs[-1], alpha*self.dx[-1])
        self.ws_try[-1][:]  = self.problem.terminalModel.state.diff(data_prev.xnext, self.xs_try[-1])
        self.merit_try += self.meritFunction(self.problem.T, data_prev) # grad_x and grad_u  at T 
        # 
        self.problem.terminalModel.calc(self.merit_terminalData, self.xs_try[-1])
        self.problem.terminalModel.calcDiff(self.merit_terminalData, self.xs_try[-1])  
        self.merit_try += self.meritFunction(-1, self.merit_terminalData) # grad_x at T+1 
        return self.merit_try

    def meritFunction(self, t, data_previous):
        """ the computation of the cost gradients will lag by one time step, 
        i.e. at t = 1 we compute dJ/dx_0  
        we will use t = -1 for terminal state 
        """
        if t == -1:
            self.x_grad[t][:] = data_previous.Lx - self.inv_mu*self.ws_try[-1].T.dot(self.invQ[-1])
            return np.linalg.norm(self.x_grad[t])
        self.u_grad[t-1][:] = data_previous.Lu + self.inv_mu*self.ws_try[t].T.dot(self.invQ[t]).dot(data_previous.Fu) 
        if t == 1:
            return np.linalg.norm(self.u_grad[t-1])
        self.x_grad[t-1][:] = data_previous.Lx - self.inv_mu*self.ws_try[t-1].T.dot(self.invQ[t-1])
        self.x_grad[t-1][:] += self.inv_mu*self.ws_try[t].T.dot(self.invQ[t]).dot(data_previous.Fx)
        
        return np.linalg.norm(self.x_grad[t-1]) + np.linalg.norm(self.u_grad[t-1])     


    def solve(self, init_xs=None, init_us=None, maxiter=10, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()] 
            init_xs [0][:] = self.problem.x0.copy()
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 
        
        self.setCandidate(init_xs, init_us, False)
        self.calc() # compute the gaps 

        self.merit = self.tryStep(1.) # compute initial value for merit function 
        print("initial merit function is %s"%self.merit)

        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    self.computeDirection(recalc=recalc)
                except:
                    raise BaseException("Backward Pass Failed")
                break 

            for a in self.alphas:
                try: 
                    # print("try step for alpha = %s"%a)
                    self.tryStep(a)
                    dV = self.merit - self.merit_try
                    
                except:
                    # repeat starting from a smaller alpha 
                    print("Try Step Faild for alpha = %s"%a) 
                    continue 
                
                if dV> 0.:
                    print("step accepted for alpha = %s \n new merit is %s"%(a, self.merit_try))
                    self.setCandidate(self.xs_try, self.us_try, self.isFeasible) 
                    self.merit = self.merit_try
                    if dV < 1.e-12:
                        
                        self.n_little_improvement += 1
                        print("little improvements")
                    break
                # else:
                #     print("no decrease for alpha = %s"%a)




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
        # 
        self.merit_runningDatas = [m.createData() for m in self.problem.runningModels]
        self.merit_terminalData = self.problem.terminalModel.createData()  
