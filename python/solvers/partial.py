""" solve for full observable case 

"""
from cv2 import split
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
    def __init__(self, shootingProblem,  mu, Q, P, measurement_trajectory): 
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
        self.measurement_trajectory = measurement_trajectory 
        self.split_t = None 
        self.initial_covariance = P 
        # 
        self.allocateData()



    def process_models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod  
    
    def measurement_models(self):
        mod = [m for m in self.measurement_trajectory.runningModels]
        return mod

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.ws[0][:] = np.zeros(self.problem.runningModels[0].state.ndx)
        self.gap_norms = 0. 
        for t, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
            self.ws[t + 1] = m.state.diff(d.xnext, x)
        
        self.measurement_trajectory.calcDiff(self.xs, self.us, recalc=True)
        for t, d in enumerate(self.measurement_trajectory.runningDatas):
            self.R[t][:,:] = d.R 

    def computeDirection(self, recalc=True):
        if recalc:
            self.calc()

        self._estimation_forward()
        self._control_backward()
        self._coupling()
        self._estimation_backward()
        self._control_forward()

    def _estimation_forward(self):
        data0 = self.measurement_trajectory.runningDatas[0]  
        model0 = self.problem.runningModels[0]
        aux0 = self.R[0] + data0.Hx.dot(self.initial_covariance).dot(data0.Hx.T)
        Lb = scl.cho_factor(aux0, lower=True) 
        K_transpose = scl.cho_solve(Lb, data0.Hx.dot(self.initial_covariance)) 
        self.K_filter[0][:,:] = K_transpose.T  
        self.P[0][:,:] = self.initial_covariance - self.K_filter[0].dot(data0.Hx).dot(self.initial_covariance)
        dx0 = model0.state.diff(self.xs[0], self.x0_est)
        self.xhat[0][:] = self.K_filter[0].dot(self.gammas[0]) \
            + (np.eye(model0.state.ndx)+ self.K_filter[0].dot(data0.Hx)).dot(dx0) 
        for t, (pmodel, pdata) in enumerate(zip(self.problem.runningModels[:self.split_t],
                                              self.problem.runningDatas[:self.split_t])):
            mdata = self.measurement_trajectory.runningDatas[t+1]
            aux0 = np.eye(pmodel.state.ndx) - self.mu*self.P[t].dot(pdata.Lxx)
            Lb0 = scl.cho_factor(aux0, lower=True) 
            Pbar =  scl.cho_solve(Lb0, self.P[t].dot(pdata.Fx.T)) 
            Pbar = self.Q[t+1] + pdata.Fx.dot(Pbar) 
            aux1 = self.R[t+1] + mdata.Hx.dot(Pbar).dot(mdata.Hx.T)
            Lb1 = scl.cho_factor(aux1, lower=True) 
            K_transpose = scl.cho_solve(Lb1, mdata.Hx.dot(Pbar)) 
            self.K_filter[t+1][:,:] = K_transpose.T
            E = np.linalg.inv(self.P[t]) + pdata.Fx.dot(self.invQ[t+1]).dot(pdata.Fx.T) - self.mu*pdata.Lxx
            self.P[t+1][:,:] = (np.eye(pmodel.state.ndx) - self.K_filter[t+1].dot(mdata.Hx)).dot(Pbar)
            self.xhat[t+1][:] = self.K_filter[t+1].dot(self.gammas[t+1]) \
                        +(np.eye(pmodel.state.ndx) - self.K_filter[t+1].dot(mdata.Hx)).dot(pdata.Fx.dot(self.xhat[t]) - self.ws[t+1]) \
                        + self.P[t+1].dot(self.invQ[t+1]).dot(pdata.Fx).dot(np.linalg.inv(E)).dot(self.mu*pdata.Lxx.dot(self.xhat[t]) + self.mu*pdata.Lx)
 
    def _control_backward(self):
        self.Vxx[-1][:,:] = self.problem.terminalData.Lxx
        self.vx[-1][:] = self.problem.terminalData.Lx 
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels[self.split_t:],
                                                  self.problem.runningDatas[self.split_t:])):
            aux0 = np.eye(model.state.ndx) - self.mu*self.Vxx[t+1].dot(self.Q[t+1]) 
            Lb = scl.cho_factor(aux0, lower=True) 
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

    def _coupling(self):
        t = self.split_t 
        aux = np.eye(self.problem.runningModels[t].state.ndx) - self.mu*self.P[t].dot(self.Vxx[t])
        Lb = scl.cho_factor(aux, lower=True) 
        self.dx[t] = scl.cho_solve(Lb, self.xhat[t] + self.mu*self.P[t].dot(self.vx[t]))

    def _estimation_backward(self):
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels[:self.split_t],
                                                  self.problem.runningDatas[:self.split_t])):
            Pinv = np.linalg.inv(self.P[t])
            right = Pinv.dot(self.xhat[t]) + data.Fx.T.dot(self.invQ[t+1]).dot(self.ws[t+1]+ self.dx[t+1]) + self.mu*data.Lx
            Lb = scl.cho_factor(Pinv + data.Fx.T.dot(self.invQ[t+1]).dot(data.Fx) - self.mu*data.Lxx, lower=True)  
            self.dx[t] = scl.cho_solve(Lb, right) 

    def _control_forward(self):
        for t, (model, data) in enumerate(zip(self.problem.runningModels[self.split_t:],
                                                  self.problem.runningDatas[self.split_t:])):
            self.du[t] = -self.K[t].dot(self.dx[t]) - self.k[t]
            right = self.mu*self.Q[t+1].dot(self.vx[t]) + data.Fx.dot(self.dx[t]) + data.Fu.dot(self.du[t]) -self.ws[t]
            Lb = scl.cho_factor(np.eye(model.state.ndx) - self.mu*self.Q[t+1].dot(self.Vxx[t+1])) 
            self.dx[t+1] = scl.cho_solve(Lb, right)
    
    def tryStep(self, alpha):
        self.merit_try = 0. 
        for t, (model, data) in enumerate(zip(self.problem.runningModels,self.merit_runningDatas)):
            self.xs_try[t][:] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t][:] = self.us[t] + alpha*self.du[t]
            model.calc(data, self.xs_try[t], self.us_try[t])
            model.calcDiff(data, self.xs_try[t], self.us_try[t]) 
        
        self.xs_try[-1][:] = self.problem.terminalModel.state.integrate(self.xs[-1], alpha*self.dx[-1])
        self.ws_try[-1][:]  = self.problem.terminalModel.state.diff(self.merit_runningDatas[-1].xnext, self.xs_try[-1])
        self.problem.terminalModel.calc(self.merit_terminalData, self.xs_try[-1])
        self.problem.terminalModel.calcDiff(self.merit_terminalData, self.xs_try[-1])  
        
        self.merit_try = self.meritFunction()
        return self.merit_try

    def meritFunction(self):
        """ the computation of the cost gradients will lag by one time step, 
        i.e. at t = 1 we compute dJ/dx_0  
        we will use t = -1 for terminal state 
        """
        if t == -1: 
            pass 

        return 0.  


    def solve(self, init_xs=None, init_us=None, init_ys=None, maxiter=1, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()] 
            init_xs [0][:] = self.problem.x0.copy()
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 
        if init_ys is None:
            self.split_t = 0 
        else:
            self.split_t = len(init_ys)
            self.ys[:self.split_t] = init_ys[:]  
        self.setCandidate(init_xs, init_us, False)

        self.merit = self.tryStep(1.)

        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    self.computeDirection(recalc=recalc)
                except:
                    raise BaseException("Backward Pass Failed")
                break 

            self.tryStep(1.)





    def allocateData(self):
        """ Allocates space for the following 
        ws       :        
        Q        :       
        invQ     :     
        ys       :  
        gammas   :
        xhat     :    
        xs_try   :     
        us_try   :  
        ws_try   :      
        ys_try   :      
        dx       :     
        du       :     
        dw       :     
        dy       :     
        Vxx      :         
        vx       :    
        dv       :    
        K        :  
        k        :  
        K_filter :    
        R        : measurement cost weight   
        P        : estimate covariance 
        x_grad   :  
        u_grad   :    
        """
        self.ws = [np.zeros(m.state.nx) for m in self.process_models()] 
        self.Q = [self.state_covariance*np.eye(m.state.ndx) for m in self.process_models()]   
        self.invQ = [self.inv_state_covariance*np.eye(m.state.ndx) for m in self.process_models()]   
        self.ys = [np.zeros(m.ny) for m in self.measurement_models()] 
        self.gammas = [np.zeros(m.ny) for m in self.measurement_models()] 
        self.xhat = [np.zeros(m.state.nx) for m in self.process_models()] 
        self.xs_try = [np.zeros(m.state.nx) for m in self.process_models()] 
        self.xs_try[0][:] = self.problem.x0.copy()
        self.x0_est = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels] 
        self.ws_try = [np.zeros(m.state.nx) for m in self.process_models()]
        self.ys_try = [np.zeros(m.ny) for m in self.measurement_models()]
        self.dx = [np.zeros(m.state.ndx) for m  in self.process_models()]
        self.du = [np.zeros(m.nu) for m  in self.problem.runningModels] 
        self.dw = [np.zeros(m.state.ndx) for m in self.process_models()]
        self.dy = [np.zeros(m.ny) for m in self.measurement_models()]
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.process_models()]   
        self.vx = [np.zeros(m.state.ndx) for m in self.process_models()]   
        self.dv = [0. for _ in self.process_models()]
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        self.K_filter = [np.zeros([m.state.ndx, m.ny]) for m in self.measurement_models()]
        self.R = [np.zeros([m.ny, m.ny]) for m in self.measurement_models()] 
        self.P = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.measurement_models()] 
        self.x_grad = [np.zeros(m.state.ndx) for m in self.process_models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]
        # 
        self.merit_runningDatas = [m.createData() for m in self.problem.runningModels]
        self.merit_terminalData = self.problem.terminalModel.createData()  