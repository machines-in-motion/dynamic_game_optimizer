import numpy as np
import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract
import eigenpy


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class PartialNeutralSolver(SolverAbstract):
    def __init__(self, shootingProblem, Q, P, measurement_trajectory, verbose=True): 
        SolverAbstract.__init__(self, shootingProblem)
        self.verbose = verbose
        self.merit = 0.
        self.merit_try = 0. 
        self.alphas = [2**(-n) for n in range(10)]
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
        
        y_pred = self.measurement_trajectory.calcDiff(self.xs, self.us, recalc=True)
        for t,(m, d) in enumerate(zip(self.measurement_trajectory.runningModels[:self.split_t+1],
                                                    self.measurement_trajectory.runningDatas[:self.split_t+1])):
            self.R[t][:,:] = d.R 
            self.gammas[t][:] = m.diff(y_pred[t], self.ys[t])
        

    def computeDirectionEstimation(self, recalc=True):
        if recalc:
            self.calc()

        self._estimation_forward()
        self._estimation_backward()

    def computeDirectionControl(self, recalc=True):
        if recalc:
            self.calc()

        self._control_backward()

    def _estimation_forward(self):
        self.P[0][:,:] = self.initial_covariance 
        self.mu_hat[0][:] = self.problem.runningModels[0].state.diff(self.xs[0], self.x0_est)
        for t, (pmodel, pdata) in enumerate(zip(self.problem.runningModels[:self.split_t],
                                              self.problem.runningDatas[:self.split_t])):
            mdata = self.measurement_trajectory.runningDatas[t+1]
            Pbar =  self.Q[t+1] + pdata.Fx.dot(self.P[t].dot(pdata.Fx.T)) 
            aux1 = self.R[t+1] + mdata.Hx.dot(Pbar).dot(mdata.Hx.T)
            Lb1 = eigenpy.LDLT(aux1) 
            K_transpose = Lb1.solve(mdata.Hx.dot(Pbar.T))
            self.K_filter[t+1][:,:] = K_transpose.T
            aux2 = np.eye(pmodel.state.ndx) - self.K_filter[t+1].dot(mdata.Hx)
            self.P[t+1][:,:] = aux2.dot(Pbar)
            self.mu_hat[t+1][:] = self.K_filter[t+1].dot(self.gammas[t+1]) \
                        +aux2.dot(pdata.Fx.dot(self.mu_hat[t]) - self.ws[t+1])
        self.dx[self.split_t] = self.mu_hat[self.split_t]

    def _control_backward(self):
        self.Vxx[-1][:,:] = self.problem.terminalData.Lxx
        self.vx[-1][:] = self.problem.terminalData.Lx 
        for t_, (model, data) in rev_enumerate(zip(self.problem.runningModels[self.split_t:],
                                                self.problem.runningDatas[self.split_t:])):
            t = self.split_t + t_
            Quu =  data.Luu + data.Fu.T.dot(self.Vxx[t+1]).dot(data.Fu) 
            Qux =  data.Lxu.T + data.Fu.T.dot(self.Vxx[t+1]).dot(data.Fx)
            if len(Qux.shape) == 1:
                Qux = np.resize(Qux,(1,Qux.shape[0]))
            Qu = data.Lu + data.Fu.T.dot(self.vx[t+1])
            #
            Lb_uu = eigenpy.LDLT(Quu)
            self.K[t][:,:] =  Lb_uu.solve(Qux)
            self.k[t][:] =  Lb_uu.solve(Qu)
            
            Vxx_tmp_ =  data.Lxx + data.Fx.T.dot(self.Vxx[t+1]).dot(data.Fx) - Qux.T.dot(self.K[t])
            self.Vxx[t][:,:] =  0.5 * (Vxx_tmp_ + Vxx_tmp_.T)
            self.vx[t][:] =  data.Lx + data.Fx.T.dot(self.vx[t+1]) - Qux.T.dot(self.k[t])
        


    def _estimation_backward(self):
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels[:self.split_t],
                                                  self.problem.runningDatas[:self.split_t])):
            Pinv = np.linalg.inv(self.P[t])
            right = Pinv.dot(self.mu_hat[t]) + data.Fx.T.dot(self.invQ[t+1]).dot(self.ws[t+1] + self.dx[t+1]) 
            aux = Pinv + data.Fx.T.dot(self.invQ[t+1]).dot(data.Fx)
            Lb = eigenpy.LDLT(aux)  
            self.dx[t] = Lb.solve(right) 

    
    def tryStepEstimation(self, alpha):
        # Estimation
        for t, (model, data) in enumerate(zip(self.problem.runningModels[:self.split_t+1], self.merit_runningDatas[:self.split_t+1])):
            self.xs_try[t][:] = model.state.integrate(self.xs[t], alpha*self.dx[t])
            self.us_try[t][:] = self.us[t] + alpha*self.du[t]
            model.calc(data, self.xs_try[t], self.us_try[t]) 
            model.calcDiff(data, self.xs_try[t], self.us_try[t]) 
            if t == 0:
                continue
            self.ws_try[t][:] = model.state.diff(self.merit_runningDatas[t-1].xnext, self.xs_try[t]) 
            
        if self.split_t == self.problem.T:
            self.xs_try[-1][:] = self.problem.terminalModel.state.integrate(self.xs[-1], alpha*self.dx[-1])
            self.ws_try[-1][:]  = self.problem.terminalModel.state.diff(self.merit_runningDatas[-1].xnext, self.xs_try[-1])
            self.problem.terminalModel.calc(self.merit_terminalData, self.xs_try[-1])
            self.problem.terminalModel.calcDiff(self.merit_terminalData, self.xs_try[-1])  
        
        self.merit_try_estimation = self.meritFunctionEstimation()
        return self.merit_try_estimation


    def meritFunctionEstimation(self):
        merit_estimation = 0 
        y_pred = self.measurement_trajectory.calcDiff(self.xs_try, self.us_try, recalc=True)

        state_err = self.problem.runningModels[0].state.diff(self.x0_est, self.xs_try[0])
        merit_estimation += state_err.T.dot(np.linalg.inv(self.initial_covariance).dot(state_err)) 
        for t, (model, data) in enumerate(zip(self.problem.runningModels[1:self.split_t+1], self.merit_runningDatas[1:self.split_t+1]), 1):
            merit_estimation += self.ws_try[t].T.dot(self.invQ[t].dot(self.ws_try[t]))
            mes_model = self.measurement_trajectory.runningModels[t]
            mes_data = self.measurement_trajectory.runningDatas[t]
            self.gammas_try[t][:] = mes_model.diff(y_pred[t] ,self.ys[t])
            merit_estimation += self.gammas_try[t].T.dot(mes_data.invR).dot(self.gammas_try[t])\
            
        if self.split_t == self.problem.T:
            mes_model = self.measurement_trajectory.runningModels[-1]
            mes_data = self.measurement_trajectory.runningDatas[-1]
            self.gammas_try[-1][:] = mes_model.diff(y_pred[-1] ,self.ys[-1])
            merit_estimation += self.gammas_try[-1].T.dot(mes_data.invR).dot(self.gammas_try[-1])\

        return merit_estimation

    
    def tryStepControl(self, alpha):
        self.merit_try_control = 0.  

        xnext = self.xs[self.split_t]
        merit_control = 0
        for t, (model, data) in enumerate(zip(self.problem.runningModels[self.split_t:], self.merit_runningDatas[self.split_t:]), self.split_t):
            dx = xnext - self.xs[t]
            self.xs_try[t][:] = xnext
            du = - self.K[t].dot(dx) - alpha*self.k[t]
            self.us_try[t][:] = self.us[t] + du
            model.calc(data, self.xs_try[t], self.us_try[t]) 
            model.calcDiff(data, self.xs_try[t], self.us_try[t]) 
            xnext = data.xnext
            merit_control += data.cost

        self.xs_try[-1][:] = xnext
        self.problem.terminalModel.calc(self.merit_terminalData, self.xs_try[-1])
        self.problem.terminalModel.calcDiff(self.merit_terminalData, self.xs_try[-1])  
        merit_control += self.merit_terminalData.cost
        # 
        self.merit_try_control = merit_control
        return self.merit_try_control


    def solve(self, init_xs=None, init_us=None, init_ys=None, maxiter=100, isFeasible=False, regInit=None):
        #___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [self.problem.x0.copy()] * (self.problem.T+1)
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels] 
        if init_ys is None:
            self.split_t = 0 
        else:
            self.split_t = len(init_ys) - 1
            self.ys[:self.split_t+1] = init_ys[:]  
        self.setCandidate(init_xs, init_us, False)
        self.calc()
        self.merit = self.tryStepEstimation(0.)
        converged = False
        if self.verbose:
            print("initial merit function = %s"%self.merit)
        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    self.computeDirectionEstimation(recalc=recalc)
                except:
                    raise BaseException("Backward Pass Failed")
                break 

            for a in self.alphas:
                try: 
                    
                    self.tryStepEstimation(a)
                    # print("try step for alpha = %s has merit = %s"%(a, self.merit_try_estimation))
                    dV = self.merit - self.merit_try_estimation
                    
                except:
                    # repeat starting from a smaller alpha 
                    print("Try Step Faild for alpha = %s"%a) 
                    continue
                
                if dV > 0:
                    if self.verbose:
                        print("step accepted for alpha = %s \n new merit is %s"%(a, self.merit_try_estimation))
                    self.setCandidate(self.xs_try, self.us_try, self.isFeasible) 
                    self.merit = self.merit_try_estimation
                    if dV < 1.e-12:
                        self.n_little_improvement += 1
                    break
            if a == self.alphas[-1]:
                if self.verbose:
                    print("No decrease found")
                break
            
            if self.n_little_improvement == 1:
                if self.verbose:
                    print("little improvements")
                converged = True 
                break

        if self.verbose:
            print("Estimation solved")
        if self.split_t == self.problem.T:
            return converged 
        self.n_little_improvement = 0
        converged = False
        self.calc()
        self.merit = self.tryStepControl(1.) # Take a feasible trajectory
        self.setCandidate(self.xs_try, self.us_try, self.isFeasible) 
        self.merit = self.tryStepControl(0.)
        if self.verbose:
            print("initial merit function = %s"%self.merit)
        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    self.computeDirectionControl(recalc=recalc)
                except:
                    raise BaseException("Backward Pass Failed")
                break 

            for a in self.alphas:
                try: 
                    self.tryStepControl(a)
                    # print("try step for alpha = %s has merit = %s"%(a, self.merit_try_control))
                    dV = self.merit - self.merit_try_control
                    
                except:
                    # repeat starting from a smaller alpha 
                    print("Try Step Faild for alpha = %s"%a) 
                    continue
                
                if dV > 0:
                    if self.verbose:
                        print("step accepted for alpha = %s \n new merit is %s"%(a, self.merit_try_control))
                    self.setCandidate(self.xs_try, self.us_try, self.isFeasible) 
                    self.merit = self.merit_try_control
                    if dV < 1.e-12:
                        self.n_little_improvement += 1
                        if self.verbose:
                            print("little improvements")
                    break
            
            if a == self.alphas[-1]:
                if self.verbose:
                    print("No decrease found")
                converged = False
                break
            
            if self.n_little_improvement == 1:
                break
        self.calc()
        return True


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
        self.Q = [self.state_covariance for _ in self.process_models()]   
        self.invQ = [self.inv_state_covariance for _ in self.process_models()]   
        self.ys = [np.zeros(m.ny) for m in self.measurement_models()] 
        self.gammas = [np.zeros(m.ny) for m in self.measurement_models()] 
        self.gammas_try = [np.zeros(m.ny) for m in self.measurement_models()] 
        self.mu_hat = [np.zeros(m.state.nx) for m in self.process_models()] 
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
        self.E = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.measurement_models()] 
        self.x_grad = [np.zeros(m.state.ndx) for m in self.process_models()]
        self.u_grad = [np.zeros(m.nu) for m in self.problem.runningModels]
        # 
        self.merit_runningDatas = [m.createData() for m in self.problem.runningModels]
        self.merit_terminalData = self.problem.terminalModel.createData()