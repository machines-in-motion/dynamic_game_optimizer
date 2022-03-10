""" this example is inspired by risk sensitive paper from farbod """

import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class PendulumDynamics:
    def __init__(self):
        self.g = 9.81
        self.mass = 1 
        self.l = 1
        self.nx = 2 
        self.ndx = 2
        self.nu = 1

    def nonlinear_dynamics(self, x, u):
        return - self.g * np.sin(x[0]) / self.l + u / (self.l**2 * self.mass) 
    
    
class DifferentialActionModelPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1.e-2, isTerminal=False):
        self.dynamics = PendulumDynamics()
        state =  crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, self.dynamics.nu, self.dynamics.ndx)
        self.ndx = self.state.ndx 
        self.isTerminal = isTerminal
        self.mass = self.dynamics.mass 
        self.l =  self.dynamics.l 
        self.g =  self.dynamics.g 
        self.dt = dt 
        self.Fxx = np.zeros([self.ndx, self.ndx, self.ndx])
        self.Fxu = np.zeros([self.ndx, self.ndx, self.nu])
        self.Fuu = np.zeros([self.ndx, self.nu, self.nu])


    def _running_cost(self, x, u):
        cost = 0.5 * u[0]**2 / 100.  
        return cost

    def _terminal_cost(self, x, u):
        cost = 2000*((x[0]-np.pi)**2) + 100*(x[1]**2)  
        return cost 
     
    def calc(self, data, x, u=None):
        if u is None: 
            u = np.zeros(self.nu)

        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
            data.xout = np.zeros(1)
        else:
            data.cost = self._running_cost(x,u)
            data.xout = self.dynamics.nonlinear_dynamics(x,u)


    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([1,2]) 
        Fu = np.zeros([1,1])
        
        Lx = np.zeros([2])
        Lu = np.zeros([1])
        Lxx = np.zeros([2,2])
        Luu = np.zeros([1,1])
        Lxu = np.zeros([2,1])
        if self.isTerminal:
            Lx[0] = 4000.*(x[0]-np.pi)
            Lx[1] = 200.*x[1]   
            Lxx[0,0] = 4000. 
            Lxx[1,1] = 200. 
        else:
            Lu[0] = u[0] / 100.
            Luu[0,0] = 1. / 100.
            # 
            Fx[0, 0] = - self.g * np.cos(x[0]) / self.l
            Fu[0,0] = 1./(self.l**2 * self.mass) 

            # I will store the 2nd order derivative of the dynamics here since crocoddyl support it 
            # this second order derivative will be of x_{t+1} = x_t + dx_t and not the continuous dynamics 
            
            self.Fxx[0,0,0] =  self.dt**2 * self.g * np.sin(x[0]) / self.l
            self.Fxx[1,0,0] =  self.dt  * self.g * np.sin(x[0]) / self.l


        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


            
     
if __name__ =="__main__":
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, '#'))
    Pendulum_diff_running =  DifferentialActionModelPendulum()
    Pendulum_diff_terminal = DifferentialActionModelPendulum(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))
    dt = 0.01 
    T = 100 
    x0 = np.zeros(2) 
    MAX_ITER = 1000
    Pendulum_running = crocoddyl.IntegratedActionModelEuler(Pendulum_diff_running, dt) 
    Pendulum_terminal = crocoddyl.IntegratedActionModelEuler(Pendulum_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, [Pendulum_running]*T, Pendulum_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    ddp = crocoddyl.SolverDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(1)]*T
    converged = ddp.solve(xs,us, MAX_ITER)
    theta =[]  
    theta_dot =[]
    time_array = dt*np.arange(T+1)
    for xi in ddp.xs:
        theta += [xi[0]]
        theta_dot += [xi[1]]
    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
        plt.figure("trajectory plot")
        # plt.plot(x,y)
        # plt.plot(np.array(ddp.xs)[:,0],np.array(ddp.xs)[:,1], label="ddp")
        plt.plot(np.array(ddp.xs)[:,0], label="ddp")

        # plt.figure("velocity plots")
        # plt.plot(time_array, vx)
        # plt.plot(time_array, vy)
        plt.show()


    