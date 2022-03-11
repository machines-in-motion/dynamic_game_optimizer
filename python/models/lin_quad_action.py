import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class PointMassDynamics:
    def __init__(self):
        self.g = np.array([0. , -9.81])
        self.mass = 1 
        self.nq = 2 
        self.nv = 2 
        self.ndx = 2 
        self.nx = self.nq + self.nv 
        self.nu = 2 
        self.c_drag = .0

    def nonlinear_dynamics(self, x, u):
        return (1/self.mass)*u + self.g - self.c_drag*x[2:]**2
    
    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([self.nv ,self.ndx])
        dfdu = np.zeros([self.nv ,self.nu])
        dfdu[0,0] = 1./self.mass 
        dfdu[1,1] = 1./self.mass 
        dfdx[0,2] = -2.*self.c_drag*x[2]
        dfdx[1,3] = -2.*self.c_drag*x[3]
        return dfdx, dfdu
    
class DifferentialActionModelLQ(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1.e-2, isTerminal=False):
        self.dynamics = PointMassDynamics()
        state =  crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, self.dynamics.nu, self.dynamics.ndx)
        self.ndx = self.state.ndx 
        self.isTerminal = isTerminal
        self.mass = self.dynamics.mass 
        self.dt = dt 
        self.Fxx = np.zeros([self.ndx, self.ndx, self.ndx])
        self.Fxu = np.zeros([self.ndx, self.ndx, self.nu])
        self.Fuu = np.zeros([self.ndx, self.nu, self.nu])


    def _running_cost(self, x, u):
        cost = (x[0]-1.)**2 + x[1]**2 + x[2]**2 + x[3]**2  + u[0]**2 + u[1]**2 
        return cost

    def _terminal_cost(self, x, u):
        cost = 20000*((x[0]-1.)**2) + 20000*(x[1]**2) + 1000*(x[2]**2) + 1000*(x[3]**2)  
        return cost 
     
    def calc(self, data, x, u=None):
        if u is None: 
            u = np.zeros(self.nu)

        
        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
            data.xout = np.zeros(self.state.nv)
        else:
            data.cost = self._running_cost(x,u)
            data.xout = self.dynamics.nonlinear_dynamics(x,u)

        # data.r = None # residuals I think, Must be crucial for finite difference derivative computation, 
        # must check it  
    
    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([2,4]) 
        Fu = np.zeros([2,2])
        
        Lx = np.zeros([4])
        Lu = np.zeros([2])
        Lxx = np.zeros([4,4])
        Luu = np.zeros([2,2])
        Lxu = np.zeros([4,2])
        if self.isTerminal:
            Lx[0] = 40000.*(x[0]-1)
            Lx[1] = 40000.*x[1]
            Lx[2] = 2000.*x[2]
            Lx[3] = 2000.*x[3]     
            Lxx[0,0] = 40000. 
            Lxx[1,1] = 40000. 
            Lxx[2,2] = 2000. 
            Lxx[3,3] = 2000. 
        else:
            Lx[0] = 2.*(x[0]-1)
            Lx[1] = 2.*x[1]
            Lx[2] = 2.*x[2]
            Lx[3] = 2.*x[3]
            Lu[0] = 2*u[0] 
            Lu[1] = 2*u[1]
            Lxx[0,0] = 2
            Lxx[1,1] = 2
            Lxx[2,2] = 2
            Lxx[3,3] = 2
            Luu[0,0] = 2. 
            Luu[1,1] = 2
            # first order dynamics derivatives are the continuous time ones, crocoddyl takes care of discretization
            Fu[0,0] = 1./self.mass 
            Fu[1,1] = 1./self.mass 
            # 
            Fx[0,2] = -2.*self.dynamics.c_drag*x[2]
            Fx[1,3] =  -2.*self.dynamics.c_drag*x[3]
            # I will store the 2nd order derivative of the dynamics here since crocoddyl doesn't support it 
            # this second order derivative will be of x_{t+1} = x_t + dx_t and not the continuous dynamics 
            self.Fxx[0,2,2] = -2 * self.dynamics.c_drag * self.dt**2  
            self.Fxx[1,3,3] = -2 * self.dynamics.c_drag * self.dt**2  
            self.Fxx[2,2,2] = -2 * self.dynamics.c_drag * self.dt 
            self.Fxx[3,3,3] = -2 * self.dynamics.c_drag * self.dt 


        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


            
     
if __name__ =="__main__":
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, '#'))
    lq_diff_running =  DifferentialActionModelLQ()
    lq_diff_terminal = DifferentialActionModelLQ(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))
    dt = 0.01 
    T = 300 
    x0 = np.zeros(4) 
    MAX_ITER = 1000
    lq_running = crocoddyl.IntegratedActionModelEuler(lq_diff_running, dt) 
    lq_terminal = crocoddyl.IntegratedActionModelEuler(lq_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, [lq_running]*T, lq_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    ddp = crocoddyl.SolverDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T
    converged = ddp.solve(xs,us, MAX_ITER)
    x =[]
    y =[]
    vx =[]
    vy =[]
    for xi in ddp.xs:
        x += [xi[0]]
        y += [xi[1]]
        vx += [xi[2]]
        vy += [xi[3]]
    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
        plt.figure("trajectory plot")
        plt.plot(np.array(ddp.xs)[:,0],np.array(ddp.xs)[:,1], label="ddp")
        plt.show()
