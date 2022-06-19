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
        return -self.g * np.sin(x[0]) / self.l + u / (self.l ** 2 * self.mass)


class DifferentialActionModelPendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, dt=1e-2, isTerminal=False):
        self.dynamics = PendulumDynamics()
        state = crocoddyl.StateVector(self.dynamics.nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, self.dynamics.nu, self.dynamics.ndx
        )
        self.ndx = self.state.ndx
        self.isTerminal = isTerminal
        self.mass = self.dynamics.mass
        self.l = self.dynamics.l
        self.g = self.dynamics.g
        self.dt = dt
        
    def _running_cost(self, x, u):
        cost = 0.5 * u[0] ** 2 / 1000 / self.dt
        return cost

    def _terminal_cost(self, x, u):
        cost = 200 * (x[0] - np.pi) ** 2 + 10 * x[1] ** 2
        return cost

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)

        if self.isTerminal:
            data.cost = self._terminal_cost(x, u)
            data.xout = np.zeros(1)
        else:
            data.cost = self._running_cost(x, u)
            data.xout = self.dynamics.nonlinear_dynamics(x, u)

    def calcDiff(self, data, x, u=None):
        Fx = np.zeros([1, 2])
        Fu = np.zeros([1, 1])

        Lx = np.zeros([2])
        Lu = np.zeros([1])
        Lxx = np.zeros([2, 2])
        Luu = np.zeros([1, 1])
        Lxu = np.zeros([2, 1])
        if self.isTerminal:
            Lx[0] = 400. * (x[0] - np.pi)
            Lx[1] = 20. * x[1]
            Lxx[0, 0] = 400.
            Lxx[1, 1] = 20.
        else:
            Lu[0] = u[0] / 1000  / self.dt 
            Luu[0, 0] = 1.0 / 1000  / self.dt
            Fx[0, 0] = -self.g * np.cos(x[0]) / self.l
            Fu[0, 0] = 1.0 / (self.l ** 2 * self.mass)

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()


class IntegratedActionModelPendulum(crocoddyl.IntegratedActionModelEuler): 
    def __init__(self, diffModel, dt=1.e-2):
        super().__init__(diffModel, dt)
        self.diffModel = diffModel 
        self.g = self.diffModel.g
        self.l = self.diffModel.l
        self.intModel = crocoddyl.IntegratedActionModelEuler(self.diffModel, dt) 
        self.Fxx = np.zeros([self.state.ndx, self.state.ndx, self.state.ndx])
        self.Fxu = np.zeros([self.state.ndx, self.state.ndx, self.nu])
        self.Fuu = np.zeros([self.state.ndx, self.nu, self.nu])
    
    def calcFxx(self, x, u): 
        self.Fxx[0, 0, 0] = self.dt ** 2 * self.g * np.sin(x[0]) / self.l
        self.Fxx[1, 0, 0] = self.dt * self.g * np.sin(x[0]) / self.l

    def calc(self, data, x, u=None):
        if u is None:
            self.intModel.calc(data, x)
        else:
            self.intModel.calc(data, x, u)
        
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.intModel.calcDiff(data, x)
            u = np.zeros(self.nu)
        else:
            self.intModel.calcDiff(data, x, u)
        self.calcFxx(x,u)
        

if __name__ == "__main__":
    print(" Testing Pendulum with DDP ".center(LINE_WIDTH, "#"))
    Pendulum_diff_running = DifferentialActionModelPendulum()
    Pendulum_diff_terminal = DifferentialActionModelPendulum(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, "-"))
    dt = 0.01
    T = 100
    x0 = np.zeros(2)
    MAX_ITER = 1000
    Pendulum_running = IntegratedActionModelPendulum(Pendulum_diff_running, dt)
    Pendulum_terminal = IntegratedActionModelPendulum(Pendulum_diff_terminal, dt)
    print(" Constructing integrated models completed ".center(LINE_WIDTH, "-"))

    problem = crocoddyl.ShootingProblem(x0, [Pendulum_running] * T, Pendulum_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, "-"))

    ddp = crocoddyl.SolverDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, "-"))
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    xs = [x0] * (T + 1)
    us = [np.zeros(1)] * T
    converged = ddp.solve(xs, us, MAX_ITER)
    time_array = dt * np.arange(T + 1)
    if converged:
        print(" DDP solver has CONVERGED ".center(LINE_WIDTH, "-"))
        plt.figure()
        plt.plot(np.array(ddp.xs)[:, 0], label="ddp")
        plt.xlabel("Time")
        plt.ylabel("$\\theta$ ")
        plt.title("DDP")
        plt.show()
