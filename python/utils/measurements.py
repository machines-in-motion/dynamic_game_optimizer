import numpy as np 


class AbstractMeasurementModel:
    def __init__(self, action_model):
        self.model = action_model
        self.state = self.model.state
        self.ndx = self.state.ndx
        self.nq = self.state.nq 
        self.nv = self.state.nv 
        self.nu = self.model.nu 
        self.data_type = MeasurementData
    
    def createData(self):
        return self.data_type(self)

    def calc(self, data, x, u=None): 
        raise NotImplementedError("calc method is not implemented for AbstractMeasurementModel")
    
    def calcDiff(self, data, x, u=None, recalc=False): 
        raise NotImplementedError("calcDiff method is not implemented for AbstractMeasurementModel")


class FullStateMeasurement(AbstractMeasurementModel):
    def __init__(self, integrated_action, m_covariance):
        """ defines a full state measurement model 
            y_{t} = g(x_t, u_t) + \gamma_{t]
            Args: 
            state_model:  state model defined in crocoddyl 
            m_covariance: covariance of \gamma_{t} in the above equation
        """
        super(FullStateMeasurement, self).__init__(integrated_action)
        self.R = m_covariance
        self.invR = np.linalg.inv(self.R)
        self.ny = self.state.ndx 
        assert self.R.shape == (self.ny, self.ny)

    def calc(self, data, x, u=None): 
        """ returns undisturbed measurement y_t = g(x_t, u_t) """
        data.R[:,:] = self.R.copy()
        data.invR[:,:] = self.invR.copy()
        return x 

    def calcDiff(self, data, x, u=None, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            if u is not None:
                self.calc(x,u)
            else:
                self.calc(x)    
        data.Hx[:,:] = np.eye(self.ny)

    def diff(self, y1, y2):
        """ return y2 - y1 """ 
        return self.state.diff(y1, y2)
    
    def integrate(self, y, dy):
        """ computes y+dy """
        return self.state.integrate(y, dy)



class PositionMeasurement(AbstractMeasurementModel):
    def __init__(self, integrated_action, m_covariance):
        """ 
            a measurement model of only positions, 
            doesn't account for geometric configuration, only vector spaces 
        """
        super(PositionMeasurement, self).__init__(integrated_action)
        self.R = m_covariance
        self.invR = np.linalg.inv(self.R)
        self.ny = self.state.nv 
        assert self.R.shape == (self.ny, self.ny)

    def calc(self, data, x, u=None): 
        """ returns undisturbed measurement y_t = g(x_t, u_t) """
        data.R[:,:] = self.R.copy()
        data.invR[:,:] = self.invR.copy()
        return x[:self.ny] 

    def calcDiff(self, data, x, u=None, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            if u is not None:
                self.calc(x,u)
            else:
                self.calc(x)    
        data.Hx[:self.ny,:self.ny] = np.eye(self.ny)

    def diff(self, y1, y2):
        """ return y2 - y1 """ 
        return y2 - y1 
    
    def integrate(self, y, dy):
        """ computes y+dy """
        return y + dy 


class PendulumCartesianMeasurement(AbstractMeasurementModel):
    def __init__(self, integrated_action, m_covariance):
        """ 
            a measurement model of only cartesian positions of the pendulum
        """
        super(PendulumCartesianMeasurement, self).__init__(integrated_action)
        self.R = m_covariance
        self.invR = np.linalg.inv(self.R)
        self.ny = 2 
        assert self.R.shape == (self.ny, self.ny)

    def calc(self, data, x, u=None): 
        """ returns undisturbed measurement y_t = g(x_t, u_t) """
        data.R[:,:] = self.R.copy()
        data.invR[:,:] = self.invR.copy()
        return np.array([np.sin(x[0]), -np.cos(x[0])])

    def calcDiff(self, data, x, u=None, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            if u is not None:
                self.calc(x,u)
            else:
                self.calc(x)    
        data.Hx[0, 0] = np.cos(x[0])
        data.Hx[1, 0] = np.sin(x[0])
        data.Hxx[0, 0, 0] = - np.sin(x[0])
        data.Hxx[1, 0, 0] = np.cos(x[0])

    def diff(self, y1, y2):
        """ return y2 - y1 """ 
        return y2 - y1 
    
    def integrate(self, y, dy):
        """ computes y+dy """
        return y + dy 


class MeasurementData: 
    def __init__(self, model):
        self.Hx = np.zeros([model.ny, model.ndx])
        self.Hxx = np.zeros([model.ny, model.ndx, model.ndx])
        self.Hu = np.zeros([model.ny, model.nu])
        self.Huu = np.zeros([model.ny, model.nu, model.nu])
        self.R =  np.zeros([model.ny, model.ny]) 
        self.invR =  np.zeros([model.ny, model.ny]) 


class MeasurementTrajectory:
    def __init__(self, models):
        self.runningModels = models[:] 
        self.runningDatas = [m.createData() for m in self.runningModels] 

    def calc(self, xs, us=None):
        if us is None:
            us = [None] * (len(xs)-1)
        y = [] 
        for i, xi in enumerate(xs):
            if i == len(xs)-1:
                y+= [self.runningModels[i].calc(self.runningDatas[i], xi)]
            else:
                y+= [self.runningModels[i].calc(self.runningDatas[i], xi, us[i])]

        return y 

    def calcDiff(self, xs, us=None, recalc=False):
        if us is None:
            us = [None] * (len(xs)-1) 
        y = []

        if recalc:
            y = self.calc(xs, us)
        for i, ui in enumerate(us):
            self.runningModels[i].calcDiff(self.runningDatas[i], xs[i], ui)
        for i, xi in enumerate(xs):
            if i == len(xs)-1:
                self.runningModels[i].calcDiff(self.runningDatas[i], xi)
            else:
                self.runningModels[i].calcDiff(self.runningDatas[i], xi, us[i])
        return y 
