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
        self.filter = np.eye(self.state.ndx)

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

    def calc(self, xs, us):
        y = [] 
        for i, ui in enumerate(us):
            y+= [self.runningModels[i].calc(self.runningDatas[i], xs[i], ui)]
        return y 

    def calcDiff(self, xs, us, recalc=False):
        y = []
        if recalc:
            y = self.calc(xs, us)
        for i, ui in enumerate(us):
            self.runningModels[i].calcDiff(self.runningDatas[i], xs[i], ui)
        return y 
