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
        self.Hx = np 
        self.ny = self.state.ndx 
        self.filter = np.eye(self.state.ndx)

    def calc(self, data, x, u=None): 
        """ returns undisturbed measurement y_t = g(x_t, u_t) """
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
        self.Hxx = np.zeros([model.ny, model.ndx])
        self.Hu = np.zeros([model.ny, model.nu])
        self.Huu = np.zeros([model.ny, model.nu, model.nu])

class MeasurementTrajectory:
    def __init__(self, models):
        self.runningModels = models[:-1] 
        self.runningDatas = [m.createData() for m in self.runningModels] 
        self.terminalModel = models[-1]
        self.terminalData = self.terminalModel.createData()

    def calc(self, xs, us):
        for i, ui in enumerate(us):
            self.runningModel[i].calc(self.runningDatas[i], xs[i], ui)
        self.terminalModel.calc(self.terminalData, xs[-1])

    def calcDiff(self, xs, us, recalc=False):
        if recalc:
            self.calc(xs, us)
        for i, ui in enumerate(us):
            self.runningModel[i].calcDiff(self.runningDatas[i], xs[i], ui)
        self.terminalModel.calcDiff(self.terminalData, xs[-1])
