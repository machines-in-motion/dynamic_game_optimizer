""" computes derivatives of the integrated action models 
for the dynamics Fx & Fu 
for the cost Lx, Lu, Lxx, Luu, Lxu 
references on finite differencing 
https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

"""

import numpy as np 

import crocoddyl 

DELTA = 1.e-6 # numerical differentiation step 

class CostNumDiff: 
    def __init__(self, model):
        self.model = model 
        self.state = self.model.state
        self.data = crocoddyl.IntegratedActionDataEuler(self.model)
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 

        self.Lx = np.zeros(self.ndx)
        self.Lu = np.zeros(self.nu)
        self.Lxx = np.zeros([self.ndx, self.ndx])
        self.Luu = np.zeros([self.nu, self.nu])
        self.Lxu = np.zeros([self.ndx, self.nu])

    def calcLx(self, x, u): 
        dx = np.zeros(self.ndx)
        for i in range(self.ndx):
            dx[i] = DELTA
            xnew = self.state.integrate(x, dx)
            self.model.calc(self.data, xnew, u)
            cost1 = self.data.cost 
            xnew = self.state.integrate(x, -dx)
            self.model.calc(self.data, xnew, u)
            cost2 = self.data.cost 
            self.Lx[i] = cost1 - cost2 
            dx[i] = 0. 
        self.Lx *= 1./(2.*DELTA) 

    def calcLu(self, x, u): 
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = DELTA
            self.model.calc(self.data, x, u+du)
            cost1 = self.data.cost 
            self.model.calc(self.data, x, u-du)
            cost2 = self.data.cost 
            self.Lu[i] = cost1 - cost2 
            du[i] = 0. 
        self.Lu *= 1./(2.*DELTA)

    def calcLxx(self, x, u): 
        dxi = np.zeros(self.ndx)
        dxj = np.zeros(self.ndx)

        for i in range(self.ndx):
            for j in range(self.ndx): 
                dxi[i] = DELTA
                dxj[j] = DELTA
                xnew = self.state.integrate(x, dxi + dxj)
                self.model.calc(self.data, xnew, u)
                cost1 = self.data.cost 
                xnew = self.state.integrate(x, dxi - dxj)
                self.model.calc(self.data, xnew, u)
                cost2 = self.data.cost 
                xnew = self.state.integrate(x, -dxi + dxj)
                self.model.calc(self.data, xnew, u)
                cost3 = self.data.cost 
                xnew = self.state.integrate(x, -dxi - dxj)
                self.model.calc(self.data, xnew, u)
                cost4 = self.data.cost 
                self.Lxx[i,j] = cost1 - cost2 - cost3 + cost4 
                dxi[i] = 0.
                dxj[j] = 0.
        self.Lxx = .5*(self.Lxx + self.Lxx.T)
        self.Lxx *= 1./(4.*DELTA*DELTA)

    def calcLuu(self, x, u): 
        dui = np.zeros(self.nu)
        duj = np.zeros(self.nu)
        for i in range(self.nu):
            for j in range(self.nu): 
                dui[i] = DELTA
                duj[j] = DELTA
                self.model.calc(self.data, x, u + dui + duj)
                cost1 = self.data.cost 
                self.model.calc(self.data, x, u + dui - duj)
                cost2 = self.data.cost 
                self.model.calc(self.data, x, u - dui + duj)
                cost3 = self.data.cost 
                self.model.calc(self.data, x, u - dui - duj)
                cost4 = self.data.cost 
                self.Luu[i,j] = cost1 - cost2 - cost3 + cost4 
                dui[i] = 0.
                duj[j] = 0.
        self.Luu = .5*(self.Luu + self.Luu.T)
        self.Luu *= 1./(4.*DELTA*DELTA)

    def calcLxu(self, x, u): 
        dx = np.zeros(self.ndx) 
        du = np.zeros(self.nu)
        for i in range(self.ndx): 
            for j in range(self.nu): 
                dx[i] = DELTA
                du[j] = DELTA
                xnew = self.state.integrate(x, dx)
                self.model.calc(self.data, xnew, u+du)
                cost1 = self.data.cost 
                xnew = self.state.integrate(x, dx)
                self.model.calc(self.data, xnew, u-du)
                cost2 = self.data.cost 
                xnew = self.state.integrate(x, -dx)
                self.model.calc(self.data, xnew, u+du)
                cost3 = self.data.cost 
                xnew = self.state.integrate(x, -dx)
                self.model.calc(self.data, xnew, u-du)
                cost4 = self.data.cost 
                self.Lxu[i,j] = cost1 - cost2 - cost3 + cost4 
                dx[i] = 0.
                du[j] = 0.
        self.Lxu *= 1./(4*DELTA*DELTA)




class DynamicsNumDiff: 
    def __init__(self, model):
        self.model = model 
        self.data = crocoddyl.IntegratedActionDataEuler(self.model)
        self.state = self.model.state
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 
        self.Fx = np.zeros([self.ndx, self.ndx])
        self.Fu = np.zeros([self.ndx, self.nu])

    def calcFx(self, x, u): 
        dx = np.zeros(self.ndx)
        for i in range(self.ndx):
            dx[i] = DELTA
            xnew = self.state.integrate(x, dx)
            self.model.calc(self.data, xnew, u)
            x_next1 = self.data.xnext.copy() 
            xnew = self.state.integrate(x, -dx)
            self.model.calc(self.data, xnew, u)
            x_next2 = self.data.xnext.copy()
            self.Fx[:,i] = self.state.diff(x_next2, x_next1)  
            dx[i] = 0. 
        self.Fx *= 1./(2.*DELTA)

    def calcFu(self, x, u): 
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = DELTA
            self.model.calc(self.data, x, u + du)
            x_next1 = self.data.xnext.copy() 
            self.model.calc(self.data, x, u - du)
            x_next2 = self.data.xnext.copy()
            self.Fu[:,i] = self.state.diff(x_next2, x_next1)  
            du[i] = 0. 
        self.Fu *= 1./(2.*DELTA)



    