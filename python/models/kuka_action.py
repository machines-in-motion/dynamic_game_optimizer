""" Action model wrapper for the Kuka to add second order derivatives """
import numpy as np 
import pinocchio as pin 
import crocoddyl
import os, sys, time
import matplotlib.pyplot as plt 

src_path = os.path.abspath('../')
sys.path.append(src_path)

from utils.measurements import FullStateMeasurement, MeasurementTrajectory
from solvers.full import SaddlePointSolver


class IntegratedActionModelKuka(crocoddyl.IntegratedActionModelEuler): 
    def __init__(self, diffModel, dt=1.e-2):
        super().__init__(diffModel, dt)
        self.diffModel = diffModel 
        self.intModel = crocoddyl.IntegratedActionModelEuler(self.diffModel, dt) 
        self.Fxx = np.zeros([self.state.ndx, self.state.ndx, self.state.ndx])
        self.Fxu = np.zeros([self.state.ndx, self.state.ndx, self.nu])
        self.Fuu = np.zeros([self.state.ndx, self.nu, self.nu])
    
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
        

if __name__ == "__main__": 
    from robot_properties_kuka.config import IiwaConfig
    robot = IiwaConfig.buildRobotWrapper()
    model = robot.model
    nq = model.nq; nv = model.nv; nu = nq; nx = nq+nv
    q0 = np.array([0.1, 0.7, 0., 0.7, -0.5, 1.5, 0.])
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0])
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)


    # # # # # # # # # # # # # # #
    ###  SETUP CROCODDYL OCP  ###
    # # # # # # # # # # # # # # #

    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Running and terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)


    # Create cost terms 
    # Control regularization cost
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    # State regularization cost
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # endeff frame translation cost
    endeff_frame_id = model.getFrameId("contact")
    # endeff_translation = robot.data.oMf[endeff_frame_id].translation.copy()
    endeff_translation = np.array([-0.4, 0.3, 0.8]) 

    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    runningCostModel.addCost("translation", frameTranslationCost, 10)
    terminalCostModel.addCost("stateReg", xRegCost, 1e-3)
    terminalCostModel.addCost("translation", frameTranslationCost, 100) 

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    dt = 1e-2
    runningModel = IntegratedActionModelKuka(running_DAM, dt)
    terminalModel = IntegratedActionModelKuka(terminal_DAM, 0.)


    # Create the shooting problem
    T = 100
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Create solver + callbacks
    ddp_solver = crocoddyl.SolverFDDP(problem)
    ddp_solver.setCallbacks([crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackVerbose()])


    # Warm start : initial state + gravity compensation
    xs_init = [x0 for i in range(T+1)]
    us_init = ddp_solver.problem.quasiStatic(xs_init[:-1])

    # Solve with DDP solver
    ddp_solver.solve(xs_init, us_init, maxiter=1000, isFeasible=False)


    # Solve with DG solver
    MU = 0.05
    pm = 1e-2 * np.eye(nx)


    dg_solver = SaddlePointSolver(problem, MU, pm)

    xs_init = ddp_solver.xs
    us_init = [np.zeros(nu) for i in range(T)]

    dg_solver.solve(init_xs=xs_init, init_us=us_init, maxiter=500)



    ### PLOTTING

    # Extract trajectories
    ddp_x = np.array(ddp_solver.xs)
    ddp_q = ddp_x[:,:nq]
    ddp_v = ddp_x[:,nv:]
    ddp_u = np.array(ddp_solver.us)

    dg_x = np.array(dg_solver.xs)
    dg_q = dg_x[:,:nq]
    dg_v = dg_x[:,nv:]
    dg_u = np.array(dg_solver.us)

    x_reg_ref = [ddp_solver.problem.runningModels[i].differential.costs.costs['stateReg'].cost.residual.reference for i in range(ddp_solver.problem.T)]
    x_reg_ref.append(ddp_solver.problem.terminalModel.differential.costs.costs['stateReg'].cost.residual.reference)
    x_reg_ref = np.array(x_reg_ref)


    # Plot State
    tspan = np.linspace(0, T*dt, T+1)
    fig, ax = plt.subplots(nq, 2, sharex='col') 

    for i in range(nq):
        # Plot positions
        ax[i,0].plot(tspan, ddp_q[:,i], linestyle='-', label='DDP')  
        ax[i,0].plot(tspan, dg_q[:,i], linestyle='-', label='DG')  
        # Plot joint position regularization reference
        ax[i,0].plot(tspan, x_reg_ref[:,i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)

        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        # Plot velocities
        ax[i,1].plot(tspan, ddp_v[:,i], linestyle='-', label='DDP') 
        ax[i,1].plot(tspan, dg_v[:,i], linestyle='-', label='DG') 
        # Plot joint velocity regularization reference
        ax[i,1].plot(tspan, x_reg_ref[:,nq+i], linestyle='-.', color='k', marker=None, label='reg_ref', alpha=0.5)
        
        # Labels, tick labels and grid
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)  
    # Common x-labels + align
    ax[-1,0].set_xlabel('time [s]', fontsize=16)
    ax[-1,1].set_xlabel('time [s]', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=18)


    # Plot Control
    tspan = np.linspace(0, T*dt, T+1)
    fig, ax = plt.subplots(nu, 1, sharex='col') 

    for i in range(nu):
        ax[i].plot(tspan[:-1], ddp_u[:,i], linestyle='-', label='DDP')  
        ax[i].plot(tspan[:-1], dg_u[:,i], linestyle='-', label='DG')  
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].yaxis.set_major_locator(plt.MaxNLocator(2))
        ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i].grid(True)

    # Common x-labels + align
    ax[-1].set_xlabel('time [s]', fontsize=16)
    fig.align_ylabels(ax[:])
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Control trajectories', size=18)



    # Plot end effector


    # Get frame position
    def get_p_(q, model, id_endeff):
        '''
        Returns end-effector positions given q trajectory 
            q         : joint positions
            model     : pinocchio model
            id_endeff : id of EE frame
        '''
        
        data = model.createData()
        if(type(q)==np.ndarray and len(q.shape)==1):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            p = data.oMf[id_endeff].translation.T
        else:
            N = np.shape(q)[0]
            p = np.empty((N,3))
            for i in range(N):
                pin.forwardKinematics(model, data, q[i])
                pin.updateFramePlacements(model, data)
                p[i,:] = data.oMf[id_endeff].translation.T
        return p


    # Get frame linear velocity
    def get_v_(q, dq, model, id_endeff, ref=pin.LOCAL):
        '''
        Returns end-effector velocities given q,dq trajectory 
            q         : joint positions
            dq        : joint velocities
            model     : pinocchio model
            id_endeff : id of EE frame
        '''
        data = model.createData()
        if(len(q) != len(dq)):
            logger.error("q and dq must have the same size !")
        if(type(q)==np.ndarray and len(q.shape)==1):
            pin.forwardKinematics(model, data, q, dq)
            spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
            v = spatial_vel.linear
        else:
            N = np.shape(q)[0]
            v = np.empty((N,3))
            for i in range(N):
                pin.forwardKinematics(model, data, q[i], dq[i])
                spatial_vel =  pin.getFrameVelocity(model, data, id_endeff, ref)
                v[i,:] = spatial_vel.linear    
        return v


    # Extract EE traj
    lin_pos_ee_DDP = get_p_(ddp_q, model, endeff_frame_id)
    lin_vel_ee_DDP = get_v_(ddp_q, ddp_v, model, endeff_frame_id)

    lin_pos_ee_DG = get_p_(dg_q, model, endeff_frame_id)
    lin_vel_ee_DG = get_v_(dg_q, dg_v, model, endeff_frame_id)


    lin_pos_ee_ref = [ddp_solver.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp_solver.problem.T)]
    lin_pos_ee_ref.append(ddp_solver.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    lin_pos_ee_ref = np.array(lin_pos_ee_ref)


    # Plots
    fig, ax = plt.subplots(3, 2, sharex='col')
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i,0].plot(tspan, lin_pos_ee_DDP[:,i], linestyle='-', color='b', label="DDP")
        ax[i,0].plot(tspan, lin_pos_ee_DG[:,i], linestyle='-', color='g', label="DG")
        # Plot EE target frame translation in WORLD frame
        ax[i,0].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, label='reference', alpha=0.5)

        # Labels, tick labels, grid
        ax[i,0].set_ylabel('$P^{EE}_%s$ [m]'%xyz[i], fontsize=16)
        # ax[i,0].yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax[i,0].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,0].grid(True)
        # Plot EE (linear) velocities in WORLD frame
        ax[i,1].plot(tspan, lin_vel_ee_DDP[:,i], linestyle='-', color='b', label="DDP")
        ax[i,1].plot(tspan, lin_vel_ee_DG[:,i], linestyle='-', color='g', label="DG")
        # Labels, tick labels, grid
        ax[i,1].set_ylabel('$V^{EE}_%s$ [m/s]'%xyz[i], fontsize=16)
        # ax[i,1].yaxis.set_major_locator(plt.MaxNLocator(2))
        # ax[i,1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2e'))
        ax[i,1].grid(True)

    #x-label + align
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    ax[i,0].set_xlabel('time [s]', fontsize=16)
    ax[i,1].set_xlabel('time [s]', fontsize=16)

    ax[0,0].legend(loc='upper right')
    # handles, labels = ax[2,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    # fig.suptitle('End-effector frame position and linear velocity', size=18)





    # Plots
    fig, ax = plt.subplots(3, 1, sharex='col')
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i].plot(tspan, lin_pos_ee_DDP[:,i], linestyle='-', color='b', label="DDP")
        ax[i].plot(tspan, lin_pos_ee_DG[:,i], linestyle='-', color='g',label="DG")
        # Plot EE target frame translation in WORLD frame
        ax[i].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, label='target', alpha=0.5)
        # Labels, tick labels, grid
        ax[i].set_ylabel('$P^{EE}_%s$ [m]'%xyz[i], fontsize=16)
        ax[i].grid(True)

    #x-label + align
    fig.align_ylabels(ax[:])
    ax[i].set_xlabel('time [s]', fontsize=16)
    ax[i].set_xlabel('time [s]', fontsize=16)

    ax[0].legend(loc='upper right')

    plt.show()