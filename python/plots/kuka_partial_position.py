import os, sys
src_path = os.path.abspath('../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import kuka_action
import matplotlib.pyplot as plt 
from utils.measurements import PositionMeasurement, FullStateMeasurement, MeasurementTrajectory
from solvers.partial import PartialDGSolver
import crocoddyl 
import pinocchio as pin 
import plotting_tools as plut 

import matplotlib as mpl 

DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE =  DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9

FILE_EXTENSIONS = ['pdf', 'png']  # ,'eps']
FIGURES_DPI = 150



# axes.hold           : True    # whether to clear the axes by default on
# axes.linewidth      : 1.0     # edge linewidth
# axes.titlesize      : large   # fontsize of the axes title
# axes.color_cycle    : b, g, r, c, m, y, k  # color cycle for plot lines
# xtick.labelsize      : medium # fontsize of the tick labels
# figure.dpi       : 80      # figure dots per inch
# image.cmap   : jet               # gray | jet etc...
# savefig.dpi         : 100      # figure dots per inch
# savefig.facecolor   : white    # figure facecolor when saving
# savefig.edgecolor   : white    # figure edgecolor when saving
# savefig.format      : png      # png, ps, pdf, svg
# savefig.jpeg_quality: 95       # when a jpeg is saved, the default quality parameter.
# savefig.directory   : ~        # default directory in savefig dialog box,
# leave empty to always use current working directory
mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['font.family'] = DEFAULT_FONT_FAMILY
mpl.rcParams['font.size'] = DEFAULT_FONT_SIZE
mpl.rcParams['font.serif'] = DEFAULT_FONT_SERIF
mpl.rcParams['text.usetex'] = DEFAULT_TEXT_USE_TEX
mpl.rcParams['axes.labelsize'] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams['axes.grid'] = True
mpl.rcParams['legend.fontsize'] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams['legend.framealpha'] = .5
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
scale = 1.0
mpl.rcParams['figure.figsize'] = 30*scale, 10*scale #23, 18  # 12, 9
# line_styles = 10*['g-', 'r--', 'b-.', 'k:', '^c', 'vm', 'yo']
line_styles = 10*['b',  'c', 'g', 'r', 'y', 'k', 'm']


LINE_WIDTH = 100 
T = 100 
plan_dt = 1.e-2 

SAVE_FIGURES = True   
FIGURE_PATH = './'

pm = 1e-2 * np.eye(14) # process error weight matrix 
mm = 5e-1 * np.eye(7) # measurement error weight matrix 
P0  = 1e-2 * np.eye(14)

t_solve = 5 # solve problem for t = 50 

MU = 1.2

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
    endeff_translation = np.array([-0.4,0.3,0.7]) 

    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, endeff_frame_id, endeff_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)


    # Add costs
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    runningCostModel.addCost("translation", frameTranslationCost, 10)
    terminalCostModel.addCost("stateReg", xRegCost, 1e-3)
    terminalCostModel.addCost("translation", frameTranslationCost, 10) 

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

    # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
    dt = 1e-2
    runningModel = kuka_action.IntegratedActionModelKuka(running_DAM, dt)
    terminalModel = kuka_action.IntegratedActionModelKuka(terminal_DAM, 0.)


    # Create the shooting problem
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


    # uncertainty models 
    measurement_models = [PositionMeasurement(runningModel, mm)]*T + [PositionMeasurement(terminalModel, mm)]

    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    measurement_trajectory =  MeasurementTrajectory(measurement_models)
    

    ys = measurement_trajectory.calc(ddp_solver.xs[:t_solve+1], ddp_solver.us[:t_solve])
    dg_solver = PartialDGSolver(problem, MU, pm, P0, measurement_trajectory)
    print(" Constructor and Data Allocation for Partial Solver Works ".center(LINE_WIDTH, '-'))

    u_init =  [np.zeros(7)]*T
    u_init[:t_solve] = ddp_solver.us[:t_solve]
    dg_solver.solve(init_xs=[xi.copy() for xi in ddp_solver.xs], init_us=u_init, init_ys=ys)

    print(" Plotting DDP and DG Solutions ".center(LINE_WIDTH, '-'))



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
    title = 'State trajectories' 
    fig.suptitle(title, size=18)
    # if SAVE_FIGURES:
    #     plt.savefig(FIGURE_PATH+title+".pdf")

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

    # Extract EE traj
    lin_pos_ee_DDP = get_p_(ddp_q, model, endeff_frame_id)

    lin_pos_ee_DG = get_p_(dg_q, model, endeff_frame_id)


    lin_pos_ee_ref = [ddp_solver.problem.runningModels[i].differential.costs.costs['translation'].cost.residual.reference for i in range(ddp_solver.problem.T)]
    lin_pos_ee_ref.append(ddp_solver.problem.terminalModel.differential.costs.costs['translation'].cost.residual.reference)
    lin_pos_ee_ref = np.array(lin_pos_ee_ref)

    # Plots
    fig, ax = plt.subplots(3, 1, figsize=(20, 20), sharex='col')
    xyz = ['x', 'y', 'z']
    for i in range(3):
        # Plot EE position in WORLD frame
        ax[i].plot(tspan, lin_pos_ee_DDP[:,i], linestyle='-', color='b', label="Neutral")
        ax[i].plot(tspan, lin_pos_ee_DG[:,i], linestyle='-', color='g',label="$\mu=%s$"%MU)
        # Plot EE target frame translation in WORLD frame
        ax[i].plot(tspan, lin_pos_ee_ref[:,i], linestyle='--', color='k', marker=None, alpha=0.5)
        # Labels, tick labels, grid
        ax[i].set_ylabel('$P^{EE}_%s$ [m]'%xyz[i], fontsize=16)
        ax[i].grid(True)
        ax[i].axvspan(tspan[0], tspan[t_solve], facecolor='lightgrey', alpha=0.5)

    #x-label + align
    fig.align_ylabels(ax[:])
    ax[-1].set_xlabel('time [s]', fontsize=16)

    ax[0].legend(loc='upper right')

    plt.show()