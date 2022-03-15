
import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'    # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 30 #DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ['pdf', 'png']  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = './'


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
mpl.rcParams['legend.framealpha'] = 0.5
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
scale = 1.0
mpl.rcParams['figure.figsize'] = 30*scale, 10*scale #23, 18  # 12, 9
line_styles = 10*['g-', 'r--', 'b-.', 'k:', '^c', 'vm', 'yo']
line_styles = 10*['g', 'r', 'b', 'k', 'c', 'm', 'y']

def plot_2d_trajectory_gaps(solvers, xnexts, solver_names, tsolve, title, xlabel, ylabel): 
    plt.figure(title, figsize=(20, 10))
    
    ax = plt.axes()
    ax.add_patch(m_patches.Rectangle(
    (-1., -0.002),1., .002,
    fill=False,
    hatch='/',
    ))
    ax.add_patch(m_patches.Rectangle(
    (10., -0.002),1., .002,
    fill=False,
    hatch='/',
    ))
    for i,name in enumerate(solver_names): 
        xs_i = np.array(solvers[i].xs)
        xnext_i = np.array(xnexts[i])
        
        xzip = np.array(list(zip(xs_i[:,0], xnext_i[:,0])))
        # xzip = np.resize(xzip, 2*xzip.shape[0])
        yzip = np.array(list(zip(xs_i[:,1], xnext_i[:,1])))
        # yzip = np.resize(yzip, 2*yzip.shape[0])
        for t in range(xzip.shape[0]):
            if t == 0:
                plt.plot(xzip[t], yzip[t],line_styles[i],  linewidth=2., label=name)
            else:
                plt.plot(xzip[t], yzip[t],line_styles[i],  linewidth=2.,)
        plt.scatter(xs_i[tsolve,0], xs_i[tsolve,1], s=45., c=line_styles[i], alpha=1., zorder=2.)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    plt.legend(loc=1)
    plt.xlim([-1., 11.])
    plt.ylim(bottom=-.002)

    if SAVE_FIGURES:
        plt.savefig(FIGURE_PATH+title+".png")


def plot_states(solvers, solver_names, dt, title, state_names, solve_time): 

    horizon = len(solvers[0].xs)
    time_id = dt*np.arange(horizon)
    f, ax = plt.subplots(len(state_names), 1, sharex=True, figsize=(20,25))

    for j, state_name in enumerate(state_names):
        for i,name in enumerate(solver_names): 
            xs_i = np.array(solvers[i].xs)
            ax[j].plot(time_id, xs_i[:, j],line_styles[i],linewidth=2., label=name)
        ax[j].set_ylabel(state_name) 
        ax[j].axvspan(time_id[0], time_id[solve_time], facecolor='lightgrey', alpha=0.5)
        if j==0:
            ax[j].legend(loc=2)
    ax[-1].set_xlabel("time [s]")
    # f.suptitle(title)
    if SAVE_FIGURES:
        plt.savefig(FIGURE_PATH+title+".png")


def plot_controls(solvers, solver_names, dt, title, control_names, solve_time): 
    horizon = len(solvers[0].us)
    time_id = dt*np.arange(horizon)
    f, ax = plt.subplots(len(control_names), 1, sharex=True, figsize=(20,15))
    if len(control_names) == 1:
        for i,name in enumerate(solver_names): 
                us_i = np.array(solvers[i].us)
                ax.plot(time_id, us_i[:],line_styles[i], linewidth=2., label=name)
        ax.set_ylabel(control_names[0]) 
        ax.axvspan(time_id[0], time_id[solve_time-1], facecolor='lightgrey', alpha=0.5)
        ax.set_xlabel("time [s]")
    else:
        for j, cntrl_name in enumerate(control_names):
            for i,name in enumerate(solver_names): 
                us_i = np.array(solvers[i].us)
                ax[j].plot(time_id, us_i[:, j],line_styles[i], linewidth=2., label=name)
            ax[j].set_ylabel(cntrl_name) 
            ax[j].axvspan(time_id[0], time_id[solve_time-1], facecolor='lightgrey', alpha=0.5)
            if j==0:
                ax[j].legend(loc=0)
        ax[-1].set_xlabel("time [s]")
    # f.suptitle(title)
    if SAVE_FIGURES:
        plt.savefig(FIGURE_PATH+title+".png")


def plot_pendulum_xy(solvers, solver_names):
    horizon = np.floor(len(solvers[0].xs) /2) 
    scale = 1./horizon
    for i, name in enumerate(solver_names):
        plt.figure("Pedulum "+name)
        xi = np.array(solvers[i].xs)
        px = np.sin(xi[:,0])
        py = -np.cos(xi[:,0])
        for i,(xi,yi) in enumerate(zip(px[::2], py[::2])):
            plt.plot([0., xi], [0., yi], 'k', linewidth=2., alpha=scale*i)
            plt.scatter(xi, yi, s=45., c='k' , alpha=scale*i)
        plt.plot([0., px[-1]], [0., py[-1]], 'k', linewidth=2., alpha=1.)
        
        plt.scatter(px[-1], py[-1], s=45., c='k', alpha=1.)
        ax = plt.gca() #you first need to get the axis handle
        ax.set_aspect(1)

