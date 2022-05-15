import enum
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection

DEFAULT_FONT_SIZE = 35
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 2  # 13
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
mpl.rcParams['legend.framealpha'] = 1.
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
scale = 1.0
mpl.rcParams['figure.figsize'] = 30*scale, 10*scale 
line_styles = 10*['k', 'r', 'm', 'b' , 'c', 'g', 'y']


horizon = 60
plan_dt = 0.05

record_cost = np.load('record_cost.npy')
record_states = np.load('record_states.npy')
record_controls = np.load('record_controls.npy')
record_cost_DDP = np.load('record_cost_DDP.npy')
record_states_DDP = np.load('record_states_DDP.npy')
record_controls_DDP = np.load('record_controls_DDP.npy')




label_dg = f'DG cost = {np.mean(record_cost):.3f}   \u00B1  {np.std(record_cost):.3f}'
label_ddp = f'DDP cost = {np.mean(record_cost_DDP):.3f}   \u00B1  {np.std(record_cost_DDP):.3f}'
print(label_dg)
print(label_ddp)

states_mean = np.mean(record_states, axis=0)
states_std = np.std(record_states, axis=0)
controls_mean = np.mean(record_controls, axis=0)
controls_std = np.std(record_controls, axis=0)
states_mean_DDP = np.mean(record_states_DDP, axis=0)
states_std_DDP = np.std(record_states_DDP, axis=0)
controls_mean_DDP = np.mean(record_controls_DDP, axis=0)
controls_std_DDP = np.std(record_controls_DDP, axis=0)

plt.figure(figsize=(20, 10))
width = 0.1
height = 2
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
ax = plt.axes()
ax.add_patch(m_patches.Ellipse((1., -0.1), width=0.55*width, height=0.55*height,fill=False,hatch='/',))
plt.plot(states_mean_DDP[:, 0], states_mean_DDP[:, 1], color="b", linewidth=2, label="Neutral")
plt.plot(states_mean[:, 0], states_mean[:, 1], color="g", linewidth=2, label="DG")
# plt.plot(state_dg_plan[:, 0], state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG initial plan")
# plt.plot(state_ddp_plan[:, 0], state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP initial plan")
# Plot shaded std
y_1 = states_mean_DDP[:, 1] - states_std_DDP[:, 1]
y_2 = states_mean_DDP[:, 1] + states_std_DDP[:, 1]
plt.fill_between(states_mean_DDP[:, 0], y_1, y_2, color="b", alpha=0.2)
y_1 = states_mean[:, 1] - states_std[:, 1]
y_2 = states_mean[:, 1] + states_std[:, 1]
plt.fill_between(states_mean[:, 0], y_1, y_2, color="g", alpha=0.2)
plt.xlabel(r"$p_x$ [m]") 
plt.ylabel(r"$p_y$ [m]")
plt.xlim([-0.05, 2.05])
plt.ylim([-0.1, 0.7])
plt.legend(loc='upper right')
plt.savefig("quadrotor_mean_trajectory.pdf", bbox_inches='tight')

time = np.linspace(0, plan_dt*(horizon+1), horizon+1)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20, 20))
ax1.plot(time, states_mean_DDP[:, 0], color="b", label="Neutral")
ax2.plot(time, states_mean_DDP[:, 1], color="b", label="Neutral")
ax3.plot(time, states_mean_DDP[:, 2], color="b", label="Neutral")
ax1.plot(time, states_mean[:, 0], color="g", label="DG")
ax2.plot(time, states_mean[:, 1], color="g", label="DG")
ax3.plot(time, states_mean[:, 2], color="g", label="DG")
# ax1.plot(time, state_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
# ax2.plot(time, state_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")
# ax3.plot(time, state_dg_plan[:, 2], color=cycle[0], linestyle='dashed', label="DG plan")
# ax1.plot(time, state_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
# ax2.plot(time, state_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")
# ax3.plot(time, state_ddp_plan[:, 2], color=cycle[1], linestyle='dashed', label="DDP plan")
ax1.fill_between(time, states_mean_DDP[:, 0] - states_std_DDP[:, 0], states_mean_DDP[:, 0] + states_std_DDP[:, 0], color="b", alpha=0.2)
ax2.fill_between(time, states_mean_DDP[:, 1] - states_std_DDP[:, 1], states_mean_DDP[:, 1] + states_std_DDP[:, 1], color="b", alpha=0.2)
ax3.fill_between(time, states_mean_DDP[:, 2] - states_std_DDP[:, 2], states_mean_DDP[:, 2] + states_std_DDP[:, 2], color="b", alpha=0.2)
ax1.fill_between(time, states_mean[:, 0] - states_std[:, 0], states_mean[:, 0] + states_std[:, 0], color="g", alpha=0.2)
ax2.fill_between(time, states_mean[:, 1] - states_std[:, 1], states_mean[:, 1] + states_std[:, 1], color="g", alpha=0.2)
ax3.fill_between(time, states_mean[:, 2] - states_std[:, 2], states_mean[:, 2] + states_std[:, 2], color="g", alpha=0.2)
ax3.set_xlabel("time [s]")
ax1.set_ylabel(r"$p_x$ [m]")
ax2.set_ylabel(r"$p_y$ [m]")
ax3.set_ylabel(r"$\theta$ [rad]")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc='lower right')
plt.savefig("quadrotor_mean_state.pdf", bbox_inches='tight')


time = np.linspace(0, plan_dt*horizon, horizon)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
ax1.plot(time, controls_mean_DDP[:, 0], color="b", label="Neutral")
ax2.plot(time, controls_mean_DDP[:, 1], color="b", label="Neutral")
ax1.plot(time, controls_mean[:, 0], color="g", label="DG")
ax2.plot(time, controls_mean[:, 1], color="g", label="DG")
# ax1.plot(time, control_dg_plan[:, 0], color=cycle[0], linestyle='dashed', label="DG plan")
# ax2.plot(time, control_dg_plan[:, 1], color=cycle[0], linestyle='dashed', label="DG plan")
# ax1.plot(time, control_ddp_plan[:, 0], color=cycle[1], linestyle='dashed', label="DDP plan")
# ax2.plot(time, control_ddp_plan[:, 1], color=cycle[1], linestyle='dashed', label="DDP plan")
ax1.fill_between(time, controls_mean_DDP[:, 0] - controls_std_DDP[:, 0], controls_mean_DDP[:, 0] + controls_std_DDP[:, 0], color="b", alpha=0.2)
ax2.fill_between(time, controls_mean_DDP[:, 1] - controls_std_DDP[:, 1], controls_mean_DDP[:, 1] + controls_std_DDP[:, 1], color="b", alpha=0.2)
ax1.fill_between(time, controls_mean[:, 0] - controls_std[:, 0], controls_mean[:, 0] + controls_std[:, 0], color="g", alpha=0.2)
ax2.fill_between(time, controls_mean[:, 1] - controls_std[:, 1], controls_mean[:, 1] + controls_std[:, 1], color="g", alpha=0.2)
ax2.set_xlabel("time [s]")
ax1.set_ylabel(r"$u_1$ [N]")
ax2.set_ylabel(r"$u_2$ [N]")
ax1.legend(loc='lower right')
plt.savefig("quadrotor_mean_control.pdf", bbox_inches='tight')

plt.show()