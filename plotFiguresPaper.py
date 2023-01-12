'''
    This script plots results of the simulations. You should specify which
    'case' you want to plot in the list 'cases' below.
'''

# %% Import packages.
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import pandas


# Some general figure settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Corbel'
mpl.rcParams['mathtext.fontset'] = 'cm'


# %% User inputs
cases = ['1','2','3','4']

lengthAthletes = np.zeros((2,7))
sprinters_lengths = np.array((1.96, 1.78, 1.80, 1.90, 1.85, 1.75, 1.75))
sprinters_mass = np.array((94, 75, 80, 87, 80, 72, 71))
marathoners_lengths = np.array((1.67,1.65,1.73,1.76,1.74,1.73,1.70))
marathoners_mass = np.array((52,54,56,57,55,58,60))
sprinter_names = ['Bolt', 'Gay', 'Blake', 'Powell', 'Gatlin', 'Coleman', 'Bromell']
marathoners_names = ['Kipchoge', 'Bekele', 'Legese', 'Geremew', 'Kimetto', 'Ekiru', 'Chebet']

# Simulations
# 1 GEN - sprinting
# 2 SPRINT_SKEL - sprinting
# 3 GEN - marathon
# 4 MARATHON_SKEL - marathon
# 5 MARATHON_SKEL - sprinting
# 6 SPRINTING_SKEL - marathon
# 7 SPRINT_MUSC - sprinting
# 8 MARATHON_MUSC - marathon
# 9 SPRINT_MUSC - marathon
# 10 MARATHON_MUSC - sprinting
cases_sprinting = ['1','2','5']
cases_marathon = ['3','6','4']
cases_models = ['1','2','4']
cases_sprinting_MUSC = ['1','7']
cases_marathon_MUSC = ['3','7'] #,'10']
cases_models_MUSC = ['1','7'] #,'4']
cmap = mpl.colormaps['Accent']
colormap_5 = [(140/255,86/255,75/255),
            (227/255,119/255,194/255),
            (127/255,127/255,127/255),
            (188/255,189/255,34/255),
            (23/255,190/255,207/255)]

colormap_5 = [cmap.colors[7],
              cmap.colors[4],
              cmap.colors[0],
              cmap.colors[1],
               cmap.colors[2]]

colormap_MUSC = [cmap.colors[7],
                 cmap.colors[1],
                 cmap.colors[2]]


labels_models = ('GEN', 'SPRINT_SKEL', 'MARATHON_SKEL')

folder = 'ResultsSprinting'
pathMain = os.getcwd()
pathTrajectories = os.path.join(pathMain, folder)
pathTrajectories =folder
optimaltrajectories = np.load(os.path.join(pathTrajectories,
                                           'optimaltrajectories.npy'),
                              allow_pickle=True).item()

## Sprinting speed

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_sprinting)):
    y_value = np.around(optimaltrajectories[cases_sprinting[i]]['speed'], decimals=2).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,12])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('sprinting speed $[m/s]$')
plt.savefig('figures_paper/SKEL_sprinting_speed.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_sprinting_speed.jpg',format = 'jpg', dpi = 600)





## Sprinting speed

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_sprinting_MUSC)):
    y_value = np.around(optimaltrajectories[cases_sprinting_MUSC[i]]['speed'], decimals=2).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_MUSC[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,12])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('sprinting speed $[m/s]$')
plt.savefig('figures_paper/MUSC_sprinting_speed.svg',format = 'svg')
plt.savefig('figures_paper/MUSC_sprinting_speed.jpg',format = 'jpg', dpi = 600)





## Marathon energy cost

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_marathon)):
    y_value = np.around(optimaltrajectories[cases_marathon[i]]['metabolic_energy_outcomes']['marathon_total_metabolic_energy_kcal_opt'], decimals=0).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,4200])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('marathon energy cost $[kcal]$')
plt.savefig('figures_paper/SKEL_marathon_energy_cost.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_marathon_energy_cost.jpg',format = 'jpg', dpi = 600)





## Marathon energy cost

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_marathon_MUSC)):
    y_value = np.around(optimaltrajectories[cases_marathon_MUSC[i]]['metabolic_energy_outcomes']['marathon_total_metabolic_energy_kcal_opt'], decimals=0).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_MUSC[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,4200])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('marathon energy cost $[kcal]$')
plt.savefig('figures_paper/MUSC_marathon_energy_cost.svg',format = 'svg')
plt.savefig('figures_paper/MUSC_marathon_energy_cost.jpg',format = 'jpg', dpi = 600)





## Marathon energy cost

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_marathon)):
    y_value = np.around(optimaltrajectories[cases_marathon[i]]['metabolic_energy_outcomes']['marathon_total_metabolic_energy_perKG_kcal_opt'], decimals=0).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,55])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('normalized marathon energy cost $[kcal/kg]$')
plt.savefig('figures_paper/SKEL_marathon_energy_perKG_cost.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_marathon_energy_perKG_cost.jpg',format = 'jpg', dpi = 600)


## Mass

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5 - 0.25/2,1 - 0.25/2))

for i in range(len(cases_models)):
    y_value = np.around(optimaltrajectories[cases_models[i]]['mass'], decimals=1).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.22, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

x_values_sprinter = 0.5 + np.array((0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21))


marathoners_names = ['Kipchoge', 'Bekele', 'Legese', 'Geremew', 'Kimetto', 'Ekiru', 'Chebet']


for i in range(len(sprinter_names)):
    ax.scatter(x_values_sprinter[i], sprinters_mass[i], color=colormap_5[1], s=10)
    if i == 0:
        ax.text(x_values_sprinter[i] - 0.03, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize = 8)
    if i == 1:
        ax.text(x_values_sprinter[i] - 0.03, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize = 8)
    if i == 2:
        ax.text(x_values_sprinter[i] - 0.13, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize = 8)
    if i == 3:
        ax.text(x_values_sprinter[i] - 0.03, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize=8)
    if i == 4:
        ax.text(x_values_sprinter[i] + 0.0, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize=8)
    if i == 5:
        ax.text(x_values_sprinter[i] - 0.13, sprinters_mass[i] + 0.7, sprinter_names[i], fontsize=8)
    if i == 6:
        ax.text(x_values_sprinter[i] - 0.06, sprinters_mass[i] - 2.5, sprinter_names[i], fontsize=8)


x_values_marathoners = 1 + np.array((0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21))
for i in range(len(marathoners_names)):
    ax.scatter(x_values_marathoners[i], marathoners_mass[i], color=colormap_5[2], s=10)
    if i == 0:
        ax.text(x_values_marathoners[i] - 0.06, marathoners_mass[i] - 2.5, marathoners_names[i], fontsize=8)
    if i == 1:
        ax.text(x_values_marathoners[i], marathoners_mass[i] - 2.5, marathoners_names[i], fontsize=8)
    if i == 2:
        ax.text(x_values_marathoners[i]-0.18, marathoners_mass[i]+0.8, marathoners_names[i], fontsize=8)
    if i == 3:
        ax.text(x_values_marathoners[i]- 0.2, marathoners_mass[i] + 1.5, marathoners_names[i], fontsize=8)
    if i == 4:
        ax.text(x_values_marathoners[i]+0.02, marathoners_mass[i] - 1.5, marathoners_names[i], fontsize=8)
    if i == 5:
        ax.text(x_values_marathoners[i]+0.02, marathoners_mass[i] - 1.5, marathoners_names[i], fontsize=8)
    if i == 6:
        ax.text(x_values_marathoners[i]+0.02, marathoners_mass[i] - 1.5, marathoners_names[i], fontsize=8)

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([45])
ax.set_yticklabels(['45'])
ax.set_ylim([45,100])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
# plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('mass $[kg]$')
plt.savefig('figures_paper/SKEL_mass.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_mass.jpg',format = 'jpg', dpi = 600)

## Height

# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5 - 0.25/2,1 - 0.25/2))

for i in range(len(cases_models)):
    y_value = np.around(optimaltrajectories[cases_models[i]]['height'], decimals=2).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.22, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label
x_values_sprinter = 0.5 + np.array((0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21))
for i in range(len(sprinter_names)):
    ax.scatter(x_values_sprinter[i], sprinters_lengths[i], color=colormap_5[1], s=10)
    if i < 5:
        ax.text(x_values_sprinter[i] - 0.05, sprinters_lengths[i] + 0.005, sprinter_names[i], fontsize = 6)
    elif i < 6:
        ax.text(x_values_sprinter[i] - 0.1, sprinters_lengths[i] + 0.005, sprinter_names[i], fontsize = 6)
    else:
        ax.text(x_values_sprinter[i] - 0.15, sprinters_lengths[i] - 0.015, sprinter_names[i], fontsize = 6)

x_values_marathoners = 1 + np.array((0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21))
for i in range(len(marathoners_names)):
    ax.scatter(x_values_marathoners[i], marathoners_lengths[i], color=colormap_5[2], s=10)
    if i < 2:
        ax.text(x_values_marathoners[i] - 0.05,  marathoners_lengths[i] + 0.005,marathoners_names[i], fontsize = 6)
    elif i != 5:
        ax.text(x_values_marathoners[i] - 0.08, marathoners_lengths[i] + 0.005, marathoners_names[i], fontsize=6)
    else:
        ax.text(x_values_marathoners[i], marathoners_lengths[i] + 0.005, marathoners_names[i], fontsize=6)


# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([1.6])
ax.set_yticklabels(['1.6'])
ax.set_ylim([1.6,2])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('height $[m]$')
plt.savefig('figures_paper/SKEL_height.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_height.jpg',format = 'jpg', dpi = 600)



## BMI
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
x_values = np.array((0,0.5,1))

for i in range(len(cases_models)):
    y_value = np.around(optimaltrajectories[cases_models[i]]['BMI'], decimals=1).flatten()
    bar_i = ax.bar(x = x_values[i] , height = y_value, width = 0.45, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# no ytick labels
ax.set_yticklabels([])
ax.set_yticks([])

# ytick labels
ax.set_yticks([0])
ax.set_yticklabels(['0'])
ax.set_ylim([0,30])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5), fontsize = 10)
plt.title('BMI $[kg/m^2]$')
plt.savefig('figures_paper/SKEL_bmi.svg',format = 'svg')
plt.savefig('figures_paper/SKEL_bmi.jpg',format = 'jpg', dpi = 600)




# Scaling
label_loc_unit_circle = np.linspace(start=0, stop=2 * np.pi, num=200)
unit_circle = np.ones((200,))
bodies = ["torso","pelvis",
              "thigh",
              "shank",
              "foot"
              ]

body_scaling = np.zeros((3,54))
body_scaling[0, :] = optimaltrajectories[cases_models[0]]['skeleton_scaling']
body_scaling[1, :] = optimaltrajectories[cases_models[1]]['skeleton_scaling']
body_scaling[2, :] = optimaltrajectories[cases_models[2]]['skeleton_scaling']

fig = plt.figure(figsize=(8,3))
fig.tight_layout(pad=5.0)
# Body Length
# Setup axes in figure
ax = fig.add_subplot((131), projection='polar')

length_indices = np.linspace(start=1, stop=52, num=18).tolist()
length_indices = [int(item) for item in length_indices]

body_scaling_length = body_scaling[:,length_indices]
body_scaling_length = body_scaling_length[:,[11,0,1,2,3]]
body_scaling_length = np.concatenate((body_scaling_length, np.reshape(body_scaling_length[:,0],(3,1))), axis = 1)
body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 0)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_length[0], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_length[1], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_length[2], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.3,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 + 20*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body length')



# Setup axes in figure
ax = fig.add_subplot((133), projection='polar')

length_indices = np.linspace(start=0, stop=51, num=18).tolist()
length_indices = [int(item) for item in length_indices]

body_scaling_length = body_scaling[:,length_indices]
body_scaling_length = body_scaling_length[:,[11,0,1,2,3]]
body_scaling_length = np.concatenate((body_scaling_length, np.reshape(body_scaling_length[:,0],(3,1))), axis = 1)
body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 0)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_length[0], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_length[1], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_length[2], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.3,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2  + 20*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body depth')



# Setup axes in figure
ax = fig.add_subplot((132), projection='polar')

length_indices = np.linspace(start=2, stop=53, num=18).tolist()
length_indices = [int(item) for item in length_indices]

body_scaling_length = body_scaling[:,length_indices]
body_scaling_length = body_scaling_length[:,[11,0,1,2,3]]
body_scaling_length = np.concatenate((body_scaling_length, np.reshape(body_scaling_length[:,0],(3,1))), axis = 1)
body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 0)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_length[0], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_length[1], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_length[2], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.3,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 + 20*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body width')






fig = plt.figure(figsize=(6,3.5))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((121), projection='polar')
for i in range(len(cases_models)):
    maximal_isometric_torques_generic = optimaltrajectories[cases_models[0]]['maximal_isometric_torques']
    maximal_isometric_torques = optimaltrajectories[cases_models[i]]['maximal_isometric_torques']
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=7)
    maximal_isometric_torques_generic_overall = np.amax(np.abs(maximal_isometric_torques_generic), 0)
    maximal_isometric_torques_overall = np.amax(np.abs(maximal_isometric_torques),0)

    maximal_isometric_torques_overall = maximal_isometric_torques_overall / maximal_isometric_torques_generic_overall
    maximal_isometric_torques_overall = np.concatenate((maximal_isometric_torques_overall, np.reshape(maximal_isometric_torques_overall[0],(1,))))
    ax.plot(label_loc, maximal_isometric_torques_overall, label=labels_models[i], color = colormap_5[i])


labels_max_iso_torque = ['hip \n extension', 'hip \n flexion', 'knee \n extension', 'knee \n flexion', 'ankle \n extension', 'ankle \n flexion']
ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(labels_max_iso_torque)+1))
ax.set_xticklabels([*labels_max_iso_torque, labels_max_iso_torque[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 5)
ax.get_yaxis().set_visible(False)
ax.set_ylim([0, 1.5])
ax.text(0.5,0.5, '0', transform=ax.transAxes)
ax.text(0.6,0.9, '1.0', transform=ax.transAxes)
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)

plt.legend(labels_models[:3],loc='upper left',bbox_to_anchor=(1.5,0.75))
plt.title('maximal isometric torque [-] \n normalized to generic')
plt.savefig('figures_paper/max_iso_torque_simple_morphology.svg',format = 'svg')
plt.savefig('figures_paper/max_iso_torque_simple_morphology.jpg',format = 'jpg', dpi =1200)




fig = plt.figure(figsize=(6,3.5))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((121), projection='polar')
for i in range(len(cases_models_MUSC)):
    maximal_isometric_torques_generic = optimaltrajectories[cases_models_MUSC[0]]['maximal_isometric_torques']
    maximal_isometric_torques = optimaltrajectories[cases_models_MUSC[i]]['maximal_isometric_torques']
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=7)
    maximal_isometric_torques_generic_overall = np.amax(np.abs(maximal_isometric_torques_generic), 0)
    maximal_isometric_torques_overall = np.amax(np.abs(maximal_isometric_torques),0)

    maximal_isometric_torques_overall = maximal_isometric_torques_overall / maximal_isometric_torques_generic_overall
    maximal_isometric_torques_overall = np.concatenate((maximal_isometric_torques_overall, np.reshape(maximal_isometric_torques_overall[0],(1,))))
    ax.plot(label_loc, maximal_isometric_torques_overall, label=labels_models[i], color = colormap_MUSC[i])


labels_max_iso_torque = ['hip \n extension', 'hip \n flexion', 'knee \n extension', 'knee \n flexion', 'ankle \n extension', 'ankle \n flexion']
ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(labels_max_iso_torque)+1))
ax.set_xticklabels([*labels_max_iso_torque, labels_max_iso_torque[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 5)
ax.get_yaxis().set_visible(False)
ax.set_ylim([0, 1.5])
ax.text(0.5,0.5, '0', transform=ax.transAxes)
ax.text(0.6,0.9, '1.0', transform=ax.transAxes)
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)

plt.legend(labels_models[:3],loc='upper left',bbox_to_anchor=(1.5,0.75))
plt.title('maximal isometric torque [-] \n normalized to generic')
plt.savefig('figures_paper/MUSC_max_iso_torque_simple_morphology.svg',format = 'svg')
plt.savefig('figures_paper/MUSC_max_iso_torque_simple_morphology.jpg',format = 'jpg', dpi =1200)





plt.show()



## Step length during sprinting
# Generate data matrix/vector to plot
sprintingSpeeds = np.zeros((3,))
sprintingSpeeds[0] = np.around(optimaltrajectories['6']['coordinate_values'][3,-1] / 2, decimals=2)
sprintingSpeeds[1] = np.around(optimaltrajectories['7']['coordinate_values'][3,-1] / 2, decimals=2)
sprintingSpeeds[2] = np.around(optimaltrajectories['2']['coordinate_values'][3,-1] / 2, decimals=2)

# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels
ax.set_yticks([0, 2])
ax.set_yticklabels(['0','2'])
ax.set_ylim([0, 2.5])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('step length sprinting [m]')
plt.savefig('figures_presentation/step_length_sprinting_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/step_length_sprinting_morphology.jpg',format = 'jpg', dpi = 600)



## Step frequency during sprinting
# Generate data matrix/vector to plot
sprintingSpeeds = np.zeros((3,))
sprintingSpeeds[0] = np.around( 1 / optimaltrajectories['6']['time'][0,-1] * 2, decimals=2)
sprintingSpeeds[1] = np.around( 1 / optimaltrajectories['7']['time'][0,-1] * 2, decimals=2)
sprintingSpeeds[2] = np.around( 1 / optimaltrajectories['2']['time'][0,-1] * 2, decimals=2)

# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels
ax.set_yticks([3])
ax.set_yticklabels(['3'])
ax.set_ylim([3, 6])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('step frequency sprinting [Hz]')
plt.savefig('figures_presentation/step_frequency_sprinting_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/step_frequency_sprinting_morphology.jpg',format = 'jpg', dpi = 600)

## Contact time during sprinting

sprintingSpeeds = np.zeros((3,))

GRF = optimaltrajectories['6']['GRF']
time = optimaltrajectories['6']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[0] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)
GRF = optimaltrajectories['7']['GRF']
time = optimaltrajectories['7']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[1] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)
GRF = optimaltrajectories['2']['GRF']
time = optimaltrajectories['2']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[2] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)


# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels

ax.set_yticklabels('0')
ax.set_yticks([0])
ax.set_ylim([0, 200])
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('contact time sprinting [ms]')
plt.savefig('figures_presentation/contact_time_sprinting_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/contact_time_sprinting_morphology.jpg',format = 'jpg', dpi = 600)

## Step length
# Generate data matrix/vector to plot
sprintingSpeeds = np.zeros((3,))
sprintingSpeeds[0] = np.around(optimaltrajectories['5']['coordinate_values'][3,-1] / 2, decimals=2)
sprintingSpeeds[1] = np.around(optimaltrajectories['1']['coordinate_values'][3,-1] / 2, decimals=2)
sprintingSpeeds[2] = np.around(optimaltrajectories['8']['coordinate_values'][3,-1] / 2, decimals=2)

# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels
ax.set_yticks([0, 2])
ax.set_yticklabels(['0','2'])
ax.set_ylim([0, 2.5])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('step length marathon [m]')
plt.savefig('figures_presentation/step_length_marathon_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/step_length_marathon_morphology.jpg',format = 'jpg', dpi = 600)



## Step frequency
# Generate data matrix/vector to plot
sprintingSpeeds = np.zeros((3,))
sprintingSpeeds[0] = np.around( 1 / optimaltrajectories['5']['time'][0,-1] * 2, decimals=2)
sprintingSpeeds[1] = np.around( 1 / optimaltrajectories['1']['time'][0,-1] * 2, decimals=2)
sprintingSpeeds[2] = np.around( 1 / optimaltrajectories['8']['time'][0,-1] * 2, decimals=2)

# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels
ax.set_yticks([3])
ax.set_yticklabels(['3'])
ax.set_ylim([3, 6])

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('step frequency marathon [Hz]')
plt.savefig('figures_presentation/step_frequency_marathon_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/step_frequency_marathon_morphology.jpg',format = 'jpg', dpi = 600)

## Contact time

sprintingSpeeds = np.zeros((3,))

GRF = optimaltrajectories['5']['GRF']
time = optimaltrajectories['5']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[0] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)
GRF = optimaltrajectories['1']['GRF']
time = optimaltrajectories['1']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[1] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)
GRF = optimaltrajectories['8']['GRF']
time = optimaltrajectories['8']['time']
idx1 = np.argmax(GRF[4,:]>10)
idx2 = idx1 + np.argmax(GRF[4,idx1:]<10)
sprintingSpeeds[2] = np.round_(1000*time[0,idx2] - 1000*time[0,idx1], decimals=0)


# Setup x-axis
sprintingSpeeds_x = np.array((1,2,3))
# Set-up figure canvas
fig = plt.figure(figsize=(6,3))
# Setup axes in figure
ax = plt.axes((0.1,0.1,0.5,0.8)) # relative positions
# Setup
for i in range(len(sprintingSpeeds)):
    bar_i = ax.bar(x = sprintingSpeeds_x[i], height = sprintingSpeeds[i], width = 0.5, color = colormap_5[i], label = labels_models[i])
    ax.bar_label(bar_i) # add bar label

# no xtick labels
ax.set_xticklabels([])
ax.set_xticks([])

# ytick labels

ax.set_yticklabels('0')
ax.set_yticks([0])
ax.set_ylim([0, 300])
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)

# generate legend
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1,0.5))
plt.title('contact time marathon [ms]')
plt.savefig('figures_presentation/contact_time_marathon_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/contact_time_marathon_morphology.jpg',format = 'jpg', dpi = 600)







########## MUSCLES #########
# Muscle main function
hip_abduction = ['glut_max1_r', 'glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r',
                  'glut_min2_r', 'glut_min3_r']

hip_adduction = ['add_brev_r', 'add_long_r', 'add_mag1_r', 'add_mag2_r', 'add_mag3_r',
                  'bifemlh_r', 'grac_r', 'pect_r', 'semimem_r', 'semiten_r']

hip_extension = ['add_long_r', 'add_mag1_r', 'add_mag2_r', 'add_mag3_r',
                  'bifemlh_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r', 'glut_med3_r', 'glut_min3_r',
                 'semimem_r','semiten_r']

hip_flexion = ['add_brev_r', 'add_long_r', 'glut_med1_r', 'glut_min1_r', 'grac_r'
                  , 'iliacus_r', 'pect_r', 'psoas_r', 'rect_fem_r', 'sar_r', 'tfl_r']

hip_inrot = ['glut_med1_r', 'glut_min1_r', 'iliacus_r', 'psoas_r', 'tfl_r']
hip_exrot = ['gem_r', 'glut_med3_r', 'glut_min3_r', 'peri_r', 'quad_fem_r']

knee_flexion = ['bifemlh_r', 'bifemsh_r', 'grac_r', 'lat_gas_r', 'med_gas_r','sar_r','semimem_r','semiten_r']
knee_extension = ['rect_fem_r', 'vas_int_r', 'vas_lat_r', 'vas_med_r']

ankle_plantarflexion = ['flex_dig_r', 'flex_hal_r', 'lat_gas_r', 'med_gas_r','per_brev_r','per_long_r','soleus_r','tib_post_r']
ankle_dorsiflexion = ['ext_dig_r', 'ext_hal_r', 'per_tert_r', 'tib_ant_r']

ankle_eversion = ['ext_dig_r', 'per_brev_r','per_long_r', 'per_tert_r']
ankle_inversion = ['ext_hal_r', 'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'tib_post_r']


##### muscle names simple
hip_abduction_simple = ['glut_max1', 'glut_med1', 'glut_med2', 'glut_med3', 'glut_min1',
                  'glut_min2', 'glut_min3']

hip_adduction_simple = ['add_brev', 'add_long', 'add_mag1', 'add_mag2', 'add_mag3',
                  'bifemlh', 'grac', 'pect', 'semimem', 'semiten']

hip_extension_simple = ['add_long', 'add_mag1', 'add_mag2', 'add_mag3',
                  'bifemlh', 'glut_max1', 'glut_max2', 'glut_max3', 'glut_med3', 'glut_min3',
                 'semimem','semiten']

hip_flexion_simple = ['add_brev', 'add_long', 'glut_med1', 'glut_min1', 'grac'
                  , 'iliacus', 'pect', 'psoas', 'rect_fem', 'sar', 'tfl']

hip_inrot_simple = ['glut_med1_r', 'glut_min1_r', 'iliacus_r', 'psoas_r', 'tfl_r']
hip_exrot_simple = ['gem_r', 'glut_med3_r', 'glut_min3_r', 'peri_r', 'quad_fem_r']

knee_flexion_simple = ['bifemlh', 'bifemsh', 'grac', 'lat_gas', 'med_gas','sar','semimem','semiten']
knee_extension_simple = ['rect_fem', 'vas_int_r', 'vas_lat', 'vas_med']

ankle_plantarflexion_simple = ['flex_dig', 'flex_hal', 'lat_gas', 'med_gas','per_brev','per_long','soleus','tib_post']
ankle_dorsiflexion_simple = ['ext_dig', 'ext_hal', 'per_tert', 'tib_ant']

ankle_eversion_simple = ['ext_dig', 'per_brev','per_long', 'per_tert']
ankle_inversion_simple = ['ext_hal', 'flex_dig', 'flex_hal', 'tib_ant', 'tib_post']



list_muscles = optimaltrajectories['1']['muscles']

max_iso_force = np.zeros((3,92))
max_iso_force[0,:] = np.reshape(optimaltrajectories['5']['maximal_isometric_force_original'],(1,92))
max_iso_force[1,:] = np.reshape(optimaltrajectories['1']['maximal_isometric_force_scaled'],(1,92)) / max_iso_force[0,:]
max_iso_force[2,:] = np.reshape(optimaltrajectories['8']['maximal_isometric_force_scaled'],(1,92)) / max_iso_force[0,:]
max_iso_force[0,:] = max_iso_force[0,:] / max_iso_force[0,:]


label_loc_unit_circle = np.linspace(start=0, stop=2 * np.pi, num=200)
unit_circle = np.ones((200,))


## Muscles in sagittal plane

fig = plt.figure(figsize=(14,8))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((331), projection='polar')

indices = []
for i in range(len(hip_flexion)):
    indices.append(list_muscles.index(hip_flexion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_flexion_simple, hip_flexion_simple[0]])

for key, spine in ax.spines.items():
    spine.set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)



ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)

plt.title('hip flexion')


# Hip extension
# Setup axes in figure
ax = fig.add_subplot((334), projection='polar')

indices = []
for i in range(len(hip_extension)):
    indices.append(list_muscles.index(hip_extension[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_extension_simple, hip_extension_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('hip extension')


# Knee flexion
# Setup axes in figure
ax = fig.add_subplot((332), projection='polar')

indices = []
for i in range(len(knee_flexion)):
    indices.append(list_muscles.index(knee_flexion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*knee_flexion_simple, knee_flexion_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
plt.title('knee flexion')


# Knee extension
# Setup axes in figure
ax = fig.add_subplot((335), projection='polar')

indices = []
for i in range(len(knee_extension)):
    indices.append(list_muscles.index(knee_extension[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*knee_extension_simple, knee_extension_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('knee extension')
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1.5,0.75))



# Ankle dorsiflexion
# Setup axes in figure
ax = fig.add_subplot((333), projection='polar')

indices = []
for i in range(len(ankle_dorsiflexion)):
    indices.append(list_muscles.index(ankle_dorsiflexion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*ankle_dorsiflexion_simple, ankle_dorsiflexion_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
plt.title('ankle dorsiflexion')


# Ankle plantarflexion
# Setup axes in figure
ax = fig.add_subplot((336), projection='polar')

indices = []
for i in range(len(ankle_plantarflexion)):
    indices.append(list_muscles.index(ankle_plantarflexion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*ankle_plantarflexion_simple, ankle_plantarflexion_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('ankle plantarflexion')


plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.savefig('figures_presentation/muscle_adaptation_sagittal_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/muscle_adaptation_sagittal_morphology.jpg',format = 'jpg', dpi = 600)


## Muscles in frontal and coronal plane


fig = plt.figure(figsize=(14,8))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((331), projection='polar')

indices = []
for i in range(len(hip_adduction)):
    indices.append(list_muscles.index(hip_adduction[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_adduction_simple, hip_adduction_simple[0]])

for key, spine in ax.spines.items():
    spine.set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)



ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)

plt.title('hip adduction')


# Hip extension
# Setup axes in figure
ax = fig.add_subplot((332), projection='polar')

indices = []
for i in range(len(hip_abduction)):
    indices.append(list_muscles.index(hip_abduction[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_abduction_simple, hip_abduction_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('hip abduction')


# Hip in rotation
# Setup axes in figure
ax = fig.add_subplot((334), projection='polar')

indices = []
for i in range(len(hip_inrot_simple)):
    indices.append(list_muscles.index(hip_inrot_simple[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_inrot_simple, hip_inrot_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
plt.title('hip internal rotation')


# Knee extension
# Setup axes in figure
ax = fig.add_subplot((335), projection='polar')

indices = []
for i in range(len(hip_exrot_simple)):
    indices.append(list_muscles.index(hip_exrot_simple[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*hip_exrot_simple, hip_exrot_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('hip external rotation')
plt.legend(labels_models,loc='upper left',bbox_to_anchor=(1.5,0.75))



# Ankle inversion
# Setup axes in figure
ax = fig.add_subplot((337), projection='polar')

indices = []
for i in range(len(ankle_inversion)):
    indices.append(list_muscles.index(ankle_inversion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*ankle_inversion_simple, ankle_inversion_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
plt.title('ankle inversion')


# Ankle plantarflexion
# Setup axes in figure
ax = fig.add_subplot((338), projection='polar')

indices = []
for i in range(len(ankle_eversion)):
    indices.append(list_muscles.index(ankle_eversion[i]))
    # max_iso_force_generic = optimaltrajectories['5']['maximal_isometric_force_original']
indices.append(indices[0])
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(indices))

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(indices)))
ax.set_xticklabels([*ankle_eversion_simple, ankle_eversion_simple[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, max_iso_force[0,indices], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, max_iso_force[1,indices], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, max_iso_force[2,indices], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('ankle eversion')


plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.savefig('figures_presentation/muscle_adaptation_frontal_coronal_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/muscle_adaptation_frontal_coronal_morphology.jpg',format = 'jpg', dpi = 600)




############################################################################################



bodies = ["torso",
              "pelvis",
              "thigh",
              "shank",
              "foot"
              ]

bodies_muscles = ["torso",
                  "pelvis",
                  "femur_l",
                  "tibia_l",
                  "talus_l",
                  "femur_r",
                  "tibia_r",
                  "talus_r"
                  ]

## Body scaling
# Generate data matrix/vector to plot
body_scaling = np.zeros((3,42))
body_scaling[0,:] = np.round_(optimaltrajectories['5']['body_scaling'], decimals=2)
body_scaling[1,:] = np.round_(optimaltrajectories['1']['body_scaling'], decimals=2)
body_scaling[2,:] = np.round_(optimaltrajectories['8']['body_scaling'], decimals=2)



fig = plt.figure(figsize=(14,8))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((321), projection='polar')

length_indices = np.linspace(start=1, stop=40, num=14).tolist()
length_indices = [int(item) for item in length_indices]
body_scaling_length = body_scaling[:,length_indices]
body_scaling_length = body_scaling_length[:,:5]
body_scaling_length = np.concatenate((body_scaling_length, np.reshape(body_scaling_length[:,0],(3,1))), axis = 1)
# body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_length[0,:], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_length[1,:], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_length[2,:], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body length scaling')


ax = fig.add_subplot((323), projection='polar')

width_indices = np.linspace(start=2, stop=41, num=14).tolist()
width_indices = [int(item) for item in width_indices]
body_scaling_width = body_scaling[:,width_indices]
body_scaling_width = body_scaling_width[:,:5]
body_scaling_width = np.concatenate((body_scaling_width, np.reshape(body_scaling_width[:,0],(3,1))), axis = 1)
# body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_width[0,:], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_width[1,:], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_width[2,:], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body width scaling')
plt.legend(labels_models[:3],loc='upper left',bbox_to_anchor=(1.5,0.75))

ax = fig.add_subplot((325), projection='polar')

depth_indices = np.linspace(start=0, stop=39, num=14).tolist()
depth_indices = [int(item) for item in depth_indices]
body_scaling_depth = body_scaling[:,depth_indices]
body_scaling_depth = body_scaling_depth[:,:5]
body_scaling_depth = np.concatenate((body_scaling_depth, np.reshape(body_scaling_depth[:,0],(3,1))), axis = 1)
# body_scaling_length = [*body_scaling_length, body_scaling_length[:,0]]

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1)

ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(bodies)+1))
ax.set_xticklabels([*bodies, bodies[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
ax.set_ylim(0.5,1.2)
ax.get_yaxis().set_visible(False)


ax.plot(label_loc, body_scaling_depth[0,:], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, body_scaling_depth[1,:], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, body_scaling_depth[2,:], label=labels_models[2], color = colormap_5[2])
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
angle_annotation = np.pi/2 - 15*np.pi/180
ax.plot([angle_annotation, angle_annotation], [1, 1.3], color='k', linestyle='-', linewidth=1)
# generate legend
plt.title('body depth scaling')

plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

plt.savefig('figures_presentation/body_scaling_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/body_scaling_morphology.jpg',format = 'jpg', dpi =1200)




#### Max torque in the following joint positions
fig = plt.figure(figsize=(14,8))

maxIsoTorque_torques = np.zeros((3,20,6))
maxIsoTorque_torques[0,:,:] = optimaltrajectories['5']['maxIsoTorque_torques']
maxIsoTorque_torques[1,:,:] = optimaltrajectories['1']['maxIsoTorque_torques']
maxIsoTorque_torques[2,:,:] = optimaltrajectories['8']['maxIsoTorque_torques']
maxIsoTorque_joint_pose = np.zeros((3,17,20,6))
maxIsoTorque_joint_pose[0,:,:,:] = optimaltrajectories['5']['maxIsoTorque_joint_pose']
maxIsoTorque_joint_pose[1,:,:,:] = optimaltrajectories['1']['maxIsoTorque_joint_pose']
maxIsoTorque_joint_pose[2,:,:,:] = optimaltrajectories['8']['maxIsoTorque_joint_pose']
maxIsoTorque_passive_torques = np.zeros((3,20,6))
maxIsoTorque_passive_torques[0,:,:] = optimaltrajectories['5']['maxIsoTorque_passivetorques']
maxIsoTorque_passive_torques[1,:,:] = optimaltrajectories['1']['maxIsoTorque_passivetorques']
maxIsoTorque_passive_torques[2,:,:] = optimaltrajectories['8']['maxIsoTorque_passivetorques']

ax = fig.add_subplot((321))
ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,0,:,0], maxIsoTorque_torques[0,:,0], label=labels_models[0], color = colormap_5[0])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,0,:,0], maxIsoTorque_torques[1,:,0], label=labels_models[1], color = colormap_5[1])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,0,:,0], maxIsoTorque_torques[2,:,0], label=labels_models[2], color = colormap_5[2])

ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,0,:,0], maxIsoTorque_passive_torques[0,:,0], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,0,:,0], maxIsoTorque_passive_torques[1,:,0], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,0,:,0], maxIsoTorque_passive_torques[2,:,0], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)

plt.title('maximal hip extension \n (knee angle = 30deg)')
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,-400])
ax.set_yticklabels(['0','-400'])
ax.set_ylim([-400, 20])
ax.set_ylabel('torque [Nm]')
ax.set_xlabel('hip angle [deg]')

ax = fig.add_subplot((322))
ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,0,:,1], maxIsoTorque_torques[0,:,1], label=labels_models[0], color = colormap_5[0])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,0,:,1], maxIsoTorque_torques[1,:,1], label=labels_models[1], color = colormap_5[1])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,0,:,1], maxIsoTorque_torques[2,:,1], label=labels_models[2], color = colormap_5[2])

ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,0,:,1], maxIsoTorque_passive_torques[0,:,1], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,0,:,1], maxIsoTorque_passive_torques[1,:,1], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,0,:,1], maxIsoTorque_passive_torques[2,:,1], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)
plt.title('maximal hip flexion \n (knee angle = 30deg)')
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,200])
ax.set_yticklabels(['0','200'])
ax.set_ylim([-110, 250])
ax.set_xlabel('hip angle [deg]')

ax = fig.add_subplot((323))
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[0,6,:,3], -maxIsoTorque_torques[0,:,3], label=labels_models[0], color = colormap_5[0])
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[1,6,:,3], -maxIsoTorque_torques[1,:,3], label=labels_models[1], color = colormap_5[1])
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[2,6,:,3], -maxIsoTorque_torques[2,:,3], label=labels_models[2], color = colormap_5[2])

ax.plot(-180/np.pi*maxIsoTorque_joint_pose[0,6,:,3], -maxIsoTorque_passive_torques[0,:,3], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[1,6,:,3], -maxIsoTorque_passive_torques[1,:,3], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[2,6,:,3], -maxIsoTorque_passive_torques[2,:,3], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,-200])
ax.set_yticklabels(['0','-200'])
ax.set_ylim([-220, 30])
ax.set_ylabel('torque [Nm]')
ax.set_xlabel('knee angle [deg]')


plt.title('maximal knee extension \n (hip angle = 45deg, ankle angle = 0deg)')
ax = fig.add_subplot((324))
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[0,6,:,2], -maxIsoTorque_torques[0,:,2], label=labels_models[0], color = colormap_5[0])
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[1,6,:,2], -maxIsoTorque_torques[1,:,2], label=labels_models[1], color = colormap_5[1])
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[2,6,:,2], -maxIsoTorque_torques[2,:,2], label=labels_models[2], color = colormap_5[2])

ax.plot(-180/np.pi*maxIsoTorque_joint_pose[0,6,:,2], -maxIsoTorque_passive_torques[0,:,2], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[1,6,:,2], -maxIsoTorque_passive_torques[1,:,2], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(-180/np.pi*maxIsoTorque_joint_pose[2,6,:,2], -maxIsoTorque_passive_torques[2,:,2], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)
plt.title('maximal knee flexion \n (hip angle = 45deg, ankle angle = 0deg)')

# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,150])
ax.set_yticklabels(['0','150'])
ax.set_ylim([-30, 180])
ax.set_xlabel('knee angle [deg]')

labels_models_torques = ('generic', 'marathon morphology', 'sprinting morphology','generic (passive)', 'marathon morphology (passive)', 'sprinting morphology (passive)')

plt.legend(labels_models_torques,loc='upper left',bbox_to_anchor=(1.5,0.75))


ax = fig.add_subplot((325))

ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,8,:,4], maxIsoTorque_torques[0,:,4], label=labels_models[0], color = colormap_5[0])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,8,:,4], maxIsoTorque_torques[1,:,4], label=labels_models[1], color = colormap_5[1])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,8,:,4], maxIsoTorque_torques[2,:,4], label=labels_models[2], color = colormap_5[2])

ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,8,:,4], maxIsoTorque_passive_torques[0,:,4], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,8,:,4], maxIsoTorque_passive_torques[1,:,4], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,8,:,4], maxIsoTorque_passive_torques[2,:,4], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,-250])
ax.set_yticklabels(['0','-250'])
ax.set_ylim([-260, 10])
ax.set_ylabel('torque [Nm]')
plt.title('maximal ankle extension \n (knee angle = 45deg)')
ax.set_xlabel('ankle angle [deg]')


ax = fig.add_subplot((326))
ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,8,:,5], maxIsoTorque_torques[0,:,5], label=labels_models[0], color = colormap_5[0])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,8,:,5], maxIsoTorque_torques[1,:,5], label=labels_models[1], color = colormap_5[1])
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,8,:,5], maxIsoTorque_torques[2,:,5], label=labels_models[2], color = colormap_5[2])

ax.plot(180/np.pi*maxIsoTorque_joint_pose[0,8,:,5], maxIsoTorque_passive_torques[0,:,5], label=labels_models[0], color = colormap_5[0], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[1,8,:,5], maxIsoTorque_passive_torques[1,:,5], label=labels_models[1], color = colormap_5[1], linestyle='--', linewidth=1)
ax.plot(180/np.pi*maxIsoTorque_joint_pose[2,8,:,5], maxIsoTorque_passive_torques[2,:,5], label=labels_models[2], color = colormap_5[2], linestyle='--', linewidth=1)
plt.title('maximal ankle flexion \n (knee angle = 45deg)')
# no spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ytick labels
ax.set_yticks([0,50])
ax.set_yticklabels(['0','50'])
ax.set_ylim([-30, 60])
ax.set_xlabel('ankle angle [deg]')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)

plt.savefig('figures_presentation/max_iso_torque_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/max_iso_torque_morphology.jpg',format = 'jpg', dpi =1200)




fig = plt.figure(figsize=(10,6))

# Hip flexion
# Setup axes in figure
ax = fig.add_subplot((121), projection='polar')
label_loc = np.linspace(start=0, stop=2 * np.pi, num=7)
maxIsoTorque_torque_overall = np.amax(np.abs(maxIsoTorque_torques),1)
maxIsoTorque_torque_overall = maxIsoTorque_torque_overall / maxIsoTorque_torque_overall[0, :]
maxIsoTorque_torque_overall = np.concatenate((maxIsoTorque_torque_overall, np.reshape(maxIsoTorque_torque_overall[:,0],(3,1))), axis = 1)
ax.plot(label_loc, maxIsoTorque_torque_overall[0,(0,1,3,2,4,5,0)], label=labels_models[0], color = colormap_5[0])
ax.plot(label_loc, maxIsoTorque_torque_overall[1,(0,1,3,2,4,5,0)], label=labels_models[1], color = colormap_5[1])
ax.plot(label_loc, maxIsoTorque_torque_overall[2,(0,1,3,2,4,5,0)], label=labels_models[2], color = colormap_5[2])

labels_max_iso_torque = ['hip extension', 'hip flexion', 'knee extension', 'knee flexion', 'ankle extension', 'ankle flexion']
ax.set_xticks(np.linspace(start=0, stop=2 * np.pi, num=len(labels_max_iso_torque)+1))
ax.set_xticklabels([*labels_max_iso_torque, labels_max_iso_torque[0]])

ax.spines['polar'].set_visible(False)
ax.tick_params(pad = 15)
# ax.get_yaxis().set_visible(False)
ax.set_ylim([0, 1.5])
ax.text(0.5,0.5, '0.5', transform=ax.transAxes)
ax.text(0.6,1.0, '1.0', transform=ax.transAxes)
ax.plot(label_loc_unit_circle, unit_circle, label='unit circle', color = (0,0,0), linewidth=1)

plt.legend(labels_models[:3],loc='upper left',bbox_to_anchor=(1.5,0.75))
plt.title('maximal isometric torque [-] \n normalized to generic')
plt.savefig('figures_presentation/max_iso_torque_simple_morphology.svg',format = 'svg')
plt.savefig('figures_presentation/max_iso_torque_simple_morphology.jpg',format = 'jpg', dpi =1200)