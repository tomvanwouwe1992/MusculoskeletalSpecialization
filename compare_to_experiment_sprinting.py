import numpy, os, matplotlib, opensim
import matplotlib.pyplot as plt
import scipy.signal as signal_processing

##########SIMULATION
import numpy as np

cmap = matplotlib.colormaps['Accent']
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


folder = 'Results'
pathMain = os.getcwd()
pathTrajectories = os.path.join(pathMain, folder)
pathTrajectories =folder

optimaltrajectories = numpy.load(os.path.join(pathTrajectories,
                                           'optimaltrajectories.npy'),
                              allow_pickle=True).item()

scaling_vector = optimaltrajectories['11']['skeleton_scaling']
Q_at_collocation = optimaltrajectories['11']['Q_at_collocation']
time_col_opt = optimaltrajectories['11']['time_col_opt']
joint_names = optimaltrajectories['11']['joint_names']

joints_of_interest = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
dict_joint_to_name = {'pelvis_tilt':'pelvis\nposterior tilt [°]' , 'pelvis_list': 'pelvis\nlist [°]', 'pelvis_rotation': 'pelvis\nrotation [°]',
                      'hip_flexion_l': 'hip\nflexion [°]', 'hip_adduction_l': 'hip\nadduction [°]', 'hip_rotation_l': 'hip\ninternal rotation [°]',
                      'knee_angle_l': 'knee\nextension [°]', 'ankle_angle_l': 'ankle\nflexion [°]', 'lumbar_extension': 'lumbar\nextension [°]',
                      'lumbar_bending': 'lumbar\nbending [°]', 'lumbar_rotation': 'lumbar\nrotation[°]'}


time_col_opt = numpy.concatenate((time_col_opt, time_col_opt + time_col_opt[-1]))
time_col_opt = time_col_opt - time_col_opt[0]

#concatenate first and second part
Q_of_interest = numpy.zeros((101,len(joints_of_interest)))
for i, joint_of_interest in enumerate(joints_of_interest):
    col_index = joint_names.index(joint_of_interest)
    Q_at_collocation_joint_part_1 = Q_at_collocation[col_index,:]
    if joint_of_interest[-1] == 'l':
        joints_of_interest_part_2 = joint_of_interest[:-1] + 'r'
        col_index_part_2 = joint_names.index(joints_of_interest_part_2)
        Q_at_collocation_joint_part_2 = Q_at_collocation[col_index_part_2, :]
    elif joint_of_interest == 'lumbar_bending' or joint_of_interest == 'lumbar_rotation' or joint_of_interest == 'pelvis_list' or joint_of_interest == 'pelvis_rotation' :
        joints_of_interest_part_2 = joint_of_interest
        col_index_part_2 = joint_names.index(joints_of_interest_part_2)
        Q_at_collocation_joint_part_2 = -Q_at_collocation[col_index_part_2, :]
    else:
        joints_of_interest_part_2 = joint_of_interest
        col_index_part_2 = joint_names.index(joints_of_interest_part_2)
        Q_at_collocation_joint_part_2 = Q_at_collocation[col_index_part_2, :]

    Q_at_collocation_joint = numpy.concatenate((Q_at_collocation_joint_part_1, Q_at_collocation_joint_part_2))

    time_col_opt_resampled = numpy.arange(0, time_col_opt[-1] + time_col_opt[-1] / 100, time_col_opt[-1] / 100)
    Q_at_collocation_joint = numpy.interp(time_col_opt_resampled, time_col_opt,
                                         Q_at_collocation_joint)
    Q_of_interest[:,i] = 180/numpy.pi*Q_at_collocation_joint


# ground reaction force
left_GRF_full_part_1 = optimaltrajectories['11']['external_function_ouput'][34:37,:]
left_GRF_full_part_2 = optimaltrajectories['11']['external_function_ouput'][31:34,:]
left_GRF_full_original = numpy.concatenate((left_GRF_full_part_1.T, left_GRF_full_part_2.T))

left_GRF_part_1 = optimaltrajectories['11']['external_function_ouput'][35,:]
left_GRF_part_2 = optimaltrajectories['11']['external_function_ouput'][32,:]
left_GRF = numpy.concatenate((left_GRF_part_1, left_GRF_part_2))
time_col_opt_resampled = numpy.arange(0, time_col_opt[-1] + time_col_opt[-1] / 100, time_col_opt[-1] / 100)
left_GRF = numpy.interp(time_col_opt_resampled, time_col_opt,
                                      left_GRF)

left_GRF_full = numpy.zeros((time_col_opt_resampled.shape[0],3))
left_GRF_full[:,0] = numpy.interp(time_col_opt_resampled, time_col_opt,
                                  left_GRF_full_original[:,0])
left_GRF_full[:,1] = numpy.interp(time_col_opt_resampled, time_col_opt,
                                  left_GRF_full_original[:,1])
left_GRF_full[:,2] = -numpy.interp(time_col_opt_resampled, time_col_opt,
                                  left_GRF_full_original[:,2])

b,a = signal_processing.butter(2,15,fs=1/time_col_opt_resampled[1])
left_GRF = signal_processing.filtfilt(b,a, left_GRF)
for i in range(left_GRF_full.shape[1]):
    left_GRF_full[:,i] = signal_processing.filtfilt(b,a, left_GRF_full[:,i])


for i in range(len(left_GRF)):
    if left_GRF[i] < 10:
        left_GRF[i] = 0

left_GRF_zeros = numpy.where(left_GRF[60:] == 0)[0]
left_foot_off_index = left_GRF_zeros[0] + 60

left_GRF = np.concatenate((left_GRF[left_foot_off_index:], left_GRF[:left_foot_off_index]))
left_GRF_full = np.concatenate((left_GRF_full[left_foot_off_index:], left_GRF_full[:left_foot_off_index]))
Q_of_interest = np.concatenate((Q_of_interest[left_foot_off_index:,:], Q_of_interest[:left_foot_off_index,:] ))

left_GRF_non_zeros = numpy.where(left_GRF > 0)[0]
left_foot_on_index = left_GRF_non_zeros[0]

left_GRF = np.concatenate((left_GRF[left_foot_on_index:], left_GRF[:left_foot_on_index]))
left_GRF_full =  np.concatenate((left_GRF_full[left_foot_on_index:,:], left_GRF_full[:left_foot_on_index,:]))
Q_of_interest = np.concatenate((Q_of_interest[left_foot_on_index:,:], Q_of_interest[:left_foot_on_index,:] ))

##########EXPERIMENT

all_motion_data = []
subject_names = ['subject01', 'subject02', 'subject03', 'subject04', 'subject08' ,'subject10', 'subject11', 'subject17', 'subject19', 'subject20']
all_GRF_data = []
all_GRF_data_full = []

root_path = os.path.dirname(os.path.abspath('compare_to_experiment_sprinting.py'))
motion_file = os.path.join(root_path, 'timDorn_sprinting_data','IK','JA1Gait35_ik.mot')
GRF_file = os.path.join(root_path, 'timDorn_sprinting_data','ID','JA1Gait35_grf.mot')

# Load the motion file
motion = opensim.Storage(motion_file)

# Get the number of time steps
num_time_steps = motion.getSize()

# Get the number of columns (degrees of freedom)
num_columns = motion.getColumnLabels().getSize()
column_names = []
for i in range(num_columns):
    column_names.append(motion.getColumnLabels().get(i))

# Create a NumPy array to store the motion data
motion_data = numpy.zeros((num_time_steps, num_columns))


# Fill the motion data array
for i in range(0,num_time_steps):
    state = motion.getStateVector(i)
    motion_data[i ,0] = state.getTime()

    for j in range(num_columns-1):
        motion_data[i, j+1] = state.getData().get(j)

GRF = opensim.Storage(GRF_file)
num_time_steps = GRF.getSize()
GRF_data = numpy.zeros((num_time_steps, 2))
GRF_data_full = numpy.zeros((num_time_steps, 4))

 # left vertical GRF
for i in range(0,174):
    GRF_data[i ,1] = GRF.getStateVector(i).getData().get(10+18)
    GRF_data[i, 0] = GRF.getStateVector(i).getTime()
    GRF_data_full[i, 0] = GRF.getStateVector(i).getTime()
    GRF_data_full[i, 1] = GRF.getStateVector(i).getData().get(10+18-1)
    GRF_data_full[i, 2] = GRF.getStateVector(i).getData().get(10+18)
    GRF_data_full[i, 3] = GRF.getStateVector(i).getData().get(10+18+1)

for i in range(174,num_time_steps):
    GRF_data[i ,1] = GRF.getStateVector(i).getData().get(10+5*9+9)
    GRF_data[i, 0] = GRF.getStateVector(i).getTime()
    GRF_data_full[i, 0] = GRF.getStateVector(i).getTime()
    GRF_data_full[i, 1] = GRF.getStateVector(i).getData().get(10+5*9+9-1)
    GRF_data_full[i, 2] = GRF.getStateVector(i).getData().get(10+5*9+9)
    GRF_data_full[i, 3] = GRF.getStateVector(i).getData().get(10+5*9+9+1)

all_motion_data.append(motion_data)
all_GRF_data.append(GRF_data)
all_GRF_data_full.append(GRF_data_full)






fig = plt.figure()

plot_counter = 1
# ax = plt.axes((0.1, 0.1, 1.1, 1.1))
for i in range(34):
    if column_names[i] in joints_of_interest:
        if column_names[i] == 'knee_angle_l':
            multiplier = -1
        else:
            multiplier = 1
        ax = plt.subplot(3,4,plot_counter)
        plot_counter = plot_counter + 1
        for k in range(len(all_motion_data)):
            GRF_data = all_GRF_data[k]
            motion_data = all_motion_data[k]

            GRF_zeros = numpy.where(GRF_data[:, 1] == 0)[0]
            left_foot_off_index = GRF_zeros[0]

            if left_foot_off_index == 0 or 1:
                left_foot_on_index = numpy.where(numpy.abs(GRF_data[left_foot_off_index + 1:, 1]) > 0)[0][
                                         0] + left_foot_off_index + 1
                left_foot_off_index = numpy.where(GRF_data[left_foot_on_index + 1:, 1] == 0)[0][
                                          0] + left_foot_on_index + 1

            left_foot_on_2_index = numpy.where(numpy.abs(GRF_data[left_foot_off_index + 1:, 1]) > 0)[0][
                                       0] + left_foot_off_index + 1
            left_foot_off_2_index = numpy.where(GRF_data[left_foot_on_2_index + 1:, 1] == 0)[0][
                                        0] + left_foot_on_2_index + 1

            left_foot_off_time = GRF_data[left_foot_off_index, 0]
            left_foot_off_index_motion_data = numpy.argmin(numpy.abs(motion_data[:, 0] - left_foot_off_time))

            left_foot_on_time = GRF_data[left_foot_on_index, 0]
            left_foot_on_index_motion_data = numpy.argmin(numpy.abs(motion_data[:, 0] - left_foot_on_time))

            left_foot_on_2_time = GRF_data[left_foot_on_2_index, 0]
            left_foot_on_2_index_motion_data = numpy.argmin(numpy.abs(motion_data[:, 0] - left_foot_on_2_time))

            left_foot_off_2_time = GRF_data[left_foot_off_2_index, 0]
            left_foot_off_2_index_motion_data = numpy.argmin(numpy.abs(motion_data[:, 0] - left_foot_off_2_time))

            time = motion_data[left_foot_on_index_motion_data:left_foot_on_2_index_motion_data, 0]

            if time[-1] - time[0] < 0.5:
                print('d')


            time = time - time[0]
            time_resampled = numpy.arange(0,time[-1]+time[-1]/100, time[-1]/100)
            motion_data_resampled = numpy.interp(time_resampled, time, motion_data[left_foot_on_index_motion_data:left_foot_on_2_index_motion_data,i])
            ax.plot(100*time_resampled/time_resampled[-1], multiplier*180/numpy.pi*motion_data_resampled,'--', color='black', alpha=0.8,linewidth=1.0)
            title = dict_joint_to_name[column_names[i]]
            # if title[-1] == 'l':
            #     title = title[:-2]
            # title = title.replace('_',' ')
            plt.title(title, fontsize = 9)
        if column_names[i] in joints_of_interest:
            index = joints_of_interest.index(column_names[i])
            ax.plot(100*time_col_opt_resampled/time_col_opt_resampled[-1], Q_of_interest[:,index],color=colormap_5[0],linewidth=1.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xticks([0,50,100])
            ax.set_xticklabels(['0%','50%','100%'], fontname = 'Corbel')
            ax.set_xlim([0, 100])
            plt.yticks(fontname = 'Corbel')


fig.tight_layout(h_pad=2)

plt.savefig('figures_paper/experimental_validation_sprinting_vJuly.svg', format='svg')
plt.savefig('figures_paper/experimental_validation_sprinting_vJuly.jpg', format='jpg', dpi=600)



fig = plt.figure()


titles_GRF = ['foreaft [BW]', 'vertical [BW]', 'mediolateral [BW]']
plot_counter = 1
# ax = plt.axes((0.1, 0.1, 1.1, 1.1))
for i in range(3):
    ax = plt.subplot(1,3,plot_counter)
    plot_counter = plot_counter + 1
    for k in range(len(all_motion_data)):
        GRF_data = all_GRF_data[k]
        motion_data = all_motion_data[k]
        GRF_data_full = all_GRF_data_full[k]
        GRF_zeros = numpy.where(GRF_data[:, 1] == 0)[0]
        left_foot_off_index = GRF_zeros[0]

        if left_foot_off_index == 0 or 1:
            left_foot_on_index = numpy.where(numpy.abs(GRF_data[left_foot_off_index + 1:, 1]) > 0)[0][0] + left_foot_off_index + 1
            left_foot_off_index = numpy.where(GRF_data[left_foot_on_index + 1:, 1] == 0)[0][
                                        0] + left_foot_on_index + 1

        left_foot_on_2_index = numpy.where(numpy.abs(GRF_data[left_foot_off_index + 1:, 1]) > 0)[0][0] + left_foot_off_index + 1
        left_foot_off_2_index = numpy.where(GRF_data[left_foot_on_2_index + 1:, 1] == 0)[0][0] + left_foot_on_2_index + 1

        left_foot_off_time = GRF_data[left_foot_off_index, 0]
        left_foot_off_index_GRF_data = numpy.argmin(numpy.abs(GRF_data[:,0] - left_foot_off_time))

        left_foot_on_time = GRF_data[left_foot_on_index, 0]
        left_foot_on_index_GRF_data = numpy.argmin(numpy.abs(GRF_data[:, 0] - left_foot_on_time))

        left_foot_on_2_time = GRF_data[left_foot_on_2_index, 0]
        left_foot_on_2_index_GRF_data = numpy.argmin(numpy.abs(GRF_data[:, 0] - left_foot_on_2_time))

        left_foot_off_2_time = GRF_data[left_foot_off_2_index, 0]
        left_foot_off_2_index_GRF_data = numpy.argmin(numpy.abs(GRF_data[:, 0] - left_foot_off_2_time))



        time = GRF_data_full[left_foot_on_index_GRF_data:left_foot_on_2_index_GRF_data,0]

        if time[-1] - time[0] < 0.5:
            print('d')


        time = time - time[0]
        time_resampled = numpy.arange(0,time[-1]+time[-1]/100, time[-1]/100)
        GRF_data_resampled = numpy.interp(time_resampled, time, GRF_data_full[left_foot_on_index_GRF_data:left_foot_on_2_index_GRF_data,i+1])
        ax.plot(100*time_resampled/time_resampled[-1], GRF_data_resampled/78/9.81,'--', color='black', alpha=0.8,linewidth=1.0)
        title = titles_GRF[i]
        # if title[-1] == 'l':
        #     title = title[:-2]
        # title = title.replace('_',' ')
        plt.title(title)


        ax.plot(100*time_col_opt_resampled/time_col_opt_resampled[-1], left_GRF_full[:,i]/75/9.81,color=colormap_5[0],linewidth=2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([0,50,100])
        ax.set_xticklabels(['0%','50%','100%'])
        ax.set_xlim([0, 100])


fig.tight_layout(h_pad=2)

plt.savefig('figures_paper/experimental_validation_sprinting_stance_first_GRF.svg', format='svg')
plt.savefig('figures_paper/experimental_validation_sprinting_stance_first_GRF.jpg', format='jpg', dpi=600)

plt.show()
