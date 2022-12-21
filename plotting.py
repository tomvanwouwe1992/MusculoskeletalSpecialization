def plot_trajectory():
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 6, sharex=True)
    fig.suptitle('Joint torques interpretation')
    for i, ax in enumerate(axs.flat):
        index = i + 6
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))

        ax.plot(time_col_opt, muscle_joint_torques_col_opt[index, :], c=next(color), label='muscle torque')
        ax.plot(time_col_opt, limit_joint_torques_col_opt[index, :], c=next(color), label='limit torque')
        ax.plot(time_col_opt, passive_joint_torques_col_opt[index, :], c=next(color), label='passive torque')
        ax.plot(time_col_opt, generalized_forces_col_opt[index, :], c=next(color), label='torque')
        ax.set_title(joints[index])
        handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')

    fig, axs = plt.subplots(4, 6, sharex=True)
    fig.suptitle('Muscle joint torques interpretation')
    for i, ax in enumerate(axs.flat):
        index = i + 6
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))

        ax.plot(time_col_opt, muscle_joint_torques_col_opt[index, :], c=next(color), label='muscle torque')
        ax.plot(time_col_opt, muscle_passive_joint_torques_col_opt[index, :], c=next(color), label='passive torque')
        ax.plot(time_col_opt, muscle_active_joint_torques_col_opt[index, :], c=next(color), label='active torque')
        ax.set_title(joints[index])
        handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, loc='upper right')

    fig, axs = plt.subplots(4, 6, sharex=True)
    fig.suptitle('Joint angles')
    for i, ax in enumerate(axs.flat):
        index = i + 6
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, 180 / np.pi * Qs_col_opt_nsc[index, :])
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Q']['upper'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Q']['lower'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))

        ax.set_title(joints[index])

    fig, axs = plt.subplots(4, 6, sharex=True)
    fig.suptitle('Joint angular velocities')
    for i, ax in enumerate(axs.flat):
        index = i + 6
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, 180 / np.pi * Qds_col_opt_nsc[index, :])
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Qd']['upper'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Qd']['lower'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))

        ax.set_title(joints[index])

    fig, axs = plt.subplots(4, 6, sharex=True)
    fig.suptitle('Joint angular accelerations')
    for i, ax in enumerate(axs.flat):
        index = i + 6
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, 180 / np.pi * Qdds_col_opt_nsc[index, :])
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Qdd']['upper'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt,
                180 / np.pi * optimaltrajectories[case]['bounds_Qdd']['lower'][joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))
        ax.set_title(joints[index])

    fig, axs = plt.subplots(2, 3, sharex=True)
    fig.suptitle('Pelvis coordinates')
    for i, ax in enumerate(axs.flat):
        index = i
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, Qs_col_opt_nsc[index, :])
        ax.plot(time_col_opt, optimaltrajectories[case]['bounds_Q']['upper'][
            joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt,
                optimaltrajectories[case]['bounds_Q']['lower'][
                    joints[index]].to_numpy() * np.ones(
                    (polynomial_order * number_of_mesh_intervals,)))

        ax.set_title(joints[index])

    fig, axs = plt.subplots(2, 3, sharex=True)
    fig.suptitle('Pelvis coordinates velocities')
    for i, ax in enumerate(axs.flat):
        index = i
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, Qds_col_opt_nsc[index, :])
        ax.plot(time_col_opt, optimaltrajectories[case]['bounds_Qd']['upper'][
            joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt, optimaltrajectories[case]['bounds_Qd']['lower'][
            joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))

        ax.set_title(joints[index])

    fig, axs = plt.subplots(2, 3, sharex=True)
    fig.suptitle('Pelvis coordinates accelerations')
    for i, ax in enumerate(axs.flat):
        index = i
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
        ax.plot(time_col_opt, Qdds_col_opt_nsc[index, :])
        ax.plot(time_col_opt, optimaltrajectories[case]['bounds_Qdd']['upper'][
            joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
        ax.plot(time_col_opt, optimaltrajectories[case]['bounds_Qdd']['lower'][
            joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))

        ax.set_title(joints[index])

    fig, axs = plt.subplots(1, 3, sharex=True)
    fig.suptitle('Scaling factors')
    titles_dim = ['depth', 'height', 'width']
    for i, ax in enumerate(axs.flat):
        index = i
        to_plot = scaling_vector_opt[i::3]
        ax.scatter(to_plot, np.arange(np.size(to_plot)))
        if i == 0:
            ax.set_yticks(np.arange(np.size(to_plot)))
            ax.set_yticklabels(optimaltrajectories[case]['skeleton_scaling_bodies'])
        ax.set_title(titles_dim[index])

    fig, axs = plt.subplots(1, 3, sharex=True)
    fig.suptitle('Ground reaction forces')
    titles_dim = ['X', 'Y', 'Z']
    for i, ax in enumerate(axs.flat):
        ax.plot(time_col_opt, GRF_col_opt[i, :])
        ax.plot(time_col_opt, GRF_col_opt[i + 3, :])
        ax.set_title(titles_dim[i])

    plt.show()