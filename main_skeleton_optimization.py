import os, sys, utilities, muscleData, casadiFunctions, NeMu_subfunctions, settings
import casadi as ca
import numpy as np

solveProblem = True # Set True to solve the optimal control problem.
saveResults = True  # Set True to save the results of the optimization.
analyzeResults = True  # Set True to analyze the results.
loadResults = True  # Set True to load the results of the optimization.
writeMotionFiles = True  # Set True to write motion files for use in OpenSim GUI
saveOptimalTrajectories = True  # Set True to save optimal trajectories
plotResults = False
set_guess_from_solution = True

# Select the case(s) for which you want to solve the associated problem(s)
cases = [str(i) for i in range(1, 7)]
settings = settings.getSettings()

for case in cases:
    ### Model and path
    model_name = 'Hamner_modified'  # default model
    path_main = os.getcwd()
    path_model_folder = os.path.join(path_main, 'Models')
    path_model = os.path.join(path_model_folder, model_name + '.osim')

    path_external_function = os.path.join(path_model_folder, 'ExternalFunction')
    path_case = 'Case_' + case
    path_trajectories = os.path.join(path_main, 'ResultsSprinting')
    path_results = os.path.join(path_trajectories, path_case)
    path_scaled_models = os.path.join(path_results, 'ScaledModels')
    os.makedirs(path_results, exist_ok=True)
    os.makedirs(path_scaled_models, exist_ok=True)

    ### Settings.
    polynomial_order = 3  # default interpolating polynomial order.
    number_of_parallel_threads = 10  # default number of threads.

    if 'bounds_scaling_factors' in settings[case]:
        bounds_skeleton_scaling_factors = settings[case]['bounds_scaling_factors']

    if isinstance(bounds_skeleton_scaling_factors, str):
        case_to_load = bounds_skeleton_scaling_factors[-1]
        optimaltrajectories = np.load(os.path.join(path_trajectories, 'optimaltrajectories.npy'),
                allow_pickle=True)
        optimaltrajectories = optimaltrajectories.item()
        skeleton_scaling_factors = optimaltrajectories[case_to_load]['skeleton_scaling']
        bounds_skeleton_scaling_factors = [skeleton_scaling_factors, skeleton_scaling_factors]

    if 'muscle_geometry' in settings[case]:
        muscle_geometry = settings[case]['muscle_geometry']

    convergence_tolerance_IPOPT = 4
    if 'tol' in settings[case]:
        convergence_tolerance_IPOPT = settings[case]['tol']

    number_of_mesh_intervals = 20  # default number of mesh intervals.
    if 'N' in settings[case]:
        number_of_mesh_intervals = settings[case]['N']

    if 'energy_cost_function_scaling' in settings[case]:
        energy_cost_function_scaling = settings[case]['energy_cost_function_scaling']

    ### Task definition
    cost_function_weights = {'metabolicEnergyRateTerm': energy_cost_function_scaling*500,  # default.
               'activationTerm': energy_cost_function_scaling*2000,
               'jointAccelerationTerm': energy_cost_function_scaling*50000,
               'armExcitationTerm': energy_cost_function_scaling*1000000,
               'passiveTorqueTerm': energy_cost_function_scaling*1000,
               'controls': 1e3*energy_cost_function_scaling*0.001}

    if energy_cost_function_scaling < 0.1:
        cost_function_weights['controls'] = energy_cost_function_scaling
    else:
        cost_function_weights['controls'] = 0.001

    if 'enforce_target_speed' in settings[case]:
        enforce_target_speed = settings[case]['enforce_target_speed']
    if 'target_speed' in settings[case]:
        target_speed = settings[case]['target_speed']
    if 'muscle_geometry' in settings[case]:
        muscle_geometry = settings[case]['muscle_geometry']
    modelMass = 75.2
    if 'modelMass' in settings[case]:
        modelMass = settings[case]['modelMass']

    ### Bodies to scale and how their scaling factors are coupled
    skeleton_scaling_bodies = ["pelvis",
                               "femur_l",
                               "tibia_l",
                               "talus_l", "calcn_l", "toes_l",
                               "femur_r",
                               "tibia_r",
                               "talus_r", "calcn_r", "toes_r",
                               "torso", "humerus_l", "radiusulnar_l", "hand_l", "humerus_r",
                               "radiusulnar_r", "hand_r"]

    bodies_skeleton_scaling_coupling = ["pelvis",
                                        ["femur_l", "femur_r"],
                                        ["tibia_l", "tibia_r"],
                                        ["talus_l", "calcn_l", "toes_l", "talus_r", "calcn_r", "toes_r"],
                                        ["torso", "pelvis", "humerus_l", "radiusulnar_l", "hand_l", "humerus_r",
                                         "radiusulnar_r", "hand_r"]]

    ### Get names, indices of joints, segments from model
    muscles, number_of_muscles, joints, number_of_joints, muscle_actuated_joints, muscle_articulated_bodies, \
    muscle_actuated_joints_indices_in_joints, number_of_muscle_actuated_joints, rotational_joints, rotational_joint_indices_in_joints, \
    non_actuated_joints, non_actuated_joints_indices_in_joints, non_muscle_actuated_joints, non_muscle_actuated_joints_indices_in_joints, \
    number_of_non_muscle_actuated_joints, number_of_muscle_actuated_and_non_actuated_joints, muscle_actuated_and_non_actuated_joints_indices_in_joints, \
    mtpJoints, limit_torque_joints, number_of_limit_torque_joints, periodic_opposite_joints, periodic_opposite_joints_indices_in_joints, \
    periodic_joints, periodic_joints_indices_start_to_end_velocity_matching, periodic_joints_indices_start_to_end_position_matching, periodic_muscles, \
    periodic_muscles_indices_start_to_end_matching, periodic_actuators, periodic_actuators_indices_start_to_end_matching, \
    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, bodies_skeleton_scaling_coupling_indices = utilities.get_names_and_indices_of_joints_and_bodies(path_model, skeleton_scaling_bodies, bodies_skeleton_scaling_coupling)

    ### Musculoskeletal geometry
    NeMu_folder = os.path.dirname(
        os.path.abspath('main.py')) + "/Models/NeMu/tanh"
    f_get_muscle_tendon_length_velocity_moment_arm = NeMu_subfunctions.NeMuApproximation(muscles, muscle_actuated_joints, muscle_articulated_bodies, NeMu_folder)
    f_muscle_length_scaling = casadiFunctions.muscle_length_scaling_vector(muscle_actuated_joints, f_get_muscle_tendon_length_velocity_moment_arm)

    ### Muscle parameters
    muscle_parameters = muscleData.getMTParameters(path_model, muscles)
    tendon_stiffness = muscleData.tendonStiffness(len(muscles))
    specific_tension = muscleData.specificTension(muscles)

    activationTimeConstant = 0.015
    deactivationTimeConstant = 0.06

    ### Dynamics
    f_hillEquilibrium = casadiFunctions.hillEquilibrium_muscle_length_scaling(muscle_parameters, tendon_stiffness,
                                                                              specific_tension)
    f_actuatorDynamics = casadiFunctions.armActivationDynamics(number_of_non_muscle_actuated_joints)

    if enforce_target_speed and target_speed < 5:
        skeleton_dynamics_and_more = ca.external('F', os.path.join(
            path_external_function, 'Hamner_modified_skeleton_scaling_running.dll'))
    else:
        skeleton_dynamics_and_more = ca.external('F', os.path.join(
            path_external_function, 'Hamner_modified_skeleton_scaling_sprinting.dll'))

    # This holds a structure with what the fields and indices are of the external function
    map_external_function_outputs = np.load(os.path.join(
        path_external_function, 'Hamner_modified' + '_map.npy'),
        allow_pickle=True).item()

    idxCalcOr_r, idxCalcOr_l, idxFemurOr_r, idxFemurOr_l, idxHandOr_r, idxHandOr_l, idxTibiaOr_r, idxTibiaOr_l, idxToesOr_r, idxToesOr_l, idxTorsoOr_r, idxTorsoOr_l, idxGRF, idxGRM = utilities.get_indices_external_function_outputs(map_external_function_outputs)

    # Helper lists to map order of joints defined here vs order in the external function output
    joint_indices_in_external_function = [joints.index(joint)
                                          for joint in list(map_external_function_outputs['residuals'].keys())]

    ### Metabolic energy model.
    maximalIsometricForce = muscle_parameters[0, :]
    optimalFiberLength = muscle_parameters[1, :]
    slow_twitch_ratio = muscleData.slowTwitchRatio(muscles)
    smoothingConstant = 10
    f_metabolicsBhargava = casadiFunctions.metabolicsBhargava_muscle_length_scaling(
        slow_twitch_ratio, maximalIsometricForce, optimalFiberLength, specific_tension, smoothingConstant)

    ### Model height and scaling of total body mass
    f_height = casadiFunctions.get_height(joints, skeleton_dynamics_and_more)
    f_body_mass_scaling = casadiFunctions.mass_scaling_with_skeleton_volume(joints, skeleton_dynamics_and_more)

    ### Limit joint torques
    f_limit_torque = {}
    damping = 0.1
    for joint in limit_torque_joints:
        f_limit_torque[joint] = casadiFunctions.getLimitTorques(
            muscleData.passiveTorqueData(joint)[0],
            muscleData.passiveTorqueData(joint)[1], damping)

    ### Passive torques in non muscle actuated joints
    stiffnessArm = 0
    dampingArm = 0.1
    f_linearPassiveTorque = casadiFunctions.getLinearPassiveTorques(stiffnessArm,
                                                                    dampingArm)

    stiffnessMtp = settings[case]['stiffnessMtp']
    dampingMtp = settings[case]['dampingMtp']
    f_linearPassiveMtpTorque = casadiFunctions.getLinearPassiveTorques(stiffnessMtp,
                                                       dampingMtp)

    ### Cost function terms
    f_nMTPreserveSum2 = casadiFunctions.normSumPow(2, 2)
    f_NMusclesSum2 = casadiFunctions.normSumPow(number_of_muscles, 2)
    f_nArmJointsSum2 = casadiFunctions.normSumPow(number_of_non_muscle_actuated_joints, 2)
    f_nNoArmJointsSum2 = casadiFunctions.normSumPow(number_of_muscle_actuated_and_non_actuated_joints, 2)
    f_nPassiveTorqueJointsSum2 = casadiFunctions.normSumPow(number_of_limit_torque_joints, 2)
    f_diffTorques = casadiFunctions.diffTorques()

    ### Initial guess and bounds
    ## Select experimental kinematics to generate guess/bounds from
    if enforce_target_speed:
        motion_walk = 'sprinting'
    else:
        motion_walk = 'sprinting'
    nametrial_walk_id = 'average_' + motion_walk + '_HGC_mtp'
    nametrial_walk_IK = 'IK_' + nametrial_walk_id
    pathIK_walk = os.path.join(path_model_folder, 'templates', 'IK',
                               nametrial_walk_IK + '.mot')

    Qs_walk_filt = utilities.getIK(pathIK_walk, joints)[1]

    ## Bounds
    import boundsVariables
    bounds = boundsVariables.bounds(Qs_walk_filt, joints, muscles, non_muscle_actuated_joints)

    # Static parameters.
    ubFinalTime, lbFinalTime = bounds.getBoundsFinalTime()

    # States.
    ubA, lbA, scaling_activation = bounds.getBoundsActivation()
    activation_upper_bound_at_mesh = ca.vec(ubA.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    activation_lower_bound_at_mesh = ca.vec(lbA.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    activation_upper_bound_at_collocation = ca.vec(ubA.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    activation_lower_bound_at_collocation = ca.vec(lbA.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    ubF, lbF, scaling_muscle_force = bounds.getBoundsForce()
    muscle_force_upper_bound_at_mesh = ca.vec(ubF.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    muscle_force_lower_bound_at_mesh = ca.vec(lbF.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    muscle_force_upper_bound_at_collocation = ca.vec(ubF.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    muscle_force_lower_bound_at_collocation = ca.vec(lbF.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    ubQs, lbQs, scaling_Q, ubQs0, lbQs0 = bounds.getBoundsPosition()
    Q_upper_bound_at_mesh = ca.vec(ubQs.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    Q_lower_bound_at_mesh = ca.vec(lbQs.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    Q_upper_bound_at_collocation = ca.vec(ubQs.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    Q_lower_bound_at_collocation = ca.vec(lbQs.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    # We want to further constraint the pelvis_tx position at the first mesh
    # point, such that the model starts running with pelvis_tx=0.
    Q_lower_bound_at_mesh[joints.index('pelvis_tx')] = lbQs0['pelvis_tx'].to_numpy()
    Q_upper_bound_at_mesh[joints.index('pelvis_tx')] = ubQs0['pelvis_tx'].to_numpy()

    ubQds, lbQds, scalingQds = bounds.getBoundsVelocity()
    ubQdsk = ca.vec(ubQds.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    lbQdsk = ca.vec(lbQds.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    ubQdsj = ca.vec(ubQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    lbQdsj = ca.vec(lbQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    ubArmA, lbArmA, scalingArmA = bounds.getBoundsArmActivation()
    ubArmAk = ca.vec(ubArmA.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    lbArmAk = ca.vec(lbArmA.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))).full()
    ubArmAj = ca.vec(ubArmA.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    lbArmAj = ca.vec(lbArmA.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    # Controls.
    ubADt, lbADt, scalingADt = bounds.getBoundsActivationDerivative()
    ubADtk = ca.vec(ubADt.to_numpy().T * np.ones((1, number_of_mesh_intervals))).full()
    lbADtk = ca.vec(lbADt.to_numpy().T * np.ones((1, number_of_mesh_intervals))).full()

    ubArmE, lbArmE, scalingArmE = bounds.getBoundsArmExcitation()
    ubArmEk = ca.vec(ubArmE.to_numpy().T * np.ones((1, number_of_mesh_intervals))).full()
    lbArmEk = ca.vec(lbArmE.to_numpy().T * np.ones((1, number_of_mesh_intervals))).full()

    # Slack controls.
    ubQdds, lbQdds, scalingQdds = bounds.getBoundsAcceleration()
    ubQddsj = ca.vec(ubQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    lbQddsj = ca.vec(lbQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    ubFDt, lbFDt, scalingFDt = bounds.getBoundsForceDerivative()
    ubFDtj = ca.vec(ubFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()
    lbFDtj = ca.vec(lbFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))).full()

    # Other.
    _, _, scalingMtpE = bounds.getBoundsMtpExcitation()

    ## Guess
    import guesses
    guess = guesses.hotStart(Qs_walk_filt, number_of_mesh_intervals, polynomial_order, joints, muscles, 1.33)

    # Static parameters.
    gFinalTime = guess.getGuessFinalTime()
    # States.
    gA = guess.getGuessActivation(scaling_activation)
    gACol = guess.getGuessActivationCol()
    gF = guess.getGuessForce(scaling_muscle_force)
    gFCol = guess.getGuessForceCol()
    gQs = guess.getGuessPosition(scaling_Q)
    gQsCol = guess.getGuessPositionCol()
    gQds = guess.getGuessVelocity(scalingQds)
    gQdsCol = guess.getGuessVelocityCol()
    gArmA = guess.getGuessTorqueActuatorActivation(non_muscle_actuated_joints)
    gArmACol = guess.getGuessTorqueActuatorActivationCol(non_muscle_actuated_joints)

    # Controls.
    gADt = guess.getGuessActivationDerivative(scalingADt)
    gArmE = guess.getGuessTorqueActuatorExcitation(non_muscle_actuated_joints)

    # Slack controls.
    gQdds = guess.getGuessAcceleration(scalingQdds)
    gQddsCol = guess.getGuessAccelerationCol()
    gFDt = guess.getGuessForceDerivative(scalingFDt)
    gFDtCol = guess.getGuessForceDerivativeCol()

    #####################################################################################################################
    ### Optimal control problem.
    if solveProblem:
        opti = ca.Opti()

        ## Generate variables, set bounds and set initial guess
        # Time
        finalTime = opti.variable()
        opti.subject_to(opti.bounded(lbFinalTime.iloc[0]['time'],
                                     finalTime,
                                     ubFinalTime.iloc[0]['time']))
        opti.set_initial(finalTime, gFinalTime)

        # States
        a = opti.variable(number_of_muscles, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(activation_lower_bound_at_mesh, ca.vec(a), activation_upper_bound_at_mesh))
        opti.set_initial(a, gA.to_numpy().T)

        a_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(activation_lower_bound_at_collocation, ca.vec(a_col), activation_upper_bound_at_collocation))
        opti.set_initial(a_col, gACol.to_numpy().T)

        normF = opti.variable(number_of_muscles, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(muscle_force_lower_bound_at_mesh, ca.vec(normF), muscle_force_upper_bound_at_mesh))
        opti.set_initial(normF, gF.to_numpy().T)

        normF_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(muscle_force_lower_bound_at_collocation, ca.vec(normF_col), muscle_force_upper_bound_at_collocation))
        opti.set_initial(normF_col, gFCol.to_numpy().T)

        Qs = opti.variable(number_of_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(Q_lower_bound_at_mesh, ca.vec(Qs), Q_upper_bound_at_mesh))
        opti.set_initial(Qs, gQs.to_numpy().T)

        Qs_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(Q_lower_bound_at_collocation, ca.vec(Qs_col), Q_upper_bound_at_collocation))
        opti.set_initial(Qs_col, gQsCol.to_numpy().T)

        Qds = opti.variable(number_of_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(lbQdsk, ca.vec(Qds), ubQdsk))
        opti.set_initial(Qds, gQds.to_numpy().T)

        Qds_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbQdsj, ca.vec(Qds_col), ubQdsj))
        opti.set_initial(Qds_col, gQdsCol.to_numpy().T)

        aArm = opti.variable(number_of_non_muscle_actuated_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(lbArmAk, ca.vec(aArm), ubArmAk))
        opti.set_initial(aArm, gArmA.to_numpy().T)

        aArm_col = opti.variable(number_of_non_muscle_actuated_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbArmAj, ca.vec(aArm_col), ubArmAj))
        opti.set_initial(aArm_col, gArmACol.to_numpy().T)

        # Controls.
        aDt = opti.variable(number_of_muscles, number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbADtk, ca.vec(aDt), ubADtk))
        opti.set_initial(aDt, gADt.to_numpy().T)

        eArm = opti.variable(number_of_non_muscle_actuated_joints, number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbArmEk, ca.vec(eArm), ubArmEk))
        opti.set_initial(eArm, gArmE.to_numpy().T)

        # Slack controls (because of implicit definition of the system dynamics
        normFDt_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbFDtj, ca.vec(normFDt_col), ubFDtj))
        opti.set_initial(normFDt_col, gFDtCol.to_numpy().T)

        Qdds_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbQddsj, ca.vec(Qdds_col),
                                     ubQddsj))
        opti.set_initial(Qdds_col, gQddsCol.to_numpy().T)

        # Model scaling parameters (body dimensions, muscle lengths, mass)
        if np.all(bounds_skeleton_scaling_factors[0] == bounds_skeleton_scaling_factors[1]):
            scaling_vector = bounds_skeleton_scaling_factors[0]
            scaling_vector_opti = opti.variable(54, polynomial_order * number_of_mesh_intervals)
            opti.subject_to(scaling_vector_opti == np.tile(scaling_vector,(1,polynomial_order * number_of_mesh_intervals)))

            muscle_length_scaling_vector_opti = opti.variable(92, polynomial_order * number_of_mesh_intervals)
            muscle_length_scaling = f_muscle_length_scaling(
                scaling_vector[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies])
            opti.subject_to(muscle_length_scaling_vector_opti == np.tile(muscle_length_scaling,(1, polynomial_order * number_of_mesh_intervals)))

            mass_scaling = f_body_mass_scaling(scaling_vector)
            model_mass_scaling_opti = opti.variable(1, polynomial_order * number_of_mesh_intervals)
            opti.subject_to(model_mass_scaling_opti == np.tile(mass_scaling,(1, polynomial_order * number_of_mesh_intervals)))

        else:
            scaling_vector_opti = opti.variable(54, polynomial_order * number_of_mesh_intervals)
            scaling_vector_guess = np.ones((54,1))
            opti.set_initial(scaling_vector_opti,scaling_vector_guess * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.subject_to(
                opti.bounded(bounds_skeleton_scaling_factors[0], scaling_vector_opti[:,0], bounds_skeleton_scaling_factors[1]))

            # Constrain scaling factors to be equal over time
            for j in range(1, polynomial_order * number_of_mesh_intervals):
                opti.subject_to(scaling_vector_opti[:, 0] == scaling_vector_opti[:, j])

            # Impose coupling of scaling as described
            for i in range(len(bodies_skeleton_scaling_coupling_indices)):
                if type(bodies_skeleton_scaling_coupling_indices[i]) is list:
                    indices_first_element = [3*bodies_skeleton_scaling_coupling_indices[i][0], 3*bodies_skeleton_scaling_coupling_indices[i][0] + 1, 3*bodies_skeleton_scaling_coupling_indices[i][0] + 2]
                    for j in range(1, len(bodies_skeleton_scaling_coupling_indices[i])):
                        indices_element = [3*bodies_skeleton_scaling_coupling_indices[i][j], 3*bodies_skeleton_scaling_coupling_indices[i][j] + 1, 3*bodies_skeleton_scaling_coupling_indices[i][j] + 2]
                        opti.subject_to(scaling_vector_opti[indices_first_element, 0] == scaling_vector_opti[indices_element, 0])

            # Impose muscle length scaling is consistent with skeleton scaling
            muscle_length_scaling_vector_opti = opti.variable(92, polynomial_order * number_of_mesh_intervals)
            muscle_length_scaling_guess = np.ones((number_of_muscles,1))
            opti.set_initial(muscle_length_scaling_vector_opti,muscle_length_scaling_guess  * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            for j in range(0, polynomial_order * number_of_mesh_intervals):
                opti.subject_to(
                    muscle_length_scaling_vector_opti[:, j] == f_muscle_length_scaling(scaling_vector_opti[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, j]))

            # Impose muscle mass is consistent with skeleton scaling
            model_mass_scaling_opti = opti.variable(1, polynomial_order * number_of_mesh_intervals)
            model_mass_scaling_guess = 1
            opti.set_initial(model_mass_scaling_opti, model_mass_scaling_guess)
            for j in range(0, polynomial_order * number_of_mesh_intervals):
                mass_scaling = f_body_mass_scaling(scaling_vector_opti[:, j])
                opti.subject_to(model_mass_scaling_opti[:, j] == mass_scaling)

            # Impose BMI to be within healthy range
            subject_height_opti = f_height(scaling_vector_opti[:, 0])
            BMI_opti = (model_mass_scaling_opti[0, 0] * modelMass) / (subject_height_opti * subject_height_opti)
            opti.subject_to(
                opti.bounded(17.5, BMI_opti, 25.5))

        # Additional controls to drive the mtp joint
        MTP_reserve_col = opti.variable(2, polynomial_order * number_of_mesh_intervals)
        if enforce_target_speed and target_speed < 5:
            opti.subject_to(opti.bounded(0, ca.vertcat(MTP_reserve_col), 0))
        else:
            opti.subject_to(opti.bounded(-30, ca.vertcat(MTP_reserve_col), 5))

        ## Adapt guess if we want start from a solution
        if set_guess_from_solution == True:
            if not os.path.exists(os.path.join(path_trajectories,
                                               'optimaltrajectories.npy')):
                optimaltrajectories = {}
            else:
                optimaltrajectories = np.load(
                    os.path.join(path_trajectories,
                                 'optimaltrajectories.npy'),
                    allow_pickle=True)
                optimaltrajectories = optimaltrajectories.item()

            optimal_trajectory = optimaltrajectories[case]
            opti.set_initial(a, optimal_trajectory['muscle_activation_at_mesh'])
            opti.set_initial(a_col, optimal_trajectory['muscle_activation_at_collocation'])
            opti.set_initial(Qs, optimal_trajectory['Q_at_mesh'] / (scaling_Q.to_numpy().T *np.ones((1, number_of_mesh_intervals + 1))))
            opti.set_initial(Qs_col, optimal_trajectory['Q_at_collocation'] / (scaling_Q.to_numpy().T *np.ones((1, polynomial_order * number_of_mesh_intervals))))
            opti.set_initial(Qds, optimal_trajectory['Qd_at_mesh'] / (scalingQds.to_numpy().T *np.ones((1, number_of_mesh_intervals + 1))))
            opti.set_initial(Qds_col, optimal_trajectory['Qd_at_collocation'] / (scalingQds.to_numpy().T *np.ones((1, polynomial_order * number_of_mesh_intervals))))
            opti.set_initial(normF, optimal_trajectory['muscle_force_at_mesh'] / (scaling_muscle_force.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))))
            opti.set_initial(normF_col, optimal_trajectory['muscle_force_at_collocation'] / (scaling_muscle_force.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
            opti.set_initial(aArm, optimal_trajectory['arm_activation_at_mesh'] / scalingArmA.iloc[0]['arm_rot_r'])
            opti.set_initial(aArm_col, optimal_trajectory['arm_activation_at_collocation'] / scalingArmA.iloc[0]['arm_rot_r'])
            opti.set_initial(aDt, optimal_trajectory['muscle_activation_derivative_at_mesh'] /  (scalingADt.to_numpy().T * np.ones((1, number_of_mesh_intervals))))
            opti.set_initial(normFDt_col, optimal_trajectory['force_derivative_at_collocation'] / (scalingFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
            opti.set_initial(eArm, optimal_trajectory['arm_excitation_at_mesh'])
            opti.set_initial(Qdds_col, optimal_trajectory['Qdd_at_collocation'] / (scalingQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
            opti.set_initial(MTP_reserve_col, optimal_trajectory['MTP_reserve_at_collocation'])
            opti.set_initial(scaling_vector_opti, np.reshape(optimal_trajectory['skeleton_scaling'],(54,1)) * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(muscle_length_scaling_vector_opti, np.reshape(optimal_trajectory['muscle_scaling'],(92,1)) * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(model_mass_scaling_opti, optimal_trajectory['model_mass_scaling'] * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(finalTime, optimal_trajectory['final_time'])


        #######################################################################
        # Parallel formulation - initialize variables.
        # Time
        tf = ca.MX.sym('tf')
        # States.
        ak = ca.MX.sym('ak', number_of_muscles)
        aj = ca.MX.sym('aj', number_of_muscles, polynomial_order)
        akj = ca.horzcat(ak, aj)
        normFk = ca.MX.sym('normFk', number_of_muscles)
        normFj = ca.MX.sym('normFj', number_of_muscles, polynomial_order)
        normFkj = ca.horzcat(normFk, normFj)
        Qsk = ca.MX.sym('Qsk', number_of_joints)
        Qsj = ca.MX.sym('Qsj', number_of_joints, polynomial_order)
        Qskj = ca.horzcat(Qsk, Qsj)
        Qdsk = ca.MX.sym('Qdsk', number_of_joints)
        Qdsj = ca.MX.sym('Qdsj', number_of_joints, polynomial_order)
        Qdskj = ca.horzcat(Qdsk, Qdsj)
        aArmk = ca.MX.sym('aArmk', number_of_non_muscle_actuated_joints)
        aArmj = ca.MX.sym('aArmj', number_of_non_muscle_actuated_joints, polynomial_order)
        aArmkj = ca.horzcat(aArmk, aArmj)
        # Controls.
        aDtk = ca.MX.sym('aDtk', number_of_muscles)
        eArmk = ca.MX.sym('eArmk', number_of_non_muscle_actuated_joints)
        # Slack controls.
        normFDtj = ca.MX.sym('normFDtj', number_of_muscles, polynomial_order);
        Qddsj = ca.MX.sym('Qddsj', number_of_joints, polynomial_order)
        # Scaling parameters
        scaling_vector_j = ca.MX.sym('scaling_vector', (len(muscle_articulated_bodies) + 6) * 3, polynomial_order)
        muscle_length_scaling_vector_j = ca.MX.sym('muscle_scaling_vector', number_of_muscles, polynomial_order)
        model_mass_scaling_j = ca.MX.sym('model_mass_scaling_vector', 1, polynomial_order)
        # MTP actuation controls
        MTP_reserve_j = ca.MX.sym('MTP_reserve_j', 2, polynomial_order)

        #######################################################################
        # Time step.
        h = tf / number_of_mesh_intervals

        #######################################################################
        # Collocation matrices.
        tau = ca.collocation_points(polynomial_order, 'radau')
        [C, D] = ca.collocation_interpolators(tau)
        # Missing matrix B, add manually.
        B = [-8.88178419700125e-16, 0.376403062700467, 0.512485826188421,
             0.111111111111111]

        #######################################################################
        # Initialize cost function and constraint vectors.
        J = 0
        eq_constr = []
        ineq_constr1 = []
        ineq_constr2 = []
        ineq_constr3 = []
        ineq_constr4 = []
        ineq_constr5 = []
        ineq_constr6 = []

        #######################################################################
        # Loop over collocation points.
        for j in range(polynomial_order):
            modelMass_opti = modelMass * model_mass_scaling_j[:,j]
            # Unscale variables.
            # States.
            normFkj_nsc = normFkj * (scaling_muscle_force.to_numpy().T * np.ones((1, polynomial_order + 1)))
            Qskj_nsc = Qskj * (scaling_Q.to_numpy().T * np.ones((1, polynomial_order + 1)))
            Qdskj_nsc = Qdskj * (scalingQds.to_numpy().T * np.ones((1, polynomial_order + 1)))
            # Controls.
            aDtk_nsc = aDtk * (scalingADt.to_numpy().T)
            # Slack controls.
            normFDtj_nsc = normFDtj * (
                    scalingFDt.to_numpy().T * np.ones((1, polynomial_order)))
            Qddsj_nsc = Qddsj * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order)))
            # Qs and Qds are intertwined in the external function.
            QsQdskj_nsc = ca.MX(number_of_joints * 2, polynomial_order + 1)
            QsQdskj_nsc[::2, :] = Qskj_nsc[joint_indices_in_external_function, :]
            QsQdskj_nsc[1::2, :] = Qdskj_nsc[joint_indices_in_external_function, :]

            ## Compute all required variables
            [lMTj_lr, vMTj_lr, dMj] = f_get_muscle_tendon_length_velocity_moment_arm(ca.vertcat(scaling_vector_j[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies,j], Qskj_nsc[muscle_actuated_joints_indices_in_joints, j + 1], Qdskj_nsc[muscle_actuated_joints_indices_in_joints, j + 1]))

            [hillEquilibriumj, Fj, activeFiberForcej, passiveFiberForcej,
             normActiveFiberLengthForcej, normFiberLengthj, fiberVelocityj,_,_] = (
                f_hillEquilibrium(akj[:, j + 1], lMTj_lr, vMTj_lr,
                                                   normFkj_nsc[:, j + 1], normFDtj_nsc[:, j], muscle_length_scaling_vector_j[:,j],
                                                   model_mass_scaling_j[:,j]))

            metabolicEnergyRatej = f_metabolicsBhargava(
                akj[:, j + 1], akj[:, j + 1], normFiberLengthj, fiberVelocityj,
                activeFiberForcej, passiveFiberForcej,
                normActiveFiberLengthForcej, muscle_length_scaling_vector_j[:,j], model_mass_scaling_j[:,j])[5]

            passiveTorque_j = {}
            passiveTorquesj = ca.MX(number_of_limit_torque_joints, 1)
            for cj, joint in enumerate(limit_torque_joints):
                passiveTorque_j[joint] = f_limit_torque[joint](
                    Qskj_nsc[joints.index(joint), j + 1],
                    Qdskj_nsc[joints.index(joint), j + 1])
                passiveTorquesj[cj, 0] = passiveTorque_j[joint]

            linearPassiveTorqueArms_j = {}
            for joint in non_muscle_actuated_joints:
                linearPassiveTorqueArms_j[joint] = f_linearPassiveTorque(
                    Qskj_nsc[joints.index(joint), j + 1],
                    Qdskj_nsc[joints.index(joint), j + 1])

            linearPassiveTorqueMtp_j = {}
            for joint in mtpJoints:
                linearPassiveTorqueMtp_j[joint] = f_linearPassiveMtpTorque(
                    Qskj_nsc[joints.index(joint), j + 1],
                    Qdskj_nsc[joints.index(joint), j + 1])

            Tj = skeleton_dynamics_and_more(ca.vertcat(QsQdskj_nsc[:, j + 1], Qddsj_nsc[joint_indices_in_external_function, j], scaling_vector_j[:, j], 0))

            # Cost function
            metEnergyRateTerm = (f_NMusclesSum2(metabolicEnergyRatej) /
                                 modelMass_opti)
            activationTerm = f_NMusclesSum2(akj[:, j + 1])
            armExcitationTerm = f_nArmJointsSum2(eArmk)
            jointAccelerationTerm = (
                f_nNoArmJointsSum2(Qddsj[muscle_actuated_and_non_actuated_joints_indices_in_joints, j]))
            passiveTorqueTerm = (
                f_nPassiveTorqueJointsSum2(passiveTorquesj))
            activationDtTerm = f_NMusclesSum2(aDtk)
            forceDtTerm = f_NMusclesSum2(normFDtj[:, j])
            armAccelerationTerm = f_nArmJointsSum2(Qddsj[non_muscle_actuated_joints_indices_in_joints, j])
            reserveTerm = f_nMTPreserveSum2(MTP_reserve_j[:, j])
            J += ((cost_function_weights['metabolicEnergyRateTerm'] * metEnergyRateTerm +
                   cost_function_weights['activationTerm'] * activationTerm +
                   cost_function_weights['armExcitationTerm'] * armExcitationTerm +
                   cost_function_weights['jointAccelerationTerm'] * jointAccelerationTerm +
                   cost_function_weights['passiveTorqueTerm'] * (passiveTorqueTerm + reserveTerm) +
                   cost_function_weights['controls'] * (forceDtTerm + activationDtTerm
                                                        + armAccelerationTerm)) * h * B[j + 1])

            # Expressions and constraints for the state derivatives at the collocation points.
            ap = ca.mtimes(akj, C[j + 1])
            normFp_nsc = ca.mtimes(normFkj_nsc, C[j + 1])
            Qsp_nsc = ca.mtimes(Qskj_nsc, C[j + 1])
            Qdsp_nsc = ca.mtimes(Qdskj_nsc, C[j + 1])
            aArmp = ca.mtimes(aArmkj, C[j + 1])
            eq_constr.append((h * aDtk_nsc - ap))
            eq_constr.append((h * normFDtj_nsc[:, j] - normFp_nsc) /
                             scaling_muscle_force.to_numpy().T)
            eq_constr.append((h * Qdskj_nsc[:, j + 1] - Qsp_nsc) /
                             scaling_Q.to_numpy().T)
            eq_constr.append((h * Qddsj_nsc[:, j] - Qdsp_nsc) /
                             scalingQds.to_numpy().T)
            aArmDtj = f_actuatorDynamics(eArmk, aArmkj[:, j + 1])
            eq_constr.append(h * aArmDtj - aArmp)



            # Path constraint implicit activation dynamics
            act1 = aDtk_nsc + akj[:, j + 1] / deactivationTimeConstant
            act2 = aDtk_nsc + akj[:, j + 1] / activationTimeConstant
            ineq_constr1.append(act1)
            ineq_constr2.append(act2)

            # Path constraint implicit muscle dynamics
            eq_constr.append(hillEquilibriumj)

            # Pelvis residuals are zero
            eq_constr.append(Tj[[map_external_function_outputs['residuals'][joint]
                                 for joint in non_actuated_joints]])

            # Muscle torque + passive torque equals inverse dynamics torque
            for i in range(len(muscle_actuated_joints_indices_in_joints)):
                joint = joints[muscle_actuated_joints_indices_in_joints[i]]
                if joint != 'mtp_angle_l' and joint != 'mtp_angle_r':
                    mTj_joint = ca.sum1(dMj[:, i] * Fj)
                    diffTj_joint = f_diffTorques(
                        Tj[map_external_function_outputs['residuals'][joint]], mTj_joint, passiveTorque_j[joint])
                    eq_constr.append(diffTj_joint)


            # Torque + passive torque equals inverse dynamics torque
            for cj, joint in enumerate(non_muscle_actuated_joints):
                diffTj_joint = f_diffTorques(
                    Tj[map_external_function_outputs['residuals'][joint]] / scalingArmE.iloc[0][joint],
                    aArmkj[cj, j + 1], linearPassiveTorqueArms_j[joint] /
                    scalingArmE.iloc[0][joint])
                eq_constr.append(diffTj_joint)

            # Torque + passive torque equals inverse dynamics torque at mtp
            for count, joint in enumerate(mtpJoints):
                diffTj_joint = f_diffTorques(
                    Tj[map_external_function_outputs['residuals'][joint]] /
                    scalingMtpE.iloc[0][joint],
                    MTP_reserve_j[count, j] /
                    scalingMtpE.iloc[0][joint], (passiveTorque_j[joint] +
                        linearPassiveTorqueMtp_j[joint]) /
                    scalingMtpE.iloc[0][joint])
                eq_constr.append(diffTj_joint)

            # Prevent collision between body parts.
            diffCalcOrs = ca.sumsqr(Tj[idxCalcOr_r] - Tj[idxCalcOr_l])
            ineq_constr3.append(diffCalcOrs)
            diffFemurHandOrs_r = ca.sumsqr(Tj[idxFemurOr_r] - Tj[idxHandOr_r])
            ineq_constr4.append(diffFemurHandOrs_r)
            diffFemurHandOrs_l = ca.sumsqr(Tj[idxFemurOr_l] - Tj[idxHandOr_l])
            ineq_constr4.append(diffFemurHandOrs_l)
            diffTibiaOrs = ca.sumsqr(Tj[idxTibiaOr_r] - Tj[idxTibiaOr_l])
            ineq_constr5.append(diffTibiaOrs)
            diffToesOrs = ca.sumsqr(Tj[idxToesOr_r] - Tj[idxToesOr_l])
            ineq_constr6.append(diffToesOrs)
            toesFrontalCoordinate_r = Tj[idxToesOr_r[1]]
            toesFrontalCoordinate_l = Tj[idxToesOr_l[1]]
            ineq_constr6.append(0.12 - toesFrontalCoordinate_r)
            ineq_constr6.append(toesFrontalCoordinate_l + 0.12)

        ## Setup parralel problem
        eq_constr = ca.vertcat(*eq_constr)
        ineq_constr1 = ca.vertcat(*ineq_constr1)
        ineq_constr2 = ca.vertcat(*ineq_constr2)
        ineq_constr3 = ca.vertcat(*ineq_constr3)
        ineq_constr4 = ca.vertcat(*ineq_constr4)
        ineq_constr5 = ca.vertcat(*ineq_constr5)
        ineq_constr6 = ca.vertcat(*ineq_constr6)

        f_coll_map = ca.Function('f_coll',
                                 [tf, scaling_vector_j, muscle_length_scaling_vector_j,
                                  model_mass_scaling_j,
                                  ak, aj, normFk, normFj, Qsk,
                                  Qsj, Qdsk, Qdsj, aArmk, aArmj,
                                  aDtk, eArmk, normFDtj, Qddsj, MTP_reserve_j],
                                 [eq_constr, ineq_constr1, ineq_constr2, ineq_constr3,
                                  ineq_constr4, ineq_constr5, ineq_constr6, J])

        f_coll_map = f_coll_map.map(number_of_mesh_intervals, 'thread', number_of_parallel_threads)

        (coll_eq_constr, coll_ineq_constr1, coll_ineq_constr2,
         coll_ineq_constr3, coll_ineq_constr4, coll_ineq_constr5,
         coll_ineq_constr6, Jall) = f_coll_map(
            finalTime, scaling_vector_opti, muscle_length_scaling_vector_opti, model_mass_scaling_opti,
            a[:, :-1], a_col, normF[:, :-1], normF_col,
            Qs[:, :-1], Qs_col, Qds[:, :-1], Qds_col,
            aArm[:, :-1], aArm_col, aDt, eArm, normFDt_col, Qdds_col, MTP_reserve_col)

        ## Set path constraints as equality or inequality constraints
        opti.subject_to(ca.vec(coll_eq_constr) == 0)
        opti.subject_to(ca.vec(coll_ineq_constr1) >= 0)
        opti.subject_to(
            ca.vec(coll_ineq_constr2) <= 1 / activationTimeConstant)
        opti.subject_to(opti.bounded(0.0081, ca.vec(coll_ineq_constr3), 4))
        opti.subject_to(opti.bounded(0.0324, ca.vec(coll_ineq_constr4), 4))
        opti.subject_to(opti.bounded(0.0121, ca.vec(coll_ineq_constr5), 4))
        opti.subject_to(opti.bounded(0.01, ca.vec(coll_ineq_constr6), 4))

        ## Continuity constraints across mesh intervals (outside of parallelization)
        for k in range(number_of_mesh_intervals):
            akj2 = (ca.horzcat(a[:, k], a_col[:, k * polynomial_order:(k + 1) * polynomial_order]))
            normFkj2 = (ca.horzcat(normF[:, k], normF_col[:, k * polynomial_order:(k + 1) * polynomial_order]))
            Qskj2 = (ca.horzcat(Qs[:, k], Qs_col[:, k * polynomial_order:(k + 1) * polynomial_order]))
            Qdskj2 = (ca.horzcat(Qds[:, k], Qds_col[:, k * polynomial_order:(k + 1) * polynomial_order]))
            aArmkj2 = (ca.horzcat(aArm[:, k], aArm_col[:, k * polynomial_order:(k + 1) * polynomial_order]))

            opti.subject_to(a[:, k + 1] == ca.mtimes(akj2, D))
            opti.subject_to(normF[:, k + 1] == ca.mtimes(normFkj2, D))
            opti.subject_to(Qs[:, k + 1] == ca.mtimes(Qskj2, D))
            opti.subject_to(Qds[:, k + 1] == ca.mtimes(Qdskj2, D))
            opti.subject_to(aArm[:, k + 1] == ca.mtimes(aArmkj2, D))

        ## Periodic constraints on states.
        opti.subject_to(Qs[periodic_joints_indices_start_to_end_position_matching[0], -1] -
                        Qs[periodic_joints_indices_start_to_end_position_matching[1], 0] == 0)
        opti.subject_to(Qds[periodic_joints_indices_start_to_end_velocity_matching[0], -1] -
                        Qds[periodic_joints_indices_start_to_end_velocity_matching[1], 0] == 0)
        opti.subject_to(Qs[periodic_opposite_joints_indices_in_joints, -1] +
                        Qs[periodic_opposite_joints_indices_in_joints, 0] == 0)
        opti.subject_to(Qds[periodic_opposite_joints_indices_in_joints, -1] +
                        Qds[periodic_opposite_joints_indices_in_joints, 0] == 0)
        opti.subject_to(a[periodic_muscles_indices_start_to_end_matching[0], -1] - a[periodic_muscles_indices_start_to_end_matching[1], 0] == 0)
        opti.subject_to(normF[periodic_muscles_indices_start_to_end_matching[0], -1] - normF[periodic_muscles_indices_start_to_end_matching[1], 0] == 0)
        opti.subject_to(aArm[periodic_actuators_indices_start_to_end_matching[0], -1] - aArm[periodic_actuators_indices_start_to_end_matching[1], 0] == 0)

        ## Average speed constraint.
        Qs_nsc = Qs * (scaling_Q.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1)))
        distTraveled = (Qs_nsc[joints.index('pelvis_tx'), -1] -
                        Qs_nsc[joints.index('pelvis_tx'), 0])
        averageSpeed = distTraveled / finalTime

        ## Setup cost function
        if enforce_target_speed:
            opti.subject_to(averageSpeed == target_speed)
            Jall_sc = (ca.sum2(Jall) / distTraveled)
        else:
            opti.subject_to(averageSpeed > 6)
            Jall_sc = (ca.sum2(Jall) / distTraveled - 100 * averageSpeed ** 2)/1000
        opti.minimize(Jall_sc)

        ## Solve
        from utilities import solve_with_bounds, solve_with_constraints
        file_path = os.path.join(path_results, 'mydiary.txt')
        # sys.stdout = open(file_path, "w")
        w_opt, stats = solve_with_bounds(opti, convergence_tolerance_IPOPT)

        ## Save results.
        if saveResults:
            np.save(os.path.join(path_results, 'w_opt.npy'), w_opt)
            np.save(os.path.join(path_results, 'stats.npy'), stats)

    ## Analyze results.
    if analyzeResults:
        if loadResults:
            w_opt = np.load(os.path.join(path_results, 'w_opt.npy'))
            stats = np.load(os.path.join(path_results, 'stats.npy'),
                            allow_pickle=True).item()

        # Warning message if no convergence.
        if not stats['success'] == True:
            print("WARNING: PROBLEM DID NOT CONVERGE - {}".format(
                stats['return_status']))

        ## Extract optimized variables
        a_opt, a_col_opt, normF_opt, normF_col_opt, Qs_opt, Qs_col_opt, Qds_opt, Qds_col_opt, aArm_opt, \
        aArm_col_opt, aDt_opt, eArm_opt, normFDt_col_opt, Qdds_col_opt, MTP_reserve_col_opt, \
        scaling_vector_opt, muscle_scaling_vector_opt, model_mass_scaling_opt, model_mass_opt, model_height_opt, \
        model_BMI_opt, finalTime_opt = utilities.get_results_opti(f_height, w_opt, number_of_muscles, number_of_mesh_intervals, polynomial_order, number_of_joints, number_of_non_muscle_actuated_joints, modelMass)

        # Generate motion file for full gait cycle
        Qs_gait_cycle_opt, tgrid_GC, Qs_gait_cycle_nsc = utilities.generate_full_gait_cycle_kinematics(path_results, joints, number_of_joints, number_of_mesh_intervals, finalTime_opt, Qs_opt, scaling_Q, rotational_joint_indices_in_joints, periodic_joints_indices_start_to_end_position_matching, periodic_opposite_joints_indices_in_joints)

        # Generate scaled model
        default_scale_tool_xml_name = path_model_folder + '/scaleTool_Default.xml'
        NeMu_subfunctions.generateScaledModels(0, default_scale_tool_xml_name, np.reshape(scaling_vector_opt, (54)),
                                  path_results, model_name)

        # Initialize, compute variables we want to extract
        QsQds_col_opt_nsc = np.zeros((number_of_joints * 2, polynomial_order * number_of_mesh_intervals))
        Qs_col_opt_nsc = (Qs_col_opt * (scaling_Q.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        Qds_col_opt_nsc = (Qds_col_opt * (scalingQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        Qdds_col_opt_nsc = (Qdds_col_opt * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        QsQds_col_opt_nsc[::2, :] = Qs_col_opt_nsc[joint_indices_in_external_function, :]
        QsQds_col_opt_nsc[1::2, :] = Qds_col_opt_nsc[joint_indices_in_external_function, :]
        speed = Qs_col_opt_nsc[joints.index('pelvis_tx'), -1] / finalTime_opt
        halfGC_length = Qs_col_opt_nsc[joints.index('pelvis_tx'), -1]
        Qsin_col_opt = Qs_col_opt_nsc[muscle_actuated_joints_indices_in_joints, :]
        Qdsin_col_opt = Qds_col_opt_nsc[muscle_actuated_joints_indices_in_joints, :]
        normF_nsc_col_opt = normF_col_opt * (scaling_muscle_force.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals)))
        normFDt_nsc_col_opt = normFDt_col_opt * (scalingFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals)))

        time_col_opt = utilities.get_time_col_opt(finalTime_opt, number_of_mesh_intervals, polynomial_order)

        lMT_col_opt, vMT_col_opt, dM_col_opt, muscle_active_joint_torques_col_opt, muscle_passive_joint_torques_col_opt, muscle_joint_torques_col_opt, biological_joint_torques_equilibrium_residual_col_opt, active_muscle_force_col_opt, passive_muscle_force_col_opt, hillEquilibrium_residual_col_opt, metabolicEnergyRate_col_opt = utilities.get_biomechanics_outputs(f_metabolicsBhargava, f_get_muscle_tendon_length_velocity_moment_arm, f_hillEquilibrium, joints, number_of_joints, number_of_muscles, number_of_mesh_intervals, muscle_actuated_joints_indices_in_joints, number_of_muscle_actuated_joints, polynomial_order, scaling_vector_opt, Qsin_col_opt, Qdsin_col_opt, a_col_opt, normF_nsc_col_opt, normFDt_nsc_col_opt, muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, muscle_scaling_vector_opt, model_mass_scaling_opt)

        metabolic_energy_outcomes = utilities.get_metabolic_energy_outcomes(metabolicEnergyRate_col_opt, time_col_opt,
                                                                            halfGC_length, model_mass_opt)

        Tj_col_opt, GRF_col_opt, GRM_col_opt, passive_joint_torques_col_opt, joint_torques_equilibrium_residual_col_opt, muscle_joint_torques_col_opt, limit_joint_torques_col_opt, passive_joint_torques_col_opt, generalized_forces_col_opt = utilities.get_skeletal_dynamics_outputs(map_external_function_outputs, f_linearPassiveTorque, f_linearPassiveMtpTorque, f_limit_torque, skeleton_dynamics_and_more, non_muscle_actuated_joints, joint_indices_in_external_function, number_of_joints, joints, mtpJoints, limit_torque_joints, polynomial_order, number_of_mesh_intervals, idxGRF, idxGRM, Qs_col_opt_nsc, Qds_col_opt_nsc, QsQds_col_opt_nsc, Qdds_col_opt_nsc, aArm_col_opt, MTP_reserve_col_opt, scaling_vector_opt, model_mass_opt, scalingArmA)

        Qs_max_iso_torque, maximal_isometric_torques, passive_isometric_torques, max_iso_torques_joints = utilities.get_max_iso_torques(f_get_muscle_tendon_length_velocity_moment_arm, f_hillEquilibrium, joints, number_of_muscles, muscle_actuated_joints, muscle_actuated_joints_indices_in_joints, muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, muscle_scaling_vector_opt, scaling_vector_opt, model_mass_scaling_opt)

        # Store and save
        if not os.path.exists(os.path.join(path_trajectories,
                                              'optimaltrajectories.npy')):
            optimaltrajectories = {}
        else:
            optimaltrajectories = np.load(
                os.path.join(path_trajectories,
                            'optimaltrajectories.npy'),
                allow_pickle=True)
            optimaltrajectories = optimaltrajectories.item()

        optimaltrajectories[case] = {
                'speed': speed,
                'height': model_height_opt,
                'mass': model_mass_opt,
                'BMI': model_BMI_opt,
                'time_col_opt': time_col_opt,
                'map_external_function_outputs': map_external_function_outputs,
                'joint_names': joints,
                'muscle_names': muscles,
                'skeleton_scaling_bodies': skeleton_scaling_bodies,
                'bounds_Q': {'upper': ubQs * scaling_Q.to_numpy(), 'lower': lbQs * scaling_Q.to_numpy(), 'scaling': scaling_Q},
                'bounds_Qd': {'upper': ubQds * scalingQds.to_numpy(), 'lower': lbQds * scalingQds.to_numpy(), 'scaling': scalingQds},
                'bounds_Qdd': {'upper': ubQdds * scalingQdds.to_numpy(), 'lower': lbQdds * scalingQdds.to_numpy(), 'scaling': scalingQdds},
                'final_time': finalTime_opt,
                'Q_at_mesh': Qs_opt * (scaling_Q.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))),
                'Q_at_collocation': Qs_col_opt * (scaling_Q.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))),
                'Qd_at_mesh': Qds_opt * (scalingQds.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))),
                'Qd_at_collocation': Qds_col_opt * (scalingQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))),
                'muscle_activation_at_mesh': a_opt ,
                'muscle_activation_at_collocation': a_col_opt ,
                'muscle_force_at_mesh': normF_opt * (scaling_muscle_force.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1))),
                'muscle_force_at_collocation': normF_col_opt * (scaling_muscle_force.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))),
                'arm_activation_at_mesh': aArm_opt * scalingArmA.iloc[0]['arm_rot_r'],
                'arm_activation_at_collocation': aArm_col_opt * scalingArmA.iloc[0]['arm_rot_r'],
                'muscle_activation_derivative_at_mesh': aDt_opt * (scalingADt.to_numpy().T * np.ones((1, number_of_mesh_intervals))),
                'arm_excitation_at_mesh': eArm_opt,
                'force_derivative_at_collocation': normFDt_col_opt * (scalingFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))),
                'Qdd_at_collocation': Qdds_col_opt * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))),
                'MTP_reserve_at_collocation': MTP_reserve_col_opt,
                'skeleton_scaling': scaling_vector_opt,
                'muscle_scaling': muscle_scaling_vector_opt,
                'model_mass_scaling': model_mass_scaling_opt,
                'external_function_ouput': Tj_col_opt,
                'generalized_forces': generalized_forces_col_opt,
                'joint_torques_equilibrium_residual_col_opt': joint_torques_equilibrium_residual_col_opt,
                'biological_joint_torques_at_collocation': muscle_joint_torques_col_opt,
                'biological_passive_joint_torques_at_collocation': muscle_passive_joint_torques_col_opt,
                'biological_active_joint_torques_at_collocation': muscle_active_joint_torques_col_opt,
                'limit_joint_torques_at_collocation': limit_joint_torques_col_opt,
                'passive_joint_torques_at_collocation': passive_joint_torques_col_opt,
                'max_iso_torques_joints': max_iso_torques_joints,
                'maximal_isometric_torques': maximal_isometric_torques,
                'passive_isometric_torques': passive_isometric_torques,
                'Qs_max_iso_torque': Qs_max_iso_torque,
                'metabolic_energy_outcomes': metabolic_energy_outcomes
                }

        np.save(os.path.join(path_trajectories, 'optimaltrajectories.npy'),
                optimaltrajectories)

