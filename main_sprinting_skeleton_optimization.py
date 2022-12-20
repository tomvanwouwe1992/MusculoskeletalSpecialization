import os
import casadi as ca
import numpy as np
import copy
import sys
import utilities
from settings_sprinting import getSettings
import muscleData
import casadiFunctions
from muscleData import getMTParameters
from muscleData import tendonStiffness
from muscleData import specificTension
from muscleData import slowTwitchRatio
import NeMu_subfunctions

# High-level settings.
# This script includes both code for solving the problem and for processing the
# results. Yet if you solved the optimal control problem and saved the results,
# you might want to latter only load and process the results without re-solving
# the problem. Playing with the settings below allows you to do exactly that.
solveProblem = False # Set True to solve the optimal control problem.
saveResults = True  # Set True to save the results of the optimization.
analyzeResults = True  # Set True to analyze the results.
loadResults = True  # Set True to load the results of the optimization.
writeMotionFiles = True  # Set True to write motion files for use in OpenSim GUI
saveOptimalTrajectories = True  # Set True to save optimal trajectories
plotResults = False
set_guess_from_solution = False
# Select the case(s) for which you want to solve the associated problem(s) or
# process the results. Specify the settings of the case(s) in the
# 'settings' module.
cases = [str(i) for i in range(1, 7)]

settings = getSettings()

for case in cases:

    ### Model and path
    model_name = 'Hamner_modified'  # default model
    mtParameters_model_name = 'Hamner_modified'
    path_main = os.getcwd()
    path_opensim_model = os.path.join(path_main, 'OpenSimModel')
    path_data = os.path.join(path_opensim_model, 'Hamner_modified')
    path_model_folder = os.path.join(path_data, 'Model')
    path_model = os.path.join(path_model_folder, model_name + '.osim')

    pathMotionFile4Polynomials = os.path.join(path_model_folder, 'templates',
                                              'MA', 'dummy_motion.mot')
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
    cost_function_weights = {'metabolicEnergyRateTerm':energy_cost_function_scaling*500,  # default.
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



    ### Name lists of components of the musculoskeletal model
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

    muscles = utilities.get_muscle_names(path_model)
    number_of_muscles = len(muscles)

    joints = utilities.get_joint_names(path_model)
    number_of_joints = len(joints)

    muscle_actuated_joints = utilities.get_muscle_actuated_joint_names(path_model)

    muscle_articulated_bodies = utilities.get_muscle_articulated_body_names(path_model)

    muscle_actuated_joints_indices_in_joints = [joints.index(i) for i in muscle_actuated_joints]
    number_of_muscle_actuated_joints = len(muscle_actuated_joints)
    rotational_joints = copy.deepcopy(joints)
    rotational_joints.remove('pelvis_tx')
    rotational_joints.remove('pelvis_ty')
    rotational_joints.remove('pelvis_tz')

    rotational_joint_indices_in_joints = [joints.index(i) for i in rotational_joints]

    non_actuated_joints = [x for x in joints if 'pelvis' in x]
    non_actuated_joints_indices_in_joints = [joints.index(i) for i in non_actuated_joints]


    non_muscle_actuated_joints = (list(set(joints) - set(muscle_actuated_joints) - set(non_actuated_joints)))
    non_muscle_actuated_joints.sort()
    non_muscle_actuated_joints_indices_in_joints = [joints.index(i) for i in non_muscle_actuated_joints]
    number_of_non_muscle_actuated_joints = len(non_muscle_actuated_joints)

    number_of_muscle_actuated_and_non_actuated_joints = number_of_joints - number_of_non_muscle_actuated_joints
    muscle_actuated_and_non_actuated_joints_indices_in_joints = muscle_actuated_joints_indices_in_joints + non_actuated_joints_indices_in_joints

    mtpJoints = ['mtp_angle_l', 'mtp_angle_r']

    limit_torque_joints = muscleData.get_limit_torque_joints()
    number_of_limit_torque_joints = len(limit_torque_joints)

    periodic_opposite_joints = ['pelvis_list', 'pelvis_rotation', 'pelvis_tz',
                              'lumbar_bending', 'lumbar_rotation']
    periodic_opposite_joints_indices_in_joints = [joints.index(i) for i in periodic_opposite_joints]

    periodic_joints = copy.deepcopy(joints)
    for periodic_opposite_joint in periodic_opposite_joints:
        periodic_joints.remove(periodic_opposite_joint)

    periodic_joints_indices_start_to_end_velocity_matching = utilities.periodic_indices_start_to_end_matching(
            periodic_joints, joints)

    periodic_joints.remove('pelvis_tx')
    periodic_joints_indices_start_to_end_position_matching = utilities.periodic_indices_start_to_end_matching(periodic_joints, joints)

    periodic_muscles = copy.deepcopy(muscles)
    periodic_muscles_indices_start_to_end_matching = utilities.periodic_indices_start_to_end_matching(periodic_muscles, muscles)

    periodic_actuators = copy.deepcopy(non_muscle_actuated_joints)
    periodic_actuators_indices_start_to_end_matching = utilities.periodic_indices_start_to_end_matching(periodic_actuators, non_muscle_actuated_joints)


    ### Musculoskeletal geometry
    NeMu_folder = os.path.dirname(
        os.path.abspath('main.py')) + "/OpenSimModel/Hamner_modified/Model/NeMu/tanh"
    f_get_muscle_tendon_length_velocity_moment_arm = NeMu_subfunctions.NeMuApproximation(muscles, muscle_actuated_joints, muscle_articulated_bodies, NeMu_folder)
    f_muscle_length_scaling = casadiFunctions.muscle_length_scaling_vector(muscle_actuated_joints, f_get_muscle_tendon_length_velocity_moment_arm)

    ### Muscle parameters
    muscle_parameters = getMTParameters(path_model, muscles)
    tendon_stiffness = tendonStiffness(len(muscles))
    specific_tension = specificTension(muscles)

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



    # skeleton_dynamics_and_more_old = ca.external('F', os.path.join(
    #                 path_external_function, 'Hamner_modified' + '_Sprinter_OurScaling_withHeight.dll'))
    #
    #
    # print("TESTING SKELETON DYNAMICS")
    # compareFunctions.compare_skeleton_dynamics_and_more(skeleton_dynamics_and_more, skeleton_dynamics_and_more_old)

    # This holds a structure with what the fields and indices are of the external function
    map_external_function_outputs = np.load(os.path.join(
        path_external_function, 'Hamner_modified' + '_map.npy'),
        allow_pickle=True).item()

    # The external function F outputs joint torques, ground reaction forces,
    # ground reaction moments, and 3D positions of body origins. The order is
    # given in the F_map dict.
    # Origins calcaneus (2D).
    idxCalcOr_r = [map_external_function_outputs['body_origins']['calcn_r'][0],
                   map_external_function_outputs['body_origins']['calcn_r'][2]]
    idxCalcOr_l = [map_external_function_outputs['body_origins']['calcn_l'][0],
                   map_external_function_outputs['body_origins']['calcn_l'][2]]
    # Origins femurs (2D).
    idxFemurOr_r = [map_external_function_outputs['body_origins']['femur_r'][0],
                    map_external_function_outputs['body_origins']['femur_r'][2]]
    idxFemurOr_l = [map_external_function_outputs['body_origins']['femur_l'][0],
                    map_external_function_outputs['body_origins']['femur_l'][2]]
    # Origins hands (2D).
    idxHandOr_r = [map_external_function_outputs['body_origins']['hand_r'][0],
                   map_external_function_outputs['body_origins']['hand_r'][2]]
    idxHandOr_l = [map_external_function_outputs['body_origins']['hand_l'][0],
                   map_external_function_outputs['body_origins']['hand_l'][2]]
    # Origins tibias (2D).
    idxTibiaOr_r = [map_external_function_outputs['body_origins']['tibia_r'][0],
                    map_external_function_outputs['body_origins']['tibia_r'][2]]
    idxTibiaOr_l = [map_external_function_outputs['body_origins']['tibia_l'][0],
                    map_external_function_outputs['body_origins']['tibia_l'][2]]
    # Origins toes (2D).
    idxToesOr_r = [map_external_function_outputs['body_origins']['toes_r'][0],
                   map_external_function_outputs['body_origins']['toes_r'][2]]
    idxToesOr_l = [map_external_function_outputs['body_origins']['toes_l'][0],
                   map_external_function_outputs['body_origins']['toes_l'][2]]
    # Origins toes (2D).
    idxTorsoOr_r = [map_external_function_outputs['body_origins']['torso'][0],
                    map_external_function_outputs['body_origins']['torso'][2]]
    idxTorsoOr_l = [map_external_function_outputs['body_origins']['torso'][0],
                    map_external_function_outputs['body_origins']['torso'][2]]
    # Ground reaction forces (GRFs).
    idxGRF_r = list(map_external_function_outputs['GRFs']['right'])
    idxGRF_l = list(map_external_function_outputs['GRFs']['left'])
    idxGRF = idxGRF_r + idxGRF_l
    NGRF = len(idxGRF)
    # Origins calcaneus (3D).
    idxCalcOr3D_r = list(map_external_function_outputs['body_origins']['calcn_r'])
    idxCalcOr3D_l = list(map_external_function_outputs['body_origins']['calcn_l'])
    idxCalcOr3D = idxCalcOr3D_r + idxCalcOr3D_l
    NCalcOr3D = len(idxCalcOr3D)
    # Ground reaction moments (GRMs).
    idxGRM_r = list(map_external_function_outputs['GRMs']['right'])
    idxGRM_l = list(map_external_function_outputs['GRMs']['left'])
    idxGRM = idxGRM_r + idxGRM_l
    NGRM = len(idxGRM)

    # Helper lists to map order of joints defined here and in F.
    joint_indices_in_external_function = [joints.index(joint)
                                          for joint in list(map_external_function_outputs['residuals'].keys())]

    ### Metabolic energy model.
    maximalIsometricForce = muscle_parameters[0, :]
    optimalFiberLength = muscle_parameters[1, :]
    slow_twitch_ratio = slowTwitchRatio(muscles)
    smoothingConstant = 10

    f_metabolicsBhargava = casadiFunctions.metabolicsBhargava_muscle_length_scaling(
        slow_twitch_ratio, maximalIsometricForce, optimalFiberLength, specific_tension, smoothingConstant)

    muscleVolume = np.multiply(maximalIsometricForce, optimalFiberLength)
    muscleMass = np.divide(np.multiply(muscleVolume, 1059.7),
                           np.multiply(specific_tension[0, :].T, 1e6))

    f_height = casadiFunctions.get_height(joints, skeleton_dynamics_and_more)
    f_body_mass_scaling = casadiFunctions.mass_scaling_with_skeleton_volume(joints, skeleton_dynamics_and_more)

    ### Passive joint torques.

    f_limit_torque = {}
    damping = 0.1
    for joint in limit_torque_joints:
        f_limit_torque[joint] = casadiFunctions.getLimitTorques(
            muscleData.passiveTorqueData(joint)[0],
            muscleData.passiveTorqueData(joint)[1], damping)

    ## Passive torques in non muscle actuated joints
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
    if enforce_target_speed:
        motion_walk = 'sprinting'
    else:
        motion_walk = 'sprinting'
    nametrial_walk_id = 'average_' + motion_walk + '_HGC_mtp'
    nametrial_walk_IK = 'IK_' + nametrial_walk_id
    pathIK_walk = os.path.join(path_opensim_model, 'templates', 'IK',
                               nametrial_walk_IK + '.mot')

    Qs_walk_filt = utilities.getIK(pathIK_walk, joints)[1]

    ### Bounds
    import boundsSprinting
    bounds = boundsSprinting.bounds(Qs_walk_filt, joints, muscles, non_muscle_actuated_joints)

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

    # %% Initial guess of the optimal control problem.
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



    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies = [
        [muscle_articulated_body in skeleton_scaling_body for skeleton_scaling_body in skeleton_scaling_bodies].index(
            True) for muscle_articulated_body in muscle_articulated_bodies]
    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies = utilities.convert1Dto3Dindices(
        muscle_articulated_bodies_indices_in_skeleton_scaling_bodies)

    # %% Optimal control problem.
    if solveProblem:
        #######################################################################
        # Initialize opti instance.
        # Opti is a collection of CasADi helper classes:
        # https://web.casadi.org/docs/#opti-stack
        opti = ca.Opti()

        #######################################################################
        # Static parameters.
        # Final time.
        finalTime = opti.variable()
        opti.subject_to(opti.bounded(lbFinalTime.iloc[0]['time'],
                                     finalTime,
                                     ubFinalTime.iloc[0]['time']))
        opti.set_initial(finalTime, gFinalTime)
        assert lbFinalTime.iloc[0]['time'] <= gFinalTime, (
            "Error lower bound final time")
        assert ubFinalTime.iloc[0]['time'] >= gFinalTime, (
            "Error upper bound final time")

        #######################################################################
        # States.
        # Muscle activation at mesh points.
        a = opti.variable(number_of_muscles, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(activation_lower_bound_at_mesh, ca.vec(a), activation_upper_bound_at_mesh))
        opti.set_initial(a, gA.to_numpy().T)
        assert np.alltrue(activation_lower_bound_at_mesh <= ca.vec(gA.to_numpy().T).full()), (
            "Error lower bound muscle activation")
        assert np.alltrue(activation_upper_bound_at_mesh >= ca.vec(gA.to_numpy().T).full()), (
            "Error upper bound muscle activation")
        # Muscle activation at collocation points.
        a_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(activation_lower_bound_at_collocation, ca.vec(a_col), activation_upper_bound_at_collocation))
        opti.set_initial(a_col, gACol.to_numpy().T)
        assert np.alltrue(activation_lower_bound_at_collocation <= ca.vec(gACol.to_numpy().T).full()), (
            "Error lower bound muscle activation collocation points")
        assert np.alltrue(activation_upper_bound_at_collocation >= ca.vec(gACol.to_numpy().T).full()), (
            "Error upper bound muscle activation collocation points")
        # Tendon force at mesh points.
        normF = opti.variable(number_of_muscles, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(muscle_force_lower_bound_at_mesh, ca.vec(normF), muscle_force_upper_bound_at_mesh))
        opti.set_initial(normF, gF.to_numpy().T)
        assert np.alltrue(muscle_force_lower_bound_at_mesh <= ca.vec(gF.to_numpy().T).full()), (
            "Error lower bound muscle force")
        assert np.alltrue(muscle_force_upper_bound_at_mesh >= ca.vec(gF.to_numpy().T).full()), (
            "Error upper bound muscle force")
        # Tendon force at collocation points.
        normF_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(muscle_force_lower_bound_at_collocation, ca.vec(normF_col), muscle_force_upper_bound_at_collocation))
        opti.set_initial(normF_col, gFCol.to_numpy().T)
        assert np.alltrue(muscle_force_lower_bound_at_collocation <= ca.vec(gFCol.to_numpy().T).full()), (
            "Error lower bound muscle force collocation points")
        assert np.alltrue(muscle_force_upper_bound_at_collocation >= ca.vec(gFCol.to_numpy().T).full()), (
            "Error upper bound muscle force collocation points")
        # Joint position at mesh points.
        Qs = opti.variable(number_of_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(Q_lower_bound_at_mesh, ca.vec(Qs), Q_upper_bound_at_mesh))
        opti.set_initial(Qs, gQs.to_numpy().T)
        # assert np.alltrue(Q_lower_bound_at_mesh <= ca.vec(gQs.to_numpy().T).full()), (
        #     "Error lower bound joint position")
        # assert np.alltrue(Q_upper_bound_at_mesh >= ca.vec(gQs.to_numpy().T).full()), (
        #     "Error upper bound joint position")
        # Joint position at collocation points.
        Qs_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(Q_lower_bound_at_collocation, ca.vec(Qs_col), Q_upper_bound_at_collocation))
        opti.set_initial(Qs_col, gQsCol.to_numpy().T)
        # if not guessType == 'coldStart':
        # assert np.alltrue(lbQsj <= ca.vec(gQsCol.to_numpy().T).full()), (
        #     "Error lower bound joint position collocation points")
        # assert np.alltrue(ubQsj >= ca.vec(gQsCol.to_numpy().T).full()), (
        #     "Error upper bound joint position collocation points")
        # Joint velocity at mesh points.
        Qds = opti.variable(number_of_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(lbQdsk, ca.vec(Qds), ubQdsk))
        opti.set_initial(Qds, gQds.to_numpy().T)
        assert np.alltrue(lbQdsk <= ca.vec(gQds.to_numpy().T).full()), (
            "Error lower bound joint velocity")
        assert np.alltrue(ubQdsk >= ca.vec(gQds.to_numpy().T).full()), (
            "Error upper bound joint velocity")
        # Joint velocity at collocation points.
        Qds_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbQdsj, ca.vec(Qds_col), ubQdsj))
        opti.set_initial(Qds_col, gQdsCol.to_numpy().T)
        assert np.alltrue(lbQdsj <= ca.vec(gQdsCol.to_numpy().T).full()), (
            "Error lower bound joint velocity collocation points")
        assert np.alltrue(ubQdsj >= ca.vec(gQdsCol.to_numpy().T).full()), (
            "Error upper bound joint velocity collocation points")
        # Arm activation at mesh points.
        aArm = opti.variable(number_of_non_muscle_actuated_joints, number_of_mesh_intervals + 1)
        opti.subject_to(opti.bounded(lbArmAk, ca.vec(aArm), ubArmAk))
        opti.set_initial(aArm, gArmA.to_numpy().T)
        assert np.alltrue(lbArmAk <= ca.vec(gArmA.to_numpy().T).full()), (
            "Error lower bound arm activation")
        assert np.alltrue(ubArmAk >= ca.vec(gArmA.to_numpy().T).full()), (
            "Error upper bound arm activation")
        # Arm activation at collocation points.
        aArm_col = opti.variable(number_of_non_muscle_actuated_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbArmAj, ca.vec(aArm_col), ubArmAj))
        opti.set_initial(aArm_col, gArmACol.to_numpy().T)
        assert np.alltrue(lbArmAj <= ca.vec(gArmACol.to_numpy().T).full()), (
            "Error lower bound arm activation collocation points")
        assert np.alltrue(ubArmAj >= ca.vec(gArmACol.to_numpy().T).full()), (
            "Error upper bound arm activation collocation points")

        #######################################################################
        # Controls.
        # Muscle activation derivative at mesh points.
        aDt = opti.variable(number_of_muscles, number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbADtk, ca.vec(aDt), ubADtk))
        opti.set_initial(aDt, gADt.to_numpy().T)
        assert np.alltrue(lbADtk <= ca.vec(gADt.to_numpy().T).full()), (
            "Error lower bound muscle activation derivative")
        assert np.alltrue(ubADtk >= ca.vec(gADt.to_numpy().T).full()), (
            "Error upper bound muscle activation derivative")
        # Arm excitation at mesh points.
        eArm = opti.variable(number_of_non_muscle_actuated_joints, number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbArmEk, ca.vec(eArm), ubArmEk))
        opti.set_initial(eArm, gArmE.to_numpy().T)
        assert np.alltrue(lbArmEk <= ca.vec(gArmE.to_numpy().T).full()), (
            "Error lower bound arm excitation")
        assert np.alltrue(ubArmEk >= ca.vec(gArmE.to_numpy().T).full()), (
            "Error upper bound arm excitation")

        #######################################################################
        # Slack controls.
        # Tendon force derivative at collocation points.
        normFDt_col = opti.variable(number_of_muscles, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbFDtj, ca.vec(normFDt_col), ubFDtj))
        opti.set_initial(normFDt_col, gFDtCol.to_numpy().T)
        assert np.alltrue(lbFDtj <= ca.vec(gFDtCol.to_numpy().T).full()), (
            "Error lower bound muscle force derivative")
        assert np.alltrue(ubFDtj >= ca.vec(gFDtCol.to_numpy().T).full()), (
            "Error upper bound muscle force derivative")
        # Joint velocity derivative (acceleration) at collocation points.
        Qdds_col = opti.variable(number_of_joints, polynomial_order * number_of_mesh_intervals)
        opti.subject_to(opti.bounded(lbQddsj, ca.vec(Qdds_col),
                                     ubQddsj))
        opti.set_initial(Qdds_col, gQddsCol.to_numpy().T)
        # assert np.alltrue(lbQddsj <= ca.vec(gQddsCol.to_numpy().T).full()), (
        #     "Error lower bound joint velocity derivative")
        # assert np.alltrue(ubQddsj >= ca.vec(gQddsCol.to_numpy().T).full()), (
        #     "Error upper bound joint velocity derivative")

        #######################################################################


        bodies_skeleton_scaling_coupling_indices = copy.deepcopy(bodies_skeleton_scaling_coupling)
        for i in range(len(bodies_skeleton_scaling_coupling_indices)):
            if type(bodies_skeleton_scaling_coupling_indices[i]) is list:
                for j in range(len(bodies_skeleton_scaling_coupling_indices[i])):
                    bodies_skeleton_scaling_coupling_indices[i][j] = skeleton_scaling_bodies.index(bodies_skeleton_scaling_coupling[i][j])
            else:
                bodies_skeleton_scaling_coupling_indices[i] = skeleton_scaling_bodies.index(bodies_skeleton_scaling_coupling[i])


        if  np.all(bounds_skeleton_scaling_factors[0] == bounds_skeleton_scaling_factors[1]):

            scaling_vector = bounds_skeleton_scaling_factors[0]
            scaling_vector_opti = opti.variable(54, polynomial_order * number_of_mesh_intervals)
            opti.subject_to(scaling_vector_opti == np.tile(scaling_vector,(polynomial_order * number_of_mesh_intervals,1)).T)

            muscle_length_scaling_vector_opti = opti.variable(92, polynomial_order * number_of_mesh_intervals)
            muscle_length_scaling =  f_muscle_length_scaling(
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

            # # Constrain scaling factors to be equal over time
            for j in range(1, polynomial_order * number_of_mesh_intervals):
                opti.subject_to(scaling_vector_opti[:, 0] == scaling_vector_opti[:, j])
            #
            # # Impose coupling of scaling as described
            for i in range(len(bodies_skeleton_scaling_coupling_indices)):
                if type(bodies_skeleton_scaling_coupling_indices[i]) is list:
                    indices_first_element = [3*bodies_skeleton_scaling_coupling_indices[i][0], 3*bodies_skeleton_scaling_coupling_indices[i][0] + 1, 3*bodies_skeleton_scaling_coupling_indices[i][0] + 2]
                    for j in range(1, len(bodies_skeleton_scaling_coupling_indices[i])):
                        indices_element = [3*bodies_skeleton_scaling_coupling_indices[i][j], 3*bodies_skeleton_scaling_coupling_indices[i][j] + 1, 3*bodies_skeleton_scaling_coupling_indices[i][j] + 2]
                        opti.subject_to(scaling_vector_opti[indices_first_element, 0] == scaling_vector_opti[indices_element, 0])

            # # Impose muscle length scaling is correct with skeleton scaling
            muscle_length_scaling_vector_opti = opti.variable(92, polynomial_order * number_of_mesh_intervals)
            muscle_length_scaling_guess = np.ones((number_of_muscles,1))
            opti.set_initial(muscle_length_scaling_vector_opti,muscle_length_scaling_guess  * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            for j in range(0, polynomial_order * number_of_mesh_intervals):
                opti.subject_to(
                    muscle_length_scaling_vector_opti[:, j] == f_muscle_length_scaling(scaling_vector_opti[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, j]))
            #
            # # Impose muscle mass is correct with skeleton scaling
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

        MTP_reserve_col = opti.variable(2, polynomial_order * number_of_mesh_intervals)
        if enforce_target_speed and target_speed < 5:
            opti.subject_to(opti.bounded(0, ca.vertcat(MTP_reserve_col), 0))
        else:
            opti.subject_to(opti.bounded(-30, ca.vertcat(MTP_reserve_col), 5))

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
            opti.set_initial(scaling_vector_opti, np.reshape(optimal_trajectory['skeleton_scaling'],(54,1))  * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(muscle_length_scaling_vector_opti, np.reshape(optimal_trajectory['muscle_scaling'],(92,1)) * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(model_mass_scaling_opti, optimal_trajectory['model_mass_scaling']  * np.ones((1, polynomial_order * number_of_mesh_intervals)))
            opti.set_initial(finalTime, optimal_trajectory['final_time'])


        #######################################################################
        # Parallel formulation - initialize variables.
        # Static parameters.

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
        scaling_vector_j = ca.MX.sym('scaling_vector', (len(muscle_articulated_bodies) + 6) * 3, polynomial_order)
        muscle_length_scaling_vector_j = ca.MX.sym('muscle_scaling_vector', number_of_muscles, polynomial_order)
        model_mass_scaling_j = ca.MX.sym('model_mass_scaling_vector', 1, polynomial_order)
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
            # Cost function.
            modelMass_opti = modelMass * model_mass_scaling_j[:,j]

            ###################################################################
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

            ###################################################################
            # Muscle lengths, velocities and moment arms
            Qsinj = Qskj_nsc[muscle_actuated_joints_indices_in_joints, j + 1]
            Qdsinj = Qdskj_nsc[muscle_actuated_joints_indices_in_joints, j + 1]
            NeMu_input_j = ca.vertcat(scaling_vector_j[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies,j], Qsinj, Qdsinj)
            [lMTj_lr, vMTj_lr, dMj] = f_get_muscle_tendon_length_velocity_moment_arm(NeMu_input_j)

            ###################################################################
            [hillEquilibriumj, Fj, activeFiberForcej, passiveFiberForcej,
             normActiveFiberLengthForcej, normFiberLengthj, fiberVelocityj,_,_] = (
                f_hillEquilibrium(akj[:, j + 1], lMTj_lr, vMTj_lr,
                                                   normFkj_nsc[:, j + 1], normFDtj_nsc[:, j], muscle_length_scaling_vector_j[:,j],
                                                   model_mass_scaling_j[:,j]))


            ###################################################################
            # Metabolic energy rate.
            metabolicEnergyRatej = f_metabolicsBhargava(
                akj[:, j + 1], akj[:, j + 1], normFiberLengthj, fiberVelocityj,
                activeFiberForcej, passiveFiberForcej,
                normActiveFiberLengthForcej, muscle_length_scaling_vector_j[:,j], model_mass_scaling_j[:,j])[5]


            ###################################################################
            # Passive joint torques.
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

            ###################################################################
            # Cost function.
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

            ###################################################################
            # Expression for the state derivatives at the collocation points.
            ap = ca.mtimes(akj, C[j + 1])
            normFp_nsc = ca.mtimes(normFkj_nsc, C[j + 1])
            Qsp_nsc = ca.mtimes(Qskj_nsc, C[j + 1])
            Qdsp_nsc = ca.mtimes(Qdskj_nsc, C[j + 1])
            aArmp = ca.mtimes(aArmkj, C[j + 1])
            # Append collocation equations.
            # Muscle activation dynamics (implicit formulation).
            eq_constr.append((h * aDtk_nsc - ap))
            # Muscle contraction dynamics (implicit formulation)  .
            eq_constr.append((h * normFDtj_nsc[:, j] - normFp_nsc) /
                             scaling_muscle_force.to_numpy().T)
            # Skeleton dynamics (implicit formulation).
            # Position derivatives.
            eq_constr.append((h * Qdskj_nsc[:, j + 1] - Qsp_nsc) /
                             scaling_Q.to_numpy().T)
            # Velocity derivatives.
            eq_constr.append((h * Qddsj_nsc[:, j] - Qdsp_nsc) /
                             scalingQds.to_numpy().T)
            # Arm activation dynamics (explicit formulation).
            aArmDtj = f_actuatorDynamics(eArmk, aArmkj[:, j + 1])
            eq_constr.append(h * aArmDtj - aArmp)

            ###################################################################
            # Path constraints.
            # Call external function (run inverse dynamics).
            Tj = skeleton_dynamics_and_more(ca.vertcat(QsQdskj_nsc[:, j + 1], Qddsj_nsc[joint_indices_in_external_function, j], scaling_vector_j[:, j], 0))


            ###################################################################
            # Null pelvis residuals.
            eq_constr.append(Tj[[map_external_function_outputs['residuals'][joint]
                                 for joint in non_actuated_joints]])

            ###################################################################
            # Implicit skeleton dynamics.
            # Muscle-driven joint torques

            for i in range(len(muscle_actuated_joints_indices_in_joints)):
                joint = joints[muscle_actuated_joints_indices_in_joints[i]]
                if joint != 'mtp_angle_l' and joint != 'mtp_angle_r':
                    mTj_joint = ca.sum1(dMj[:, i] * Fj)
                    diffTj_joint = f_diffTorques(
                        Tj[map_external_function_outputs['residuals'][joint]], mTj_joint, passiveTorque_j[joint])
                    eq_constr.append(diffTj_joint)


            # Torque-driven joint torques
            for cj, joint in enumerate(non_muscle_actuated_joints):
                diffTj_joint = f_diffTorques(
                    Tj[map_external_function_outputs['residuals'][joint]] / scalingArmE.iloc[0][joint],
                    aArmkj[cj, j + 1], linearPassiveTorqueArms_j[joint] /
                    scalingArmE.iloc[0][joint])
                eq_constr.append(diffTj_joint)

            # Passive joint torques (mtp joints).
            for count, joint in enumerate(mtpJoints):
                diffTj_joint = f_diffTorques(
                    Tj[map_external_function_outputs['residuals'][joint]] /
                    scalingMtpE.iloc[0][joint],
                    MTP_reserve_j[count, j] /
                    scalingMtpE.iloc[0][joint], (passiveTorque_j[joint] +
                        linearPassiveTorqueMtp_j[joint]) /
                    scalingMtpE.iloc[0][joint])
                eq_constr.append(diffTj_joint)


                    ###################################################################
            # Implicit activation dynamics.
            act1 = aDtk_nsc + akj[:, j + 1] / deactivationTimeConstant
            act2 = aDtk_nsc + akj[:, j + 1] / activationTimeConstant
            ineq_constr1.append(act1)
            ineq_constr2.append(act2)

            ###################################################################
            # Implicit contraction dynamics.
            eq_constr.append(hillEquilibriumj)

            ###################################################################
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


        # End loop over collocation points.

        #######################################################################
        # Flatten constraint vectors.
        eq_constr = ca.vertcat(*eq_constr)
        ineq_constr1 = ca.vertcat(*ineq_constr1)
        ineq_constr2 = ca.vertcat(*ineq_constr2)
        ineq_constr3 = ca.vertcat(*ineq_constr3)
        ineq_constr4 = ca.vertcat(*ineq_constr4)
        ineq_constr5 = ca.vertcat(*ineq_constr5)
        ineq_constr6 = ca.vertcat(*ineq_constr6)
        # Create function for map construct (parallel computing).
        f_coll_map = ca.Function('f_coll',
                                 [tf, scaling_vector_j, muscle_length_scaling_vector_j,
                                  model_mass_scaling_j,
                                  ak, aj, normFk, normFj, Qsk,
                                  Qsj, Qdsk, Qdsj, aArmk, aArmj,
                                  aDtk, eArmk, normFDtj, Qddsj, MTP_reserve_j],
                                 [eq_constr, ineq_constr1, ineq_constr2, ineq_constr3,
                                  ineq_constr4, ineq_constr5, ineq_constr6, J])
        # Create map construct (N mesh intervals).
        f_coll_map = f_coll_map.map(number_of_mesh_intervals, 'thread', number_of_parallel_threads)
        # Call function with opti variables.
        (coll_eq_constr, coll_ineq_constr1, coll_ineq_constr2,
         coll_ineq_constr3, coll_ineq_constr4, coll_ineq_constr5,
         coll_ineq_constr6, Jall) = f_coll_map(
            finalTime, scaling_vector_opti, muscle_length_scaling_vector_opti, model_mass_scaling_opti,
            a[:, :-1], a_col, normF[:, :-1], normF_col,
            Qs[:, :-1], Qs_col, Qds[:, :-1], Qds_col,
            aArm[:, :-1], aArm_col, aDt, eArm, normFDt_col, Qdds_col, MTP_reserve_col)



        # # Set constraints.
        opti.subject_to(ca.vec(coll_eq_constr) == 0)
        opti.subject_to(ca.vec(coll_ineq_constr1) >= 0)
        opti.subject_to(
            ca.vec(coll_ineq_constr2) <= 1 / activationTimeConstant)
        opti.subject_to(opti.bounded(0.0081, ca.vec(coll_ineq_constr3), 4))
        opti.subject_to(opti.bounded(0.0324, ca.vec(coll_ineq_constr4), 4))
        opti.subject_to(opti.bounded(0.0121, ca.vec(coll_ineq_constr5), 4))
        opti.subject_to(opti.bounded(0.01, ca.vec(coll_ineq_constr6), 4))
        #######################################################################
        # Equality / continuity constraints.
        # Loop over mesh points.
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

            #######################################################################
        # Periodic constraints on states.
        # Joint positions and velocities.
        opti.subject_to(Qs[periodic_joints_indices_start_to_end_position_matching[0], -1] -
                        Qs[periodic_joints_indices_start_to_end_position_matching[1], 0] == 0)
        opti.subject_to(Qds[periodic_joints_indices_start_to_end_velocity_matching[0], -1] -
                        Qds[periodic_joints_indices_start_to_end_velocity_matching[1], 0] == 0)
        opti.subject_to(Qs[periodic_opposite_joints_indices_in_joints, -1] +
                        Qs[periodic_opposite_joints_indices_in_joints, 0] == 0)
        opti.subject_to(Qds[periodic_opposite_joints_indices_in_joints, -1] +
                        Qds[periodic_opposite_joints_indices_in_joints, 0] == 0)
        # Muscle activations.
        opti.subject_to(a[periodic_muscles_indices_start_to_end_matching[0], -1] - a[periodic_muscles_indices_start_to_end_matching[1], 0] == 0)
        # Tendon forces.
        opti.subject_to(normF[periodic_muscles_indices_start_to_end_matching[0], -1] - normF[periodic_muscles_indices_start_to_end_matching[1], 0] == 0)
        # Arm activations.
        opti.subject_to(aArm[periodic_actuators_indices_start_to_end_matching[0], -1] - aArm[periodic_actuators_indices_start_to_end_matching[1], 0] == 0)

        #######################################################################
        # Average speed constraint.
        Qs_nsc = Qs * (scaling_Q.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1)))
        distTraveled = (Qs_nsc[joints.index('pelvis_tx'), -1] -
                        Qs_nsc[joints.index('pelvis_tx'), 0])
        averageSpeed = distTraveled / finalTime

        if enforce_target_speed:
            opti.subject_to(averageSpeed == target_speed)
            Jall_sc = (ca.sum2(Jall) / distTraveled)
        else:
            opti.subject_to(averageSpeed > 6)
            Jall_sc = (ca.sum2(Jall) / distTraveled - 100 * averageSpeed ** 2)/1000


        #######################################################################
        # Create NLP solver.
        opti.minimize(Jall_sc)

        #######################################################################
        # Solve problem.
        from utilities import solve_with_bounds, solve_with_constraints

        # When using the default opti, bounds are replaced by constraints,
        # which is not what we want. This functions allows using bounds and not
        # constraints.
        file_path = os.path.join(path_results, 'mydiary.txt')
        sys.stdout = open(file_path, "w")
        w_opt, stats = solve_with_bounds(opti, convergence_tolerance_IPOPT)

        #######################################################################
        # Save results.
        if saveResults:
            np.save(os.path.join(path_results, 'w_opt.npy'), w_opt)
            np.save(os.path.join(path_results, 'stats.npy'), stats)

    # %% Analyze results.
    if analyzeResults:
        if loadResults:
            w_opt = np.load(os.path.join(path_results, 'w_opt.npy'))
            stats = np.load(os.path.join(path_results, 'stats.npy'),
                            allow_pickle=True).item()

        # Warning message if no convergence.
        if not stats['success'] == True:
            print("WARNING: PROBLEM DID NOT CONVERGE - {}".format(
                stats['return_status']))

        # %% Extract optimal results.
        # Because we had to replace bounds by constraints, we cannot retrieve
        # the optimal values using opti. The order below follows the order in
        # which the opti variables were declared.
        NParameters = 1
        finalTime_opt = w_opt[:NParameters]
        starti = NParameters
        n_muscles = number_of_muscles
        a_opt = (np.reshape(w_opt[starti:starti + n_muscles * (number_of_mesh_intervals + 1)],
                            (number_of_mesh_intervals + 1, n_muscles))).T
        starti = starti + n_muscles * (number_of_mesh_intervals + 1)

        a_col_opt = (np.reshape(w_opt[starti:starti + n_muscles * (polynomial_order * number_of_mesh_intervals)],
                                (polynomial_order * number_of_mesh_intervals, n_muscles))).T
        starti = starti + n_muscles * (polynomial_order * number_of_mesh_intervals)

        normF_opt = (np.reshape(w_opt[starti:starti + n_muscles * (number_of_mesh_intervals + 1)],
                                (number_of_mesh_intervals + 1, n_muscles))).T
        starti = starti + n_muscles * (number_of_mesh_intervals + 1)

        normF_col_opt = (np.reshape(w_opt[starti:starti + n_muscles * (polynomial_order * number_of_mesh_intervals)],
                                    (polynomial_order * number_of_mesh_intervals, n_muscles))).T
        starti = starti + n_muscles * (polynomial_order * number_of_mesh_intervals)

        Qs_opt = (np.reshape(w_opt[starti:starti + number_of_joints * (number_of_mesh_intervals + 1)],
                             (number_of_mesh_intervals + 1, number_of_joints))).T
        starti = starti + number_of_joints * (number_of_mesh_intervals + 1)

        Qs_col_opt = (np.reshape(w_opt[starti:starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)],
                                 (polynomial_order * number_of_mesh_intervals, number_of_joints))).T
        starti = starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)

        Qds_opt = (np.reshape(w_opt[starti:starti + number_of_joints * (number_of_mesh_intervals + 1)],
                              (number_of_mesh_intervals + 1, number_of_joints))).T
        starti = starti + number_of_joints * (number_of_mesh_intervals + 1)

        Qds_col_opt = (np.reshape(w_opt[starti:starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)],
                                  (polynomial_order * number_of_mesh_intervals, number_of_joints))).T
        starti = starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)

        aArm_opt = (np.reshape(w_opt[starti:starti + number_of_non_muscle_actuated_joints * (number_of_mesh_intervals + 1)],
                               (number_of_mesh_intervals + 1, number_of_non_muscle_actuated_joints))).T
        starti = starti + number_of_non_muscle_actuated_joints * (number_of_mesh_intervals + 1)

        aArm_col_opt = (np.reshape(w_opt[starti:starti + number_of_non_muscle_actuated_joints * (polynomial_order * number_of_mesh_intervals)],
                                   (polynomial_order * number_of_mesh_intervals, number_of_non_muscle_actuated_joints))).T
        starti = starti + number_of_non_muscle_actuated_joints * (polynomial_order * number_of_mesh_intervals)

        aDt_opt = (np.reshape(w_opt[starti:starti + n_muscles * number_of_mesh_intervals],
                              (number_of_mesh_intervals, n_muscles))).T
        starti = starti + n_muscles * number_of_mesh_intervals

        eArm_opt = (np.reshape(w_opt[starti:starti + number_of_non_muscle_actuated_joints * number_of_mesh_intervals],
                               (number_of_mesh_intervals, number_of_non_muscle_actuated_joints))).T
        starti = starti + number_of_non_muscle_actuated_joints * number_of_mesh_intervals

        normFDt_col_opt = (np.reshape(w_opt[starti:starti + n_muscles * (polynomial_order * number_of_mesh_intervals)],
                                      (polynomial_order * number_of_mesh_intervals, n_muscles))).T
        starti = starti + n_muscles * (polynomial_order * number_of_mesh_intervals)

        Qdds_col_opt = (np.reshape(w_opt[starti:starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)],
                                   (polynomial_order * number_of_mesh_intervals, number_of_joints))).T
        starti = starti + number_of_joints * (polynomial_order * number_of_mesh_intervals)


        scaling_vector_opt = (np.reshape(w_opt[starti:starti + 54 * (polynomial_order * number_of_mesh_intervals)],
                                         (polynomial_order * number_of_mesh_intervals, 54))).T
        starti = starti + 54 * (polynomial_order * number_of_mesh_intervals)

        muscle_scaling_vector_opt = (np.reshape(w_opt[starti:starti + 92 * (polynomial_order * number_of_mesh_intervals)],
                                                (polynomial_order * number_of_mesh_intervals, 92))).T
        starti = starti + 92 * polynomial_order * number_of_mesh_intervals

        model_mass_scaling_opt = (np.reshape(w_opt[starti:starti + 1 * (polynomial_order * number_of_mesh_intervals)],
                                             (polynomial_order * number_of_mesh_intervals, 1))).T
        starti = starti + 1 * polynomial_order * number_of_mesh_intervals

        MTP_reserve_col_opt = (np.reshape(w_opt[starti:starti + 2 * (polynomial_order * number_of_mesh_intervals)],
                                   (polynomial_order * number_of_mesh_intervals, 2))).T
        starti = starti + 2 * (polynomial_order * number_of_mesh_intervals)

        scaling_vector_opt = scaling_vector_opt[:,0]
        muscle_scaling_vector_opt = muscle_scaling_vector_opt[:,0]
        model_mass_scaling_opt = model_mass_scaling_opt[:,0]
        model_mass_opt = modelMass * model_mass_scaling_opt
        model_height_opt = f_height(scaling_vector_opt)
        model_BMI_opt = model_mass_opt / (model_height_opt * model_height_opt)
        assert (starti == w_opt.shape[0]), "error when extracting results"

        # Go from half gait cycle to full gait cycle
        Qs_opt_part_2 = np.zeros((number_of_joints, number_of_mesh_intervals))
        Qs_opt_part_2[:,:] = Qs_opt[:,1:]
        Qs_opt_part_2[periodic_joints_indices_start_to_end_position_matching[1],:] = Qs_opt[periodic_joints_indices_start_to_end_position_matching[0],1:]
        Qs_opt_part_2[periodic_opposite_joints_indices_in_joints, :] = -Qs_opt[periodic_opposite_joints_indices_in_joints, 1:]
        Qs_opt_part_2[joints.index('pelvis_tx'),:] = Qs_opt[joints.index('pelvis_tx'),1:] + Qs_opt[joints.index('pelvis_tx'),-1]
        Qs_gait_cycle_opt = np.concatenate((Qs_opt,Qs_opt_part_2),1)
        if writeMotionFiles:
            # muscleLabels = [bothSidesMuscle + '/activation'
            #                 for bothSidesMuscle in bothSidesMuscles]
            labels = ['time'] + joints
            tgrid_GC = tgrid = np.linspace(0, 2*finalTime_opt[0], 2*number_of_mesh_intervals + 1)
            Qs_gait_cycle_nsc = (Qs_gait_cycle_opt * (scaling_Q.to_numpy().T * np.ones((1, 2*number_of_mesh_intervals + 1)))).T
            Qs_gait_cycle_nsc[:,rotational_joint_indices_in_joints] = 180/np.pi*Qs_gait_cycle_nsc[:,rotational_joint_indices_in_joints]
            data = np.concatenate((tgrid_GC, Qs_gait_cycle_nsc), axis=1)
            from utilities import numpy2storage

            numpy2storage(labels, data, os.path.join(path_results, 'motion.mot'))


        default_scale_tool_xml_name = path_model_folder + '/scaleTool_Default.xml'
        NeMu_subfunctions.generateScaledModels(0, default_scale_tool_xml_name, np.reshape(scaling_vector_opt, (54)),
                                  path_results, model_name)


        if not os.path.exists(os.path.join(path_trajectories,
                                              'optimaltrajectories.npy')):
            optimaltrajectories = {}
        else:
            optimaltrajectories = np.load(
                os.path.join(path_trajectories,
                            'optimaltrajectories.npy'),
                allow_pickle=True)
            optimaltrajectories = optimaltrajectories.item()

        # Evaluate skeleton dynamics at each collocation point:
        QsQds_col_opt_nsc = np.zeros((number_of_joints * 2, polynomial_order * number_of_mesh_intervals))
        Qs_col_opt_nsc = (Qs_col_opt * (scaling_Q.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        Qds_col_opt_nsc = (Qds_col_opt * (scalingQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        Qdds_col_opt_nsc = (Qdds_col_opt * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))

        speed = Qs_col_opt_nsc[joints.index('pelvis_tx'), -1] / finalTime_opt

        QsQds_col_opt_nsc[::2, :] = Qs_col_opt_nsc[joint_indices_in_external_function, :]
        QsQds_col_opt_nsc[1::2, :] = Qds_col_opt_nsc[joint_indices_in_external_function, :]

        # Muscle lengths, velocities and moment arms
        Qsin_col_opt = Qs_col_opt_nsc[muscle_actuated_joints_indices_in_joints, :]
        Qdsin_col_opt = Qds_col_opt_nsc[muscle_actuated_joints_indices_in_joints, :]
        lMT_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
        vMT_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
        dM_col_opt = np.zeros((number_of_muscles, number_of_muscle_actuated_joints, polynomial_order * number_of_mesh_intervals))
        normF_nsc_col_opt = normF_col_opt * (scaling_muscle_force.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals)))
        normFDt_nsc_col_opt = normFDt_col_opt * (scalingFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals)))
        active_muscle_force_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
        passive_muscle_force_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
        hillEquilibrium_residual_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
        muscle_active_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        muscle_passive_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        muscle_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        biological_joint_torques_equilibrium_residual_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        time_col_opt = np.zeros((1, polynomial_order * number_of_mesh_intervals))
        metabolicEnergyRate_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))

        index = 0
        previous_time = 0
        h_opt = finalTime_opt / number_of_mesh_intervals
        for i in range(number_of_mesh_intervals):
            time_col_opt[0, index]   = previous_time  + 0.155 * h_opt
            time_col_opt[0, index+1] = previous_time  + 0.645 *h_opt
            time_col_opt[0, index+2] = previous_time + 1 * h_opt
            previous_time = time_col_opt[0, index+2]
            index = index + 3
        time_col_opt = np.reshape(time_col_opt,(polynomial_order * number_of_mesh_intervals,))


        for i in range(polynomial_order * number_of_mesh_intervals):
            NeMu_input_j = ca.vertcat(scaling_vector_opt[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies],
                                      Qsin_col_opt[:,i], Qdsin_col_opt[:,i])
            [lMTj_lr, vMTj_lr, dMj] = f_get_muscle_tendon_length_velocity_moment_arm(NeMu_input_j)
            lMT_col_opt[:, i] = np.reshape(lMTj_lr,(number_of_muscles,))
            vMT_col_opt[:, i] = np.reshape(vMTj_lr,(number_of_muscles,))
            dM_col_opt[:, :, i] = dMj



            [hillEquilibriumj, Fj, activeFiberForcej, passiveFiberForcej,
             normActiveFiberLengthForcej, normFiberLengthj, fiberVelocityj, activeFiberForce_effectivej, passiveFiberForce_effectivej] = (
                f_hillEquilibrium(a_col_opt[:, i], lMTj_lr, vMTj_lr,
                                  normF_nsc_col_opt[:, i], normFDt_nsc_col_opt[:, i], muscle_scaling_vector_opt,
                                  muscle_scaling_vector_opt))
            active_muscle_force_col_opt[:, i] = np.reshape(activeFiberForce_effectivej,(number_of_muscles,))
            passive_muscle_force_col_opt[:, i] = np.reshape(passiveFiberForce_effectivej,(number_of_muscles,))
            hillEquilibrium_residual_col_opt[:, i] = np.reshape(hillEquilibriumj,(number_of_muscles,))

            metabolicEnergyRate_col_opt[:, i] = f_metabolicsBhargava(
                a_col_opt[:, i], a_col_opt[:, i], normFiberLengthj, fiberVelocityj,
                activeFiberForcej, passiveFiberForcej,
                normActiveFiberLengthForcej, muscle_scaling_vector_opt, muscle_scaling_vector_opt)[5]


            for j in range(len(muscle_actuated_joints_indices_in_joints)):
                joint = joints[muscle_actuated_joints_indices_in_joints[j]]
                if joint != 'mtp_angle_l' and joint != 'mtp_angle_r':
                    muscle_active_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(dMj[:, j] * activeFiberForce_effectivej)
                    muscle_passive_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(dMj[:, j] * passiveFiberForce_effectivej)
                    muscle_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(dMj[:, j] * Fj)
            biological_joint_torques_equilibrium_residual_col_opt[:, i] = muscle_joint_torques_col_opt[:, i] - (muscle_passive_joint_torques_col_opt[:, i] + muscle_active_joint_torques_col_opt[:, i])

        Tj_test = skeleton_dynamics_and_more(
            ca.vertcat(QsQds_col_opt_nsc[:, 0], Qdds_col_opt_nsc[:, 0], scaling_vector_opt, modelMass))
        Tj_col_opt = np.zeros((Tj_test.shape[0], polynomial_order * number_of_mesh_intervals))

        for i in range(polynomial_order * number_of_mesh_intervals):
            Tj_col_opt[:, i] = np.reshape(skeleton_dynamics_and_more(ca.vertcat(QsQds_col_opt_nsc[:, i], Qdds_col_opt_nsc[joint_indices_in_external_function, i], scaling_vector_opt, modelMass)), (Tj_test.shape[0],))

        generalized_forces_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        for i, joint in enumerate(joints):
            index = map_external_function_outputs['residuals'][joint]
            generalized_forces_col_opt[i, :] = Tj_col_opt[index, :]

        # generalized forces (joint torques) are the result of passive joint torques, limit joint torques, biological joint torques (active contribution +  passive contribution)
        passive_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        for joint in non_muscle_actuated_joints:
            passive_joint_torques = f_linearPassiveTorque(Qs_col_opt_nsc[joints.index(joint), :], Qds_col_opt_nsc[joints.index(joint), :])
            passive_joint_torques_col_opt[joints.index(joint), :] = np.reshape(passive_joint_torques,(polynomial_order * number_of_mesh_intervals,))

        for joint in mtpJoints:
            passive_joint_torques = f_linearPassiveMtpTorque(Qs_col_opt_nsc[joints.index(joint), :], Qds_col_opt_nsc[joints.index(joint), :])
            passive_joint_torques_col_opt[joints.index(joint), :] = np.reshape(passive_joint_torques,(polynomial_order * number_of_mesh_intervals,))

        limit_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        for joint in limit_torque_joints:
            limit_joint_torques =  f_limit_torque[joint](Qs_col_opt_nsc[joints.index(joint), :], Qds_col_opt_nsc[joints.index(joint), :])
            limit_joint_torques_col_opt[joints.index(joint), :] = np.reshape(limit_joint_torques,(polynomial_order * number_of_mesh_intervals,))

        for i, joint in enumerate(non_muscle_actuated_joints):
            index = joints.index(joint)
            muscle_joint_torques_col_opt[index, :] = aArm_col_opt[i, :] * scalingArmA.iloc[0]['arm_rot_r']

        for i, joint in enumerate(mtpJoints):
            index = joints.index(joint)
            muscle_joint_torques_col_opt[index, :] = MTP_reserve_col_opt[i, :]

        joint_torques_equilibrium_residual_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
        for joint in joints:
            index = joints.index(joint)
            joint_torques_equilibrium_residual_col_opt[index, :] = generalized_forces_col_opt[index, :] - (passive_joint_torques_col_opt[index, :] + limit_joint_torques_col_opt[index, :] + muscle_joint_torques_col_opt[index, :])



        # Evaluate maximal isometric torques

        max_iso_torques_joints = ['hip_flexion_l', 'hip_flexion_l', 'knee_angle_l', 'knee_angle_l', 'ankle_angle_l',
                           'ankle_angle_l']
        max_iso_torques_sign = [1, -1, 1, -1, 1, -1]

        # We evaluate for every max iso torque of interest (6) across 20 positions.
        Qs_max_iso_torque = np.zeros((len(joints), 20, 6))

        Qs_max_iso_torque[joints.index('hip_flexion_l'), :, 0] = np.linspace(-45, 90, num=20)
        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 0] = np.linspace(-30, -30, num=20)

        Qs_max_iso_torque[joints.index('hip_flexion_l'), :, 1] = np.linspace(-45, 90, num=20)
        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 1] = np.linspace(-30, -30, num=20)

        Qs_max_iso_torque[joints.index('hip_flexion_l'), :, 2] = np.linspace(45, 45, num=20)
        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 2] = np.linspace(-120, 0, num=20)

        Qs_max_iso_torque[joints.index('hip_flexion_l'), :, 3] = np.linspace(45, 45, num=20)
        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 3] = np.linspace(-120, 0, num=20)

        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 4] = np.linspace(45, 45, num=20)
        Qs_max_iso_torque[joints.index('ankle_angle_l'), :, 4] = np.linspace(-30, 10, num=20)

        Qs_max_iso_torque[joints.index('knee_angle_l'), :, 5] = np.linspace(45, 45, num=20)
        Qs_max_iso_torque[joints.index('ankle_angle_l'), :, 5] = np.linspace(-30, 10, num=20)

        Qs_max_iso_torque = Qs_max_iso_torque * np.pi / 180

        maximal_isometric_torques = np.zeros((20, 6))
        passive_isometric_torques = np.zeros((20, 6))
        for j in range(len(max_iso_torques_joints)):
            for k in range(20):
                # Compute muscle-tendon length in default pose (all zeros) for scaled model (scaling factors are the optimization variables)
                Q_max_iso_torque = Qs_max_iso_torque[:, k, j]
                # isometric: speeds are zero
                Qd_max_iso_torque = np.zeros((len(joints), 1))

                Qsinj = Q_max_iso_torque[muscle_actuated_joints_indices_in_joints]
                Qdsinj = Qd_max_iso_torque[muscle_actuated_joints_indices_in_joints]
                NeMu_input = ca.vertcat(
                    scaling_vector_opt[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies], Qsinj, Qdsinj)
                [lMT_max_iso_torque, vMT_max_iso_torque, dM_max_iso_torque] = f_get_muscle_tendon_length_velocity_moment_arm(NeMu_input)

                from utilities import solve_with_bounds

                opti_maxIso = ca.Opti()
                a_max_iso_torque = opti_maxIso.variable(number_of_muscles, 1)
                opti_maxIso.subject_to(opti_maxIso.bounded(0, a_max_iso_torque, 1))
                F_max_iso_torque = opti_maxIso.variable(number_of_muscles, 1)
                opti_maxIso.subject_to(opti_maxIso.bounded(0, F_max_iso_torque, 10000))

                [hillEquilibrium_neutral, Ft_neutral, activeFiberForce_neutral,
                 passiveFiberForce_neutral, normActiveFiberLengthForce_neutral,
                 normFiberLength_neutral, fiberVelocity_neutral, _, passiveEffectiveFiberForce_neutral] = (
                    f_hillEquilibrium(a_max_iso_torque, lMT_max_iso_torque, vMT_max_iso_torque,
                        F_max_iso_torque, np.zeros((number_of_muscles, 1)), muscle_scaling_vector_opt,
                        model_mass_scaling_opt))

                opti_maxIso.subject_to(hillEquilibrium_neutral == 0)
                joint_maximized = max_iso_torques_joints[j]
                i = muscle_actuated_joints.index(joint_maximized)
                tau = ca.sum1(dM_max_iso_torque[:, i] * Ft_neutral)
                opti_maxIso.minimize(max_iso_torques_sign[j] * tau + 0.0001 * ca.sumsqr(a_max_iso_torque))
                w_maxIso, stats = solve_with_bounds(opti_maxIso, 1e-4)
                a_max_iso_torque = w_maxIso[:92]
                F_max_iso_torque = w_maxIso[92:]
                [hillEquilibrium_neutral, Ft_neutral, activeFiberForce_neutral,
                 passiveFiberForce_neutral, normActiveFiberLengthForce_neutral,
                 normFiberLength_neutral, fiberVelocity_neutral, _, passiveEffectiveFiberForce_neutral] = (
                    f_hillEquilibrium(a_max_iso_torque, lMT_max_iso_torque, vMT_max_iso_torque,
                                      F_max_iso_torque, np.zeros((number_of_muscles, 1)), muscle_scaling_vector_opt,
                                      model_mass_scaling_opt))
                maximal_isometric_torques[k, j] = ca.sum1(dM_max_iso_torque[:, i] * Ft_neutral)
                passive_isometric_torques[k, j] = ca.sum1(dM_max_iso_torque[:, i] * passiveEffectiveFiberForce_neutral)



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
                'Qs_max_iso_torque': Qs_max_iso_torque
                }

        if plotResults:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4, 6, sharex=True)
            fig.suptitle('Joint torques interpretation')
            for i, ax in enumerate(axs.flat):
                index = i + 6
                color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))

                ax.plot(time_col_opt, muscle_joint_torques_col_opt[index, :], c=next(color), label='muscle torque')
                ax.plot(time_col_opt,limit_joint_torques_col_opt[index, :], c=next(color), label='limit torque')
                ax.plot(time_col_opt,passive_joint_torques_col_opt[index, :] , c=next(color), label='passive torque')
                ax.plot(time_col_opt,generalized_forces_col_opt[index, :], c=next(color), label='torque')
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
                ax.plot(time_col_opt,180/np.pi * Qs_col_opt_nsc[index, :])
                ax.plot(time_col_opt, 180 / np.pi * optimaltrajectories[case]['bounds_Q']['upper'][joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
                ax.plot(time_col_opt,
                        180 / np.pi * optimaltrajectories[case]['bounds_Q']['lower'][joints[index]].to_numpy() * np.ones(
                            (polynomial_order * number_of_mesh_intervals,)))

                ax.set_title(joints[index])

            fig, axs = plt.subplots(4, 6, sharex=True)
            fig.suptitle('Joint angular velocities')
            for i, ax in enumerate(axs.flat):
                index = i + 6
                color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
                ax.plot(time_col_opt,180/np.pi * Qds_col_opt_nsc[index, :])
                ax.plot(time_col_opt, 180 / np.pi * optimaltrajectories[case]['bounds_Qd']['upper'][joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
                ax.plot(time_col_opt, 180 / np.pi * optimaltrajectories[case]['bounds_Qd']['lower'][joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))

                ax.set_title(joints[index])

            fig, axs = plt.subplots(4, 6, sharex=True)
            fig.suptitle('Joint angular accelerations')
            for i, ax in enumerate(axs.flat):
                index = i + 6
                color = iter(plt.cm.rainbow(np.linspace(0, 1, 6)))
                ax.plot(time_col_opt, 180 / np.pi * Qdds_col_opt_nsc[index, :])
                ax.plot(time_col_opt, 180 / np.pi * optimaltrajectories[case]['bounds_Qdd']['upper'][joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
                ax.plot(time_col_opt, 180 / np.pi * optimaltrajectories[case]['bounds_Qdd']['lower'][joints[index]].to_numpy() * np.ones((polynomial_order * number_of_mesh_intervals,)))
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
            titles_dim = ['depth','height','width']
            for i, ax in enumerate(axs.flat):
                index = i
                to_plot = scaling_vector_opt[i::3]
                ax.scatter(to_plot, np.arange(np.size(to_plot)))
                if i == 0:
                    ax.set_yticks(np.arange(np.size(to_plot)))
                    ax.set_yticklabels(optimaltrajectories[case]['skeleton_scaling_bodies'])
                ax.set_title(titles_dim[index])




            plt.show()


        np.save(os.path.join(path_trajectories, 'optimaltrajectories.npy'),
                optimaltrajectories)

        # modelMass = modelMass * model_mass_scaling_opt
        # muscleVolume_original = (muscle_parameters[0, :] * muscle_parameters[1, :]).T
        # muscleVolume_scaled = muscleVolume_original * model_mass_scaling_opt
        # optimalFiberLength_scaled = (muscle_scaling_vector_opt * muscle_parameters[1, :]).T
        # tendonSlackLength_scaled = (muscle_scaling_vector_opt * muscle_parameters[2, :]).T
        # maximalIsometricForce_scaled = muscleVolume_scaled.T / optimalFiberLength_scaled
        # maxFiberVelocity_scaled = (muscle_scaling_vector_opt * muscle_parameters[4, :]).T
        #
        #




        # # %% Unscale some of the optimal variables.
        # normF_opt_nsc = normF_opt * (scalingF.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1)))
        # normF_col_opt_nsc = (
        #         normF_col_opt * (scalingF.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        # Qs_opt_nsc = Qs_opt * (scalingQs.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1)))
        # Qs_col_opt_nsc = (
        #         Qs_col_opt * (scalingQs.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        # Qds_opt_nsc = Qds_opt * (scalingQds.to_numpy().T * np.ones((1, number_of_mesh_intervals + 1)))
        # Qds_col_opt_nsc = (
        #         Qds_col_opt * (scalingQds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        # aDt_opt_nsc = aDt_opt * (scalingADt.to_numpy().T * np.ones((1, number_of_mesh_intervals)))
        # Qdds_col_opt_nsc = (
        #         Qdds_col_opt * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        # normFDt_col_opt_nsc = (
        #         normFDt_col_opt * (scalingFDt.to_numpy().T * np.ones((1, polynomial_order * number_of_mesh_intervals))))
        # normFDt_opt_nsc = normFDt_col_opt_nsc[:, polynomial_order - 1::polynomial_order]
        # aArm_opt_nsc = aArm_opt * scalingArmE.iloc[0]['arm_rot_r']
        #
        # # %% Extract ground reaction forces to later identify heel-strike and
        # # reconstruct a full gait cycle. Also do some sanity checks with
        # # non-muscle-driven joints.
        #
        # # Passive torques.
        # # Arms.
        # linearPassiveTorqueArms_opt = np.zeros((number_of_non_muscle_actuated_joints, number_of_mesh_intervals + 1))
        # for k in range(number_of_mesh_intervals + 1):
        #     for cj, joint in enumerate(non_muscle_actuated_joints):
        #         linearPassiveTorqueArms_opt[cj, k] = f_linearPassiveTorque(
        #             Qs_opt_nsc[joints.index(joint), k],
        #             Qds_opt_nsc[joints.index(joint), k])
        #
        # # Mtps.
        # linearPassiveTorqueMtp_opt = np.zeros((nMtpJoints, number_of_mesh_intervals + 1))
        # passiveTorqueMtp_opt = np.zeros((nMtpJoints, number_of_mesh_intervals + 1))
        # for k in range(number_of_mesh_intervals + 1):
        #     for cj, joint in enumerate(mtpJoints):
        #         linearPassiveTorqueMtp_opt[cj, k] = (
        #             f_linearPassiveMtpTorque(
        #                 Qs_opt_nsc[joints.index(joint), k],
        #                 Qds_opt_nsc[joints.index(joint), k]))
        #         passiveTorqueMtp_opt[cj, k] = f_limit_torque[joint](
        #             Qs_opt_nsc[joints.index(joint), k],
        #             Qds_opt_nsc[joints.index(joint), k])
        #
        # # Ground reactions forces
        # QsQds_opt_nsc = np.zeros((number_of_joints * 2, number_of_mesh_intervals + 1))
        # QsQds_opt_nsc[::2, :] = Qs_opt_nsc[joint_acceleration_indices_in_F, :]
        # QsQds_opt_nsc[1::2, :] = Qds_opt_nsc[joint_acceleration_indices_in_F, :]
        # Qdds_opt = Qdds_col_opt_nsc[:, polynomial_order - 1::polynomial_order]
        # Qdds_opt_nsc = Qdds_opt[joint_acceleration_indices_in_F, :]
        # if optimize_skeleton_scaling == True:
        #     Tj_temp = F_scalefactors(ca.vertcat(QsQds_opt_nsc[:, 1], Qdds_opt_nsc[:, 0], scaling_vector_opt, modelMass))
        # else:
        #     Tj_temp = F(ca.vertcat(QsQds_opt_nsc[:, 1], Qdds_opt_nsc[:, 0]))
        # F1_out = np.zeros((Tj_temp.shape[0], number_of_mesh_intervals))
        # armT = np.zeros((nArmJoints, number_of_mesh_intervals))
        # if withMTP:
        #     mtpT = np.zeros((nMtpJoints, number_of_mesh_intervals))
        # for k in range(number_of_mesh_intervals):
        #     if optimize_skeleton_scaling == True:
        #         Tj = F_scalefactors(ca.vertcat(QsQds_opt_nsc[:, k + 1], Qdds_opt_nsc[:, k], scaling_vector_opt, modelMass))
        #     else:
        #         Tj = F(ca.vertcat(QsQds_opt_nsc[:, k + 1], Qdds_opt_nsc[:, k]))
        #     F1_out[:, k] = Tj.full().T
        #     for cj, joint in enumerate(non_muscle_actuated_joints):
        #         armT[cj, k] = f_diffTorques(
        #             F1_out[F_map['residuals'][joint], k] /
        #             scalingArmE.iloc[0][joint],
        #             aArm_opt[cj, k + 1],
        #             linearPassiveTorqueArms_opt[cj, k + 1] /
        #             scalingArmE.iloc[0][joint])
        #     if withMTP:
        #         count = 0
        #         for cj, joint in enumerate(mtpJoints):
        #             mtpT[cj, k] = f_diffTorques(
        #                 F1_out[F_map['residuals'][joint], k] /
        #                 scalingMtpE.iloc[0][joint],
        #                 MTP_reserve_opt[count, k] /
        #                 scalingMtpE.iloc[0][joint],
        #                 (linearPassiveTorqueMtp_opt[cj, k + 1] +
        #                  passiveTorqueMtp_opt[cj, k + 1]) /
        #                 scalingMtpE.iloc[0][joint])
        #             count = count + 1
        # GRF_opt = F1_out[idxGRF, :]
        #
        # # Sanity checks.
        # assert np.alltrue(np.abs(armT) < 10 ** (-convergence_tolerance_IPOPT)), (
        #     "Error arm torques balance")
        # if withMTP:
        #     assert np.alltrue(np.abs(mtpT) < 10 ** (-convergence_tolerance_IPOPT)), (
        #         "error mtp torques balance")
        #
        #     # %% Reconstruct entire gait cycle starting at right heel-strike.
        # from utilities import getIdxIC_3D
        #
        # threshold = 30
        # idxIC, legIC = getIdxIC_3D(GRF_opt, threshold)
        # if legIC == "undefined":
        #     np.disp("Problem with gait reconstruction")
        # idxIC_s = idxIC + 1  # GRF_opt obtained at mesh points starting at k=1.
        # idxIC_c = idxIC
        #
        # # Joint positions.
        # Qs_GC = np.zeros((nJoints, 2 * number_of_mesh_intervals))
        # Qs_GC[:, :number_of_mesh_intervals - idxIC_s[0]] = Qs_opt_nsc[:, idxIC_s[0]:-1]
        # Qs_GC[idxPerQsJointsA, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     Qs_opt_nsc[idxPerQsJointsB, :-1])
        # Qs_GC[periodic_opposite_joints_indices_in_joints, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     -Qs_opt_nsc[periodic_opposite_joints_indices_in_joints, :-1])
        # Qs_GC[joints.index('pelvis_tx'), number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #         Qs_opt_nsc[joints.index('pelvis_tx'), :-1] +
        #         Qs_opt_nsc[joints.index('pelvis_tx'), -1])
        # Qs_GC[:, number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = Qs_opt_nsc[:, :idxIC_s[0]]
        # Qs_GC[joints.index('pelvis_tx'), number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = (
        #         Qs_opt_nsc[joints.index('pelvis_tx'), :idxIC_s[0]] +
        #         2 * Qs_opt_nsc[joints.index('pelvis_tx'), -1])
        # if legIC == "left":
        #     Qs_GC[idxPerQsJointsA, :] = Qs_GC[idxPerQsJointsB, :]
        #     Qs_GC[periodic_opposite_joints_indices_in_joints, :] = (
        #         -Qs_GC[periodic_opposite_joints_indices_in_joints, :])
        # Qs_GC[joints.index('pelvis_tx'), :] -= (
        #     Qs_GC[joints.index('pelvis_tx'), 0])
        # Qs_GC[rotational_joint_indices_in_joints, :] = Qs_GC[rotational_joint_indices_in_joints, :] * 180 / np.pi
        #
        # # Joint velocities.
        # Qds_GC = np.zeros((nJoints, 2 * number_of_mesh_intervals))
        # Qds_GC[:, :number_of_mesh_intervals - idxIC_s[0]] = Qds_opt_nsc[:, idxIC_s[0]:-1]
        # Qds_GC[idxPerQdsJointsA, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     Qds_opt_nsc[idxPerQdsJointsB, :-1])
        # Qds_GC[idxPerQdsJointsA, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     Qds_opt_nsc[idxPerQdsJointsB, :-1])
        # Qds_GC[periodic_opposite_joints_indices_in_joints, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     -Qds_opt_nsc[periodic_opposite_joints_indices_in_joints, :-1])
        # Qds_GC[:, number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = Qds_opt_nsc[:, :idxIC_s[0]]
        # if legIC == "left":
        #     Qds_GC[idxPerQdsJointsA, :] = Qds_GC[idxPerQdsJointsB, :]
        #     Qds_GC[periodic_opposite_joints_indices_in_joints, :] = -Qds_GC[periodic_opposite_joints_indices_in_joints, :]
        # Qds_GC[rotational_joint_indices_in_joints, :] = Qds_GC[rotational_joint_indices_in_joints, :] * 180 / np.pi
        #
        # # Joint accelerations.
        # Qdds_GC = np.zeros((nJoints, 2 * number_of_mesh_intervals))
        # Qdds_GC[:, :number_of_mesh_intervals - idxIC_c[0]] = Qdds_opt[:, idxIC_c[0]:]
        # Qdds_GC[idxPerQdsJointsA, number_of_mesh_intervals - idxIC_c[0]:number_of_mesh_intervals - idxIC_c[0] + number_of_mesh_intervals] = (
        #     Qdds_opt[idxPerQdsJointsB, :])
        # Qdds_GC[periodic_opposite_joints_indices_in_joints, number_of_mesh_intervals - idxIC_c[0]:number_of_mesh_intervals - idxIC_c[0] + number_of_mesh_intervals] = (
        #     -Qdds_opt[periodic_opposite_joints_indices_in_joints, :])
        # Qdds_GC[:, number_of_mesh_intervals - idxIC_c[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = Qdds_opt[:, :idxIC_c[0]]
        # if legIC == "left":
        #     Qdds_GC[idxPerQdsJointsA, :] = Qdds_GC[idxPerQdsJointsB, :]
        #     Qdds_GC[periodic_opposite_joints_indices_in_joints, :] = -Qdds_GC[periodic_opposite_joints_indices_in_joints, :]
        # Qdds_GC[rotational_joint_indices_in_joints, :] = Qdds_GC[rotational_joint_indices_in_joints, :] * 180 / np.pi
        #
        # # Muscle activations.
        # A_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # A_GC[:, :number_of_mesh_intervals - idxIC_s[0]] = a_opt[:, idxIC_s[0]:-1]
        # A_GC[:, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = a_opt[idxPerMuscles, :-1]
        # A_GC[:, number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = a_opt[:, :idxIC_s[0]]
        # if legIC == "left":
        #     A_GC = A_GC[idxPerMuscles, :]
        #
        # # Tendon forces.
        # F_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # F_GC[:, :number_of_mesh_intervals - idxIC_s[0]] = normF_opt_nsc[:, idxIC_s[0]:-1]
        # F_GC[:, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     normF_opt_nsc[idxPerMuscles, :-1])
        # F_GC[:, number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = normF_opt_nsc[:, :idxIC_s[0]]
        # if legIC == "left":
        #     F_GC = F_GC[idxPerMuscles, :]
        #
        # # Tendon force derivatives.
        # FDt_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # FDt_GC[:, :number_of_mesh_intervals - idxIC_c[0]] = normFDt_opt_nsc[:, idxIC_c[0]:]
        # FDt_GC[:, number_of_mesh_intervals - idxIC_c[0]:number_of_mesh_intervals - idxIC_c[0] + number_of_mesh_intervals] = (
        #     normFDt_opt_nsc[idxPerMuscles, :])
        # FDt_GC[:, number_of_mesh_intervals - idxIC_c[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = normFDt_opt_nsc[:, :idxIC_c[0]]
        # if legIC == "left":
        #     FDt_GC = FDt_GC[idxPerMuscles, :]
        #
        # # Arm actuator activations.
        # aArm_GC = np.zeros((nArmJoints, 2 * number_of_mesh_intervals))
        # aArm_GC[:, :number_of_mesh_intervals - idxIC_s[0]] = aArm_opt_nsc[:, idxIC_s[0]:-1]
        # aArm_GC[:, number_of_mesh_intervals - idxIC_s[0]:number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals] = (
        #     aArm_opt_nsc[idxPerArmJoints, :-1])
        # aArm_GC[:, number_of_mesh_intervals - idxIC_s[0] + number_of_mesh_intervals:2 * number_of_mesh_intervals] = aArm_opt_nsc[:, :idxIC_s[0]]
        # if legIC == "left":
        #     aArm_GC = aArm_GC[idxPerArmJoints, :]
        #
        # # Time grid.
        # tgrid = np.linspace(0, finalTime_opt[0], number_of_mesh_intervals + 1)
        # tgrid_GC = np.zeros((1, 2 * number_of_mesh_intervals))
        # tgrid_GC[:, :number_of_mesh_intervals] = tgrid[:number_of_mesh_intervals].T
        # tgrid_GC[:, number_of_mesh_intervals:] = tgrid[:number_of_mesh_intervals].T + tgrid[-1].T
        #
        # # %% Compute metabolic cost of transport over entire gait cycle.
        # Qs_GC_rad = Qs_GC.copy()
        # Qs_GC_rad[rotational_joint_indices_in_joints, :] = Qs_GC_rad[rotational_joint_indices_in_joints, :] * np.pi / 180
        # Qds_GC_rad = Qds_GC.copy()
        # Qds_GC_rad[rotational_joint_indices_in_joints, :] = Qds_GC_rad[rotational_joint_indices_in_joints, :] * np.pi / 180
        # basal_coef = 1.2
        # basal_exp = 1
        # metERatePerMuscle = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # tolMetERate = np.zeros((1, 2 * number_of_mesh_intervals))
        # actHeatRate = np.zeros((1, 2 * number_of_mesh_intervals))
        # mtnHeatRate = np.zeros((1, 2 * number_of_mesh_intervals))
        # shHeatRate = np.zeros((1, 2 * number_of_mesh_intervals))
        # mechWRate = np.zeros((1, 2 * number_of_mesh_intervals))
        # normFiberLength_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # fiberVelocity_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # actHeatRate_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # mtnHeatRate_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # shHeatRate_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # mechWRate_GC = np.zeros((n_muscles, 2 * number_of_mesh_intervals))
        # for k in range(2 * number_of_mesh_intervals):
        #     ###################################################################
        #     if use_NeMu == True:
        #         # NeMu approximation
        #         NeMu_input_ = np.concatenate((np.reshape(scaling_vector_opt[:24], (24)),
        #                                       Qs_GC_rad[muscle_driven_joints_indices, k],
        #                                       Qds_GC_rad[muscle_driven_joints_indices, k]), 0)
        #
        #         NeMu_output_ = f_NeMu(NeMu_input_)
        #         lMTk_GC_lr = np.array(NeMu_output_[0])
        #         vMTk_GC_lr = np.array(NeMu_output_[1])
        #     else:
        #         # Polynomial approximations.
        #         # Left leg.
        #         Qsk_GC_l = Qs_GC_rad[leftPolJointIdx, k]
        #         Qdsk_GC_l = Qds_GC_rad[leftPolJointIdx, k]
        #         [lMTk_GC_l, vMTk_GC_l, _] = f_polynomial(Qsk_GC_l, Qdsk_GC_l)
        #         # Right leg.
        #         Qsk_GC_r = Qs_GC_rad[rightPolJointIdx, k]
        #         Qdsk_GC_r = Qds_GC_rad[rightPolJointIdx, k]
        #         [lMTk_GC_r, vMTk_GC_r, _] = f_polynomial(Qsk_GC_r, Qdsk_GC_r)
        #         # Both leg.
        #         lMTk_GC_lr = ca.vertcat(lMTk_GC_l[leftPolMuscleIdx],
        #                                 lMTk_GC_r[rightPolMuscleIdx])
        #         vMTk_GC_lr = ca.vertcat(vMTk_GC_l[leftPolMuscleIdx],
        #                                 vMTk_GC_r[rightPolMuscleIdx])
        #
        #     ###################################################################
        #     # Derive Hill-equilibrium.
        #
        #     if optimize_skeleton_scaling == True:
        #         [hillEquilibriumk_GC, Fk_GC, activeFiberForcek_GC,
        #          passiveFiberForcek_GC, normActiveFiberLengthForcek_GC,
        #          normFiberLengthk_GC, fiberVelocityk_GC] = (
        #             f_hillEquilibrium_optimize_scaling(A_GC[:, k], lMTk_GC_lr, vMTk_GC_lr,
        #                                                F_GC[:, k], FDt_GC[:, k], muscle_scaling_vector_opt,
        #                                                model_mass_scaling_opt))
        #     else:
        #         [hillEquilibriumk_GC, Fk_GC, activeFiberForcek_GC,
        #          passiveFiberForcek_GC, normActiveFiberLengthForcek_GC,
        #          normFiberLengthk_GC, fiberVelocityk_GC] = (
        #             f_hillEquilibrium(A_GC[:, k], lMTk_GC_lr, vMTk_GC_lr,
        #                               F_GC[:, k], FDt_GC[:, k]))
        #
        #     if stats['success'] == True:
        #         assert np.alltrue(
        #             np.abs(hillEquilibriumk_GC.full()) <= 10 ** (convergence_tolerance_IPOPT)), (
        #             "Error in Hill equilibrium")
        #
        #     normFiberLength_GC[:, k] = normFiberLengthk_GC.full().flatten()
        #     fiberVelocity_GC[:, k] = fiberVelocityk_GC.full().flatten()
        #
        #     ###################################################################
        #     # Get metabolic energy rate.
        #     if optimize_skeleton_scaling == True:
        #         [actHeatRatek_GC, mtnHeatRatek_GC,
        #          shHeatRatek_GC, mechWRatek_GC, _,
        #          metabolicEnergyRatek_GC] = f_metabolicsBhargava(
        #             A_GC[:, k], A_GC[:, k], normFiberLengthk_GC,
        #             fiberVelocityk_GC, activeFiberForcek_GC,
        #             passiveFiberForcek_GC, normActiveFiberLengthForcek_GC, muscle_scaling_vector_opt, model_mass_scaling_opt)
        #     else:
        #         [actHeatRatek_GC, mtnHeatRatek_GC,
        #          shHeatRatek_GC, mechWRatek_GC, _,
        #          metabolicEnergyRatek_GC] = f_metabolicsBhargava(
        #             A_GC[:, k], A_GC[:, k], normFiberLengthk_GC,
        #             fiberVelocityk_GC, activeFiberForcek_GC,
        #             passiveFiberForcek_GC, normActiveFiberLengthForcek_GC, muscle_scaling_vector_opt, model_mass_scaling_opt)
        #
        #     metERatePerMuscle[:, k:k + 1] = (
        #         metabolicEnergyRatek_GC.full())
        #
        #     # Sum over all muscles.
        #     metabolicEnergyRatek_allMuscles = np.sum(
        #         metabolicEnergyRatek_GC.full())
        #     actHeatRatek_allMuscles = np.sum(actHeatRatek_GC.full())
        #     mtnHeatRatek_allMuscles = np.sum(mtnHeatRatek_GC.full())
        #     shHeatRatek_allMuscles = np.sum(shHeatRatek_GC.full())
        #     mechWRatek_allMuscles = np.sum(mechWRatek_GC.full())
        #     actHeatRate_GC[:, k] = actHeatRatek_GC.full().flatten()
        #     mtnHeatRate_GC[:, k] = mtnHeatRatek_GC.full().flatten()
        #     shHeatRate_GC[:, k] = shHeatRatek_GC.full().flatten()
        #     mechWRate_GC[:, k] = mechWRatek_GC.full().flatten()
        #
        #     # Add basal rate.
        #     basalRatek = basal_coef * modelMass ** basal_exp
        #     tolMetERate[0, k] = (metabolicEnergyRatek_allMuscles + basalRatek)
        #     actHeatRate[0, k] = actHeatRatek_allMuscles
        #     mtnHeatRate[0, k] = mtnHeatRatek_allMuscles
        #     shHeatRate[0, k] = shHeatRatek_allMuscles
        #     mechWRate[0, k] = mechWRatek_allMuscles
        #
        #     # Integrate.
        # metERatePerMuscle_int = np.trapz(
        #     metERatePerMuscle, tgrid_GC)
        # tolMetERate_int = np.trapz(tolMetERate, tgrid_GC)
        # actHeatRate_int = np.trapz(actHeatRate, tgrid_GC)
        # mtnHeatRate_int = np.trapz(mtnHeatRate, tgrid_GC)
        # shHeatRate_int = np.trapz(shHeatRate, tgrid_GC)
        # mechWRate_int = np.trapz(mechWRate, tgrid_GC)
        #
        # # Total distance traveled.
        # distTraveled_GC = (Qs_GC_rad[joints.index('pelvis_tx'), -1] -
        #                    Qs_GC_rad[joints.index('pelvis_tx'), 0])
        # # Cost of transport (COT).
        # COT_GC = tolMetERate_int / modelMass / distTraveled_GC
        # COT_activation_GC = actHeatRate_int / modelMass / distTraveled_GC
        # COT_maintenance_GC = mtnHeatRate_int / modelMass / distTraveled_GC
        # COT_shortening_GC = shHeatRate_int / modelMass / distTraveled_GC
        # COT_mechanical_GC = mechWRate_int / modelMass / distTraveled_GC
        # COT_perMuscle_GC = metERatePerMuscle_int / modelMass / distTraveled_GC
        #
        # # %% Compute stride length and extract GRFs, GRMs, and joint torques
        # # over the entire gait cycle.
        # QsQds_opt_nsc_GC = np.zeros((nJoints * 2, number_of_mesh_intervals * 2))
        # QsQds_opt_nsc_GC[::2, :] = Qs_GC_rad[idxJoints4F, :]
        # QsQds_opt_nsc_GC[1::2, :] = Qds_GC_rad[idxJoints4F, :]
        # Qdds_GC_rad = Qdds_GC.copy()
        # Qdds_GC_rad[rotational_joint_indices_in_joints, :] = (
        #         Qdds_GC_rad[rotational_joint_indices_in_joints, :] * np.pi / 180)
        # F1_GC = np.zeros((Tj_temp.shape[0], number_of_mesh_intervals * 2))
        # for k_GC in range(number_of_mesh_intervals * 2):
        #     if optimize_skeleton_scaling == True:
        #         Tk_GC = F_scalefactors(ca.vertcat(QsQds_opt_nsc_GC[:, k_GC],
        #                                           Qdds_GC_rad[idxJoints4F, k_GC], scaling_vector_opt, modelMass))
        #     else:
        #         Tk_GC = F(ca.vertcat(QsQds_opt_nsc_GC[:, k_GC],
        #                              Qdds_GC_rad[idxJoints4F, k_GC]))
        #     F1_GC[:, k_GC] = Tk_GC.full().T
        # stride_length_GC = ca.norm_2(F1_GC[idxCalcOr3D_r, 0] -
        #                              F1_GC[idxCalcOr3D_r, -1]).full()[0][0]
        # GRF_GC = F1_GC[idxGRF, :]
        # GRM_GC = F1_GC[idxGRM, :]
        #
        # torques_GC = F1_GC[getJointIndices(list(F_map["residuals"].keys()), joints), :]
        #
        # passiveTorques_GC = np.zeros((number_of_limit_torque_joints, 2 * number_of_mesh_intervals))
        #
        # Qs_GC_rad = Qs_GC * np.pi / 180
        # Qds_GC_rad = Qds_GC * np.pi / 180
        # for k_GC in range(number_of_mesh_intervals * 2):
        #     for cj, joint in enumerate(limit_torque_joints):
        #         passiveTorques_GC[cj, k_GC] = f_limit_torque[joint](
        #             Qs_GC_rad[joints.index(joint), k_GC],
        #             Qds_GC_rad[joints.index(joint), k_GC])
        #
        # distTraveled_opt = (Qs_opt_nsc[joints.index('pelvis_tx'), -1] -
        #                     Qs_opt_nsc[joints.index('pelvis_tx'), 0])
        # averageSpeed = distTraveled_opt / finalTime_opt
        #
        # # %% Decompose optimal cost and check that the recomputed optimal cost
        # # matches the one from CasADi's stats.
        # # Missing matrix B, add manually (again in case only analyzing).
        # B = [-8.88178419700125e-16, 0.376403062700467, 0.512485826188421,
        #      0.111111111111111]
        # metabolicEnergyRateTerm_opt_all = 0
        # activationTerm_opt_all = 0
        # armExcitationTerm_opt_all = 0
        # jointAccelerationTerm_opt_all = 0
        # passiveTorqueTerm_opt_all = 0
        # reserveTerm_opt_all = 0
        # activationDtTerm_opt_all = 0
        # forceDtTerm_opt_all = 0
        # armAccelerationTerm_opt_all = 0
        # h_opt = finalTime_opt / number_of_mesh_intervals
        #
        #
        # for k in range(number_of_mesh_intervals):
        #     # States.
        #     akj_opt = (ca.horzcat(a_opt[:, k], a_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]))
        #     normFkj_opt = (
        #         ca.horzcat(normF_opt[:, k], normF_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]))
        #     normFkj_opt_nsc = (
        #             normFkj_opt * (scalingF.to_numpy().T * np.ones((1, polynomial_order + 1))))
        #     Qskj_opt = (
        #         ca.horzcat(Qs_opt[:, k], Qs_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]))
        #     Qskj_opt_nsc = (
        #             Qskj_opt * (scalingQs.to_numpy().T * np.ones((1, polynomial_order + 1))))
        #     Qdskj_opt = (
        #         ca.horzcat(Qds_opt[:, k], Qds_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]))
        #     Qdskj_opt_nsc = (
        #             Qdskj_opt * (scalingQds.to_numpy().T * np.ones((1, polynomial_order + 1))))
        #     # Controls.
        #     aDtk_opt = aDt_opt[:, k]
        #     aDtk_opt_nsc = aDt_opt_nsc[:, k]
        #     eArmk_opt = eArm_opt[:, k]
        #     # Slack controls.
        #     Qddsj_opt = Qdds_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]
        #     MTP_reserve_j_opt = MTP_reserve_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]
        #     Qddsj_opt_nsc = (
        #             Qddsj_opt * (scalingQdds.to_numpy().T * np.ones((1, polynomial_order))))
        #     normFDtj_opt = normFDt_col_opt[:, k * polynomial_order:(k + 1) * polynomial_order]
        #     normFDtj_opt_nsc = (
        #             normFDtj_opt * (scalingFDt.to_numpy().T * np.ones((1, polynomial_order))))
        #     # Qs and Qds are intertwined in external function.
        #     QsQdskj_opt_nsc = ca.DM(nJoints * 2, polynomial_order + 1)
        #     QsQdskj_opt_nsc[::2, :] = Qskj_opt_nsc
        #     QsQdskj_opt_nsc[1::2, :] = Qdskj_opt_nsc
        #     # Loop over collocation points.
        #     for j in range(polynomial_order):
        #         ###############################################################
        #         # Passive joint torques.
        #         passiveTorquesj_opt = np.zeros((number_of_limit_torque_joints, 1))
        #         for cj, joint in enumerate(limit_torque_joints):
        #             passiveTorquesj_opt[cj, 0] = f_limit_torque[joint](
        #                 Qskj_opt_nsc[joints.index(joint), j + 1],
        #                 Qdskj_opt_nsc[joints.index(joint), j + 1])
        #
        #
        #         ###############################################################
        #         # Extract muscle length and velocities
        #         if use_NeMu == True:
        #             # NeMu approximation
        #             NeMu_input_initial_guess = np.concatenate((np.reshape(scaling_vector_opt[:24], (24, 1)),
        #                                                        Qskj_opt_nsc[muscle_driven_joints_indices, j + 1],
        #                                                        Qdskj_opt_nsc[muscle_driven_joints_indices, j + 1]), 0)
        #
        #             NeMu_output_initial_guess = f_NeMu(NeMu_input_initial_guess)
        #             lMTj_opt_lr = np.array(NeMu_output_initial_guess[0])
        #             vMTj_opt_lr = np.array(NeMu_output_initial_guess[1])
        #             dM_NeMu = np.array(NeMu_output_initial_guess[2])
        #         else:
        #             # Polynomial approximations.
        #             # Left leg.
        #             Qsinj_opt_l = Qskj_opt_nsc[leftPolJointIdx, j + 1]
        #             Qdsinj_opt_l = Qdskj_opt_nsc[leftPolJointIdx, j + 1]
        #             [lMTj_opt_l, vMTj_opt_l, _] = f_polynomial(Qsinj_opt_l,
        #                                                        Qdsinj_opt_l)
        #             # Right leg.
        #             Qsinj_opt_r = Qskj_opt_nsc[rightPolJointIdx, j + 1]
        #             Qdsinj_opt_r = Qdskj_opt_nsc[rightPolJointIdx, j + 1]
        #             [lMTj_opt_r, vMTj_opt_r, _] = f_polynomial(Qsinj_opt_r,
        #                                                        Qdsinj_opt_r)
        #             # Both legs        .
        #             lMTj_opt_lr = ca.vertcat(lMTj_opt_l[leftPolMuscleIdx],
        #                                      lMTj_opt_r[rightPolMuscleIdx])
        #             vMTj_opt_lr = ca.vertcat(vMTj_opt_l[leftPolMuscleIdx],
        #                                      vMTj_opt_r[rightPolMuscleIdx])
        #
        #         ###############################################################
        #         # Derive Hill-equilibrium.
        #
        #         if optimize_skeleton_scaling == True:
        #             [hillEquilibriumj_opt, Fj_opt, activeFiberForcej_opt,
        #              passiveFiberForcej_opt, normActiveFiberLengthForcej_opt,
        #              normFiberLengthj_opt, fiberVelocityj_opt] = (
        #                 f_hillEquilibrium_optimize_scaling(
        #                     akj_opt[:, j + 1], lMTj_opt_lr, vMTj_opt_lr,
        #                     normFkj_opt_nsc[:, j + 1], normFDtj_opt_nsc[:, j], muscle_scaling_vector_opt,
        #                     model_mass_scaling_opt))
        #         else:
        #             [hillEquilibriumj_opt, Fj_opt, activeFiberForcej_opt,
        #              passiveFiberForcej_opt, normActiveFiberLengthForcej_opt,
        #              normFiberLengthj_opt, fiberVelocityj_opt] = (
        #                 f_hillEquilibrium(
        #                     akj_opt[:, j + 1], lMTj_opt_lr, vMTj_opt_lr,
        #                     normFkj_opt_nsc[:, j + 1], normFDtj_opt_nsc[:, j]))
        #
        #         ###############################################################
        #         # Get metabolic energy rate.
        #         if optimize_skeleton_scaling == True:
        #             [actHeatRatej_opt, mtnHeatRatej_opt,
        #              shHeatRatej_opt, mechWRatej_opt, _,
        #              metabolicEnergyRatej_opt] = f_metabolicsBhargava(
        #                 akj_opt[:, j + 1], akj_opt[:, j + 1],
        #                 normFiberLengthj_opt, fiberVelocityj_opt,
        #                 activeFiberForcej_opt, passiveFiberForcej_opt,
        #                 normActiveFiberLengthForcej_opt, muscle_scaling_vector_opt, model_mass_scaling_opt)
        #         else:
        #             [actHeatRatej_opt, mtnHeatRatej_opt,
        #              shHeatRatej_opt, mechWRatej_opt, _,
        #              metabolicEnergyRatej_opt] = f_metabolicsBhargava(
        #                 akj_opt[:, j + 1], akj_opt[:, j + 1],
        #                 normFiberLengthj_opt, fiberVelocityj_opt,
        #                 activeFiberForcej_opt, passiveFiberForcej_opt,
        #                 normActiveFiberLengthForcej_opt, muscle_scaling_vector_opt, model_mass_scaling_opt)
        #         ###############################################################
        #         # Cost function terms.
        #         activationTerm_opt = f_NMusclesSum2(akj_opt[:, j + 1])
        #         jointAccelerationTerm_opt = f_nNoArmJointsSum2(
        #             Qddsj_opt[idxNoArmJoints, j])
        #         passiveTorqueTerm_opt = f_nPassiveTorqueJointsSum2(
        #             passiveTorquesj_opt)
        #         reserveTerm_opt = f_nMTPreserveSum2(
        #             MTP_reserve_j_opt[:, j])
        #         activationDtTerm_opt = f_NMusclesSum2(aDtk_opt)
        #         forceDtTerm_opt = f_NMusclesSum2(normFDtj_opt[:, j])
        #         armAccelerationTerm_opt = f_nArmJointsSum2(
        #             Qddsj_opt[non_muscle_actuated_joints_indices_in_joints, j])
        #         armExcitationTerm_opt = f_nArmJointsSum2(eArmk_opt)
        #         metabolicEnergyRateTerm_opt = (
        #                 f_NMusclesSum2(metabolicEnergyRatej_opt) / modelMass)
        #
        #         metabolicEnergyRateTerm_opt_all += (
        #                 weights['metabolicEnergyRateTerm'] *
        #                 metabolicEnergyRateTerm_opt * h_opt * B[j + 1] /
        #                 distTraveled_opt)
        #         activationTerm_opt_all += (
        #                 weights['activationTerm'] * activationTerm_opt *
        #                 h_opt * B[j + 1] / distTraveled_opt)
        #         armExcitationTerm_opt_all += (
        #                 weights['armExcitationTerm'] * armExcitationTerm_opt *
        #                 h_opt * B[j + 1] / distTraveled_opt)
        #         jointAccelerationTerm_opt_all += (
        #                 weights['jointAccelerationTerm'] *
        #                 jointAccelerationTerm_opt * h_opt * B[j + 1] /
        #                 distTraveled_opt)
        #         passiveTorqueTerm_opt_all += (
        #                 weights['passiveTorqueTerm'] * passiveTorqueTerm_opt *
        #                 h_opt * B[j + 1] / distTraveled_opt)
        #         reserveTerm_opt_all += (
        #                 weights['passiveTorqueTerm'] * reserveTerm_opt *
        #                 h_opt * B[j + 1] / distTraveled_opt)
        #         activationDtTerm_opt_all += (
        #                 weights['controls'] * activationDtTerm_opt * h_opt *
        #                 B[j + 1] / distTraveled_opt)
        #         forceDtTerm_opt_all += (
        #                 weights['controls'] * forceDtTerm_opt * h_opt *
        #                 B[j + 1] / distTraveled_opt)
        #         armAccelerationTerm_opt_all += (
        #                 weights['controls'] * armAccelerationTerm_opt *
        #                 h_opt * B[j + 1] / distTraveled_opt)
        #
        # objective_terms = {
        #     "metabolicEnergyRateTerm": metabolicEnergyRateTerm_opt_all.full(),
        #     "activationTerm": activationTerm_opt_all.full(),
        #     "armExcitationTerm": armExcitationTerm_opt_all.full(),
        #     "jointAccelerationTerm": jointAccelerationTerm_opt_all.full(),
        #     "passiveTorqueTerm": passiveTorqueTerm_opt_all.full(),
        #     "reserveTerm": reserveTerm_opt_all.full(),
        #     "activationDtTerm": activationDtTerm_opt_all.full(),
        #     "forceDtTerm": forceDtTerm_opt_all.full(),
        #     "armAccelerationTerm": armAccelerationTerm_opt_all.full()}
        #
        # JAll_opt = (metabolicEnergyRateTerm_opt_all.full() +
        #             activationTerm_opt_all.full() +
        #             armExcitationTerm_opt_all.full() +
        #             jointAccelerationTerm_opt_all.full() +
        #             passiveTorqueTerm_opt_all.full() +
        #             reserveTerm_opt_all.full() +
        #             activationDtTerm_opt_all.full() +
        #             forceDtTerm_opt_all.full() +
        #             armAccelerationTerm_opt_all.full())
        #
        #
        #
        # if stats['success'] == True:
        #     assert np.alltrue(
        #         np.abs((JAll_opt[0][0]-100*averageSpeed**2)/1000 - stats['iterations']['obj'][-1])
        #         <= 1e-1), "decomposition cost"
        # # %% Write motion files for visualization in OpenSim GUI.
        # if writeMotionFiles:
        #     muscleLabels = [bothSidesMuscle + '/activation'
        #                     for bothSidesMuscle in bothSidesMuscles]
        #     labels = ['time'] + joints + muscleLabels
        #     data = np.concatenate((tgrid_GC.T, Qs_GC.T, A_GC.T), axis=1)
        #     from utilities import numpy2storage
        #
        #     numpy2storage(labels, data, os.path.join(path_results, 'motion.mot'))
        #
        #     # Compute center of pressure (COP) and free torque (freeT).
        #     from utilities import getCOP
        #
        #     COPr_GC, freeTr_GC = getCOP(GRF_GC[:3, :], GRM_GC[:3, :])
        #     COPl_GC, freeTl_GC = getCOP(GRF_GC[3:, :], GRM_GC[3:, :])
        #     COP_GC = np.concatenate((COPr_GC, COPl_GC))
        #     freeT_GC = np.concatenate((freeTr_GC, freeTl_GC))
        #     # Post-processing.
        #     GRF_GC_toPrint = np.copy(GRF_GC)
        #     COP_GC_toPrint = np.copy(COP_GC)
        #     freeT_GC_toPrint = np.copy(freeT_GC)
        #     idx_r = np.argwhere(GRF_GC_toPrint[1, :] < 30)
        #     for tr in range(idx_r.shape[0]):
        #         GRF_GC_toPrint[:3, idx_r[tr, 0]] = 0
        #         COP_GC_toPrint[:3, idx_r[tr, 0]] = 0
        #         freeT_GC_toPrint[:3, idx_r[tr, 0]] = 0
        #     idx_l = np.argwhere(GRF_GC_toPrint[4, :] < 30)
        #     for tl in range(idx_l.shape[0]):
        #         GRF_GC_toPrint[3:, idx_l[tl, 0]] = 0
        #         COP_GC_toPrint[3:, idx_l[tl, 0]] = 0
        #         freeT_GC_toPrint[3:, idx_l[tl, 0]] = 0
        #     grf_cop_Labels = [
        #         'r_ground_force_vx', 'r_ground_force_vy', 'r_ground_force_vz',
        #         'r_ground_force_px', 'r_ground_force_py', 'r_ground_force_pz',
        #         'l_ground_force_vx', 'l_ground_force_vy', 'l_ground_force_vz',
        #         'l_ground_force_px', 'l_ground_force_py', 'l_ground_force_pz']
        #     grmLabels = [
        #         'r_ground_torque_x', 'r_ground_torque_y', 'r_ground_torque_z',
        #         'l_ground_torque_x', 'l_ground_torque_y', 'l_ground_torque_z']
        #     GRFNames = ['GRF_x_r', 'GRF_y_r', 'GRF_z_r',
        #                 'GRF_x_l', 'GRF_y_l', 'GRF_z_l']
        #     grLabels = grf_cop_Labels + grmLabels
        #     labels = ['time'] + grLabels
        #     data = np.concatenate(
        #         (tgrid_GC.T, GRF_GC_toPrint[:3, :].T, COP_GC_toPrint[:3, :].T,
        #          GRF_GC_toPrint[3:, :].T, COP_GC_toPrint[3:, :].T,
        #          freeT_GC_toPrint.T), axis=1)
        #     numpy2storage(labels, data, os.path.join(path_results, 'GRF.mot'))
        #
        # # %% Save optimal trajectories for further analysis.
        # if saveOptimalTrajectories:
        #     if not os.path.exists(os.path.join(path_trajectories,
        #                                        'optimaltrajectories.npy')):
        #         optimaltrajectories = {}
        #     else:
        #         optimaltrajectories = np.load(
        #             os.path.join(path_trajectories,
        #                          'optimaltrajectories.npy'),
        #             allow_pickle=True)
        #         optimaltrajectories = optimaltrajectories.item()
        #     GC_percent = np.linspace(1, 100, 2 * number_of_mesh_intervals)
        #     optimaltrajectories[case] = {
        #         'coordinate_values': Qs_GC,
        #         'coordinate_speeds': Qds_GC,
        #         'coordinate_accelerations': Qdds_GC,
        #         'muscle_activations': A_GC,
        #         'arm_activations': aArm_GC,
        #         'joint_torques': torques_GC,
        #         'passive_joint_torques_limit_forces': passiveTorques_GC,
        #         'MTP_reserves': MTP_reserve_opt,
        #         'GRF': GRF_GC,
        #         'time': tgrid_GC,
        #         'norm_fiber_lengths': normFiberLength_GC,
        #         'fiber_velocity': fiberVelocity_GC,
        #         'joints': joints,
        #         'passive_torque_joints': limit_torque_joints,
        #         'muscles': bothSidesMuscles,
        #         'mtp_joints': mtpJoints,
        #         'GRF_labels': GRFNames,
        #         'COT': COT_GC[0],
        #         'COT_perMuscle': COT_perMuscle_GC,
        #         'GC_percent': GC_percent,
        #         'objective': stats['iterations']['obj'][-1],
        #         'objective_terms': objective_terms,
        #         'iter_count': stats['iter_count'],
        #         "stride_length": stride_length_GC,
        #         "maximal_isometric_force_original": muscle_parameters[0, :],
        #         "maximal_isometric_force_scaled": maximalIsometricForce_scaled,
        #         "muscle_volume_original":muscleVolume_original,
        #         "muscle_volume_scaled":muscleVolume_scaled,
        #         "muscle_length_scaling": muscle_scaling_vector_opt,
        #         "body_scaling": scaling_vector_opt,
        #         "model_mass": modelMass,
        #         "speed": averageSpeed}

