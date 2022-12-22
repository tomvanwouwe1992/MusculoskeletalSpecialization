'''
    This script contains helper functions used in this project.
'''

# %% Import packages.
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import casadi as ca
import matplotlib.pyplot as plt
import opensim
import copy, os
import muscleData

def generate_full_gait_cycle_kinematics(path_results, joints, number_of_joints, number_of_mesh_intervals, finalTime_opt, Qs_opt, scaling_Q, rotational_joint_indices_in_joints, periodic_joints_indices_start_to_end_position_matching, periodic_opposite_joints_indices_in_joints):
    Qs_opt_part_2 = np.zeros((number_of_joints, number_of_mesh_intervals))
    Qs_opt_part_2[:, :] = Qs_opt[:, 1:]
    Qs_opt_part_2[periodic_joints_indices_start_to_end_position_matching[1], :] = Qs_opt[
                                                                                  periodic_joints_indices_start_to_end_position_matching[
                                                                                      0], 1:]
    Qs_opt_part_2[periodic_opposite_joints_indices_in_joints, :] = -Qs_opt[periodic_opposite_joints_indices_in_joints,
                                                                    1:]
    Qs_opt_part_2[joints.index('pelvis_tx'), :] = Qs_opt[joints.index('pelvis_tx'), 1:] + Qs_opt[
        joints.index('pelvis_tx'), -1]
    Qs_gait_cycle_opt = np.concatenate((Qs_opt, Qs_opt_part_2), 1)

    labels = ['time'] + joints
    tgrid_GC =  np.linspace(0, 2 * finalTime_opt[0], 2 * number_of_mesh_intervals + 1)
    Qs_gait_cycle_nsc = (
                Qs_gait_cycle_opt * (scaling_Q.to_numpy().T * np.ones((1, 2 * number_of_mesh_intervals + 1)))).T
    Qs_gait_cycle_nsc[:, rotational_joint_indices_in_joints] = 180 / np.pi * Qs_gait_cycle_nsc[:,
                                                                             rotational_joint_indices_in_joints]
    data = np.concatenate((np.reshape(tgrid_GC, (2 * number_of_mesh_intervals + 1, 1)), Qs_gait_cycle_nsc), axis=1)
    from utilities import numpy2storage

    numpy2storage(labels, data, os.path.join(path_results, 'motion.mot'))

    return Qs_gait_cycle_opt, tgrid_GC, Qs_gait_cycle_nsc

def get_results_opti(f_height, w_opt, number_of_muscles, number_of_mesh_intervals, polynomial_order, number_of_joints, number_of_non_muscle_actuated_joints, modelMass):
    finalTime_opt = w_opt[0]
    starti = 1
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

    aArm_col_opt = (np.reshape(
        w_opt[starti:starti + number_of_non_muscle_actuated_joints * (polynomial_order * number_of_mesh_intervals)],
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

    muscle_cross_section_multiplier_opt = (np.reshape(w_opt[starti:starti + 92 * (polynomial_order * number_of_mesh_intervals)],
                                            (polynomial_order * number_of_mesh_intervals, 92))).T
    starti = starti + 92 * polynomial_order * number_of_mesh_intervals

    MTP_reserve_col_opt = (np.reshape(w_opt[starti:starti + 2 * (polynomial_order * number_of_mesh_intervals)],
                                      (polynomial_order * number_of_mesh_intervals, 2))).T
    starti = starti + 2 * (polynomial_order * number_of_mesh_intervals)

    scaling_vector_opt = scaling_vector_opt[:, 0]
    muscle_scaling_vector_opt = muscle_scaling_vector_opt[:, 0]
    model_mass_scaling_opt = model_mass_scaling_opt[:, 0]
    muscle_cross_section_multiplier_opt = muscle_cross_section_multiplier_opt[:,0]
    model_mass_opt = modelMass * model_mass_scaling_opt
    model_height_opt = f_height(scaling_vector_opt)
    model_BMI_opt = model_mass_opt / (model_height_opt * model_height_opt)
    assert (starti == w_opt.shape[0]), "error when extracting results"

    return a_opt, a_col_opt, normF_opt, normF_col_opt, Qs_opt, Qs_col_opt, Qds_opt, Qds_col_opt, aArm_opt, \
           aArm_col_opt, aDt_opt, eArm_opt, normFDt_col_opt, Qdds_col_opt, MTP_reserve_col_opt, \
           scaling_vector_opt, muscle_scaling_vector_opt, model_mass_scaling_opt, model_mass_opt, model_height_opt, \
           model_BMI_opt, finalTime_opt, muscle_cross_section_multiplier_opt


def get_names_and_indices_of_joints_and_bodies(path_model, skeleton_scaling_bodies, bodies_skeleton_scaling_coupling):

    muscles = get_muscle_names(path_model)
    number_of_muscles = len(muscles)

    joints = get_joint_names(path_model)
    number_of_joints = len(joints)

    muscle_actuated_joints = get_muscle_actuated_joint_names(path_model)

    muscle_articulated_bodies = get_muscle_articulated_body_names(path_model)

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

    periodic_joints_indices_start_to_end_velocity_matching = periodic_indices_start_to_end_matching(
            periodic_joints, joints)

    periodic_joints.remove('pelvis_tx')
    periodic_joints_indices_start_to_end_position_matching = periodic_indices_start_to_end_matching(periodic_joints, joints)

    periodic_muscles = copy.deepcopy(muscles)
    periodic_muscles_indices_start_to_end_matching = periodic_indices_start_to_end_matching(periodic_muscles, muscles)

    periodic_actuators = copy.deepcopy(non_muscle_actuated_joints)
    periodic_actuators_indices_start_to_end_matching = periodic_indices_start_to_end_matching(periodic_actuators, non_muscle_actuated_joints)

    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies = [
        [muscle_articulated_body in skeleton_scaling_body for skeleton_scaling_body in skeleton_scaling_bodies].index(
            True) for muscle_articulated_body in muscle_articulated_bodies]
    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies = convert1Dto3Dindices(
        muscle_articulated_bodies_indices_in_skeleton_scaling_bodies)

    bodies_skeleton_scaling_coupling_indices = copy.deepcopy(bodies_skeleton_scaling_coupling)
    for i in range(len(bodies_skeleton_scaling_coupling_indices)):
        if type(bodies_skeleton_scaling_coupling_indices[i]) is list:
            for j in range(len(bodies_skeleton_scaling_coupling_indices[i])):
                bodies_skeleton_scaling_coupling_indices[i][j] = skeleton_scaling_bodies.index(
                    bodies_skeleton_scaling_coupling[i][j])
        else:
            bodies_skeleton_scaling_coupling_indices[i] = skeleton_scaling_bodies.index(
                bodies_skeleton_scaling_coupling[i])

    return muscles, number_of_muscles, joints, number_of_joints, muscle_actuated_joints, muscle_articulated_bodies, \
           muscle_actuated_joints_indices_in_joints, number_of_muscle_actuated_joints, rotational_joints, rotational_joint_indices_in_joints, \
           non_actuated_joints, non_actuated_joints_indices_in_joints, non_muscle_actuated_joints, non_muscle_actuated_joints_indices_in_joints, \
           number_of_non_muscle_actuated_joints, number_of_muscle_actuated_and_non_actuated_joints, muscle_actuated_and_non_actuated_joints_indices_in_joints, \
           mtpJoints, limit_torque_joints, number_of_limit_torque_joints, periodic_opposite_joints, periodic_opposite_joints_indices_in_joints, \
           periodic_joints, periodic_joints_indices_start_to_end_velocity_matching, periodic_joints_indices_start_to_end_position_matching, periodic_muscles,\
           periodic_muscles_indices_start_to_end_matching, periodic_actuators, periodic_actuators_indices_start_to_end_matching, \
           muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, bodies_skeleton_scaling_coupling_indices



def get_indices_external_function_outputs(map_external_function_outputs):
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

    return idxCalcOr_r, idxCalcOr_l, idxFemurOr_r, idxFemurOr_l, idxHandOr_r, idxHandOr_l, idxTibiaOr_r, idxTibiaOr_l, idxToesOr_r, idxToesOr_l, idxTorsoOr_r, idxTorsoOr_l, idxGRF, idxGRM

def get_skeletal_dynamics_outputs(map_external_function_outputs, f_linearPassiveTorque, f_linearPassiveMtpTorque, f_limit_torque, skeleton_dynamics_and_more, non_muscle_actuated_joints, joint_indices_in_external_function, number_of_joints, joints, mtpJoints, limit_torque_joints, polynomial_order, number_of_mesh_intervals, idxGRF, idxGRM, Qs_col_opt_nsc, Qds_col_opt_nsc, QsQds_col_opt_nsc, Qdds_col_opt_nsc, aArm_col_opt, MTP_reserve_col_opt, scaling_vector_opt, model_mass_opt, scalingArmA):
    muscle_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))


    Tj_test = skeleton_dynamics_and_more(
        ca.vertcat(QsQds_col_opt_nsc[:, 0], Qdds_col_opt_nsc[:, 0], scaling_vector_opt, model_mass_opt))
    Tj_col_opt = np.zeros((Tj_test.shape[0], polynomial_order * number_of_mesh_intervals))

    for i in range(polynomial_order * number_of_mesh_intervals):
        Tj_col_opt[:, i] = np.reshape(skeleton_dynamics_and_more(
            ca.vertcat(QsQds_col_opt_nsc[:, i], Qdds_col_opt_nsc[joint_indices_in_external_function, i],
                       scaling_vector_opt, model_mass_opt)), (Tj_test.shape[0],))

    GRF_col_opt = Tj_col_opt[idxGRF, :]
    GRM_col_opt = Tj_col_opt[idxGRM, :]

    generalized_forces_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    for i, joint in enumerate(joints):
        index = map_external_function_outputs['residuals'][joint]
        generalized_forces_col_opt[i, :] = Tj_col_opt[index, :]

    # generalized forces (joint torques) are the result of passive joint torques, limit joint torques, biological joint torques (active contribution +  passive contribution)
    passive_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    for joint in non_muscle_actuated_joints:
        passive_joint_torques = f_linearPassiveTorque(Qs_col_opt_nsc[joints.index(joint), :],
                                                      Qds_col_opt_nsc[joints.index(joint), :])
        passive_joint_torques_col_opt[joints.index(joint), :] = np.reshape(passive_joint_torques, (
        polynomial_order * number_of_mesh_intervals,))

    for joint in mtpJoints:
        passive_joint_torques = f_linearPassiveMtpTorque(Qs_col_opt_nsc[joints.index(joint), :],
                                                         Qds_col_opt_nsc[joints.index(joint), :])
        passive_joint_torques_col_opt[joints.index(joint), :] = np.reshape(passive_joint_torques, (
        polynomial_order * number_of_mesh_intervals,))

    limit_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    for joint in limit_torque_joints:
        limit_joint_torques = f_limit_torque[joint](Qs_col_opt_nsc[joints.index(joint), :],
                                                    Qds_col_opt_nsc[joints.index(joint), :])
        limit_joint_torques_col_opt[joints.index(joint), :] = np.reshape(limit_joint_torques,
                                                                         (polynomial_order * number_of_mesh_intervals,))

    for i, joint in enumerate(non_muscle_actuated_joints):
        index = joints.index(joint)
        muscle_joint_torques_col_opt[index, :] = aArm_col_opt[i, :] * scalingArmA.iloc[0]['arm_rot_r']

    for i, joint in enumerate(mtpJoints):
        index = joints.index(joint)
        muscle_joint_torques_col_opt[index, :] = MTP_reserve_col_opt[i, :]

    joint_torques_equilibrium_residual_col_opt = np.zeros(
        (number_of_joints, polynomial_order * number_of_mesh_intervals))
    for joint in joints:
        index = joints.index(joint)
        joint_torques_equilibrium_residual_col_opt[index, :] = generalized_forces_col_opt[index, :] - (
                    passive_joint_torques_col_opt[index, :] + limit_joint_torques_col_opt[index,
                                                              :] + muscle_joint_torques_col_opt[index, :])

    return Tj_col_opt, GRF_col_opt, GRM_col_opt, passive_joint_torques_col_opt, joint_torques_equilibrium_residual_col_opt, muscle_joint_torques_col_opt, limit_joint_torques_col_opt, passive_joint_torques_col_opt, generalized_forces_col_opt



def get_biomechanics_outputs(f_metabolicsBhargava, f_get_muscle_tendon_length_velocity_moment_arm, f_hillEquilibrium, joints, number_of_joints, number_of_muscles, number_of_mesh_intervals, muscle_actuated_joints_indices_in_joints, number_of_muscle_actuated_joints, polynomial_order, scaling_vector_opt, Qsin_col_opt, Qdsin_col_opt, a_col_opt, normF_nsc_col_opt, normFDt_nsc_col_opt, muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, muscle_scaling_vector_opt, model_mass_scaling_opt, muscle_cross_section_multiplier_opt):
    lMT_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
    vMT_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
    dM_col_opt = np.zeros(
        (number_of_muscles, number_of_muscle_actuated_joints, polynomial_order * number_of_mesh_intervals))
    muscle_active_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    muscle_passive_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    muscle_joint_torques_col_opt = np.zeros((number_of_joints, polynomial_order * number_of_mesh_intervals))
    biological_joint_torques_equilibrium_residual_col_opt = np.zeros(
        (number_of_joints, polynomial_order * number_of_mesh_intervals))
    active_muscle_force_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
    passive_muscle_force_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
    hillEquilibrium_residual_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))
    metabolicEnergyRate_col_opt = np.zeros((number_of_muscles, polynomial_order * number_of_mesh_intervals))

    for i in range(polynomial_order * number_of_mesh_intervals):
        NeMu_input_j = ca.vertcat(scaling_vector_opt[muscle_articulated_bodies_indices_in_skeleton_scaling_bodies],
                                  Qsin_col_opt[:, i], Qdsin_col_opt[:, i])
        [lMTj_lr, vMTj_lr, dMj] = f_get_muscle_tendon_length_velocity_moment_arm(NeMu_input_j)
        lMT_col_opt[:, i] = np.reshape(lMTj_lr, (number_of_muscles,))
        vMT_col_opt[:, i] = np.reshape(vMTj_lr, (number_of_muscles,))
        dM_col_opt[:, :, i] = dMj

        [hillEquilibriumj, Fj, activeFiberForcej, passiveFiberForcej,
         normActiveFiberLengthForcej, normFiberLengthj, fiberVelocityj, activeFiberForce_effectivej,
         passiveFiberForce_effectivej] = (
            f_hillEquilibrium(a_col_opt[:, i], lMTj_lr, vMTj_lr,
                              normF_nsc_col_opt[:, i], normFDt_nsc_col_opt[:, i], muscle_scaling_vector_opt,
                              model_mass_scaling_opt, muscle_cross_section_multiplier_opt))
        active_muscle_force_col_opt[:, i] = np.reshape(activeFiberForce_effectivej, (number_of_muscles,))
        passive_muscle_force_col_opt[:, i] = np.reshape(passiveFiberForce_effectivej, (number_of_muscles,))
        hillEquilibrium_residual_col_opt[:, i] = np.reshape(hillEquilibriumj, (number_of_muscles,))

        metabolicEnergyRate_col_opt[:, i] = np.reshape(f_metabolicsBhargava(
            a_col_opt[:, i], a_col_opt[:, i], normFiberLengthj, fiberVelocityj,
            activeFiberForcej, passiveFiberForcej,
            normActiveFiberLengthForcej, muscle_scaling_vector_opt, model_mass_scaling_opt, muscle_cross_section_multiplier_opt)[5], (number_of_muscles,))

        for j in range(len(muscle_actuated_joints_indices_in_joints)):
            joint = joints[muscle_actuated_joints_indices_in_joints[j]]
            if joint != 'mtp_angle_l' and joint != 'mtp_angle_r':
                muscle_active_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(
                    dMj[:, j] * activeFiberForce_effectivej)
                muscle_passive_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(
                    dMj[:, j] * passiveFiberForce_effectivej)
                muscle_joint_torques_col_opt[joints.index(joint), i] = ca.sum1(dMj[:, j] * Fj)
        biological_joint_torques_equilibrium_residual_col_opt[:, i] = muscle_joint_torques_col_opt[:, i] - (
                    muscle_passive_joint_torques_col_opt[:, i] + muscle_active_joint_torques_col_opt[:, i])


    return lMT_col_opt, vMT_col_opt, dM_col_opt, muscle_active_joint_torques_col_opt, muscle_passive_joint_torques_col_opt, muscle_joint_torques_col_opt, biological_joint_torques_equilibrium_residual_col_opt, active_muscle_force_col_opt, passive_muscle_force_col_opt, hillEquilibrium_residual_col_opt, metabolicEnergyRate_col_opt

def get_max_iso_torques(f_get_muscle_tendon_length_velocity_moment_arm, f_hillEquilibrium, joints, number_of_muscles, muscle_actuated_joints, muscle_actuated_joints_indices_in_joints, muscle_articulated_bodies_indices_in_skeleton_scaling_bodies, muscle_scaling_vector_opt, scaling_vector_opt, model_mass_scaling_opt, muscle_cross_section_multiplier_opt):
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
            [lMT_max_iso_torque, vMT_max_iso_torque,
             dM_max_iso_torque] = f_get_muscle_tendon_length_velocity_moment_arm(NeMu_input)

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
                                  model_mass_scaling_opt, muscle_cross_section_multiplier_opt))

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
                                  model_mass_scaling_opt, muscle_cross_section_multiplier_opt))
            maximal_isometric_torques[k, j] = ca.sum1(dM_max_iso_torque[:, i] * Ft_neutral)
            passive_isometric_torques[k, j] = ca.sum1(dM_max_iso_torque[:, i] * passiveEffectiveFiberForce_neutral)
    return Qs_max_iso_torque, maximal_isometric_torques, passive_isometric_torques, max_iso_torques_joints

def get_time_col_opt(finalTime_opt, number_of_mesh_intervals, polynomial_order):
    tau = ca.collocation_points(polynomial_order, 'radau')
    time_col_opt = np.zeros((1, polynomial_order * number_of_mesh_intervals))
    index = 0
    previous_time = 0
    h_opt = finalTime_opt / number_of_mesh_intervals
    for i in range(number_of_mesh_intervals):
        time_col_opt[0, index] = previous_time + tau[0] * h_opt
        time_col_opt[0, index + 1] = previous_time + tau[1] * h_opt
        time_col_opt[0, index + 2] = previous_time + tau[2] * h_opt
        previous_time = time_col_opt[0, index + 2]
        index = index + 3
    time_col_opt = np.reshape(time_col_opt, (polynomial_order * number_of_mesh_intervals,))
    return time_col_opt



def get_metabolic_energy_outcomes(metabolicEnergyRate_col_opt, time_col_opt, halfGC_length, model_mass_opt):
    basal_coef = 1.2
    basal_exp = 1
    basal_metabolic_rate_opt = basal_coef * model_mass_opt ** basal_exp

    basal_metabolic_energy_halfGC_opt = np.trapz(
        basal_metabolic_rate_opt * np.ones((len(time_col_opt),)), time_col_opt)
    absolute_muscle_metabolic_energy_halfGC_opt = np.trapz(np.sum(metabolicEnergyRate_col_opt, 0), time_col_opt)
    relative_muscle_metabolic_energy_halfGC_opt = absolute_muscle_metabolic_energy_halfGC_opt / model_mass_opt

    marathon_muscle_metabolic_energy_joule_opt = absolute_muscle_metabolic_energy_halfGC_opt * (42196 / halfGC_length)
    marathon_muscle_metabolic_energy_cal_opt = marathon_muscle_metabolic_energy_joule_opt / 4.183
    marathon_muscle_metabolic_energy_kcal_opt = marathon_muscle_metabolic_energy_joule_opt / 4183

    marathon_basal_metabolic_energy_joule_opt = basal_metabolic_energy_halfGC_opt * (42196 / halfGC_length)
    marathon_basal_metabolic_energy_cal_opt = marathon_basal_metabolic_energy_joule_opt / 4.183
    marathon_basal_metabolic_energy_kcal_opt = marathon_basal_metabolic_energy_joule_opt / 4183

    marathon_total_metabolic_energy_joule_opt = marathon_basal_metabolic_energy_joule_opt + marathon_muscle_metabolic_energy_joule_opt
    marathon_total_metabolic_energy_cal_opt = marathon_basal_metabolic_energy_cal_opt + marathon_muscle_metabolic_energy_cal_opt
    marathon_total_metabolic_energy_kcal_opt = marathon_basal_metabolic_energy_kcal_opt + marathon_muscle_metabolic_energy_kcal_opt

    marathon_total_metabolic_energy_perKG_joule_opt = marathon_total_metabolic_energy_joule_opt / model_mass_opt
    marathon_total_metabolic_energy_perKG_cal_opt = marathon_total_metabolic_energy_cal_opt / model_mass_opt
    marathon_total_metabolic_energy_perKG_kcal_opt = marathon_total_metabolic_energy_kcal_opt / model_mass_opt

    metabolic_energy_outcomes = {}

    metabolic_energy_outcomes['basal_metabolic_energy_halfGC_opt'] = basal_metabolic_energy_halfGC_opt
    metabolic_energy_outcomes[
        'absolute_muscle_metabolic_energy_halfGC_opt'] = absolute_muscle_metabolic_energy_halfGC_opt
    metabolic_energy_outcomes[
        'relative_muscle_metabolic_energy_halfGC_opt'] = relative_muscle_metabolic_energy_halfGC_opt

    metabolic_energy_outcomes['marathon_muscle_metabolic_energy_joule_opt'] = marathon_muscle_metabolic_energy_joule_opt
    metabolic_energy_outcomes['marathon_muscle_metabolic_energy_cal_opt'] = marathon_muscle_metabolic_energy_cal_opt
    metabolic_energy_outcomes['marathon_muscle_metabolic_energy_kcal_opt'] = marathon_muscle_metabolic_energy_kcal_opt

    metabolic_energy_outcomes['marathon_basal_metabolic_energy_joule_opt'] = marathon_basal_metabolic_energy_joule_opt
    metabolic_energy_outcomes['marathon_basal_metabolic_energy_cal_opt'] = marathon_basal_metabolic_energy_cal_opt
    metabolic_energy_outcomes['marathon_basal_metabolic_energy_kcal_opt'] = marathon_basal_metabolic_energy_kcal_opt

    metabolic_energy_outcomes['marathon_total_metabolic_energy_joule_opt'] = marathon_total_metabolic_energy_joule_opt
    metabolic_energy_outcomes['marathon_total_metabolic_energy_cal_opt'] = marathon_total_metabolic_energy_cal_opt
    metabolic_energy_outcomes['marathon_total_metabolic_energy_kcal_opt'] = marathon_total_metabolic_energy_kcal_opt

    metabolic_energy_outcomes[
        'marathon_total_metabolic_energy_perKG_joule_opt'] = marathon_total_metabolic_energy_perKG_joule_opt
    metabolic_energy_outcomes[
        'marathon_total_metabolic_energy_perKG_cal_opt'] = marathon_total_metabolic_energy_perKG_cal_opt
    metabolic_energy_outcomes[
        'marathon_total_metabolic_energy_perKG_kcal_opt'] = marathon_total_metabolic_energy_perKG_kcal_opt

    return metabolic_energy_outcomes


def convert1Dto3Dindices(muscle_articulated_bodies_indices_in_skeleton_scaling_bodies):
    muscle_articulated_bodies_indices_in_skeleton_scaling_bodies_new = []
    for i in range(len(muscle_articulated_bodies_indices_in_skeleton_scaling_bodies)):
        muscle_articulated_bodies_indices_in_skeleton_scaling_bodies_new.append(3*muscle_articulated_bodies_indices_in_skeleton_scaling_bodies[i])
        muscle_articulated_bodies_indices_in_skeleton_scaling_bodies_new.append(
            3 * muscle_articulated_bodies_indices_in_skeleton_scaling_bodies[i]+1)
        muscle_articulated_bodies_indices_in_skeleton_scaling_bodies_new.append(
            3 * muscle_articulated_bodies_indices_in_skeleton_scaling_bodies[i]+2)
    return muscle_articulated_bodies_indices_in_skeleton_scaling_bodies_new

def periodic_indices_start_to_end_matching(periodic_variables, variables):
    listA = []
    listB = []
    for periodic_variable in periodic_variables:
        if periodic_variable[-1] == 'l':
            listA.append(variables.index(periodic_variable))
            listB.append(variables.index(periodic_variable[:-1] + 'r'))
        elif periodic_variable[-1] == 'r':
            listA.append(variables.index(periodic_variable))
            listB.append(variables.index(periodic_variable[:-1] + 'l'))
        else:
            listA.append(variables.index(periodic_variable))
            listB.append(variables.index(periodic_variable))
    periodic_indices_start_to_end_matching = [listA , listB]
    return periodic_indices_start_to_end_matching


def get_muscle_names(opensim_model_name):
    opensim_model = opensim.Model(opensim_model_name)
    muscle_names = []
    for m in range(opensim_model.getMuscles().getSize()):
        muscle_names.append(opensim_model.getMuscles().get(m).getName())
    return muscle_names

def get_joint_names(opensim_model_name):
    opensim_model = opensim.Model(opensim_model_name)
    joint_names = []
    for j in range(opensim_model.getCoordinateSet().getSize()):
        joint_names.append(opensim_model.getCoordinateSet().get(j).getName())
    return joint_names

def get_muscle_actuated_joint_names(opensim_model_name):
    opensim_model = opensim.Model(opensim_model_name)

    state = opensim_model.initSystem()
    opensim_model.equilibrateMuscles(state)

    muscle_driven_joint_names = []
    for j in range(opensim_model.getCoordinateSet().getSize()):
        joint = opensim_model.getCoordinateSet().get(j)
        for m in range(opensim_model.getMuscles().getSize()):
            muscle = opensim_model.getMuscles().get(m)
            moment_arm = muscle.computeMomentArm(state, joint)
            if abs(moment_arm) > 0.0001:
                muscle_driven_joint_names.append(joint.getName())
                break
    return muscle_driven_joint_names

def get_muscle_articulated_body_names(opensim_model_name):
    opensim_model = opensim.Model(opensim_model_name)

    state = opensim_model.initSystem()
    opensim_model.equilibrateMuscles(state)

    muscle_articulated_body_names = []
    for j in range(opensim_model.getCoordinateSet().getSize()):
        coordinate = opensim_model.getCoordinateSet().get(j)
        for m in range(opensim_model.getMuscles().getSize()):
            muscle = opensim_model.getMuscles().get(m)
            moment_arm = muscle.computeMomentArm(state, coordinate)
            if abs(moment_arm) > 0.0001:
                joint = coordinate.getJoint()
                child_frame = joint.getChildFrame().getName()
                if child_frame.endswith('_offset'):
                    child_frame = child_frame[:-7]
                parent_frame = joint.getParentFrame().getName()
                if parent_frame.endswith('_offset'):
                    parent_frame = parent_frame[:-7]
                if child_frame not in muscle_articulated_body_names:
                    muscle_articulated_body_names.append(child_frame)
                if parent_frame not in muscle_articulated_body_names:
                    muscle_articulated_body_names.append(parent_frame)
    return muscle_articulated_body_names

# %% Storage file to numpy array.
# Found here: https://github.com/chrisdembia/perimysium/
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data
    
# %% Storage file to dataframe.
def storage2df(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Extract IK results from storage file.
def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    # Filter data    
    fs=1/np.mean(np.diff(Qs['time']))    
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

# %% Extract activations from storage file.
def getActivations(storage_file, muscles):
    # Extract data
    data = storage2numpy(storage_file)
    activations = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, muscle in enumerate(muscles):  
            activations.insert(count + 1, muscle, data[muscle + "activation"])              
                
    return activations

# %% Extract ground reaction forces from storage file.
def getGRF(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        GRFs.insert(count + 1, header, data[header])    
    
    return GRFs

# %% Extract ID results from storage file.
def getID(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        if ((header == 'pelvis_tx') or (header == 'pelvis_ty') or 
            (header == 'pelvis_tz')):
            out.insert(count + 1, header, data[header + '_force'])              
        else:
            out.insert(count + 1, header, data[header + '_moment'])    
    
    return out

# %% Extract from storage file.
def getFromStorage(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

# %% Compute ground reaction moments (GRM).
def getGRM_wrt_groundOrigin(storage_file, fHeaders, pHeaders, mHeaders):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame()    
    for count, fheader in enumerate(fHeaders):
        GRFs.insert(count, fheader, data[fheader])  
    PoAs = pd.DataFrame()    
    for count, pheader in enumerate(pHeaders):
        PoAs.insert(count, pheader, data[pheader]) 
    GRMs = pd.DataFrame()    
    for count, mheader in enumerate(mHeaders):
        GRMs.insert(count, mheader, data[mheader])  
        
    # GRT_x = PoA_y*GRF_z - PoA_z*GRF_y
    # GRT_y = PoA_z*GRF_x - PoA_z*GRF_z + T_y
    # GRT_z = PoA_x*GRF_y - PoA_y*GRF_x
    GRM_wrt_groundOrigin = pd.DataFrame(data=data['time'], columns=['time'])    
    GRM_wrt_groundOrigin.insert(1, mHeaders[0], PoAs[pHeaders[1]] * GRFs[fHeaders[2]]  - PoAs[pHeaders[2]] * GRFs[fHeaders[1]])
    GRM_wrt_groundOrigin.insert(2, mHeaders[1], PoAs[pHeaders[2]] * GRFs[fHeaders[0]]  - PoAs[pHeaders[0]] * GRFs[fHeaders[2]] + GRMs[mHeaders[1]])
    GRM_wrt_groundOrigin.insert(3, mHeaders[2], PoAs[pHeaders[0]] * GRFs[fHeaders[1]]  - PoAs[pHeaders[1]] * GRFs[fHeaders[0]])        
    
    return GRM_wrt_groundOrigin

# %% Compite center of pressure (COP).
def getCOP(GRF, GRM):
    
    COP = np.zeros((3, GRF.shape[1]))
    torques = np.zeros((3, GRF.shape[1]))
    GRF[1, :] = np.maximum(0.0001, GRF[1, :])
    COP[0, :] = GRM[2, :] / GRF[1, :]    
    COP[2, :] = -GRM[0, :] / GRF[1, :]
    
    torques[1, :] = GRM[1, :] - COP[2, :]*GRF[0, :] + COP[0, :]*GRF[2, :]
    
    return COP, torques

# %% Get indices from list.
def getJointIndices(joints, selectedJoints):
    
    jointIndices = []
    for joint in selectedJoints:
        jointIndices.append(joints.index(joint))
            
    return jointIndices

# %% Get moment arm indices.
def getMomentArmIndices(rightMuscles, leftPolynomialJoints,
                        rightPolynomialJoints, polynomialData):
         
    momentArmIndices = {}
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        leftPolynomialJoints[i], []).append(count)
    for count, muscle in enumerate(rightMuscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        rightPolynomialJoints[i], []).append(
                                count + len(rightMuscles))                
        
    return momentArmIndices


# %% Solve OCP using bounds instead of constraints.
def solve_with_bounds(opti, tolerance):
        # Get guess
        constraintViolation_guess = opti.debug.value(opti.g, opti.initial())
        guess = opti.debug.value(opti.x, opti.initial())
        # Sparsity pattern of the constraint Jacobian
        jac = ca.jacobian(opti.g, opti.x)
        sp = (ca.DM(jac.sparsity(), 1)).sparse()
        # Find constraints dependent on one variable
        is_single = np.sum(sp, axis=1)
        is_single_num = np.zeros(is_single.shape[0])
        for i in range(is_single.shape[0]):
            is_single_num[i] = np.equal(is_single[i, 0], 1)
        # Find constraints with linear dependencies or no dependencies
        is_nonlinear = ca.which_depends(opti.g, opti.x, 2, True)
        is_linear = [not i for i in is_nonlinear]
        is_linear_np = np.array(is_linear)
        is_linear_np_num = is_linear_np * 1
        # Constraints dependent linearly on one variable should become bounds
        is_simple = is_single_num.astype(int) & is_linear_np_num
        idx_is_simple = [i for i, x in enumerate(is_simple) if x]
        ## Find corresponding variables
        col_sort = np.argsort(np.nonzero(sp[idx_is_simple, :].T)[1])
        col = np.nonzero(sp[idx_is_simple, :].T)[0]
        col = col[col_sort]
        # Contraint values
        lbg = opti.lbg
        lbg = opti.value(lbg)
        ubg = opti.ubg
        ubg = opti.value(ubg)
        # Detect  f2(p)x+f1(p)==0
        # This is important if  you have scaled variables: x = 10*opti.variable()
        # with a constraint -10 < x < 10. Because in the reformulation we read out
        # the original variable and thus we need to scale the bounds appropriately.
        g = opti.g
        gf = ca.Function('gf', [opti.x, opti.p], [g[idx_is_simple, 0],
                                                  ca.jtimes(g[idx_is_simple, 0], opti.x,
                                                            np.ones((opti.nx, 1)))])
        [f1, f2] = gf(0, opti.p)
        f1 = (ca.evalf(f1)).full()  # maybe a problem here
        f2 = (ca.evalf(f2)).full()
        lb = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
        ub = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
        # Initialize bound vector
        lbx = -np.inf * np.ones((opti.nx))
        ubx = np.inf * np.ones((opti.nx))
        # Fill bound vector. For unbounded variables, we keep +/- inf.
        for i in range(col.shape[0]):
            lbx[col[i]] = np.maximum(lbx[col[i]], lb[i])
            ubx[col[i]] = np.minimum(ubx[col[i]], ub[i])

        lbx_ = lbx
        ubx_ = lbx
        lbx[col] = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
        ubx[col] = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
        # Updated constraint value vector
        not_idx_is_simple = np.delete(range(0, is_simple.shape[0]), idx_is_simple)
        new_g = g[not_idx_is_simple, 0]
        # Updated bounds
        llb = lbg[not_idx_is_simple]
        uub = ubg[not_idx_is_simple]

        prob = {'x': opti.x, 'f': opti.f, 'g': new_g}
        s_opts = {}
        s_opts["expand"] = False
        s_opts["ipopt.hessian_approximation"] = "limited-memory"
        s_opts["ipopt.mu_strategy"] = "adaptive"
        s_opts["ipopt.max_iter"] = 100000
        s_opts["ipopt.tol"] = 10 ** (-tolerance)
        s_opts["ipopt.constr_viol_tol"] = 10 ** (-5)
        s_opts["ipopt.linear_solver"] = "mumps"
        s_opts["ipopt.bound_push"] = 1e-32
        s_opts["ipopt.bound_frac"] = 1e-32
        s_opts["ipopt.slack_bound_push"] = 1e-32
        s_opts["ipopt.slack_bound_frac"] = 1e-32
        # s_opts["ipopt.nlp_scaling_method"] = "none"
        # s_opts["ipopt.derivative_test"] = "first-order"
        s_opts["ipopt.print_frequency_iter"] = 1
        solver = ca.nlpsol("solver", "ipopt", prob, s_opts)
        # Solve
        arg = {}
        arg["x0"] = guess
        # Bounds on x
        arg["lbx"] = lbx
        arg["ubx"] = ubx
        # Bounds on g
        arg["lbg"] = llb
        arg["ubg"] = uub
        sol = solver(**arg)
        # Extract and save results
        w_opt = sol['x'].full()
        stats = solver.stats()

        return w_opt, stats

# %% Solve OCP with constraints.
def solve_with_constraints(opti, tolerance):
    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 0,
              "tol": 10**(-tolerance)}
    opti.callback(lambda i: opti.debug.show_infeasibilities(0.3))
    p_opts = {"expand":False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()  
    
    return sol

# %% Write storage file from numpy array.    
def numpy2storage(labels, data, storage_file):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    f.write('name %s\n' %storage_file)
    f.write('datacolumns %d\n' %data.shape[1])
    f.write('datarows %d\n' %data.shape[0])
    f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()
    
# %% Interpolate dataframe and return numpy array.    
def interpolateDataFrame2Numpy(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)
    dataInterp = np.zeros([N, len(dataFrame.columns)])
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])
        dataInterp[:,i] = set_interp(tOut)
        
    return dataInterp    

# %% Interpolate dataframe.
def interpolateDataFrame(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)    
    dataInterp = pd.DataFrame() 
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])        
        dataInterp.insert(i, col, set_interp(tOut))
        
    return dataInterp

# %% Scale dataframe.
def scaleDataFrame(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, 
                                dataFrame[header] / scaling.iloc[0][header])
        
    return dataFrame_scaled

# %% Unscale dataframe.
def unscaleDataFrame2(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, 
                                dataFrame[header] * scaling.iloc[0][header])
        
    return dataFrame_scaled

# %% Plot variables against their bounds.
def plotVSBounds(y,lb,ub,title=''):    
    ny = np.floor(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny+1), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.hlines(lb[i,0],x[0],x[-1],'r')
            ax.hlines(ub[i,0],x[0],x[-1],'b')
    plt.show()
         
# %% Plot variables against their bounds, which might be time-dependent.
def plotVSvaryingBounds(y,lb,ub,title=''):    
    ny = np.floor(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny+1), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.plot(x,lb[i,:],'r')
            ax.plot(x,ub[i,:],'b')
    plt.show()
            
# %% Plot paraeters.
def plotParametersVSBounds(y,lb,ub,title='',xticklabels=[]):    
    x = np.linspace(1,y.shape[0],y.shape[0])   
    plt.figure()
    ax = plt.gca()
    ax.scatter(x,lb,c='r',marker='_')
    ax.scatter(x,ub,c='b',marker='_')
    ax.scatter(x,y,c='k')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels) 
    ax.set_title(title)
    
# %% Calculate number of subplots.
def nSubplots(N):
    
    ny_0 = (np.sqrt(N)) 
    ny = np.round(ny_0) 
    ny_a = int(ny)
    ny_b = int(ny)
    if (ny == ny_0) == False:
        if ny_a == 1:
            ny_b = N
        if ny < ny_0:
            ny_b = int(ny+1)
            
    return ny_a, ny_b

# %% Compute index initial contact from GRFs.
def getIdxIC_3D(GRF_opt, threshold):    
    idxIC = np.nan
    N = GRF_opt.shape[1]
    legIC = "undefined"    
    GRF_opt_rl = np.concatenate((GRF_opt[1,:], GRF_opt[4,:]))
    last_noContact = np.argwhere(GRF_opt_rl < threshold)[-1]
    if last_noContact == 2*N - 1:
        first_contact = np.argwhere(GRF_opt_rl > threshold)[0]
    else:
        first_contact = last_noContact + 1
    if first_contact >= N:
        idxIC = first_contact - N
        legIC = "left"
    else:
        idxIC = first_contact
        legIC = "right"
            
    return idxIC, legIC
      
# %% Compute RMSE.      
def getRMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# %% Compute RMSE normalized by signal range.
def getRMSENormMinMax(predictions, targets):    
    ROM = np.max(targets) - np.min(targets)    
    return (np.sqrt(((predictions - targets) ** 2).mean()))/ROM

# %% Compute RMSE normalized by standard deviation.
def getRMSENormStd(predictions, targets):    
    std = np.std(targets)
    return (np.sqrt(((predictions - targets) ** 2).mean()))/std

# %% Compute R2.
def getR2(predictions, targets):
    return (np.corrcoef(predictions, targets)[0,1])**2 

# %% Return some metrics.
def getMetrics(predictions, targets):
    r2 = np.zeros((predictions.shape[0]))
    rmse = np.zeros((predictions.shape[0]))
    rmseNormMinMax = np.zeros((predictions.shape[0]))
    rmseNormStd = np.zeros((predictions.shape[0]))
    for i in range(predictions.shape[0]):
        r2[i] = getR2(predictions[i,:], targets[i,:]) 
        rmse[i] = getRMSE(predictions[i,:],targets[i,:])  
        rmseNormMinMax[i] = getRMSENormMinMax(predictions[i,:],targets[i,:])   
        rmseNormStd[i] = getRMSENormStd(predictions[i,:],targets[i,:])        
    return r2, rmse, rmseNormMinMax, rmseNormStd

# %% Euler integration error.
def eulerIntegration(xk_0, xk_1, uk, delta_t):
    
    return (xk_1 - xk_0) - uk * delta_t

# %% Get initial contacts.
def getInitialContact(GRF_y, time, threshold):
    
    idxIC = np.argwhere(GRF_y >= threshold)[0]
    timeIC = time[idxIC]
    
    timeIC_round2 = np.round(timeIC, 2)
    idxIC_round2 = np.argwhere(time >= timeIC_round2)[0]
    
    return idxIC, timeIC, idxIC_round2, timeIC_round2    
