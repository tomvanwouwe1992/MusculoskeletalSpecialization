import os.path
import numpy as np
import pickle
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot
import opensim
from matplotlib.backends.backend_pdf import PdfPages
# from keras.models import Sequential
# from keras.layers import Dense

class MuscleInfo:
    def __init__(self, name, index, actuated_coordinate_names, actuated_coordinate_full_names,
                 actuated_coordinate_indices, actuated_body_names, actuated_body_indices,
                 maximal_isometric_force,
                 optimal_fiber_length, tendon_slack_length, pennation_at_optimal_fiber_length,
                 maximal_fiber_velocity, muscle_width, tendon_stiffness):
        self.name = name
        self.index = index
        self.actuated_coordinate_names = actuated_coordinate_names
        self.actuated_coordinate_full_names = actuated_coordinate_full_names
        self.actuated_coordinate_indices = actuated_coordinate_indices
        self.actuated_body_names = actuated_body_names
        self.actuated_body_indices = actuated_body_indices
        self.maximal_isometric_force = maximal_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_at_optimal_fiber_length = pennation_at_optimal_fiber_length
        self.maximal_fiber_velocity = maximal_fiber_velocity
        self.muscle_width = muscle_width
        self.tendon_stiffness = tendon_stiffness
        self.q_samples = []
        self.moment_arm_samples = []
        self.muscle_tendon_length_samples = []
        self.scaling_vector_samples = []
        self.NeMu_LMT_dM = []
        self.min_muscle_tendon_length_generic = []
        self.max_muscle_tendon_length_generic = []


def generateScaledModels(i, default_scale_tool_xml_name, scale_vector,
                         save_path, model_name):
    # import opensim
    opensim.Logger.setLevelString('error')
    ScaleTool_OS = opensim.ScaleTool(default_scale_tool_xml_name)
    ModelMaker_OS = ScaleTool_OS.getGenericModelMaker()
    ModelMaker_OS.setModelFileName(model_name + '.osim')

    # Get model scaler and scaleset - then adapt the weights in the scalesets
    ModelScaler_OS = ScaleTool_OS.getModelScaler()
    ScaleSet_OS = ModelScaler_OS.getScaleSet()

    # pelvis
    Scale_OS = ScaleSet_OS.get(0)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[0])
    newScaleFactors.set(1, scale_vector[1])
    newScaleFactors.set(2, scale_vector[2])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_l
    Scale_OS = ScaleSet_OS.get(1)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[3])
    newScaleFactors.set(1, scale_vector[4])
    newScaleFactors.set(2, scale_vector[5])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_l
    Scale_OS = ScaleSet_OS.get(2)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[6])
    newScaleFactors.set(1, scale_vector[7])
    newScaleFactors.set(2, scale_vector[8])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_l
    Scale_OS = ScaleSet_OS.get(3)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[9])
    newScaleFactors.set(1, scale_vector[10])
    newScaleFactors.set(2, scale_vector[11])
    Scale_OS.setScaleFactors(newScaleFactors)

    # calcn_l
    Scale_OS = ScaleSet_OS.get(4)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    # toes_l
    Scale_OS = ScaleSet_OS.get(5)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[15])
    newScaleFactors.set(1, scale_vector[16])
    newScaleFactors.set(2, scale_vector[17])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_r
    Scale_OS = ScaleSet_OS.get(6)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[18])
    newScaleFactors.set(1, scale_vector[19])
    newScaleFactors.set(2, scale_vector[20])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_r
    Scale_OS = ScaleSet_OS.get(7)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[21])
    newScaleFactors.set(1, scale_vector[22])
    newScaleFactors.set(2, scale_vector[23])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_r
    Scale_OS = ScaleSet_OS.get(8)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[24])
    newScaleFactors.set(1, scale_vector[25])
    newScaleFactors.set(2, scale_vector[26])
    Scale_OS.setScaleFactors(newScaleFactors)

    # calcn_r
    Scale_OS = ScaleSet_OS.get(9)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[27])
    newScaleFactors.set(1, scale_vector[28])
    newScaleFactors.set(2, scale_vector[29])
    Scale_OS.setScaleFactors(newScaleFactors)

    # toes_r
    Scale_OS = ScaleSet_OS.get(10)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[30])
    newScaleFactors.set(1, scale_vector[31])
    newScaleFactors.set(2, scale_vector[32])
    Scale_OS.setScaleFactors(newScaleFactors)

    # torso
    Scale_OS = ScaleSet_OS.get(11)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[33])
    newScaleFactors.set(1, scale_vector[34])
    newScaleFactors.set(2, scale_vector[35])
    Scale_OS.setScaleFactors(newScaleFactors)

    # humerus_l
    Scale_OS = ScaleSet_OS.get(12)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[36])
    newScaleFactors.set(1, scale_vector[37])
    newScaleFactors.set(2, scale_vector[38])
    Scale_OS.setScaleFactors(newScaleFactors)

    # radioulnar_l
    Scale_OS = ScaleSet_OS.get(13)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[39])
    newScaleFactors.set(1, scale_vector[40])
    newScaleFactors.set(2, scale_vector[41])
    Scale_OS.setScaleFactors(newScaleFactors)

    # hand_l
    Scale_OS = ScaleSet_OS.get(14)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[42])
    newScaleFactors.set(1, scale_vector[43])
    newScaleFactors.set(2, scale_vector[44])
    Scale_OS.setScaleFactors(newScaleFactors)

    # humerus_r
    Scale_OS = ScaleSet_OS.get(15)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[45])
    newScaleFactors.set(1, scale_vector[46])
    newScaleFactors.set(2, scale_vector[47])
    Scale_OS.setScaleFactors(newScaleFactors)

    # radioulnar_r
    Scale_OS = ScaleSet_OS.get(16)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[48])
    newScaleFactors.set(1, scale_vector[49])
    newScaleFactors.set(2, scale_vector[50])
    Scale_OS.setScaleFactors(newScaleFactors)

    # hand_r
    Scale_OS = ScaleSet_OS.get(17)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[51])
    newScaleFactors.set(1, scale_vector[52])
    newScaleFactors.set(2, scale_vector[53])
    Scale_OS.setScaleFactors(newScaleFactors)


    ScaleTool_OS.setName('scaledModel_' + str(i))
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_withoutKneeTranslation_' + str(i) + '.osim')
    ModelScaler_OS.setOutputModelFileName(scaled_model_path)
    print('Scaling model #' + str(i))
    # Run the scale tool
    ScaleTool_OS.run()

    scaled_model_opensim = opensim.Model(scaled_model_path)

    knee_joint_l = scaled_model_opensim.getJointSet().get('knee_l')
    knee_joint_l_dc = opensim.CustomJoint_safeDownCast(knee_joint_l)
    knee_spatial_transform = knee_joint_l_dc.getSpatialTransform()

    # tranform axis translation 1
    TA_translation1 = knee_spatial_transform.getTransformAxis(3)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation1.get_function())
    TA_MultiplierFunction.setScale(scale_vector[3])
    knee_spatial_transform.updTransformAxis(3)
    # tranform axis translation 2
    TA_translation2 = knee_spatial_transform.getTransformAxis(4)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation2.get_function())
    TA_MultiplierFunction.setScale(scale_vector[4])
    knee_spatial_transform.updTransformAxis(4)
    # tranform axis translation 3
    TA_translation3 = knee_spatial_transform.getTransformAxis(5)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation3.get_function())
    TA_MultiplierFunction.setScale(scale_vector[5])
    knee_spatial_transform.updTransformAxis(5)
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_' + str(i) + '.osim')
    scaled_model_opensim.printToXML(scaled_model_path)


    knee_joint_r = scaled_model_opensim.getJointSet().get('knee_r')
    knee_joint_r_dc = opensim.CustomJoint_safeDownCast(knee_joint_r)
    knee_spatial_transform = knee_joint_r_dc.getSpatialTransform()

    # tranform axis translation 1
    TA_translation1 = knee_spatial_transform.getTransformAxis(3)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation1.get_function())
    TA_MultiplierFunction.setScale(scale_vector[18])
    knee_spatial_transform.updTransformAxis(3)
    # tranform axis translation 2
    TA_translation2 = knee_spatial_transform.getTransformAxis(4)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation2.get_function())
    TA_MultiplierFunction.setScale(scale_vector[19])
    knee_spatial_transform.updTransformAxis(4)
    # tranform axis translation 3
    TA_translation3 = knee_spatial_transform.getTransformAxis(5)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation3.get_function())
    TA_MultiplierFunction.setScale(scale_vector[20])
    knee_spatial_transform.updTransformAxis(5)
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_' + str(i) + '.osim')
    scaled_model_opensim.printToXML(scaled_model_path)



def generateScaledModels_NeMu(i, default_scale_tool_xml_name, scale_vector, root_path,
                              save_path, model_name):
    # import opensim
    opensim.Logger.setLevelString('error')
    ScaleTool_OS = opensim.ScaleTool(default_scale_tool_xml_name)
    ModelMaker_OS = ScaleTool_OS.getGenericModelMaker()
    ModelMaker_OS.setModelFileName(model_name)

    # Get model scaler and scaleset - then adapt the weights in the scalesets
    ModelScaler_OS = ScaleTool_OS.getModelScaler()
    ScaleSet_OS = ModelScaler_OS.getScaleSet()

    # torso
    Scale_OS = ScaleSet_OS.get(11)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[0])
    newScaleFactors.set(1, scale_vector[1])
    newScaleFactors.set(2, scale_vector[2])
    Scale_OS.setScaleFactors(newScaleFactors)

    # pelvis
    Scale_OS = ScaleSet_OS.get(0)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[3])
    newScaleFactors.set(1, scale_vector[4])
    newScaleFactors.set(2, scale_vector[5])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_l
    Scale_OS = ScaleSet_OS.get(6)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[6])
    newScaleFactors.set(1, scale_vector[7])
    newScaleFactors.set(2, scale_vector[8])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_l
    Scale_OS = ScaleSet_OS.get(7)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[9])
    newScaleFactors.set(1, scale_vector[10])
    newScaleFactors.set(2, scale_vector[11])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_l - calcaneus_l - toes (same scaling values)
    Scale_OS = ScaleSet_OS.get(8)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(9)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(10)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    ScaleTool_OS.setName('scaledModel_' + str(i))
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_withoutKneeTranslation_' + str(i) + '.osim')
    ModelScaler_OS.setOutputModelFileName(scaled_model_path)
    print('Scaling model #' + str(i))
    # Run the scale tool
    ScaleTool_OS.run()

    scaled_model_opensim = opensim.Model(scaled_model_path)

    knee_joint_r = scaled_model_opensim.getJointSet().get('knee_l')
    knee_joint_r_dc = opensim.CustomJoint_safeDownCast(knee_joint_r)
    knee_spatial_transform = knee_joint_r_dc.getSpatialTransform()

    # tranform axis translation 1
    TA_translation1 = knee_spatial_transform.getTransformAxis(3)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation1.get_function())
    TA_MultiplierFunction.setScale(scale_vector[6])
    knee_spatial_transform.updTransformAxis(3)
    # tranform axis translation 2
    TA_translation2 = knee_spatial_transform.getTransformAxis(4)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation2.get_function())
    TA_MultiplierFunction.setScale(scale_vector[7])
    knee_spatial_transform.updTransformAxis(4)
    # tranform axis translation 3
    TA_translation3 = knee_spatial_transform.getTransformAxis(5)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation3.get_function())
    TA_MultiplierFunction.setScale(scale_vector[8])
    knee_spatial_transform.updTransformAxis(5)
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_' + str(i) + '.osim')
    scaled_model_opensim.printToXML(scaled_model_path)

def sampleMomentArmsMuscleTendonLengths(muscle, indices_of_included_coordinates, ROM_all, min_angle_all, max_angle_all, number_of_used_scaled_models, scale_vectors, save_path):
        import opensim
        opensim.Logger.setLevelString('error') # Do not print errors to console
        number_of_actuated_coordinates = len(muscle.actuated_coordinate_indices)

        if number_of_actuated_coordinates > 0:

            # We first get to LMT and dM since these only depend on q
            number_of_samples = 1
            for d in range(number_of_actuated_coordinates):
                actuated_coordinate_index_in_model = muscle.actuated_coordinate_indices[d]
                index_d = indices_of_included_coordinates.index(actuated_coordinate_index_in_model)
                # Take a sample every 1deg
                number_of_samples = int(number_of_samples * (ROM_all[index_d] / np.pi * 180))
            number_of_samples = int(number_of_samples / (np.power(7, (number_of_actuated_coordinates - 1))))
            if number_of_actuated_coordinates < 2:
                number_of_samples = number_of_samples * 4
            if number_of_samples < 20000:
                number_of_samples = 20000
            number_of_samples = int(number_of_samples / number_of_used_scaled_models) * number_of_used_scaled_models

            q_samples = np.zeros((number_of_samples, number_of_actuated_coordinates))
            moment_arm_samples = np.zeros((number_of_samples, number_of_actuated_coordinates))
            muscle_tendon_length_samples = np.zeros((number_of_samples, 1))
            scaling_vector_samples = np.zeros((number_of_samples, np.shape(scale_vectors)[1]))

            scaled_model_optimal_fiber_length_samples = np.zeros((number_of_samples, 1))
            scaled_model_tendon_slack_length_samples = np.zeros((number_of_samples, 1))
            scaled_model_maximal_fiber_velocity_samples = np.zeros((number_of_samples, 1))
            scaled_model_muscle_width_samples = np.zeros((number_of_samples, 1))
            scaled_model_tendon_stiffness_samples = np.zeros((number_of_samples, 1))

            for d in range(number_of_actuated_coordinates):
                actuated_coordinate_index_in_model = muscle.actuated_coordinate_indices[d]
                index_d = indices_of_included_coordinates.index(actuated_coordinate_index_in_model)
                q_samples[:, d] = ROM_all[index_d] * (np.random.random_sample(number_of_samples) - 0.5) + (
                        min_angle_all[index_d] + max_angle_all[index_d]) / 2
            print(muscle.name + ' data generation with ' + str(number_of_samples) + ' samples, for ' + str(
                number_of_used_scaled_models) + ' different models')
            # We are going to take samples for moment arms and muscle tendon lengths from differently scaled models
            # We generated 2000 randomly scaled models (with some correlation between different scaling variables)
            blocks = int(number_of_samples / number_of_used_scaled_models)
            for j in range(number_of_used_scaled_models):
                q_samples_block = q_samples[j * blocks:(j + 1) * blocks, :]
                model_name = 'ScaledModels/scaledModel_' + str(j) + '.osim'
                model_path = os.path.join(save_path, model_name)
                model_opensim = opensim.Model(model_path)
                muscles = model_opensim.getMuscles()
                opensim_muscle = muscles.get(muscle.name)
                muscle_tendon_length_samples_block, moment_arm_samples_block = get_mtu_length_and_moment_arm(model_opensim,
                                                                                                             muscle,
                                                                                                             q_samples_block)
                moment_arm_samples[j * blocks:(j + 1) * blocks, :] = moment_arm_samples_block
                muscle_tendon_length_samples[j * blocks:(j + 1) * blocks, :] = muscle_tendon_length_samples_block
                scaling_vector_samples[j * blocks:(j + 1) * blocks, :] = np.tile(scale_vectors[j, :], (blocks, 1))
                scaled_model_optimal_fiber_length_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getOptimalFiberLength()
                scaled_model_tendon_slack_length_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getTendonSlackLength()
                scaled_model_maximal_fiber_velocity_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getMaxContractionVelocity()
                scaled_model_muscle_width_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getOptimalFiberLength() * np.sin(
                    opensim_muscle.getPennationAngleAtOptimalFiberLength())
                scaled_model_tendon_stiffness_samples[j * blocks:(j + 1) * blocks, :] = 35

            muscle.q_samples = q_samples
            muscle.moment_arm_samples = moment_arm_samples
            muscle.muscle_tendon_length_samples = muscle_tendon_length_samples
            scaling_vector_indices = np.zeros(3*len(muscle.actuated_body_indices))
            for i in range(len(muscle.actuated_body_indices)):
                scaling_vector_indices[3 * i + 0] = 3 * muscle.actuated_body_indices[i] + 0
                scaling_vector_indices[3 * i + 1] = 3 * muscle.actuated_body_indices[i] + 1
                scaling_vector_indices[3 * i + 2] = 3 * muscle.actuated_body_indices[i] + 2
            muscle.scaling_vector_samples = scaling_vector_samples[:, scaling_vector_indices.astype(int)]

            file_pi = open(save_path + muscle.name, 'wb')
            pickle.dump(muscle, file_pi)
            file_pi.close()


def get_muscle_and_coordinate_information(model_os, bodies_scaling_list):
    state = model_os.initSystem()
    model_os.equilibrateMuscles(state)

    coordinate_set = model_os.getCoordinateSet()
    number_of_coordinates = coordinate_set.getSize()

    coordinate_names = []
    coordinate_full_names = []
    coordinate_indices_in_model = []
    for i in range(0, number_of_coordinates):
        coordinate_name = coordinate_set.get(i).getName()
        name_joint_of_coordinate_of_interest_os = coordinate_set.get(i).getJoint().getName()
        if np.char.endswith(coordinate_name, '_l') or not np.char.endswith(coordinate_name, '_r') and not np.char.startswith(coordinate_name, 'pelvis_'):
            coordinate_names.append(coordinate_name)
            coordinate_indices_in_model.append(i)
            coordinate_full_names.append(name_joint_of_coordinate_of_interest_os + '/' + coordinate_name)
    muscle_set = model_os.getMuscles()
    number_of_muscles = muscle_set.getSize()

    included_muscles = []
    for m in range(0, number_of_muscles):
        muscle = model_os.getMuscles().get(m)
        muscle_name = muscle.getName()
        if np.char.endswith(muscle_name, '_l'):
            actuated_coordinate_names_for_this_muscle = []
            actuated_coordinate_full_names_for_this_muscle = []
            index_of_actuated_coordinate_for_this_muscle = []
            actuated_body_names_for_this_muscle = []
            for c in range(len(coordinate_indices_in_model)):
                index_of_coordinate_in_model_coordinates = coordinate_indices_in_model[c]
                coordinate_of_interest_os = model_os.getCoordinateSet().get(index_of_coordinate_in_model_coordinates)
                name_joint_of_coordinate_of_interest_os = coordinate_of_interest_os.getJoint().getName()
                moment_arm = muscle.computeMomentArm(state, coordinate_of_interest_os)
                if abs(moment_arm) > 0.0001:
                    actuated_coordinate_names_for_this_muscle.append(coordinate_of_interest_os.getName())
                    full_name_actuated_coordinate = name_joint_of_coordinate_of_interest_os + '/' + coordinate_of_interest_os.getName()
                    actuated_coordinate_full_names_for_this_muscle.append(full_name_actuated_coordinate)
                    index_of_actuated_coordinate_for_this_muscle.append(index_of_coordinate_in_model_coordinates)
                    joint_os = coordinate_of_interest_os.getJoint()
                    child_frame = joint_os.getChildFrame().getName()
                    if child_frame.endswith('_offset'):
                        child_frame = child_frame[:-7]
                    parent_frame = joint_os.getParentFrame().getName()
                    if parent_frame.endswith('_offset'):
                        parent_frame = parent_frame[:-7]
                    if child_frame not in actuated_body_names_for_this_muscle:
                        actuated_body_names_for_this_muscle.append(child_frame)
                    if parent_frame not in actuated_body_names_for_this_muscle:
                        actuated_body_names_for_this_muscle.append(parent_frame)

            actuated_body_indices_for_this_muscle = []
            for i in range(len(actuated_body_names_for_this_muscle)):
                for j in range(len(bodies_scaling_list)):
                    if actuated_body_names_for_this_muscle[i] in bodies_scaling_list[j]:
                        actuated_body_indices_for_this_muscle.append(j)
            actuated_body_indices_for_this_muscle = list(set(actuated_body_indices_for_this_muscle))
            actuated_body_indices_for_this_muscle.sort()

            included_muscle = MuscleInfo(muscle_name, m,
                                         actuated_coordinate_names_for_this_muscle,
                                         actuated_coordinate_full_names_for_this_muscle,
                                         index_of_actuated_coordinate_for_this_muscle,
                                         actuated_body_names_for_this_muscle,
                                         actuated_body_indices_for_this_muscle,
                                         muscle.getMaxIsometricForce(),
                                         muscle.getOptimalFiberLength(),
                                         muscle.getTendonSlackLength(),
                                         muscle.getPennationAngleAtOptimalFiberLength(),
                                         muscle.getMaxContractionVelocity(),
                                         muscle.getOptimalFiberLength() * np.sin(
                                             muscle.getPennationAngleAtOptimalFiberLength()),
                                         35)
            included_muscles.append(included_muscle)

    return included_muscles, coordinate_indices_in_model, coordinate_names, coordinate_full_names

def get_mtu_length_and_moment_arm(model_os, muscle, q):
    number_of_samples = np.shape(q)[0]
    actuated_coordinate_indices = muscle.actuated_coordinate_indices
    trajectory_mtu_length = np.zeros((number_of_samples, 1))
    trajectory_moment_arm = np.zeros((number_of_samples, len(actuated_coordinate_indices)))
    state = model_os.initSystem()
    for i in range(number_of_samples):
        for j in range(len(actuated_coordinate_indices)):
            actuated_coordinate_name = muscle.actuated_coordinate_full_names[j]
            state_name_q = '/jointset/' + actuated_coordinate_name + '/value'
            value_q = q[i, j]
            model_os.setStateVariableValue(state, state_name_q, value_q)
        state = model_os.updWorkingState()
        model_os.realizePosition(state)

        muscle_os = model_os.getMuscles().get(muscle.index)

        trajectory_mtu_length[i] = muscle_os.getLength(state)
        coordinate_set = model_os.getCoordinateSet()
        for k in range(len(actuated_coordinate_indices)):
            coordinate = coordinate_set.get(actuated_coordinate_indices[k])
            trajectory_moment_arm[i, k] = muscle_os.computeMomentArm(state,
                                                                     coordinate)
    return trajectory_mtu_length, trajectory_moment_arm


def trainNeMu_Geometry_efficient(muscle_name, save_path, activation_function):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # '-1'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(save_path + muscle_name, 'rb') as f:
        muscle = pickle.load(f)
    in_full = np.concatenate((muscle.scaling_vector_samples, muscle.q_samples), axis=1)
    out_full = np.concatenate((muscle.moment_arm_samples, muscle.muscle_tendon_length_samples), axis=1)


    # We remove feature-label combinations that are out of distribution and potentially erroneous
    to_keep = np.ones((np.size(out_full,0),))
    for i in range(np.size(out_full,1)):
        labels_i = out_full[:,i]
        mean = np.mean(labels_i)
        std = np.std(labels_i)
        zero_based = abs(labels_i - mean)
        max_deviations = 3
        to_keep = to_keep * (zero_based < max_deviations * std)

    to_keep = to_keep == 1

    in_full = in_full[to_keep, :]
    out_full = out_full[to_keep, :]

    permutation = np.random.permutation(np.shape(in_full)[0])
    in_full = in_full[permutation, :]
    out_full = out_full[permutation, :]

    length = np.shape(in_full)[0]
    x_train = in_full[0:int(length * 0.9), :]
    x_test = in_full[int(length * 0.9):, :]

    y_train = out_full[0:int(length * 0.9), :]
    y_test = out_full[int(length * 0.9):, :]
    model = keras.models.Sequential()

    if len(muscle.actuated_coordinate_indices) == 1:
        model.add(keras.layers.Dense(8, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) == 2:
        model.add(keras.layers.Dense(8, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) == 3:
        model.add(keras.layers.Dense(16, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) > 3:
        model.add(keras.layers.Dense(16, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    out_normalization_range = np.amax(out_full, axis=0) - np.amin(out_full, axis=0)
    weights_array = 1 / out_normalization_range

    n_epoch = 1000
    model.compile(loss='mean_squared_error', optimizer='adam', loss_weights=weights_array.tolist())
    history = model.fit(x_train, y_train, epochs=n_epoch, validation_split=0.10, batch_size=64)
    score = model.evaluate(x_test, y_test, batch_size=64)

    print(score)

    model.save(save_path + muscle.name + '_Geometry.h5')
    with open(save_path + muscle.name + '_Geometry_trainingHistory', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def testNeMu_Geometry_efficient(muscle_name, save_path, pp):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # '-1'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with open(save_path + muscle_name, 'rb') as f:
        muscle = pickle.load(f)
    in_full = np.concatenate((muscle.scaling_vector_samples, muscle.q_samples), axis=1)
    out_full = np.concatenate((muscle.moment_arm_samples, muscle.muscle_tendon_length_samples), axis=1)

    # We remove feature-label combinations that are out of distribution and potentially erroneous
    to_keep = np.ones((np.size(out_full,0),))
    for i in range(np.size(out_full,1)):
        labels_i = out_full[:,i]
        mean = np.mean(labels_i)
        std = np.std(labels_i)
        zero_based = abs(labels_i - mean)
        max_deviations = 3
        to_keep = to_keep * (zero_based < max_deviations * std)

    to_keep = to_keep == 1

    in_full = in_full[to_keep, :]
    out_full = out_full[to_keep, :]


    from tensorflow.keras.models import load_model
    model = load_model(save_path + muscle.name + '_Geometry.h5', compile=True)
    model_prediction = model.predict(in_full)
    error = model_prediction - out_full
    error = abs(error)

    for k in range(0,np.size(error,1)):
        # Get indices of data samples that produce largest absolute error
        if k < np.size(error,1) - 1:
            title = "moment arm of " + muscle_name + " wrt " + muscle.actuated_coordinate_names[k]
        else:
            title = "length of " + muscle_name
        corr_matrix = np.corrcoef(out_full[:, k], model_prediction[:, k])
        corr = corr_matrix[0, 1]
        R_sq = corr ** 2
        RMS_error =  np.sqrt(np.mean(error[:,k]**2))
        partition_size = int(np.size(error,0)/50)
        ordered_indices = np.flip(np.argsort(np.abs(error[:,k])))[:partition_size]
        largest_errors = error[ordered_indices,k]

        nr_skeleton_indices = 3*len(muscle.actuated_body_indices)
        pyplot.figure(figsize=(6,9), dpi=10)
        pyplot.subplot(3,1,1)
        pyplot.scatter(out_full[:1000, k], model_prediction[:1000, k])
        pyplot.title('label vs prediction: R^2 = ' + str(np.round_(R_sq,3)) + ' - RMS = ' + str(np.round_(100*RMS_error,3)) + 'cm', fontsize=10)
        pyplot.subplot(3,1,2)
        pyplot.hist(error[:, k], weights=np.ones(len(error[:, k])) / len(error[:, k]))
        pyplot.xlabel('error [m]', fontsize=8)
        pyplot.title('error distribution', fontsize=10)
        pyplot.subplot(3,1,3)
        pyplot.hist(largest_errors, weights=np.ones(len(largest_errors)) / len(largest_errors))
        pyplot.xlabel('error [m]', fontsize=8)
        pyplot.title('error distribution of 2% largest errors', fontsize=10)
        pyplot.suptitle(title, fontsize=12)
        pyplot.tight_layout()
        pp.savefig()



def conversionNeMu(activation_function,root_path):
    from konverter import Konverter
    from tensorflow.keras.models import load_model
    from sys import path
    path.append(r"C:\Program Files\casadi-windows-py37-v3.5.5-64bit")
    global ca


    save_path = root_path + '/Models/NeMu/' + activation_function + '/'

    path_NeMu_models = save_path
    list_model_names = os.listdir(path_NeMu_models)

    for i in range(len(list_model_names)):
        model_name = list_model_names[i]
        if model_name[-2:] == 'h5':
            print(model_name)
            model = load_model(path_NeMu_models + '/' + model_name, compile = False)
            Konverter(model, output_file=save_path + '/' + model_name[:-3] + '_Konverted')# creates the numpy model from the keras model

def NeMuApproximation(muscle_names, joint_names, body_names, NeMu_folder):
    import pickle
    import imp, os
    import casadi as ca

    NeMu_input = ca.SX.sym("NeMu_input", 1, len(joint_names) * 2 + len(body_names) * 3)
    scaling_in = NeMu_input[0,0:3*len(body_names)]
    q_in = NeMu_input[0,3*len(body_names):3*len(body_names)+len(joint_names)]
    qdot_in = NeMu_input[0,3*len(body_names)+len(joint_names):]

    lMT = ca.SX(len(muscle_names), 1)
    vMT = ca.SX(len(muscle_names), 1)
    dM = ca.SX(len(muscle_names), len(joint_names))

    list_modules = os.listdir(NeMu_folder)
    for muscle_index, muscle_name in enumerate(muscle_names):
        module_name = []

        # Check if we are evaluating for a right-side muscle
        right_side_muscle = False
        if muscle_name[-1] == 'r':
            right_side_muscle = True
        muscle_name = muscle_name[:-1] + 'l'

        # find the numpy module for the selected muscle left side equivalent
        for j in range(len(list_modules)):
            module_name_test = list_modules[j]
            if muscle_name[:-2] == module_name_test[:len(muscle_name[:-2])] and module_name_test[-2:] == 'py':
                module_name = module_name_test
                break
        if module_name == []:
            print('issue finding python version of the muscle')

        # load the numpy module
        foo = imp.load_source('module', NeMu_folder + os.sep + module_name)


        # load the information of the muscle (which dofs/bodies are actuated etc.)
        NeMu_info = NeMu_folder + '/' + muscle_name
        with open(NeMu_info, 'rb') as f:
            NeMu_info = pickle.load(f)


        # select from input to the specific neural network from the full model input
        ## scaling factor indices
        scaling_factor_indices = []
        for j in range(len(NeMu_info.actuated_body_indices)):
            body_name = NeMu_info.actuated_body_names[j]
            if right_side_muscle == True and body_name[-1] == 'l':
                body_name = body_name[:-1] + 'r'
            index = body_names.index(body_name)
            scaling_factor_indices.append(3 * index)
            scaling_factor_indices.append(3 * index + 1)
            scaling_factor_indices.append(3 * index + 2)


        ## joint position/velocity indices
        ## flip input sign? - we have build models for left side muscles. To get the values for right side muscles we can just evaluate these functions but then for the input of the corresponding right side coordinates.
        ## This is true except for lumbar bending and rotation, where we need to sign of input of the coordinates (note that here there is no corresponding right side coordinate)

        flip_input = ca.SX(1, len(NeMu_info.actuated_coordinate_names))
        q_indices = []
        for j, coordinate_name in enumerate(NeMu_info.actuated_coordinate_names):
            if right_side_muscle == True and coordinate_name[-1] == 'l':
                coordinate_name = coordinate_name[:-1] + 'r'
            index = joint_names.index(coordinate_name)
            q_indices.append(index)
            if (coordinate_name == 'lumbar_bending' or coordinate_name == 'lumbar_rotation') and right_side_muscle == True:
                flip_input[j] = -1
            else:
                flip_input[j] = 1
        qdot_indices = q_indices


        # Input to the NumPy function
        q_in_muscle = flip_input * q_in[0,q_indices]
        qdot_in_muscle = qdot_in[0,qdot_indices]
        scaling_in_muscle = scaling_in[0,scaling_factor_indices]

        #Compute output: moment arms, muscle length
        ## flip_output - if the momentarms are about lumbar bending and rotation we also need to flip their signs when it is about right side muscles. Muscle lenghts outputs stay preserverd (always positive)
        flip_output = ca.horzcat(flip_input, 1)
        network_input = ca.horzcat(scaling_in_muscle, q_in_muscle)
        network_output = flip_output * foo.predict(network_input)

        #lMT, dM
        lMT[muscle_index, 0] = network_output[-1]
        dM[muscle_index, q_indices] = network_output[:-1]

        # vMT
        vMT[muscle_index, 0] = - ca.dot(dM[muscle_index, q_indices], qdot_in_muscle)

    f_NeMu = ca.Function('f_NeMu', [NeMu_input], [lMT, vMT, dM])
    return f_NeMu


