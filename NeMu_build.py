import opensim
import numpy as np
import os.path
import pickle
import multiprocessing
import NeMu_subfunctions
from NeMu_subfunctions import get_muscle_and_coordinate_information
from NeMu_subfunctions import testNeMu_Geometry_efficient

if __name__ == '__main__':

    #####################################################################
    # FLOW CONTROL
    #####################################################################
    generate_randomly_scaled_models = True
    generate_ground_truth_data_all_muscles = True
    train_model = True
    convert_to_NumPy = True
    build_NeMu_CasADi = True
    activation_function = 'tanh'
    model_name = 'Hamner_modified.osim'
    number_of_randomly_scaled_models = 2000
    root_path = os.path.dirname(os.path.abspath('NeMu_build.py'))
    model_path = root_path + '/Models/'
    save_path = root_path + '/Models/NeMu_accurate/' + activation_function + '/'

    # list of bodies to be scaled - bodies with identical scaling factors should be grouped in a sublist
    bodies_scaling_list = ["torso",
                           "pelvis",
                           "femur_l",
                           "tibia_l",
                           ["talus_l", "calcn_l", "toes_l"]]

    # Import OpenSim
    os.add_dll_directory(r"C:/OpenSim 4.3/bin")
    opensim.Logger.setLevelString('error') # Do not print errors to console

    # load OpenSim model
    model_ = os.path.join(model_path, model_name)
    model_opensim = opensim.Model(model_)

    # extract information on the included muscles and coordinates from the opensim model (just look at left side of the model)
    included_muscles, indices_of_included_coordinates, coordinate_names, coordinate_full_names = get_muscle_and_coordinate_information(
        model_opensim, bodies_scaling_list)
    moment_arm_indices = []
    moment_arm_names = []
    muscle_names = []
    for i in range(len(included_muscles)):
        muscle = included_muscles[i]
        moment_arm_indices.append(muscle.actuated_coordinate_indices)
        moment_arm_names.append(muscle.actuated_coordinate_names)
        muscle_names.append(included_muscles[i].name)

    coordinate_names = coordinate_names[:10]
    coordinate_full_names = coordinate_full_names[:10]
    indices_of_included_coordinates = indices_of_included_coordinates[:10]
    ######################################################################
    # --- PART 1: Generate scaled models over which we will randomize
    ######################################################################
    if generate_randomly_scaled_models == True:
        # INPUT
        # - model name on which the approximation for moment arms and muscle tendon lengths will be based
        # - name of default scale tool .xml setup file
        # - number of randomly scaled models we want to consider
        # - properties of how scaling factors are distributed for the randomly scaled model
        default_scale_tool_xml_name = model_path + '/scaleTool_Default.xml'
        correlation_scaling_factors = 0.6
        standard_deviation_scaling_factors = 0.08

        # OUTPUT
        # - generate & store scaling vectors, generate & store scaled versions of the original model
        # number_of_bodies = model_opensim.getNumBodies()
        number_of_scaling_factors = 3*len(bodies_scaling_list)
        mean_scaling_factor = np.ones(number_of_scaling_factors)
        variance_scaling_factors = standard_deviation_scaling_factors**2
        covariance_scaling_factors = variance_scaling_factors*np.identity(number_of_scaling_factors)
        indices_off_diagonal = np.where(~np.eye(covariance_scaling_factors.shape[0],dtype=bool))
        covariance_scaling_factors[indices_off_diagonal] = correlation_scaling_factors*np.sqrt(variance_scaling_factors)**2


        # scale_vectors = np.random.multivariate_normal(mean_scaling_factor, covariance_scaling_factors, size=number_of_randomly_scaled_models)
        scale_vectors = np.random.uniform(low=0.8, high=1.2, size=(number_of_randomly_scaled_models,number_of_scaling_factors))
        iterable = []
        from NeMu_subfunctions import generateScaledModels_NeMu
        for i in range(0, number_of_randomly_scaled_models):
            iterable.append((i, default_scale_tool_xml_name, scale_vectors[i, :], root_path, save_path, model_name))
            # print(scale_vectors[i, :])
            # generateScaledModels(i, default_scale_tool_xml_name, scale_vectors[i, :], root_path, save_path, model_name)
        # Parallel generation of randomly scaled models
        pool = multiprocessing.Pool(processes=16)
        pool.starmap(NeMu_subfunctions.generateScaledModels_NeMu, iterable)
        pool.close()
        pool.join()
        # Save scale vectors to file
        file_pi = open(save_path + 'ScaledModels/scale_vectors', 'wb')
        pickle.dump(scale_vectors, file_pi)
        file_pi.close()
    else:
        print('Skipped generating randomly scaled models')


    ######################################################################
    # --- PART 2: Randomly sample moment arms and muscle tendon length for muscles included in the model
    ######################################################################
    if generate_ground_truth_data_all_muscles == True:
        with open(save_path + 'ScaledModels/scale_vectors', 'rb') as f:
            scale_vectors = pickle.load(f)
        # --- PART 2.a: Provide range of motion for different coordinates

        max_angle_hip_flexion = 70 * np.pi / 180
        min_angle_hip_flexion = -30 * np.pi / 180
        ROM_hip_flexion = max_angle_hip_flexion - min_angle_hip_flexion

        max_angle_hip_adduction = 20 * np.pi / 180
        min_angle_hip_adduction = -20 * np.pi / 180
        ROM_hip_adduction = max_angle_hip_adduction - min_angle_hip_adduction

        max_angle_hip_rotation = 35 * np.pi / 180
        min_angle_hip_rotation = -20 * np.pi / 180
        ROM_hip_rotation = max_angle_hip_rotation - min_angle_hip_rotation

        max_angle_knee = 10 * np.pi / 180
        min_angle_knee = -120 * np.pi / 180
        ROM_knee = max_angle_knee - min_angle_knee

        max_angle_ankle = 50 * np.pi / 180
        min_angle_ankle = -50 * np.pi / 180
        ROM_ankle = max_angle_ankle - min_angle_ankle

        max_angle_subtalar = 35 * np.pi / 180
        min_angle_subtalar = -35 * np.pi / 180
        ROM_subtalar = max_angle_subtalar - min_angle_subtalar

        max_angle_mtp = 60 * np.pi / 180
        min_angle_mtp = -10 * np.pi / 180
        ROM_mtp = max_angle_mtp - min_angle_mtp

        max_angle_lumbar_extension = 45 * np.pi / 180
        min_angle_lumbar_extension = -45 * np.pi / 180
        ROM_lumbar_extension = max_angle_lumbar_extension - min_angle_lumbar_extension

        max_angle_lumbar_bending = 45 * np.pi / 180
        min_angle_lumbar_bending = -45 * np.pi / 180
        ROM_lumbar_bending = max_angle_lumbar_bending - min_angle_lumbar_bending

        max_angle_lumbar_rotation = 45 * np.pi / 180
        min_angle_lumbar_rotation = -45 * np.pi / 180
        ROM_lumbar_rotation = max_angle_lumbar_rotation - min_angle_lumbar_rotation

        ROM_all = np.array(
            [ROM_hip_flexion, ROM_hip_adduction, ROM_hip_rotation, ROM_knee, ROM_ankle, ROM_subtalar, ROM_mtp, ROM_lumbar_extension, ROM_lumbar_bending, ROM_lumbar_rotation])
        min_angle_all = np.array(
            [min_angle_hip_flexion, min_angle_hip_adduction, min_angle_hip_rotation, min_angle_knee, min_angle_ankle,
             min_angle_subtalar, min_angle_mtp, min_angle_lumbar_extension, min_angle_lumbar_bending, min_angle_lumbar_rotation])
        max_angle_all = np.array(
            [max_angle_hip_flexion, max_angle_hip_adduction, max_angle_hip_rotation, max_angle_knee, max_angle_ankle,
             max_angle_subtalar, max_angle_mtp, max_angle_lumbar_extension, max_angle_lumbar_bending, max_angle_lumbar_rotation])

        if not len(ROM_all) == len(coordinate_names) or not len(min_angle_all) == len(coordinate_names) or not len(max_angle_all) == len(coordinate_names):
            raise Exception("Please provide ROM/min/max for all included dofs and in the following order: " + str(coordinate_names[:]))

        # --- PART 2.b: Sample throughout state space for different muscles (in parallel)

        number_of_samples = 20000
        q_samples = np.zeros((len(coordinate_names), number_of_samples))
        for d in range(len(coordinate_names)):
            q_samples[d, :] = ROM_all[d] * (np.random.random_sample(number_of_samples) - 0.5) + (
                    min_angle_all[d] + max_angle_all[d]) / 2


        iterable = []
        for i in range(len(included_muscles)):
            iterable.append((included_muscles[i], indices_of_included_coordinates, ROM_all, min_angle_all, max_angle_all, number_of_randomly_scaled_models, scale_vectors, save_path))

        # Parallel generation of randomly scaled models
        pool = multiprocessing.Pool(processes=16)
        pool.starmap(NeMu_subfunctions.sampleMomentArmsMuscleTendonLengths, iterable)
        pool.close()
        pool.join()

    ######################################################################
    # --- PART 3: Train NeMu's based on generated data - tensorflow models
    ######################################################################

    os.add_dll_directory(r"C:\OpenSim 4.3\bin")
    muscle_list = os.listdir(save_path)
    if train_model == True:
        iterable = []
        # NeMu_subfunctions.trainNeMu_Geometry_efficient(included_muscles[0].name, save_path, activation_function)
        for i in range(0,len(included_muscles)):
            iterable.append((included_muscles[i].name, save_path, activation_function))

        pool = multiprocessing.Pool(processes=16)
        pool.starmap(NeMu_subfunctions.trainNeMu_Geometry_accurate, iterable)
        pool.close()
        pool.join()
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages('test.pdf')
        for i in range(0,len(included_muscles)):
            testNeMu_Geometry_efficient(included_muscles[i].name, save_path, pp)
        pp.close()

    ######################################################################
    # --- PART 4: Convert tensorflow models to NumPy scripts
    ######################################################################
    from NeMu_subfunctions import conversionNeMu
    root_path = os.path.dirname(os.path.abspath('NeMu_build.py'))
    if convert_to_NumPy == True:
        conversionNeMu(activation_function, root_path)


    ##############################################################################################
    # --- PART 5: Build CasADi function to evaluate the musculoskeletal geometry for a whole model
    ##############################################################################################
    from NeMu_subfunctions import NeMuApproximation
    if build_NeMu_CasADi == True:
        bodies_muscles = ["torso",
                          "pelvis",
                          "femur_l",
                          "tibia_l",
                          "talus_l",
                          "calcn_l",
                          "talus_l",
                          "femur_r",
                          "tibia_r",
                          "talus_r",
                          "calcn_r",
                          "talus_r"
                          ]
        muscles = [
            'glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r',
            'glut_min2_r', 'glut_min3_r', 'semimem_r', 'semiten_r', 'bifemlh_r',
            'bifemsh_r', 'sar_r', 'add_long_r', 'add_brev_r', 'add_mag1_r',
            'add_mag2_r', 'add_mag3_r', 'tfl_r', 'pect_r', 'grac_r',
            'glut_max1_r', 'glut_max2_r', 'glut_max3_r', 'iliacus_r', 'psoas_r',
            'quad_fem_r', 'gem_r', 'peri_r', 'rect_fem_r', 'vas_med_r',
            'vas_int_r', 'vas_lat_r', 'med_gas_r', 'lat_gas_r', 'soleus_r',
            'tib_post_r', 'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'per_brev_r',
            'per_long_r', 'per_tert_r', 'ext_dig_r', 'ext_hal_r', 'ercspn_r',
            'intobl_r', 'extobl_r', 'ercspn_l', 'intobl_l', 'extobl_l']

        rightSideMuscles = muscles[:-3]
        leftSideMuscles = [muscle[:-1] + 'l' for muscle in rightSideMuscles]
        bothSidesMuscles = leftSideMuscles + rightSideMuscles
        muscle_driven_joints = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                                'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
                                'knee_angle_l', 'knee_angle_r',
                                'ankle_angle_l', 'ankle_angle_r',
                                'subtalar_angle_l', 'subtalar_angle_r',
                                'mtp_angle_l', 'mtp_angle_r',
                                'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
        NeMu_folder = root_path + '/Models/NeMu/' + activation_function + '/'
        f_NeMu = NeMuApproximation(bothSidesMuscles, muscle_driven_joints, bodies_muscles, NeMu_folder)
        input = np.concatenate((np.ones((36, 1)),np.zeros((17, 1)),np.ones((17, 1))))
        f_NeMu(input)


        # Time it takes to evaluate the function 1000 times
        import time
        start_time = time.time()
        for i in range(1000):
            [lMT, vMT, dM] = f_NeMu(input)
        print("--- %s seconds ---" % (time.time() - start_time))



        print(f_NeMu)
