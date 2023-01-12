import numpy as np

# 1 :: GEN - sprinting
# 2 :: SKEL_SPRINT - sprinting
# 3 :: GEN - marathon
# 4 :: SKEL_MARATHON - marathon
# 5 :: SKEL_MARATHON - sprint
# 6 :: SKEL_SPRINT - marathon

def getSettings():

    settings = {
        '1': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1e-4},

        '2': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [0.8 * np.ones((54, 1)), 1.2 * np.ones((54, 1))],
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1e-4},

        '3': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1},

        '4': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [0.8 * np.ones((54, 1)), 1.2 * np.ones((54, 1))],
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1},

        '5': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': 'case 4',
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1e-4},

        '6': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': 'case 2',
            'strength_training': False,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), np.ones((92, 1))],
            'energy_cost_function_scaling': 1},

        '7': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors':  [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': True,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), 1.2*np.ones((92, 1))],
            'energy_cost_function_scaling': 1},

        '8': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': True,
            'bounds_cross_sectional_area_scaling': [np.ones((92, 1)), 1.2*np.ones((92, 1))],
            'energy_cost_function_scaling':  1e-4},

        '9': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': True,
            'bounds_cross_sectional_area_scaling': 'case 8',
            'energy_cost_function_scaling': 1},

        '10': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 2,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'NeMu_folder': 'NeMu/tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'strength_training': True,
            'bounds_cross_sectional_area_scaling': 'case 7',
            'energy_cost_function_scaling': 1e-4},

    }

    return settings
