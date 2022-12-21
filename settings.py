import numpy as np
def getSettings():

    settings = {
        '1': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'activation_function': 'tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1e-4},

        '2': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'activation_function': 'tanh',
            'bounds_scaling_factors': [0.8 * np.ones((54, 1)), 1.2 * np.ones((54, 1))],
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1e-4},

        '3': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'activation_function': 'tanh',
            'bounds_scaling_factors': [np.ones((54, 1)), np.ones((54, 1))],
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1},

        '4': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'activation_function': 'tanh',
            'bounds_scaling_factors': [0.8 * np.ones((54, 1)), 1.2 * np.ones((54, 1))],
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1},

        '5': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': False,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 0.4,
            'stiffnessMtp': 40,
            'activation_function': 'tanh',
            'bounds_scaling_factors': 'case 4',
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1e-4},

        '6': {
            'model': 'Hamner_modified',
            'target_speed': 3.33,
            'enforce_target_speed': True,
            'tol': 3,
            'N': 50,
            'nThreads': 10,
            'modelMass': 75.2,
            'dampingMtp': 1.9,
            'stiffnessMtp': 25,
            'activation_function': 'tanh',
            'bounds_scaling_factors': 'case 2',
            'muscle_geometry': 'NeMu',
            'energy_cost_function_scaling': 1},



    }

    return settings
