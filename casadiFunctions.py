'''
    This script contains several CasADi functions for use when setting up
    the optimal control problem.
'''

# %% Import packages.
import casadi as ca
import numpy as np

# %% CasADi function to approximate muscle-tendon lenghts, velocities,
# and moment arms based on joint positions and velocities.
def polynomialApproximation(musclesPolynomials, polynomialData, NPolynomial):    
    
    from polynomials import polynomials
    
    # Function variables.
    qin = ca.SX.sym('qin', 1, NPolynomial)
    qdotin  = ca.SX.sym('qdotin', 1, NPolynomial)
    lMT = ca.SX(len(musclesPolynomials), 1)
    vMT = ca.SX(len(musclesPolynomials), 1)
    dM = ca.SX(len(musclesPolynomials), NPolynomial)
    
    for count, musclePolynomials in enumerate(musclesPolynomials):
        
        coefficients = polynomialData[musclePolynomials]['coefficients']
        dimension = polynomialData[musclePolynomials]['dimension']
        order = polynomialData[musclePolynomials]['order']        
        spanning = polynomialData[musclePolynomials]['spanning']          
        
        polynomial = polynomials(coefficients, dimension, order)
        
        idxSpanning = [i for i, e in enumerate(spanning) if e == 1]        
        lMT[count] = polynomial.calcValue(qin[0, idxSpanning])
        
        dM[count, :] = 0
        vMT[count] = 0        
        for i in range(len(idxSpanning)):
            dM[count, idxSpanning[i]] = - polynomial.calcDerivative(
                    qin[0, idxSpanning], i)
            vMT[count] += (-dM[count, idxSpanning[i]] * 
               qdotin[0, idxSpanning[i]])
        
    f_polynomial = ca.Function('f_polynomial',[qin, qdotin],[lMT, vMT, dM])
    
    return f_polynomial

def muscle_length_scaling_vector(muscle_actuated_joints, f_NeMu):
    # The optimal fiber length and tendon slack length ('muscle length parameters') depend on the skeletal scaling.
    # This means that when the skeleton scaling variables are optimized, the scaling of the optimal fiber length
    # and tendon slack length will depend on this.
    # We thus generate a function that maps the skeleton scaling vector into a muscle length scaling vector.

    # Mapping of skeleton scaling, joint pose and joint velocity into muscle tendon length, velocity and moment arm
    # Compute muscle-tendon length in default pose (all zeros) for generic model (all scaling factors = 1)
    Qs_default = np.zeros((len(muscle_actuated_joints), 1))
    Qds_default = np.zeros((len(muscle_actuated_joints), 1))
    scale_vector_default = np.ones((36, 1))
    NeMu_input_default = ca.vertcat(scale_vector_default, Qs_default, Qds_default)
    [lMT_default, vMT_default, dM_default] = f_NeMu(NeMu_input_default)

    # Compute muscle-tendon length in default pose (all zeros) for scaled model (scaling factors are the optimization variables)
    skeleton_scaling_vector_SX = ca.SX.sym("scaling_vector_SX", 12 * 3, 1)
    NeMu_input_SX = ca.vertcat(skeleton_scaling_vector_SX, Qs_default, Qds_default)
    [lMT_scaled, vMT_scaled, dM_scaled] = f_NeMu(NeMu_input_SX)

    # The ratio of the scaled vs the default muscle-tendon lengths determines the scaling of the optimal fiber and tendon slack length
    muscle_scaling_vector = lMT_scaled / lMT_default
    f_muscle_length_scaling_vector = ca.Function('f_muscle_length_scaling_vector',
                                                 [skeleton_scaling_vector_SX],
                                                 [muscle_scaling_vector])

    return f_muscle_length_scaling_vector

# %% CasADi function to derive the Hill equilibrium.
def hillEquilibrium(mtParameters, tendonCompliance, specificTension):
    
    from muscleModels import DeGrooteFregly2016MuscleModel
    
    NMuscles = mtParameters.shape[1]
    
    # Function variables.
    activation = ca.SX.sym('activation', NMuscles)
    mtLength = ca.SX.sym('mtLength', NMuscles)
    mtVelocity = ca.SX.sym('mtVelocity', NMuscles)
    normTendonForce = ca.SX.sym('normTendonForce', NMuscles)
    normTendonForceDT = ca.SX.sym('normTendonForceDT', NMuscles)
     
    hillEquilibrium = ca.SX(NMuscles, 1)
    tendonForce = ca.SX(NMuscles, 1)
    activeFiberForce = ca.SX(NMuscles, 1)
    normActiveFiberLengthForce = ca.SX(NMuscles, 1)
    passiveFiberForce = ca.SX(NMuscles, 1)
    activeFiberForce_effective = ca.SX(NMuscles, 1)
    passiveFiberForce_effective = ca.SX(NMuscles, 1)
    normFiberLength = ca.SX(NMuscles, 1)
    fiberVelocity = ca.SX(NMuscles, 1)    
    
    for m in range(NMuscles):    
        muscle = DeGrooteFregly2016MuscleModel(
            mtParameters[:, m], activation[m], mtLength[m], mtVelocity[m], 
            normTendonForce[m], normTendonForceDT[m], tendonCompliance[:, m],
            specificTension[:, m])
        
        hillEquilibrium[m] = muscle.deriveHillEquilibrium()
        tendonForce[m] = muscle.getTendonForce()
        activeFiberForce[m] = muscle.getActiveFiberForce()[0]
        passiveFiberForce[m] = muscle.getPassiveFiberForce()[0]
        activeFiberForce_effective[m] = muscle.getActiveFiberForce()[0] * muscle.cosPennationAngle
        passiveFiberForce_effective[m] = muscle.getPassiveFiberForce()[0] * muscle.cosPennationAngle
        normActiveFiberLengthForce[m] = muscle.getActiveFiberLengthForce()
        normFiberLength[m] = muscle.getFiberLength()[1]
        fiberVelocity[m] = muscle.getFiberVelocity()[0]
        
    f_hillEquilibrium = ca.Function('f_hillEquilibrium',
                                    [activation, mtLength, mtVelocity, 
                                     normTendonForce, normTendonForceDT], 
                                     [hillEquilibrium, tendonForce,
                                      activeFiberForce, passiveFiberForce,
                                      normActiveFiberLengthForce,
                                      normFiberLength, fiberVelocity, activeFiberForce_effective, passiveFiberForce_effective])
    
    return f_hillEquilibrium

def mass_scaling_with_skeleton_volume(joints, F_skeleton_scaling):
    # This scaling factor to scale up muscle mass and force in the metabolic energy model (and the hillEquilibrium dynamics)
    # is computed from the ratio of the scaled model mass over the generic model mass
    Q_Qdot_Qdotdot_default = np.zeros((3 * len(joints), 1))
    scaling_vector_MX = ca.MX.sym("scaling_vector_MX",  18 * 3, 1)
    input_SkeletonDyn = ca.vertcat(Q_Qdot_Qdotdot_default, scaling_vector_MX, 0)
    mass_scaling_MX = F_skeleton_scaling(input_SkeletonDyn)[103,0]
    f_mass_scaling = ca.Function('f_mass_scaling',
                                                 [scaling_vector_MX],
                                                 [mass_scaling_MX])
    return f_mass_scaling

def get_height(joints, F_skeleton_scaling):
    # This scaling factor to scale up muscle mass and force in the metabolic energy model (and the hillEquilibrium dynamics)
    # is computed from the ratio of the scaled model mass over the generic model mass
    Q_Qdot_Qdotdot_default = np.zeros((3 * len(joints), 1))
    scaling_vector_MX = ca.MX.sym("scaling_vector_MX",  18 * 3, 1)
    input_SkeletonDyn = ca.vertcat(Q_Qdot_Qdotdot_default, scaling_vector_MX, 0)
    height_MX = F_skeleton_scaling(input_SkeletonDyn)[104,0]
    f_height = ca.Function('f_height',
                                                 [scaling_vector_MX],
                                                 [height_MX])
    return f_height

def hillEquilibrium_muscle_length_scaling(mtParameters, tendonCompliance, specific_tension):
    from muscleModels import DeGrooteFregly2016MuscleModel

    NMuscles = mtParameters.shape[1]

    # Function variables.
    muscle_length_scaling_SX = ca.SX.sym("scaling_vector_SX", NMuscles, 1)

    activation = ca.SX.sym('activation', NMuscles)
    mtLength = ca.SX.sym('mtLength', NMuscles)
    mtVelocity = ca.SX.sym('mtVelocity', NMuscles)
    normTendonForce = ca.SX.sym('normTendonForce', NMuscles)
    normTendonForceDT = ca.SX.sym('normTendonForceDT', NMuscles)
    mtParameters_SX = ca.SX.sym('scaling_vector', 5, NMuscles)
    model_mass_scaling = ca.SX.sym('model_mass_scaling', 1, 1)

    muscleVolume_original = (mtParameters[0, :] / (1e6 * specific_tension) * mtParameters[1, :]).T
    muscleVolume_scaled = model_mass_scaling * muscleVolume_original
    optimalFiberLength_scaled = (muscle_length_scaling_SX * mtParameters[1, :]).T
    tendonSlackLength_scaled = (muscle_length_scaling_SX * mtParameters[2, :]).T
    maximalIsometricForce_scaled = muscleVolume_scaled.T / optimalFiberLength_scaled * (1e6 * specific_tension)
    maxFiberVelocity_scaled = (muscle_length_scaling_SX * mtParameters[4, :]).T

    mtParameters_SX[0, :] = maximalIsometricForce_scaled
    mtParameters_SX[1, :] = optimalFiberLength_scaled
    mtParameters_SX[2, :] = tendonSlackLength_scaled
    mtParameters_SX[3, :] = mtParameters[3, :]
    mtParameters_SX[4, :] = maxFiberVelocity_scaled

    hillEquilibrium = ca.SX(NMuscles, 1)
    tendonForce = ca.SX(NMuscles, 1)
    activeFiberForce = ca.SX(NMuscles, 1)
    normActiveFiberLengthForce = ca.SX(NMuscles, 1)
    passiveFiberForce = ca.SX(NMuscles, 1)
    passiveEffectiveFiberForce = ca.SX(NMuscles, 1)
    activeEffectiveFiberForce = ca.SX(NMuscles, 1)
    normFiberLength = ca.SX(NMuscles, 1)
    fiberVelocity = ca.SX(NMuscles, 1)

    for m in range(NMuscles):
        muscle = DeGrooteFregly2016MuscleModel(
            mtParameters_SX[:, m], activation[m], mtLength[m], mtVelocity[m],
            normTendonForce[m], normTendonForceDT[m], tendonCompliance[:, m],
            specific_tension[:, m])

        hillEquilibrium[m] = muscle.deriveHillEquilibrium()
        tendonForce[m] = muscle.getTendonForce()
        activeFiberForce[m] = muscle.getActiveFiberForce()[0]
        passiveFiberForce[m] = muscle.getPassiveFiberForce()[0]
        passiveEffectiveFiberForce[m] = muscle.getPassiveFiberForce()[0] * muscle.cosPennationAngle
        activeEffectiveFiberForce[m] = muscle.getActiveFiberForce()[0] * muscle.cosPennationAngle

        normActiveFiberLengthForce[m] = muscle.getActiveFiberLengthForce()
        normFiberLength[m] = muscle.getFiberLength()[1]
        fiberVelocity[m] = muscle.getFiberVelocity()[0]

    f_hillEquilibrium = ca.Function('f_hillEquilibrium_skeleton_scaling',
                                    [activation, mtLength, mtVelocity,
                                     normTendonForce, normTendonForceDT, muscle_length_scaling_SX, model_mass_scaling],
                                    [hillEquilibrium, tendonForce,
                                     activeFiberForce, passiveFiberForce,
                                     normActiveFiberLengthForce,
                                     normFiberLength, fiberVelocity,activeEffectiveFiberForce,passiveEffectiveFiberForce])

    return f_hillEquilibrium

# %% CasADi function to explicitly describe the dynamic equations governing 
# the arm movements.
def armActivationDynamics(NArmJoints):
    
    t = 0.035 # time constant       
    
    # Function variables.
    eArm = ca.SX.sym('eArm',NArmJoints)
    aArm = ca.SX.sym('aArm',NArmJoints)
    
    aArmDt = (eArm - aArm) / t
    
    f_armActivationDynamics = ca.Function('f_armActivationDynamics',
                                          [eArm, aArm], [aArmDt])
    
    return f_armActivationDynamics  

# %% CasADi function to compute the metabolic cost of transport based on 
# Bhargava et al. (2004).

def metabolicsBhargava_muscle_length_scaling(slow_twitch_ratio, maximalIsometricForce,
                       optimalFiberLength, specific_tension, smoothingConstant,
                       use_fiber_length_dep_curve=False,
                       use_force_dependent_shortening_prop_constant=True,
                       include_negative_mechanical_work=False):
    NMuscles = maximalIsometricForce.shape[0]

    # Function variables.
    muscle_length_scaling_SX = ca.SX.sym("scaling_vector_SX", NMuscles, 1)

    # Function variables.
    excitation_SX = ca.SX.sym('excitation', NMuscles)
    activation_SX = ca.SX.sym('activation', NMuscles)
    normFiberLength_SX = ca.SX.sym('normFiberLength', NMuscles)
    fiberVelocity_SX = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce_SX = ca.SX.sym('activeFiberForce', NMuscles)
    passiveFiberForce_SX = ca.SX.sym('passiveFiberForce', NMuscles)
    normActiveFiberLengthForce_SX = (
        ca.SX.sym('normActiveFiberLengthForce', NMuscles))
    model_mass_scaling_SX = ca.SX.sym('model_mass_scaling', 1, 1)

    activationHeatRate = ca.SX(NMuscles, 1)
    maintenanceHeatRate = ca.SX(NMuscles, 1)
    shorteningHeatRate = ca.SX(NMuscles, 1)
    mechanicalWork = ca.SX(NMuscles, 1)
    totalHeatRate = ca.SX(NMuscles, 1)
    metabolicEnergyRate = ca.SX(NMuscles, 1)
    slowTwitchExcitation = ca.SX(NMuscles, 1)
    fastTwitchExcitation = ca.SX(NMuscles, 1)
    muscleVolume_original = ca.SX(NMuscles, 1)
    muscleVolume_scaled = ca.SX(NMuscles, 1)
    maximalIsometricForce_scaled = ca.SX(NMuscles, 1)
    optimalFiberLength_scaled = ca.SX(NMuscles, 1)
    muscleMass_scaled = ca.SX(NMuscles, 1)



    from metabolicEnergyModels import Bhargava2004SmoothedMuscleMetabolics


    for m in range(NMuscles):
        muscleVolume_original[m] = maximalIsometricForce[m] / (1e6 * specific_tension[0, m]) * optimalFiberLength[m]
        muscleVolume_scaled[m] = muscleVolume_original[m] * model_mass_scaling_SX
        optimalFiberLength_scaled[m] = optimalFiberLength[m] * muscle_length_scaling_SX[m]
        maximalIsometricForce_scaled[m] = muscleVolume_scaled[m] / optimalFiberLength_scaled[m] * (1e6 * specific_tension[0, m])
        muscleMass_scaled[m] = muscleVolume_scaled[m] * 1059.7

        metabolics = (Bhargava2004SmoothedMuscleMetabolics(
            excitation_SX[m], activation_SX[m],
            normFiberLength_SX[m],
            fiberVelocity_SX[m],
            activeFiberForce_SX[m],
            passiveFiberForce_SX[m],
            normActiveFiberLengthForce_SX[m],
            slow_twitch_ratio[m],
            maximalIsometricForce_scaled[m],
            muscleMass_scaled[m], smoothingConstant))

        slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0]
        fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1]
        activationHeatRate[m] = metabolics.getActivationHeatRate()
        maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
            use_fiber_length_dep_curve)
        shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
            use_force_dependent_shortening_prop_constant)
        mechanicalWork[m] = metabolics.getMechanicalWork(
            include_negative_mechanical_work)
        totalHeatRate[m] = metabolics.getTotalHeatRate()
        metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()

    f_metabolicsBhargava = ca.Function('metabolicsBhargava',
                                       [excitation_SX, activation_SX, normFiberLength_SX,
                                        fiberVelocity_SX, activeFiberForce_SX,
                                        passiveFiberForce_SX,
                                        normActiveFiberLengthForce_SX, muscle_length_scaling_SX, model_mass_scaling_SX],
                                       [activationHeatRate, maintenanceHeatRate,
                                        shorteningHeatRate, mechanicalWork,
                                        totalHeatRate, metabolicEnergyRate])

    return f_metabolicsBhargava


def metabolicsBhargava(slowTwitchRatio, maximalIsometricForce,
                       muscleMass, smoothingConstant,
                       use_fiber_length_dep_curve=False,
                       use_force_dependent_shortening_prop_constant=True,
                       include_negative_mechanical_work=False):
    NMuscles = maximalIsometricForce.shape[0]

    # Function variables.
    excitation = ca.SX.sym('excitation', NMuscles)
    activation = ca.SX.sym('activation', NMuscles)
    normFiberLength = ca.SX.sym('normFiberLength', NMuscles)
    fiberVelocity = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce = ca.SX.sym('activeFiberForce', NMuscles)
    passiveFiberForce = ca.SX.sym('passiveFiberForce', NMuscles)
    normActiveFiberLengthForce = (
        ca.SX.sym('normActiveFiberLengthForce', NMuscles))

    activationHeatRate = ca.SX(NMuscles, 1)
    maintenanceHeatRate = ca.SX(NMuscles, 1)
    shorteningHeatRate = ca.SX(NMuscles, 1)
    mechanicalWork = ca.SX(NMuscles, 1)
    totalHeatRate = ca.SX(NMuscles, 1)
    metabolicEnergyRate = ca.SX(NMuscles, 1)
    slowTwitchExcitation = ca.SX(NMuscles, 1)
    fastTwitchExcitation = ca.SX(NMuscles, 1)

    from metabolicEnergyModels import Bhargava2004SmoothedMuscleMetabolics

    for m in range(NMuscles):
        metabolics = (Bhargava2004SmoothedMuscleMetabolics(
            excitation[m], activation[m],
            normFiberLength[m],
            fiberVelocity[m],
            activeFiberForce[m],
            passiveFiberForce[m],
            normActiveFiberLengthForce[m],
            slowTwitchRatio[m],
            maximalIsometricForce[m],
            muscleMass[m], smoothingConstant))

        slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0]
        fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1]
        activationHeatRate[m] = metabolics.getActivationHeatRate()
        maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
            use_fiber_length_dep_curve)
        shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
            use_force_dependent_shortening_prop_constant)
        mechanicalWork[m] = metabolics.getMechanicalWork(
            include_negative_mechanical_work)
        totalHeatRate[m] = metabolics.getTotalHeatRate()
        metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()

    f_metabolicsBhargava = ca.Function('metabolicsBhargava',
                                       [excitation, activation, normFiberLength,
                                        fiberVelocity, activeFiberForce,
                                        passiveFiberForce,
                                        normActiveFiberLengthForce],
                                       [activationHeatRate, maintenanceHeatRate,
                                        shorteningHeatRate, mechanicalWork,
                                        totalHeatRate, metabolicEnergyRate])

    return f_metabolicsBhargava


def metabolicsBhargava_muscleScaling(slowTwitchRatio, maximalIsometricForce,
                       optimalFiberLength, specificTension, smoothingConstant, optimize_scaling, scaling_algorithm,
                       use_fiber_length_dep_curve=False,
                       use_force_dependent_shortening_prop_constant=True,
                       include_negative_mechanical_work=False):
    NMuscles = maximalIsometricForce.shape[0]

    # Function variables.
    excitation = ca.SX.sym('excitation', NMuscles)
    activation = ca.SX.sym('activation', NMuscles)
    normFiberLength = ca.SX.sym('normFiberLength', NMuscles)
    fiberVelocity = ca.SX.sym('fiberVelocity', NMuscles)
    activeFiberForce = ca.SX.sym('activeFiberForce', NMuscles)
    passiveFiberForce = ca.SX.sym('passiveFiberForce', NMuscles)
    normActiveFiberLengthForce = (
        ca.SX.sym('normActiveFiberLengthForce', NMuscles))
    model_mass_scaling = ca.SX.sym('model_mass_scaling', NMuscles)
    muscle_length_scaling_vector = ca.SX.sym('muscle_scaling_vector', NMuscles)
    muscleVolume_scaling_vector = ca.SX.sym('muscleVolume_scaling_vector', NMuscles)

    activationHeatRate = ca.SX(NMuscles, 1)
    maintenanceHeatRate = ca.SX(NMuscles, 1)
    shorteningHeatRate = ca.SX(NMuscles, 1)
    mechanicalWork = ca.SX(NMuscles, 1)
    totalHeatRate = ca.SX(NMuscles, 1)
    metabolicEnergyRate = ca.SX(NMuscles, 1)
    slowTwitchExcitation = ca.SX(NMuscles, 1)
    fastTwitchExcitation = ca.SX(NMuscles, 1)
    muscleVolume_original = ca.SX(NMuscles, 1)
    muscleVolume_scaled = ca.SX(NMuscles, 1)
    maximalIsometricForce_scaled = ca.SX(NMuscles, 1)
    optimalFiberLength_scaled = ca.SX(NMuscles, 1)
    muscleMass_scaled = ca.SX(NMuscles, 1)

    from metabolicEnergyModels import Bhargava2004SmoothedMuscleMetabolics

    if optimize_scaling == True and scaling_algorithm == 'OurScaling':
        for m in range(NMuscles):
            muscleVolume_original[m] = np.multiply(maximalIsometricForce[m], optimalFiberLength[m])
            muscleVolume_scaled[m] = np.multiply(muscleVolume_scaling_vector[m],
                                                 np.multiply(muscleVolume_original[m], model_mass_scaling[m]))
            optimalFiberLength_scaled[m] = np.multiply(optimalFiberLength[m], muscle_length_scaling_vector[m])
            maximalIsometricForce_scaled[m] = np.divide(muscleVolume_scaled[m], optimalFiberLength_scaled[m])
            muscleMass_scaled[m] = np.divide(np.multiply(muscleVolume_scaled[m], 1059.7),
                                             np.multiply(specificTension[0, m].T, 1e6))
            metabolics = (Bhargava2004SmoothedMuscleMetabolics(
                excitation[m], activation[m],
                normFiberLength[m],
                fiberVelocity[m],
                activeFiberForce[m],
                passiveFiberForce[m],
                normActiveFiberLengthForce[m],
                slowTwitchRatio[m],
                maximalIsometricForce_scaled[m],
                muscleMass_scaled[m], smoothingConstant))

            slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0]
            fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1]
            activationHeatRate[m] = metabolics.getActivationHeatRate()
            maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
                use_fiber_length_dep_curve)
            shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
                use_force_dependent_shortening_prop_constant)
            mechanicalWork[m] = metabolics.getMechanicalWork(
                include_negative_mechanical_work)
            totalHeatRate[m] = metabolics.getTotalHeatRate()
            metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()

    if optimize_scaling == True and scaling_algorithm == 'OpenSimScaling':
        for m in range(NMuscles):
            optimalFiberLength_scaled[m] = np.multiply(optimalFiberLength[m], muscle_length_scaling_vector[m])
            muscleVolume_scaled[m] = np.multiply(maximalIsometricForce[m], optimalFiberLength_scaled[m])
            muscleMass_scaled[m] = np.divide(np.multiply(muscleVolume_scaled[m], 1059.7),
                                             np.multiply(specificTension[0, m].T, 1e6))

            metabolics = (Bhargava2004SmoothedMuscleMetabolics(
                excitation[m], activation[m],
                normFiberLength[m],
                fiberVelocity[m],
                activeFiberForce[m],
                passiveFiberForce[m],
                normActiveFiberLengthForce[m],
                slowTwitchRatio[m],
                maximalIsometricForce[m],
                muscleMass_scaled[m], smoothingConstant))

            slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0]
            fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1]
            activationHeatRate[m] = metabolics.getActivationHeatRate()
            maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
                use_fiber_length_dep_curve)
            shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
                use_force_dependent_shortening_prop_constant)
            mechanicalWork[m] = metabolics.getMechanicalWork(
                include_negative_mechanical_work)
            totalHeatRate[m] = metabolics.getTotalHeatRate()
            metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()

    if optimize_scaling == False:
        for m in range(NMuscles):
            muscleVolume_scaled[m] = np.multiply(maximalIsometricForce[m], optimalFiberLength[m])
            muscleMass_scaled[m] = np.divide(np.multiply(muscleVolume_scaled[m], 1059.7),
                                             np.multiply(specificTension[0, m].T, 1e6))
            metabolics = (Bhargava2004SmoothedMuscleMetabolics(
                excitation[m], activation[m],
                normFiberLength[m],
                fiberVelocity[m],
                activeFiberForce[m],
                passiveFiberForce[m],
                normActiveFiberLengthForce[m],
                slowTwitchRatio[m],
                maximalIsometricForce[m],
                muscleMass_scaled[m], smoothingConstant))

            slowTwitchExcitation[m] = metabolics.getTwitchExcitation()[0]
            fastTwitchExcitation[m] = metabolics.getTwitchExcitation()[1]
            activationHeatRate[m] = metabolics.getActivationHeatRate()
            maintenanceHeatRate[m] = metabolics.getMaintenanceHeatRate(
                use_fiber_length_dep_curve)
            shorteningHeatRate[m] = metabolics.getShorteningHeatRate(
                use_force_dependent_shortening_prop_constant)
            mechanicalWork[m] = metabolics.getMechanicalWork(
                include_negative_mechanical_work)
            totalHeatRate[m] = metabolics.getTotalHeatRate()
            metabolicEnergyRate[m] = metabolics.getMetabolicEnergyRate()

    f_metabolicsBhargava = ca.Function('metabolicsBhargava',
                                       [excitation, activation, normFiberLength,
                                        fiberVelocity, activeFiberForce,
                                        passiveFiberForce,
                                        normActiveFiberLengthForce, muscle_length_scaling_vector, model_mass_scaling,
                                        muscleVolume_scaling_vector],
                                       [activationHeatRate, maintenanceHeatRate,
                                        shorteningHeatRate, mechanicalWork,
                                        totalHeatRate, metabolicEnergyRate])

    return f_metabolicsBhargava

# %% CasADi function to compute passive (limit) joint torques.
def getLimitTorques(k, theta, d):
    
    # Function variables.
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = (k[0] * np.exp(k[1] * (Q - theta[1])) + k[2] *
                           np.exp(k[3] * (Q - theta[0])) - d * Qdot)
    
    f_passiveJointTorque = ca.Function('f_passiveJointTorque', [Q, Qdot], 
                                       [passiveJointTorque])
    
    return f_passiveJointTorque

# %% CasADi function to compute linear passive joint torques.
def getLinearPassiveTorques(k, d):
    
    # Function variables.
    Q = ca.SX.sym('Q', 1)
    Qdot = ca.SX.sym('Qdot', 1)
    
    passiveJointTorque = -k * Q - d * Qdot
    f_passiveMtpTorque = ca.Function('f_passiveMtpTorque', [Q, Qdot], 
                                     [passiveJointTorque])
    
    return f_passiveMtpTorque    

# %% CasADi function to compute normalized sum of elements to given power.
def normSumPow(N, exp):
    
    # Function variables.
    x = ca.SX.sym('x', N,  1)
      
    nsp = ca.sum1(x**exp)       
    nsp = nsp / N
    
    f_normSumPow = ca.Function('f_normSumPow', [x], [nsp])
    
    return f_normSumPow

# %% CasADi function to compute difference in torques.
def diffTorques():
    
    # Function variables.
    jointTorque = ca.SX.sym('x', 1) 
    muscleTorque = ca.SX.sym('x', 1) 
    passiveTorque = ca.SX.sym('x', 1)
    
    diffTorque = jointTorque - (muscleTorque + passiveTorque)
    
    f_diffTorques = ca.Function(
            'f_diffTorques', [jointTorque, muscleTorque, passiveTorque], 
            [diffTorque])
        
    return f_diffTorques

# %% CasADi function to compute foot-ground contact forces.
# Note: this function is unused for the predictive simulations, but could be
# useful in other studies.
def smoothSphereHalfSpaceForce(dissipation, transitionVelocity,
                               staticFriction, dynamicFriction, 
                               viscousFriction, normal):
    
    from contactModels import smoothSphereHalfSpaceForce_ca
    
    # Function variables.
    stiffness = ca.SX.sym('stiffness', 1) 
    radius = ca.SX.sym('radius', 1)     
    locSphere_inB = ca.SX.sym('locSphere_inB', 3) 
    posB_inG = ca.SX.sym('posB_inG', 3) 
    lVelB_inG = ca.SX.sym('lVelB_inG', 3) 
    aVelB_inG = ca.SX.sym('aVelB_inG', 3) 
    RBG_inG = ca.SX.sym('RBG_inG', 3, 3) 
    TBG_inG = ca.SX.sym('TBG_inG', 3) 
    
    contactElement = smoothSphereHalfSpaceForce_ca(
        stiffness, radius, dissipation, transitionVelocity, staticFriction,
        dynamicFriction, viscousFriction, normal)
    
    contactForce = contactElement.getContactForce(locSphere_inB, posB_inG,
                                                  lVelB_inG, aVelB_inG,
                                                  RBG_inG, TBG_inG)
    
    f_smoothSphereHalfSpaceForce = ca.Function(
            'f_smoothSphereHalfSpaceForce',[stiffness, radius, locSphere_inB,
                                            posB_inG, lVelB_inG, aVelB_inG,
                                            RBG_inG, TBG_inG], [contactForce])
    
    return f_smoothSphereHalfSpaceForce 
