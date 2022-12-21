

'''
    This script contains classes to set bounds to the optimization variables.
'''

# %% Import packages
import scipy.interpolate as interpolate
import pandas as pd
import numpy as np


# %% Class bounds.
class bounds:

    def __init__(self, Qs, joints, muscles, armJoints,
                 mtpJoints=['mtp_angle_l', 'mtp_angle_r']):

        self.Qs = Qs
        self.joints = joints
        self.muscles = muscles
        self.armJoints = armJoints
        self.mtpJoints = mtpJoints

    def splineQs(self):
        self.Qs_spline = self.Qs.copy()
        self.Qdots_spline = self.Qs.copy()
        self.Qdotdots_spline = self.Qs.copy()

        for joint in self.joints:
            spline = interpolate.InterpolatedUnivariateSpline(self.Qs['time'],
                                                              self.Qs[joint],
                                                              k=3)
            self.Qs_spline[joint] = spline(self.Qs['time'])
            splineD1 = spline.derivative(n=1)
            self.Qdots_spline[joint] = splineD1(self.Qs['time'])
            splineD2 = spline.derivative(n=2)
            self.Qdotdots_spline[joint] = splineD2(self.Qs['time'])

    joints = ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
              'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
              'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r',
              'subtalar_angle_l', 'subtalar_angle_r',
              'mtp_angle_l', 'mtp_angle_r',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
              'arm_flex_l', 'arm_add_l', 'arm_rot_l',
              'arm_flex_r', 'arm_add_r', 'arm_rot_r',
              'elbow_flex_l', 'elbow_flex_r']

    def getBoundsPosition(self):
        self.splineQs()
        upperBoundsPosition = pd.DataFrame()
        lowerBoundsPosition = pd.DataFrame()
        scalingPosition = pd.DataFrame()
        for count, joint in enumerate(self.joints):

            upperBoundsPosition.insert(count, joint, [-1])
            lowerBoundsPosition.insert(count, joint, [1])

            # Special cases
            if joint == 'pelvis_tilt':
                lowerBoundsPosition[joint] = -45 * np.pi / 180
                upperBoundsPosition[joint] = 15 * np.pi / 180
                scalingPosition.insert(count, joint, [20 * np.pi / 180])
            elif joint == 'pelvis_list':
                lowerBoundsPosition[joint] = -23.9 * np.pi / 180
                upperBoundsPosition[joint] = 20.7 * np.pi / 180
                scalingPosition.insert(count, joint, [23.9 * np.pi / 180])
            elif joint == 'pelvis_rotation':
                lowerBoundsPosition[joint] = -45 * np.pi / 180
                upperBoundsPosition[joint] = 45 * np.pi / 180
                scalingPosition.insert(count, joint, [14.9 * np.pi / 180])
            elif joint == 'pelvis_tx':
                lowerBoundsPosition[joint] = 0
                upperBoundsPosition[joint] = 3
                scalingPosition.insert(count, joint, [2])
            elif joint == 'pelvis_ty':
                lowerBoundsPosition[joint] = 0.6
                upperBoundsPosition[joint] = 1.1
                scalingPosition.insert(count, joint, [1.1])
            elif joint == 'pelvis_tz':
                lowerBoundsPosition[joint] = -0.3
                upperBoundsPosition[joint] = 0.3
                scalingPosition.insert(count, joint, [0.1])
            elif joint == 'lumbar_extension':
                lowerBoundsPosition[joint] = -20 * np.pi / 180
                upperBoundsPosition[joint] = 20 * np.pi / 180
                scalingPosition.insert(count, joint, [16 * np.pi / 180])
            elif joint == 'lumbar_rotation':
                lowerBoundsPosition[joint] = -38.1 * np.pi / 180
                upperBoundsPosition[joint] = 37.7 * np.pi / 180
                scalingPosition.insert(count, joint, [38.1 * np.pi / 180])

            elif joint == 'lumbar_bending':
                lowerBoundsPosition[joint] = -25.5 * np.pi / 180
                upperBoundsPosition[joint] = 30.4 * np.pi / 180
                scalingPosition.insert(count, joint, [30.4 * np.pi / 180])

            elif (joint == 'hip_flexion_l') or (joint == 'hip_flexion_r'):
                lowerBoundsPosition[joint] = -109.9 * np.pi / 180
                upperBoundsPosition[joint] = 121.4 * np.pi / 180
                scalingPosition.insert(count, joint, [121.4 * np.pi / 180])

            elif (joint == 'hip_adduction_l') or (joint == 'hip_adduction_r'):
                lowerBoundsPosition[joint] = -36.6 * np.pi / 180
                upperBoundsPosition[joint] = 41.2 * np.pi / 180
                scalingPosition.insert(count, joint, [41.2 * np.pi / 180])

            elif (joint == 'hip_rotation_l') or (joint == 'hip_rotation_r'):
                lowerBoundsPosition[joint] = -10 * np.pi / 180
                upperBoundsPosition[joint] = 10 * np.pi / 180
                scalingPosition.insert(count, joint, [29 * np.pi / 180])

            elif (joint == 'knee_angle_l') or (joint == 'knee_angle_r'):
                lowerBoundsPosition[joint] = -175 * np.pi / 180
                upperBoundsPosition[joint] = 10 * np.pi / 180
                scalingPosition.insert(count, joint, [175 * np.pi / 180])

            elif (joint == 'ankle_angle_l') or (joint == 'ankle_angle_r'):
                lowerBoundsPosition[joint] = -81.2 * np.pi / 180
                upperBoundsPosition[joint] = 77.6 * np.pi / 180
                scalingPosition.insert(count, joint, [81.2 * np.pi / 180])

            elif (joint == 'subtalar_angle_l') or (joint == 'subtalar_angle_r'):
                lowerBoundsPosition[joint] = -47 * np.pi / 180
                upperBoundsPosition[joint] = 51.8 * np.pi / 180
                scalingPosition.insert(count, joint, [51.8 * np.pi / 180])

            elif (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                lowerBoundsPosition[joint] = -60 * np.pi / 180
                upperBoundsPosition[joint] = 60 * np.pi / 180
                scalingPosition.insert(count, joint, [60 * np.pi / 180])

            elif (joint == 'elbow_flex_l') or (joint == 'elbow_flex_r'):
                lowerBoundsPosition[joint] = 0 * np.pi / 180
                upperBoundsPosition[joint] = 115 * np.pi / 180
                scalingPosition.insert(count, joint, [115 * np.pi / 180])

            elif (joint == 'arm_add_l') or (joint == 'arm_add_r'):
                lowerBoundsPosition[joint] = -42.6 * np.pi / 180
                upperBoundsPosition[joint] = 5 * np.pi / 180
                scalingPosition.insert(count, joint, [42.6 * np.pi / 180])

            elif (joint == 'arm_rot_l') or (joint == 'arm_rot_r'):
                lowerBoundsPosition[joint] = -13.2 * np.pi / 180
                upperBoundsPosition[joint] = 19.4 * np.pi / 180
                scalingPosition.insert(count, joint, [19.4 * np.pi / 180])

            elif (joint == 'arm_flex_l') or (joint == 'arm_flex_r'):
                lowerBoundsPosition[joint] = -70 * np.pi / 180
                upperBoundsPosition[joint] = 50 * np.pi / 180
                scalingPosition.insert(count, joint, [34.66 * np.pi / 180])

            lowerBoundsPosition[joint] /= scalingPosition[joint]
            upperBoundsPosition[joint] /= scalingPosition[joint]

        # Hard bounds at initial position.
        lowerBoundsPositionInitial = lowerBoundsPosition.copy()
        lowerBoundsPositionInitial['pelvis_tx'] = [0]
        upperBoundsPositionInitial = upperBoundsPosition.copy()
        upperBoundsPositionInitial['pelvis_tx'] = [0]

        return (upperBoundsPosition, lowerBoundsPosition, scalingPosition,
                upperBoundsPositionInitial, lowerBoundsPositionInitial)

    def getBoundsVelocity(self):
        self.splineQs()
        upperBoundsVelocity = pd.DataFrame()
        lowerBoundsVelocity = pd.DataFrame()
        scalingVelocity = pd.DataFrame()
        for count, joint in enumerate(self.joints):
            lowerBoundsVelocity.insert(count, joint, [-1])
            upperBoundsVelocity.insert(count, joint, [1])
            # Special cases
            basicLimit = 1000
            if joint == 'pelvis_tilt':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [70.6 * np.pi / 180])
            elif joint == 'pelvis_list':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [355.7 * np.pi / 180])
            elif joint == 'pelvis_rotation':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [171.7 * np.pi / 180])
            elif joint == 'pelvis_tx':
                lowerBoundsVelocity[joint] = 0
                upperBoundsVelocity[joint] = 20
                scalingVelocity.insert(count, joint, [4])
            elif joint == 'pelvis_ty':
                lowerBoundsVelocity[joint] = -4
                upperBoundsVelocity[joint] = 4
                scalingVelocity.insert(count, joint, [1.375])
            elif joint == 'pelvis_tz':
                lowerBoundsVelocity[joint] = -4
                upperBoundsVelocity[joint] = 4
                scalingVelocity.insert(count, joint, [0.48])
            elif joint == 'lumbar_extension':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [199 * np.pi / 180])
            elif joint == 'lumbar_rotation':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [206.3 * np.pi / 180])
            elif joint == 'lumbar_bending':
                lowerBoundsVelocity[joint] = -1000 * np.pi / 180
                upperBoundsVelocity[joint] = 1000 * np.pi / 180
                scalingVelocity.insert(count, joint, [442.5 * np.pi / 180])

            elif (joint == 'hip_flexion_l') or (joint == 'hip_flexion_r'):
                lowerBoundsVelocity[joint] = -2000 * np.pi / 180
                upperBoundsVelocity[joint] = 2000 * np.pi / 180
                scalingVelocity.insert(count, joint, [1252 * np.pi / 180])

            elif (joint == 'hip_adduction_l') or (joint == 'hip_adduction_r'):
                lowerBoundsVelocity[joint] = -2000 * np.pi / 180
                upperBoundsVelocity[joint] = 2000 * np.pi / 180
                scalingVelocity.insert(count, joint, [587 * np.pi / 180])

            elif (joint == 'hip_rotation_l') or (joint == 'hip_rotation_r'):
                lowerBoundsVelocity[joint] = -2500 * np.pi / 180
                upperBoundsVelocity[joint] = 2500 * np.pi / 180
                scalingVelocity.insert(count, joint, [399 * np.pi / 180])

            elif (joint == 'knee_angle_l') or (joint == 'knee_angle_r'):
                lowerBoundsVelocity[joint] = -2500 * np.pi / 180
                upperBoundsVelocity[joint] = 2500 * np.pi / 180
                scalingVelocity.insert(count, joint, [2263 * np.pi / 180])

            elif (joint == 'ankle_angle_l') or (joint == 'ankle_angle_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [1489 * np.pi / 180])
            elif (joint == 'subtalar_angle_l') or (joint == 'subtalar_angle_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [853.2 * np.pi / 180])
            elif (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [744.9 * np.pi / 180])
            elif (joint == 'elbow_flex_l') or (joint == 'elbow_flex_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [739 * np.pi / 180])
            elif (joint == 'arm_add_l') or (joint == 'arm_add_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [239 * np.pi / 180])
            elif (joint == 'arm_rot_l') or (joint == 'arm_rot_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [346 * np.pi / 180])
            elif (joint == 'arm_flex_l') or (joint == 'arm_flex_r'):
                lowerBoundsVelocity[joint] = -5000 * np.pi / 180
                upperBoundsVelocity[joint] = 5000 * np.pi / 180
                scalingVelocity.insert(count, joint, [309.6 * np.pi / 180])
            # Scaling.
            upperBoundsVelocity[joint] /= scalingVelocity[joint]
            lowerBoundsVelocity[joint] /= scalingVelocity[joint]

        return upperBoundsVelocity, lowerBoundsVelocity, scalingVelocity

    def getBoundsAcceleration(self):
        self.splineQs()
        upperBoundsAcceleration = pd.DataFrame()
        lowerBoundsAcceleration = pd.DataFrame()
        scalingAcceleration = pd.DataFrame()
        for count, joint in enumerate(self.joints):
            lowerBoundsAcceleration.insert(count, joint, [-1])
            upperBoundsAcceleration.insert(count, joint, [1])
            # Special cases
            basicLimit = 1000
            if joint == 'pelvis_tilt':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [1133 * np.pi / 180])

            elif joint == 'pelvis_list':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [5273 * np.pi / 180])

            elif joint == 'pelvis_rotation':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [3526.5 * np.pi / 180])

            elif joint == 'pelvis_tx':
                lowerBoundsAcceleration[joint] = -100
                upperBoundsAcceleration[joint] = 100
                scalingAcceleration.insert(count, joint, [61.36])

            elif joint == 'pelvis_ty':
                lowerBoundsAcceleration[joint] = -300
                upperBoundsAcceleration[joint] = 300
                scalingAcceleration.insert(count, joint, [16.9])

            elif joint == 'pelvis_tz':
                lowerBoundsAcceleration[joint] = -200
                upperBoundsAcceleration[joint] = 200
                scalingAcceleration.insert(count, joint, [5.91])

            elif joint == 'lumbar_extension':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [3027 * np.pi / 180])

            elif joint == 'lumbar_rotation':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [2220 * np.pi / 180])

            elif joint == 'lumbar_bending':
                lowerBoundsAcceleration[joint] = -100000 * np.pi / 180
                upperBoundsAcceleration[joint] = 100000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [6288 * np.pi / 180])

            elif (joint == 'hip_flexion_l') or (joint == 'hip_flexion_r'):
                lowerBoundsAcceleration[joint] = -150000 * np.pi / 180
                upperBoundsAcceleration[joint] = 150000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [11542 * np.pi / 180])

            elif (joint == 'hip_adduction_l') or (joint == 'hip_adduction_r'):
                lowerBoundsAcceleration[joint] = -150000 * np.pi / 180
                upperBoundsAcceleration[joint] = 150000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [8339 * np.pi / 180])

            elif (joint == 'hip_rotation_l') or (joint == 'hip_rotation_r'):
                lowerBoundsAcceleration[joint] = -1000000 * np.pi / 180
                upperBoundsAcceleration[joint] = 1000000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [7842 * np.pi / 180])

            elif (joint == 'knee_angle_l') or (joint == 'knee_angle_r'):
                lowerBoundsAcceleration[joint] = -1000000 * np.pi / 180
                upperBoundsAcceleration[joint] = 1000000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [25312 * np.pi / 180])

            elif (joint == 'ankle_angle_l') or (joint == 'ankle_angle_r'):
                lowerBoundsAcceleration[joint] = -4000000 * np.pi / 180
                upperBoundsAcceleration[joint] = 4000000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [24657 * np.pi / 180])

            elif (joint == 'subtalar_angle_l') or (joint == 'subtalar_angle_r'):
                lowerBoundsAcceleration[joint] = -2000000 * np.pi / 180
                upperBoundsAcceleration[joint] = 2000000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [13344 * np.pi / 180])

            elif (joint == 'mtp_angle_l') or (joint == 'mtp_angle_r'):
                lowerBoundsAcceleration[joint] = -2000000 * np.pi / 180
                upperBoundsAcceleration[joint] = 2000000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [28648 * np.pi / 180])

            elif (joint == 'elbow_flex_l') or (joint == 'elbow_flex_r'):
                lowerBoundsAcceleration[joint] = -400000 * np.pi / 180
                upperBoundsAcceleration[joint] = 400000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [5401 * np.pi / 180])

            elif (joint == 'arm_add_l') or (joint == 'arm_add_r'):
                lowerBoundsAcceleration[joint] = -400000 * np.pi / 180
                upperBoundsAcceleration[joint] = 400000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [2259.6 * np.pi / 180])

            elif (joint == 'arm_rot_l') or (joint == 'arm_rot_r'):
                lowerBoundsAcceleration[joint] = -400000 * np.pi / 180
                upperBoundsAcceleration[joint] = 400000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [6582 * np.pi / 180])

            elif (joint == 'arm_flex_l') or (joint == 'arm_flex_r'):
                lowerBoundsAcceleration[joint] = -400000 * np.pi / 180
                upperBoundsAcceleration[joint] = 400000 * np.pi / 180
                scalingAcceleration.insert(count, joint, [3824 * np.pi / 180])

            # Scaling.
            upperBoundsAcceleration[joint] /= scalingAcceleration[joint]
            lowerBoundsAcceleration[joint] /= scalingAcceleration[joint]

        return upperBoundsAcceleration, lowerBoundsAcceleration, scalingAcceleration

    def getBoundsActivation(self):
        lb = [0.05]
        lb_vec = lb * len(self.muscles)
        ub = [1]
        ub_vec = ub * len(self.muscles)
        s = [1]
        s_vec = s * len(self.muscles)
        upperBoundsActivation = pd.DataFrame([ub_vec], columns=self.muscles)
        lowerBoundsActivation = pd.DataFrame([lb_vec], columns=self.muscles)
        scalingActivation = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsActivation = upperBoundsActivation.div(scalingActivation)
        lowerBoundsActivation = lowerBoundsActivation.div(scalingActivation)

        return upperBoundsActivation, lowerBoundsActivation, scalingActivation

    def getBoundsForce(self):
        lb = [0]
        lb_vec = lb * len(self.muscles)
        ub = [5]
        ub_vec = ub * len(self.muscles)
        s = max([abs(lbi) for lbi in lb], [abs(ubi) for ubi in ub])
        s_vec = s * len(self.muscles)
        upperBoundsForce = pd.DataFrame([ub_vec], columns=self.muscles)
        lowerBoundsForce = pd.DataFrame([lb_vec], columns=self.muscles)
        scalingForce = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsForce = upperBoundsForce.div(scalingForce)
        lowerBoundsForce = lowerBoundsForce.div(scalingForce)

        return upperBoundsForce, lowerBoundsForce, scalingForce

    def getBoundsActivationDerivative(self):
        activationTimeConstant = 0.015
        deactivationTimeConstant = 0.06
        lb = [-1 / deactivationTimeConstant]
        lb_vec = lb * len(self.muscles)
        ub = [1 / activationTimeConstant]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsActivationDerivative = pd.DataFrame([ub_vec],
                                                       columns=self.muscles)
        lowerBoundsActivationDerivative = pd.DataFrame([lb_vec],
                                                       columns=self.muscles)
        scalingActivationDerivative = pd.DataFrame([s_vec],
                                                   columns=self.muscles)
        upperBoundsActivationDerivative = upperBoundsActivationDerivative.div(
            scalingActivationDerivative)
        lowerBoundsActivationDerivative = lowerBoundsActivationDerivative.div(
            scalingActivationDerivative)

        return (upperBoundsActivationDerivative,
                lowerBoundsActivationDerivative, scalingActivationDerivative)

    def getBoundsForceDerivative(self):
        lb = [-100]
        lb_vec = lb * len(self.muscles)
        ub = [100]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsForceDerivative = pd.DataFrame([ub_vec],
                                                  columns=self.muscles)
        lowerBoundsForceDerivative = pd.DataFrame([lb_vec],
                                                  columns=self.muscles)
        scalingForceDerivative = pd.DataFrame([s_vec],
                                              columns=self.muscles)
        upperBoundsForceDerivative = upperBoundsForceDerivative.div(
            scalingForceDerivative)
        lowerBoundsForceDerivative = lowerBoundsForceDerivative.div(
            scalingForceDerivative)


        return (upperBoundsForceDerivative, lowerBoundsForceDerivative,
                scalingForceDerivative)

    def getBoundsArmExcitation(self):
        lb = [-1]
        lb_vec = lb * len(self.armJoints)
        ub = [1]
        ub_vec = ub * len(self.armJoints)
        s = [150]
        s_vec = s * len(self.armJoints)
        upperBoundsArmExcitation = pd.DataFrame([ub_vec],
                                                columns=self.armJoints)
        lowerBoundsArmExcitation = pd.DataFrame([lb_vec],
                                                columns=self.armJoints)
        # Scaling.
        scalingArmExcitation = pd.DataFrame([s_vec], columns=self.armJoints)

        return (upperBoundsArmExcitation, lowerBoundsArmExcitation,
                scalingArmExcitation)

    def getBoundsArmActivation(self):
        lb = [-1]
        lb_vec = lb * len(self.armJoints)
        ub = [1]
        ub_vec = ub * len(self.armJoints)
        s = [150]
        s_vec = s * len(self.armJoints)
        upperBoundsArmActivation = pd.DataFrame([ub_vec],
                                                columns=self.armJoints)
        lowerBoundsArmActivation = pd.DataFrame([lb_vec],
                                                columns=self.armJoints)
        # Scaling.
        scalingArmActivation = pd.DataFrame([s_vec], columns=self.armJoints)

        return (upperBoundsArmActivation, lowerBoundsArmActivation,
                scalingArmActivation)

    def getBoundsMtpExcitation(self):
        lb = [-1]
        lb_vec = lb * len(self.mtpJoints)
        ub = [1]
        ub_vec = ub * len(self.mtpJoints)
        s = [60]
        s_vec = s * len(self.mtpJoints)
        upperBoundsMtpExcitation = pd.DataFrame([ub_vec],
                                                columns=self.mtpJoints)
        lowerBoundsMtpExcitation = pd.DataFrame([lb_vec],
                                                columns=self.mtpJoints)
        # Scaling.
        scalingMtpExcitation = pd.DataFrame([s_vec], columns=self.mtpJoints)

        return (upperBoundsMtpExcitation, lowerBoundsMtpExcitation,
                scalingMtpExcitation)

    def getBoundsMtpActivation(self):
        lb = [-1]
        lb_vec = lb * len(self.mtpJoints)
        ub = [1]
        ub_vec = ub * len(self.mtpJoints)
        s = [30]
        s_vec = s * len(self.mtpJoints)
        upperBoundsMtpActivation = pd.DataFrame([ub_vec],
                                                columns=self.mtpJoints)
        lowerBoundsMtpActivation = pd.DataFrame([lb_vec],
                                                columns=self.mtpJoints)
        # Scaling.
        scalingMtpActivation = pd.DataFrame([s_vec], columns=self.mtpJoints)

        return (upperBoundsMtpActivation, lowerBoundsMtpActivation,
                scalingMtpActivation)

    def getBoundsFinalTime(self):
        upperBoundsFinalTime = pd.DataFrame([1], columns=['time'])
        lowerBoundsFinalTime = pd.DataFrame([0.1], columns=['time'])

        return upperBoundsFinalTime, lowerBoundsFinalTime
