"""
  Generated using Konverter: https://github.com/ShaneSmiskol/Konverter
"""

import numpy as np
import casadi as ca

wb = np.load('C:/Users/tom_v/MusculoskeletalSpecialization/Models/NeMu/tanh/flex_dig_l_Geometry_Konverted_weights.npz', allow_pickle=True)
w, b = wb['wb']

def predict(x):
  x = ca.vertcat(x)
  l0 = ca.mtimes(x, w[0]) + ca.transpose(b[0])
  l0 = ca.tanh(l0)
  l1 = ca.mtimes(l0, w[1]) + ca.transpose(b[1])
  return l1
