import numpy as np
import matplotlib.pyplot as plt
from config_ENV import CONFIG_ENV
# from param_matlab import param, m, NI
import load_mat

Cp_na=load_mat.load_mat(f"2023_BC/bc{CONFIG_ENV['bc']}/raw/{CONFIG_ENV['date']}/Cp_phavg.mat")
print(Cp_na)