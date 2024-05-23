import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tools import *
from datetime import datetime
from nls_optimizer import *
from my_time_series import tests_gaussian_white_noise

## Load data

dam_id = 169
path = from_id_to_time_series(dam_id)
ts = pd.read_csv(path)

## define the model

def model_inflow(coefs_inflow, t):
    return coefs_inflow[0] + coefs_inflow[1] * (np.sin((t+coefs_inflow[2])*2*np.pi/365))

## Calibrate the NLS

inflows = ts.inflow.values
times   = np.arange(0 + 1,len(inflows) + 1,1) # + 1 because it starts in January 1st

observed_values = np.logical_not(np.isnan(inflows))
inflows = inflows[observed_values]
times   = times[observed_values]
times   = times[1:]

def residuals_inflow(coefs):
    return model_inflow(coefs, times) - inflows[1:]

num_coefs = 3
p0 = initial_values(num_coefs)
inflow_model = NLS(residuals_inflow, p0, xdata=times, ydata=inflows[1:])

print(inflow_model.summary())
# tests_gaussian_white_noise(inflow_model.mod1[2]['fvec'])
# plt.show()

## Save results

# results = {
#             "param estimate": inflow_model.parmEsts,
#             "std": np.std(inflow_model.mod1[2]['fvec'])
# }

np.save("Estimates/inflow_model.npy", [inflow_model.parmEsts, np.std(inflow_model.mod1[2]['fvec'])])

## Load results

# np.load("Estimates/inflow_model.npy", allow_pickle=True)
