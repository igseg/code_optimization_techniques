import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tools import *
from datetime import datetime
from nls_optimizer import *
from my_time_series import tests_gaussian_white_noise

#####################

## Load data

ele_ts = pd.read_csv("electricity_data_clean.csv")

## define the model

def model_price_e(coefs_prices_e,t,prev):
    return (coefs_prices_e[1] * (coefs_prices_e[0] - prev) +
        coefs_prices_e[2] * np.sin((t + coefs_prices_e[3]) * 2 * np.pi / (6 * 30 * 24)) +
                             coefs_prices_e[4] * np.sin((t + coefs_prices_e[5]) * 2 * np.pi / (12 * 30 * 24)) +
                             coefs_prices_e[6] * np.sin((t + coefs_prices_e[7]) * 2 * np.pi / 24))

## Calibrate the NLS
ele_prices = ele_ts.price.values
times      = np.arange(0+5,len(ele_prices)+5,1) # + 5 becauase the data starts in may

observed_values = np.logical_not(np.isnan(ele_prices))
ele_prices = ele_prices[observed_values]
times   = times[observed_values]
times   = times[1:]

def residuals_ele(coefs):
    return model_price_e(coefs, times, ele_prices[:-1]) - ele_prices[1:]

num_coefs = 8
p0 = initial_values(num_coefs)
ele_model = NLS(residuals_ele, p0, xdata=times, ydata=ele_prices[1:])

print(ele_model.summary())
# tests_gaussian_white_noise(ele_model.mod1[2]['fvec'])
# plt.show()

## Save results

# results = {
#             "param estimate": ele_model.parmEsts,
#             "std": np.std(ele_model.mod1[2]['fvec'])
# }

np.save("Estimates/ele_model.npy", [ele_model.parmEsts, np.std(ele_model.mod1[2]['fvec'])])

## Load results

# np.load("Estimates/ele_model.npy", allow_pickle=True)
