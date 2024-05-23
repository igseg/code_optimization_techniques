import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_tools import *
from datetime import datetime

size = 100
dt = 0.001
times  = np.arange(0,1+dt,dt)

def BM(T,n,seed):
    np.random.seed(seed)
    Z = np.random.randn(n)
    B = np.ones(len(T))*0
    for i in range(n):
        xi = np.sqrt(2)*(Z[i])/((i+0.5)*np.pi)
        B = B + xi*np.array([np.sin((i+0.5)*np.pi*t) for t in T])
    return B


def generate_forecast(size_ele, size_inflow,dt,times, seed_ele, seed_inflow):
    results_ele    = np.load("Estimates/ele_model.npy", allow_pickle=True)
    results_inflow = np.load("Estimates/inflow_model.npy", allow_pickle=True)

    params_ele    = results_ele[0]
    std_ele       = results_ele[1]
    params_inflow = results_inflow[0]
    std_inflow    = results_inflow[1]

    initial_electricity_price = 40.31 # Sample mean

    ## Models forecasting

    def model_inflow(coefs_inflow, t, shock, std_inflow):
        return coefs_inflow[0] + coefs_inflow[1] * (np.sin((t+coefs_inflow[2])*2*np.pi/365)) + std_inflow * shock

    def model_price_e(coefs_prices_e,t,prev, shock, std_ele):
        return (coefs_prices_e[1] * (coefs_prices_e[0] - prev) +
            coefs_prices_e[2] * np.sin((t + coefs_prices_e[3]) * 2 * np.pi / (6 * 30 * 24)) +
                                 coefs_prices_e[4] * np.sin((t + coefs_prices_e[5]) * 2 * np.pi / (12 * 30 * 24)) +
                                 coefs_prices_e[6] * np.sin((t + coefs_prices_e[7]) * 2 * np.pi / 24) + shock * std_ele)

    ## Generate random processes

    last_elem = np.arange(0,len(times))[::24][-1]
    shocks_ele    = BM(times[:last_elem + 1], size_ele, seed_ele) #* np.sqrt(last_elem + 1)    ## hourly shocks  & variability adjusted by time window
    shocks_inflow = BM(times[::24], size_inflow, seed_inflow)           * np.sqrt(len(times[::24])) ## Daily shocks   & variability adjusted by time window

    forecast_ele = []
    prev =  initial_electricity_price
    for idx, t in enumerate(times[:last_elem + 1]):
        forecasted_ele = model_price_e(params_ele,
                                          idx,
                                          prev,
                                          shocks_ele[idx] * 1,
                                          std_ele
        )
        forecast_ele.append(forecasted_ele)
        prev = forecasted_ele

    forecast_inflow = []
    for idx,t in enumerate(times[:last_elem + 1]):
        time = idx // 24
        forecasted_inflow = model_inflow(params_inflow,
                                         time,
                                         shocks_inflow[idx//24],
                                         std_inflow
        )
        if forecasted_inflow < 0:
            forecasted_inflow = 0
        forecast_inflow.append(forecasted_inflow / 24)
    
    return forecast_ele, forecast_inflow