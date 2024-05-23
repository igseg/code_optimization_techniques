import pyomo.environ as pyo
from pyomo.environ import (AbstractModel, Set, Var, Param, Constraint, Objective,
                           NonNegativeReals, Reals, Binary,
                           maximize)

## Params ##

def get_data(t_values, elev_efficiencies, elev_thershold, forecast_inflow, forecast_ele):
#     t_values  = list(range(3))
#     elev_efficiencies = [3,6]
    tr_values = list(range(len(elev_efficiencies)))
#     elev_thershold = [0,5]

    data = dict(
        p_s0   = {None: 45.158},
        p_smin = {None: 26.224},
        p_smax = {None: 136.521},

        p_rmin = {None: 0},
        p_rmax = {None: 1.01},
        T = {None: t_values},
        Tr = {None: tr_values},

#         p_inflow = {t:v for t,v in zip(t_values,[10,2,2])},
#         p_prices = {t:v for t,v in zip(t_values,[10, 10, 10])},
        p_inflow = {t:v for t,v in zip(t_values,forecast_inflow)},
        p_prices = {t:v for t,v in zip(t_values,forecast_ele)},

        p_elev_eff = {tr:v for tr,v in zip(tr_values, elev_efficiencies)},
        p_elev_thershold = {tr:v for tr,v in zip(tr_values, elev_thershold)} ,
    )
    return {None: data}

def get_model():

    model    = AbstractModel(name="Reservoir")
    model.T  = Set(doc="time steps")
    model.Tr = Set(doc="elevation step")

    model.TrT = model.Tr * model.T

    model.p_inflow = Param(model.T, domain=NonNegativeReals, doc="water inflow")
    model.p_prices = Param(model.T, domain=Reals, doc="prices")

    model.p_rmax = Param(domain=NonNegativeReals, doc="max outflow")
    model.p_rmin = Param(domain=NonNegativeReals, doc="min outflow")

    model.p_s0   = Param(domain=NonNegativeReals, doc="initial storage")

    model.p_smax = Param(domain=NonNegativeReals, doc="max storage")
    model.p_smin = Param(domain=NonNegativeReals, doc="min storage")

    model.p_elev_eff = Param(model.Tr, domain=NonNegativeReals, doc="elevation")
    model.p_elev_thershold = Param(model.Tr, domain=NonNegativeReals, doc="elevation thershold")

    model.v_outflow = Var(model.TrT, domain=NonNegativeReals, doc="water outflow")
    model.v_storage = Var(model.T  , domain=NonNegativeReals, doc="water storage")
    model.v_water_curtailment = Var(model.T, domain=NonNegativeReals, doc="water curtailment")

    model.b_elev_step = Var(model.TrT, domain=Binary, doc="active elevation step")

    def constraint_water_balance(m,t):

        return m.v_storage[t] + m.v_water_curtailment[t] == (
               (m.v_storage[m.T.prev(t)] if m.T.first() != t
                    else m.p_s0)
                + m.p_inflow[t] - sum(m.v_outflow[tr,t] for tr in m.Tr)
                )

    def constraint_outflow_limit_max(m, tr, t):
        return m.v_outflow[tr,t] <= m.p_rmax * m.b_elev_step[tr,t]

    def constraint_outflow_limit_min(m, tr, t):
        return m.v_outflow[tr,t] >= m.p_rmin * m.b_elev_step[tr,t]

    def constraint_storage_limit_max(m,t):
        return m.v_storage[t] <= m.p_smax

    def constraint_storage_limit_min(m,t):
        return m.v_storage[t] >= m.p_smin

    def constraint_tram_activation(m,t):
        return sum(m.b_elev_step[tr,t] for tr in m.Tr) <= 1

    def constraint_thershold_efficiency(m, tr, t):
        return m.p_elev_thershold[tr] * m.b_elev_step[tr,t] <= (m.v_storage[m.T.prev(t)] if m.T.first() != t
             else m.p_s0)

    model.c_water_balance     = Constraint(model.T  , rule=constraint_water_balance    )
    model.c_outflow_limit_max = Constraint(model.TrT, rule=constraint_outflow_limit_max)
    model.c_outflow_limit_min = Constraint(model.TrT, rule=constraint_outflow_limit_min)
    model.c_storage_limit_max = Constraint(model.T  , rule=constraint_storage_limit_max)
    model.c_storage_limit_min = Constraint(model.T  , rule=constraint_storage_limit_min)
    model.c_tram              = Constraint(model.T  , rule=constraint_tram_activation  )
    model.c_thershold_efficiency    = Constraint(model.TrT, rule=constraint_thershold_efficiency)

    def objective(m):
        return sum([ m.p_prices[t] * sum(m.v_outflow[tr,t] * m.p_elev_eff[tr] for tr in m.Tr) for t in m.T])

    model.obj = Objective(rule=objective, sense=maximize)

    return model
