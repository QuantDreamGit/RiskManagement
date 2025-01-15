"""
Problem Description:
We want to hedge the risk of a company that needs to buy foreign currency at time t.
The company can use forward contracts and call options to hedge the risk.

Assumptions:
Domestic currency: EUR
Foreign currency: USD

We will analyze 3 main scenarios:
    1.  The interest rate in the domestic country is higher than the foreign country. (strong EUR)
        Then, rd = 0.08 and rf = 0.02

    2.  The interest rate in the domestic country is equal to the foreign country. (neutral)
        Then, rd = 0.05 and rf = 0.05

    3.  The interest rate in the domestic country is lower than the foreign country. (weak EUR)
        Then, rd = 0.02 and rf = 0.08
"""
from Simulator.optimizer import run_simulation
from Simulator.models import basic_model
from Simulator.utils import set_environment

# Set the environment
env = set_environment()

# Constants
SCENARIOS = [100, 1_000, ]
ALPHAS = [0.68, 0.95, 0.99]         # Confidence level
foreign_currency = 1e6              # Amount of need foreign currency at time t
s0 = 1                              # Initial spot price
rds = [0.08, 0.05, 0.02]            # Domestic risk-free rate (EUR)
rfs = [0.02, 0.05, 0.08]            # Foreign risk-free rate (USD)
sigma = 0.2                         # Volatility of price
t = 1                               # Years to maturity
n_steps = 365                       # Number of time steps
k = [s0*0.8,                        # Strike prices
     s0*0.9,
     s0*1.0,
     s0*1.1,
     s0*1.2]

# Run the simulation
run_simulation(s0=s0,
               rds=rds,
               rfs=rfs,
               sigma=sigma,
               t=t,
               n_steps=n_steps,
               k=k,
               benchmark_cost=s0*foreign_currency,
               foreign_currency=foreign_currency,
               model=basic_model,
               model_parameters=None,
               alphas=ALPHAS,
               scenarios=SCENARIOS,
               env=env,
               base_folder="output/base_model")



