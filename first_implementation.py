import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt

from gurobipy import GRB, quicksum
from scipy.stats import norm

def forward_price(s0, rd, rf, t):
    """
    Calculate the forward price.
    s0: Spot price of foreign currency
    rd: Domestic risk-free rate (continuously compounded)
    rf: Foreign risk-free rate (continuously compounded)
    t: Time to maturity
    """
    return s0 * np.exp((rd - rf) * t)

def call_price_bsm(s0, k, rd, rf, sigma, t):
    """
    Calculate the call option price using the Black-Scholes-Merton formula.
    s0: Spot price of foreign currency
    k: Strike price
    rd: Domestic risk-free rate (continuously compounded)
    rf: Foreign risk-free rate (continuously compounded)
    sigma: Volatility of price
    t: Time to maturity
    """
    # Calculate d1 and d2
    d1 = (np.log(s0 / k) + (rd - rf + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    # Calculate call price
    call_price = (s0 * np.exp(-rf * t) * norm.cdf(d1)) - (k * np.exp(-rd * t) * norm.cdf(d2))
    return call_price

def simulate_gbm(s0, mu, sigma, t, n_steps):
    """
    Simulate a Geometric Brownian Motion (GBM) time series.

    Parameters:
    s0 : float : Initial stock price
    mu : float : Drift (expected return)
    sigma : float : Volatility of the stock
    t : float : Total time period (in years)
    n_steps : int : Number of time steps

    Returns:
    np.array : Simulated GBM time series
    """
    dt = t / n_steps
    time_series = [s0]

    for _ in range(n_steps-1):
        z = np.random.normal(0, 1)
        ds = time_series[-1] * (mu * dt + sigma * np.sqrt(dt) * z)
        time_series.append(time_series[-1] + ds)

    return np.array(time_series)

def plot_scenarios(time_series, title="Generated Scenarios"):
    """
    Plot the time series data.

    Parameters:
    time_series : each row is a time series, with same length
    title : str : Title of the plot
    """
    plt.figure(figsize=(10, 6))
    for i, ts in enumerate(time_series):
        plt.plot(ts, label=f"Scenario {i+1}")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Spot Price")
    plt.show()

def compute_var(money, s0, rd, rf, sigma, t, alpha):
    """
    Compute the Value at Risk (VaR) for a future spot price S_T modeled with GBM.

    Parameters:
    money : float : Amount of money to be hedged
    s0 : float : Initial spot price
    rd : float : Domestic risk-free rate
    rf : float : Foreign risk-free rate
    sigma : float : Volatility
    t : float : Time to maturity
    alpha : float : Confidence level (e.g., 0.95 for 95%)

    Returns:
    float : Value at Risk (VaR) at confidence level alpha
    """
    # Compute mean and standard deviation of log returns
    mu_X = (rd - rf - 0.5 * sigma ** 2) * t
    sigma_X = sigma * np.sqrt(t)

    # Compute quantile of the normal distribution
    q_X = mu_X + norm.ppf(alpha) * sigma_X
    st = s0 * np.exp(q_X)

    # Compute the loss
    loss = money * (st - s0)

    # Compute VaR
    return loss

def print_gurobi_summary(model):
    """
    Print a summary of the Gurobi optimization results.

    Parameters:
    model : Gurobi Model : The optimized model.
    """
    # Check the optimization status
    status = model.Status
    print("========== Gurobi Optimization Summary ==========")

    if status == GRB.OPTIMAL:
        print("Status: Optimal solution found!")
        print(f"Objective Value: {model.ObjVal:.2f}")
        print(f"Runtime: {model.Runtime:.2f} seconds")
        print(f"Number of Variables: {model.NumVars}")
        print(f"Number of Constraints: {model.NumConstrs}")

        print("\nDecision Variables:")
        for var in model.getVars():
            print(f"{var.VarName} = {var.X:.2f}")

    elif status == GRB.INFEASIBLE:
        print("Status: Model is infeasible.")
    elif status == GRB.UNBOUNDED:
        print("Status: Model is unbounded.")
    else:
        print(f"Status: Optimization was stopped with status code {status}.")

    print("=================================================")

# Constants
N_SCENARIOS = 200                           # Number of scenarios
pi = np.full(N_SCENARIOS, 1/N_SCENARIOS)  # Probability of each scenario
assert round(np.sum(pi), 5) == 1, "Probabilities must sum to 1"

alpha = 0.95                        # Confidence level

s0 = 1                              # Initial spot price
rd = 0.05                           # Domestic risk-free rate
rf = 0.03                           # Foreign risk-free rate
sigma = 0.2                         # Volatility of price
mu = rd - rf - 0.5 * sigma**2       # Drift under the risk-neutral measure
t = 1                               # Years to maturity
n_steps = 365                       # Number of time steps
k = [s0*0.8, s0*1.0, s0*1.2]        # Strike prices

foreign_currency = 1e6              # Amount of need foreign currency at time t


# Define the trivial exchange cost for the company
# This can be computed in a smarter way if we consider that we can borrow money before usage!
trivial_exchange_cost = foreign_currency / s0

# Spot price time series
s = np.array([simulate_gbm(s0, mu, sigma, t, n_steps) for _ in range(N_SCENARIOS)])
# Last spot price of each scenario
s_t = s[:, -1]

# Forward price of the foreign currency
f = forward_price(s0, rd, rf, t)

# Call option price for each strike price
# Calculate the call option price for each strike price
c = np.array([call_price_bsm(s0, ki, rd, rf, sigma, t) for ki in k])

# Compute the V@R
value_at_risk = compute_var(foreign_currency, s0, rd, rf, sigma, t, alpha=alpha)

print("Gurobi Model ==================================================")
# Problem Definition
# min {v@r + 1/(1-alpha) * sum[pi_i * excess_loss_i]}
# s.t. excess_loss >= loss - v@r
#      excess_loss >= 0
#      h_i <= y_i
#
# Variables description:
# v@r: Value at Risk
# alpha: Confidence level
# pi_i: Probability of scenario i
# excess_loss_i: Loss of scenario i
# loss: Loss of the portfolio
# h_i: exercised call option of scenario i
# y_i: owned call option of scenario i

# Create a model
m = gp.Model("forex_currency_hedging")

# Create variables
x = m.addVar(name="Number of forward contracts", lb=0, vtype=GRB.CONTINUOUS)
y = m.addVars(len(k), name="Number of call options", lb=0, vtype=GRB.CONTINUOUS)
h = m.addVars(len(k), N_SCENARIOS, name="Exercised call options", lb=0, vtype=GRB.CONTINUOUS)
z = m.addVars(N_SCENARIOS, name="Currency to buy at time T", lb=0, vtype=GRB.CONTINUOUS)
w = m.addVars(N_SCENARIOS, name="Currency to sell at time T", lb=0, vtype=GRB.CONTINUOUS)
excess_loss = m.addVars(N_SCENARIOS, name="Excess loss", lb=0, vtype=GRB.CONTINUOUS)

# Add constraints
# Exercised call options
for i in range(N_SCENARIOS):
    for j in range(len(k)):
        m.addConstr(h[j, i] <= y[j], name=f"Exercised call options {j} scenario {i}")
        m.addConstr(h[j, i] >= 0, name=f"Exercised call options {j} scenario {i}")

# Add constraints to ensure that the company owns the total amount of foreign currency at time t
for i in range(N_SCENARIOS):
    m.addConstr(x + quicksum(h[j, i] for j in range(len(k))) + z[i] - w[i] == foreign_currency, name="Currency balance")

# Compute loss for each scenario
loss = []
for i in range(N_SCENARIOS):
    loss.append(x * f
                + z[i] * s_t[i] - w[i] * s_t[i]
                + quicksum(y[j] * c[j] for j in range(len(k)))
                + quicksum(h[j, i] * k[j] for j in range(len(k)))
                - s0 * foreign_currency)

# Add constraints for excess loss
for i in range(N_SCENARIOS):
    m.addConstr(excess_loss[i] >= loss[i] - value_at_risk, name=f"Excess loss {i}")
    m.addConstr(excess_loss[i] >= 0, name=f"Excess loss {i}")

# Objective: Minimize VaR + (1/(1-alpha)) * Expected Excess Loss
objective = value_at_risk + (1 / (1 - alpha)) * quicksum(pi[i] * excess_loss[i] for i in range(N_SCENARIOS))
m.setObjective(objective, GRB.MINIMIZE)

# Optimize the model
m.optimize()

# Print Results
print("Example results of last scenario =============================")
print(f"Probability of this scenario: \t{pi[-1]:.2f}")
print(f"Trivial Exchange Cost: \t\t\t{trivial_exchange_cost:.2f} $")
print(f"Value at risk: \t\t\t\t\t{value_at_risk:.2f} $ ({alpha*100}% confidence)")
print(f"Last Spot Price: \t\t\t\t{s_t[0]:.2f} $")
print(f"Forward Price: \t\t\t\t\t{f:.2f} $")
print(f"Call Option Price (K1): \t\t{c[0]:.2f} $")
print(f"Call Option Price (K2): \t\t{c[1]:.2f} $")
print(f"Call Option Price (K3): \t\t{c[2]:.2f} $")
print(f"X: {x.X}")
print(f"Y: {[y[j].X for j in range(len(k))]}")
print(f"H: {[h[j, 0].X for j in range(len(k))]}")
print(f"Z: {z[0].X}")
print(f"W: {w[0].X}")
print(f"Loss: {loss[0].getValue()}")
print(f"manual loss: {x.X * f} + {z[0].X} * {s_t[0]} - {w[0].X} * {s_t[0]} + {y[0].X} * {c[0]} + {h[0, 0].X} * {k[0]} - {s0 * foreign_currency} = {loss[0].getValue()}")
print(f"Excess Loss: {excess_loss[0].X}")
print("===============================================================\n")

# Print the summary of the optimization
# print_gurobi_summary(m)
# print(excess_loss)

# Plot time series
# plot_scenarios(s)
