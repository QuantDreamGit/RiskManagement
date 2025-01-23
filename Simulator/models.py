import gurobipy as gp
import numpy as np
from time import time
from gurobipy import GRB, quicksum

def basic_model(k, foreign_currency, pi, alpha, coeff, s_t, f, c,
                  value_at_risk, benchmark_cost, n_scenarios, env,
                  print_results=True, model_parameters=None):
    """
    Problem Definition
    min {v@r + 1/(1-alpha) * sum[pi_i * excess_loss_i]}
    s.t. excess_loss >= loss - v@r
         excess_loss >= 0
         gamma_i * (s_t - k) >= 0
         h_i <= y_i
         x + sum[h_i * gamma_i] + z - w == foreign_currency

    where:
    loss = x * f + z * s_t - w * s_t + sum[y_i * c_i] + sum[gamma_i * k_i * y_i] - benchmark_cost

    Variables description:
    v@r: Value at Risk
    alpha: Confidence level
    pi_i: Probability of scenario i
    excess_loss_i: Loss of scenario i
    loss: Loss of the portfolio
    h_i: Number of exercised call options of scenario i
    y_i: owned call option of scenario i
    """
    # Start the timer
    time_start = time()
    # Create a new model
    m = gp.Model("forex_currency_hedging", env=env)
    # Create variables
    x = m.addVar(name="Number of forward contracts", lb=0, vtype=GRB.CONTINUOUS)
    y = m.addVars(len(k), name="Number of call options", lb=0, vtype=GRB.CONTINUOUS)
    z = m.addVars(n_scenarios, name="Currency to buy at time T", lb=0, vtype=GRB.CONTINUOUS)
    w = m.addVars(n_scenarios, name="Currency to sell at time T", lb=0, vtype=GRB.CONTINUOUS)
    gamma = m.addVars(len(k), n_scenarios, name="Exercise condition", lb=0, vtype=GRB.BINARY)
    h = m.addVars(len(k), n_scenarios, name="Number of exercised call options", lb=0, vtype=GRB.CONTINUOUS)
    excess_loss = m.addVars(n_scenarios, name="Excess loss", lb=0, vtype=GRB.CONTINUOUS)

    # Add constraints
    for i in range(n_scenarios):
        for j in range(len(k)):
            # Exercised call options
            m.addConstr(gamma[j, i] * (s_t[i] - k[j]) >= 0, name=f"Payoff call options {j} scenario {i}")
            # Add constraints to ensure that the company exercises the call options that they own
            m.addConstr(h[j, i] <= y[j])
            m.addConstr(h[j, i] >= 0)

    for i in range(n_scenarios):
        # Add constraints to ensure that the company owns the total amount of foreign currency at time t
        m.addConstr(x
                    + quicksum(gamma[j, i] * h[j, i] for j in range(len(k)))
                    + z[i] - w[i]
                    == foreign_currency, name="Currency balance")

    loss = []
    for i in range(n_scenarios):
        # Compute the loss of the portfolio
        loss.append(x * f
                    + z[i] * s_t[i] - w[i] * s_t[i]
                    + quicksum(y[j] * c[j] for j in range(len(k)))
                    + quicksum(gamma[j, i] * k[j] * h[j, i] for j in range(len(k)))
                    - benchmark_cost)

        # Add constraints for excess loss
        m.addConstr(excess_loss[i] >= loss[i] - value_at_risk, name=f"Excess loss {i}")
        m.addConstr(excess_loss[i] >= 0, name=f"Excess loss {i}")

    # Objective Function
    objective = (value_at_risk
                 + (1 / (1 - alpha)) * quicksum(pi[i] * excess_loss[i] for i in range(n_scenarios)))
    # Set objective
    m.setObjective(objective, GRB.MINIMIZE)
    # Optimize the model
    m.optimize()

    if m.Status == GRB.OPTIMAL and print_results:
        # Print Results
        print("Example results of last scenario =============================")
        print(f"Probability of this scenario: \t{pi[-1]:.2f}")
        print(f"Benchmark cost: \t\t\t{benchmark_cost:.2f} $")
        print(f"Value at risk: \t\t\t\t\t{value_at_risk:.2f} $ ({alpha * 100}% confidence)")
        print(f"Last Spot Price: \t\t\t\t{s_t[0]:.2f} $")
        print(f"Forward Price: \t\t\t\t\t{f:.2f} $")
        print(f"Call Option Price (K = {k[0]} $): \t{c[0]:.4f} $")
        print(f"Call Option Price (K = {k[1]}$): \t{c[1]:.4f} $")
        print(f"Call Option Price (K = {k[2]} $): \t{c[2]:.4f} $")
        print(f"Call Option Price (K = {k[3]}$): \t{c[3]:.4f} $")
        print(f"Call Option Price (K = {k[4]} $): \t{c[4]:.4f} $")
        print("Gurobi Model ==================================================")
        print(f"X: {x.X}")
        print(f"Y: {[y[j].X for j in range(len(k))]}")
        print(f"h: {[h[j, 0].X for j in range(len(k))]}")
        print(f"gamma: {[gamma[j, 0].X for j in range(len(k))]}")
        print(f"Z: {z[0].X}")
        print(f"W: {w[0].X}")
        print(f"Loss: {loss[0].getValue()}")
        print(f"AVG Loss: {np.mean([loss_i.getValue() for loss_i in loss])} $")
        print(f"AVG Excess Loss: {np.mean([excess_loss[i].X for i in range(n_scenarios)])} $")
        print("===============================================================\n")
    # Return results
    end_time = time()

    # Return the results
    if m.Status == GRB.OPTIMAL:
        return (x.X, [y[j].X for j in range(len(k))],
                [loss_i.getValue() for loss_i in loss], [excess_loss[i].X for i in range(n_scenarios)],
                m.objVal, end_time - time_start)
    else:
        return None, None, None, None, None, end_time - time_start

def var_model(k, foreign_currency, pi, alpha, coeff, s_t, f, c,
                  value_at_risk, benchmark_cost, n_scenarios, env,
                  print_results=True, model_parameters=None):
    """
    Problem Definition
    min {v@r + 1/(1-alpha) * sum[pi_i * excess_loss_i] + rho * sum[pi_i * (excess_loss_i)^2]}
    s.t. excess_loss >= loss - v@r
         excess_loss >= 0
         gamma_i * (s_t - k) >= 0
         h_i <= y_i
         x + sum[h_i * gamma_i] + z - w == foreign_currency

    where:
    loss = x * f + z * s_t - w * s_t + sum[y_i * c_i] + sum[gamma_i * k_i * y_i] - benchmark_cost

    Variables description:
    v@r: Value at Risk
    alpha: Confidence level
    pi_i: Probability of scenario i
    excess_loss_i: Loss of scenario i
    loss: Loss of the portfolio
    h_i: Number of exercised call options of scenario i
    y_i: owned call option of scenario i
    rho: Multiplier for the constraint sum[pi_i * (excess_loss_i)^2]
    """
    # Check if the model parameters are provided
    if model_parameters is None:
        rho = coeff
    # Otherwise, get the value from the dictionary
    else:
        rho = model_parameters["rho"]

    # Start the timer
    time_start = time()
    # Create a new model
    m = gp.Model("forex_currency_hedging", env=env)
    # Create variables
    x = m.addVar(name="Number of forward contracts", lb=0, vtype=GRB.CONTINUOUS)
    y = m.addVars(len(k), name="Number of call options", lb=0, vtype=GRB.CONTINUOUS)
    z = m.addVars(n_scenarios, name="Currency to buy at time T", lb=0, vtype=GRB.CONTINUOUS)
    w = m.addVars(n_scenarios, name="Currency to sell at time T", lb=0, vtype=GRB.CONTINUOUS)
    gamma = m.addVars(len(k), n_scenarios, name="Exercise condition", lb=0, vtype=GRB.BINARY)
    h = m.addVars(len(k), n_scenarios, name="Number of exercised call options", lb=0, vtype=GRB.CONTINUOUS)
    excess_loss = m.addVars(n_scenarios, name="Excess loss", lb=0, vtype=GRB.CONTINUOUS)

    # Add constraints
    for i in range(n_scenarios):
        for j in range(len(k)):
            # Exercised call options
            m.addConstr(gamma[j, i] * (s_t[i] - k[j]) >= 0, name=f"Payoff call options {j} scenario {i}")
            # Add constraints to ensure that the company exercises the call options that they own
            m.addConstr(h[j, i] <= y[j])
            m.addConstr(h[j, i] >= 0)

    for i in range(n_scenarios):
        # Add constraints to ensure that the company owns the total amount of foreign currency at time t
        m.addConstr(x
                    + quicksum(gamma[j, i] * h[j, i] for j in range(len(k)))
                    + z[i] - w[i]
                    == foreign_currency, name="Currency balance")

    loss = []
    for i in range(n_scenarios):
        # Compute the loss of the portfolio
        loss.append(x * f
                    + z[i] * s_t[i] - w[i] * s_t[i]
                    + quicksum(y[j] * c[j] for j in range(len(k)))
                    + quicksum(gamma[j, i] * k[j] * h[j, i] for j in range(len(k)))
                    - benchmark_cost)

        # Add constraints for excess loss
        m.addConstr(excess_loss[i] >= loss[i] - value_at_risk, name=f"Excess loss {i}")
        m.addConstr(excess_loss[i] >= 0, name=f"Excess loss {i}")

    # Objective Function
    objective = (value_at_risk
                 + (1 / (1 - alpha)) * quicksum(pi[i] * excess_loss[i] for i in range(n_scenarios))
                 + rho * quicksum(pi[i] * excess_loss[i] ** 2 for i in range(n_scenarios)))

    # Set objective
    m.setObjective(objective, GRB.MINIMIZE)
    # Optimize the model
    m.optimize()

    if m.Status == GRB.OPTIMAL and print_results:
        # Print Results
        print("Example results of last scenario =============================")
        print(f"Probability of this scenario: \t{pi[-1]:.2f}")
        print(f"Benchmark cost: \t\t\t{benchmark_cost:.2f} $")
        print(f"Value at risk: \t\t\t\t\t{value_at_risk:.2f} $ ({alpha * 100}% confidence)")
        print(f"Last Spot Price: \t\t\t\t{s_t[0]:.2f} $")
        print(f"Forward Price: \t\t\t\t\t{f:.2f} $")
        print(f"Call Option Price (K = {k[0]} $): \t{c[0]:.4f} $")
        print(f"Call Option Price (K = {k[1]}$): \t{c[1]:.4f} $")
        print(f"Call Option Price (K = {k[2]} $): \t{c[2]:.4f} $")
        print(f"Call Option Price (K = {k[3]}$): \t{c[3]:.4f} $")
        print(f"Call Option Price (K = {k[4]} $): \t{c[4]:.4f} $")
        print("Gurobi Model ==================================================")
        print(f"X: {x.X}")
        print(f"Y: {[y[j].X for j in range(len(k))]}")
        print(f"h: {[h[j, 0].X for j in range(len(k))]}")
        print(f"gamma: {[gamma[j, 0].X for j in range(len(k))]}")
        print(f"Z: {z[0].X}")
        print(f"W: {w[0].X}")
        print(f"Loss: {loss[0].getValue()}")
        print(f"AVG Loss: {np.mean([loss_i.getValue() for loss_i in loss])} $")
        print(f"AVG Excess Loss: {np.mean([excess_loss[i].X for i in range(n_scenarios)])} $")
        print("===============================================================\n")

    # Return results
    end_time = time()

    # Return the results
    if m.Status == GRB.OPTIMAL:
        return (x.X, [y[j].X for j in range(len(k))],
                [loss_i.getValue() for loss_i in loss], [excess_loss[i].X for i in range(n_scenarios)],
                m.objVal, end_time - time_start)
    else:
        return None, None, None, None, None, end_time - time_start

def var_mix_model(k, foreign_currency, pi, alpha, coeff, s_t, f, c,
                  value_at_risk, benchmark_cost, n_scenarios, env,
                  print_results=True, model_parameters=None):
    """
    Problem Definition
    min {v@r + 1/(1-alpha) * sum[pi_i * excess_loss_i] + rho * sum[pi_i * (excess_loss_i)^2] + lambda * (x - sum[y_i])^2}
    s.t. excess_loss >= loss - v@r
         excess_loss >= 0
         gamma_i * (s_t - k) >= 0
         h_i <= y_i
         x + sum[h_i * gamma_i] + z - w == foreign_currency

    where:
    loss = x * f + z * s_t - w * s_t + sum[y_i * c_i] + sum[gamma_i * k_i * y_i] - benchmark_cost

    Variables description:
    v@r: Value at Risk
    alpha: Confidence level
    pi_i: Probability of scenario i
    excess_loss_i: Loss of scenario i
    loss: Loss of the portfolio
    h_i: Number of exercised call options of scenario i
    y_i: owned call option of scenario i
    rho: Multiplier for the constraint sum[pi_i * (excess_loss_i)^2]
    lambda: Multiplier for the constraint x - sum(y_i)
    """
    # Check if the model parameters are provided
    if model_parameters is None:
        rho, lambda_ = coeff, coeff
    # Otherwise, get the value from the dictionary
    else:
        rho, lambda_ = model_parameters["rho"], model_parameters["lambda"]

    # Start the timer
    time_start = time()
    # Create a new model
    m = gp.Model("forex_currency_hedging", env=env)
    # Create variables
    x = m.addVar(name="Number of forward contracts", lb=0, vtype=GRB.CONTINUOUS)
    y = m.addVars(len(k), name="Number of call options", lb=0, vtype=GRB.CONTINUOUS)
    z = m.addVars(n_scenarios, name="Currency to buy at time T", lb=0, vtype=GRB.CONTINUOUS)
    w = m.addVars(n_scenarios, name="Currency to sell at time T", lb=0, vtype=GRB.CONTINUOUS)
    gamma = m.addVars(len(k), n_scenarios, name="Exercise condition", lb=0, vtype=GRB.BINARY)
    h = m.addVars(len(k), n_scenarios, name="Number of exercised call options", lb=0, vtype=GRB.CONTINUOUS)
    excess_loss = m.addVars(n_scenarios, name="Excess loss", lb=0, vtype=GRB.CONTINUOUS)

    # Add constraints
    for i in range(n_scenarios):
        for j in range(len(k)):
            # Exercised call options
            m.addConstr(gamma[j, i] * (s_t[i] - k[j]) >= 0, name=f"Payoff call options {j} scenario {i}")
            # Add constraints to ensure that the company exercises the call options that they own
            m.addConstr(h[j, i] <= y[j])
            m.addConstr(h[j, i] >= 0)

    for i in range(n_scenarios):
        # Add constraints to ensure that the company owns the total amount of foreign currency at time t
        m.addConstr(x
                    + quicksum(gamma[j, i] * h[j, i] for j in range(len(k)))
                    + z[i] - w[i]
                    == foreign_currency, name="Currency balance")

    loss = []
    for i in range(n_scenarios):
        # Compute the loss of the portfolio
        loss.append(x * f
                    + z[i] * s_t[i] - w[i] * s_t[i]
                    + quicksum(y[j] * c[j] for j in range(len(k)))
                    + quicksum(gamma[j, i] * k[j] * h[j, i] for j in range(len(k)))
                    - benchmark_cost)

        # Add constraints for excess loss
        m.addConstr(excess_loss[i] >= loss[i] - value_at_risk, name=f"Excess loss {i}")
        m.addConstr(excess_loss[i] >= 0, name=f"Excess loss {i}")

    # Objective Function
    objective = (value_at_risk
                 + (1 / (1 - alpha)) * quicksum(pi[i] * excess_loss[i] for i in range(n_scenarios))
                 + rho * quicksum(pi[i] * excess_loss[i] ** 2 for i in range(n_scenarios))
                 + lambda_ * (x - quicksum(y[j] for j in range(len(k)))) ** 2)

    # Set objective
    m.setObjective(objective, GRB.MINIMIZE)
    # Optimize the model
    m.optimize()

    if m.Status == GRB.OPTIMAL and print_results:
        # Print Results
        print("Example results of last scenario =============================")
        print(f"Probability of this scenario: \t{pi[-1]:.2f}")
        print(f"Benchmark cost: \t\t\t{benchmark_cost:.2f} $")
        print(f"Value at risk: \t\t\t\t\t{value_at_risk:.2f} $ ({alpha * 100}% confidence)")
        print(f"Last Spot Price: \t\t\t\t{s_t[0]:.2f} $")
        print(f"Forward Price: \t\t\t\t\t{f:.2f} $")
        print(f"Call Option Price (K = {k[0]} $): \t{c[0]:.4f} $")
        print(f"Call Option Price (K = {k[1]}$): \t{c[1]:.4f} $")
        print(f"Call Option Price (K = {k[2]} $): \t{c[2]:.4f} $")
        print(f"Call Option Price (K = {k[3]}$): \t{c[3]:.4f} $")
        print(f"Call Option Price (K = {k[4]} $): \t{c[4]:.4f} $")
        print("Gurobi Model ==================================================")
        print(f"X: {x.X}")
        print(f"Y: {[y[j].X for j in range(len(k))]}")
        print(f"h: {[h[j, 0].X for j in range(len(k))]}")
        print(f"gamma: {[gamma[j, 0].X for j in range(len(k))]}")
        print(f"Z: {z[0].X}")
        print(f"W: {w[0].X}")
        print(f"Loss: {loss[0].getValue()}")
        print(f"AVG Loss: {np.mean([loss_i.getValue() for loss_i in loss])} $")
        print(f"AVG Excess Loss: {np.mean([excess_loss[i].X for i in range(n_scenarios)])} $")
        print("===============================================================\n")

    # Return results
    end_time = time()

    # Return the results
    if m.Status == GRB.OPTIMAL:
        return (x.X, [y[j].X for j in range(len(k))],
                [loss_i.getValue() for loss_i in loss], [excess_loss[i].X for i in range(n_scenarios)],
                m.objVal, end_time - time_start)
    else:
        return None, None, None, None, None, end_time - time_start