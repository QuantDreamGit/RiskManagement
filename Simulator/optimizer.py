import os
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from .models import dro_mix_model
from .plotting import plot_results, plot_loss_distribution, plot_scenarios

def forward_price(s0, rd, rf, t):
    """
    Calculate the forward price.

    Parameters:
    s0 (float): Spot price of foreign currency
    rd (float): Domestic risk-free rate (continuously compounded)
    rf (float): Foreign risk-free rate (continuously compounded)
    t (float): Time to maturity

    Returns:
    float: Forward price
    """
    return s0 * np.exp((rd - rf) * t)


def call_price_bsm(s0, k, rd, rf, sigma, t):
    """
    Calculate the call option price using the Black-Scholes-Merton formula.

    Parameters:
    s0 (float): Spot price of foreign currency
    k (float): Strike price
    rd (float): Domestic risk-free rate (continuously compounded)
    rf (float): Foreign risk-free rate (continuously compounded)
    sigma (float): Volatility of price
    t (float): Time to maturity

    Returns:
    float: Call option price
    """
    # Calculate d1 and d2
    d1 = (np.log(s0 / k) + (rd - rf + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    # Calculate call price
    call_price = (s0 * np.exp(-rf * t) * norm.cdf(d1)) - (k * np.exp(-rd * t) * norm.cdf(d2))
    return call_price


def simulate_gbm(s0, rd, rf, sigma, t, n_steps):
    """
    Simulate Geometric Brownian Motion (GBM) under risk-neutral measure.

    Parameters:
    s0 (float): Initial spot price
    rd (float): Domestic risk-free rate
    rf (float): Foreign risk-free rate
    sigma (float): Volatility
    t (float): Time period (in years)
    n_steps (int): Number of time steps

    Returns:
    np.array: Simulated GBM time series
    """
    dt = t / n_steps
    time_series = [s0]

    for _ in range(n_steps - 1):
        z = np.random.normal(0, 1)
        x = (rd - rf - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        time_series.append(time_series[-1] * np.exp(x))

    return np.array(time_series)


def compute_var(money, s0, rd, rf, sigma, t, alpha):
    """
    Compute the Value at Risk (VaR) for a future spot price S_T modeled with GBM.
    Value at risk is the maximum loss that can occur at a given confidence level if the spot price falls below a certain
    threshold.

    Parameters:
    money (float): Amount of money to be hedged
    s0 (float): Initial spot price
    rd (float): Domestic risk-free rate
    rf (float): Foreign risk-free rate
    sigma (float): Volatility
    t (float): Time to maturity
    alpha (float): Confidence level (e.g., 0.95 for 95%)

    Returns:
    float: Value at Risk (VaR) at confidence level alpha
    """
    # Compute mean and standard deviation of log returns
    mu_X = (rd - rf - 0.5 * sigma ** 2) * t
    sigma_X = sigma * np.sqrt(t)
    # Compute quantile of the normal distribution (lower tail for loss)
    q_X = mu_X + norm.ppf(1 - alpha) * sigma_X
    st = s0 * np.exp(q_X)
    # Compute the loss (negative because VaR represents loss)
    loss = money * max((s0 - st), 0)

    return loss


def save_to_file(base_folder, alpha, coeff, rd, rf, n_scenarios, x, y, losses, obj_val, times, first_iteration):
    """
    Save optimization results to a file.

    Parameters:
    base_folder (str): Base folder to save the results
    alpha (float): Confidence level
    coeff (float): Risk aversion coefficient
    rd (float): Domestic risk-free rate
    rf (float): Foreign risk-free rate
    n_scenarios (int): Number of scenarios
    x (float): Optimized number of forward contracts
    y (float): Optimized number of call options
    losses (list of float): List of loss values from the optimization
    obj_val (float): Objective value of the optimization
    times (float): Optimization time
    first_iteration (bool): Flag indicating if it is the first iteration

    Returns:
    bool: False to indicate that it is not the first iteration
    """
    # Compute average and standard deviation of losses
    avg_loss = int(np.mean(losses))
    std_loss = int(np.std(losses))

    # Check if the folder exists, if not create it
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Check if the file exists, if not create it
    if not os.path.exists(f"{base_folder}/results.txt"):
        with open(f"{base_folder}/results.txt", "w") as file:
            file.write("Coeff;Alpha;Rd;Rf;n_scenarios;OptimizedX;OptimizedY;ObjVal;AvgLoss;StdDevLoss;OptimizationTime\n")
            file.write(f"{coeff}; {alpha}; {rd}; {rf}; {n_scenarios}; {x}; {y}; {obj_val}; {avg_loss}; {std_loss}; {round(times, 3)}\n")
    # If the file exists and is the first iteration, remove the file and create a new one
    elif first_iteration:
        # if the file exists and is the first iteration, remove the file and create a new one
        os.remove(f"{base_folder}/results.txt")
        with open(f"{base_folder}/results.txt", "w") as file:
            file.write("Coeff;Alpha;Rd;Rf;n_scenarios;OptimizedX;OptimizedY;ObjVal;AvgLoss;StdDevLoss;OptimizationTime\n")
            file.write(f"{coeff}; {alpha}; {rd}; {rf}; {n_scenarios}; {x}; {y}; {obj_val}; {avg_loss}; {std_loss}; {round(times, 3)}\n")
    else:
        # otherwise append the results
        with open(f"{base_folder}/results.txt", "a") as file:
            file.write(f"{coeff}; {alpha}; {rd}; {rf}; {n_scenarios}; {x}; {y}; {obj_val}; {avg_loss}; {std_loss}; {round(times, 3)}\n")
    # Return False to indicate that it is not the first iteration
    return False


def run_simulation(s0, rds, rfs, sigma, t, n_steps, k, benchmark_cost, foreign_currency,
                   model, alphas, coeffs, scenarios, env, base_folder,
                   model_parameters=None, single_run=False, debug_model=False, save_loss_distributions=False):
    """
    Run the simulation for different parameters and save the results.

    Parameters:
    s0 (float): Initial spot price
    rds (list of float): List of domestic risk-free rates
    rfs (list of float): List of foreign risk-free rates
    sigma (float): Volatility
    t (float): Time period (in years)
    n_steps (int): Number of time steps
    k (list of float): List of strike prices
    benchmark_cost (float): Benchmark cost of the company (value to compare the loss of the portfolio)
    foreign_currency (float): Amount of foreign currency to be hedged
    model (function): Optimization model function
    model_parameters (dict): Additional parameters for the optimization model
    alphas (list of float): List of confidence levels
    coeffs (list of float): List of risk aversion coefficients
    scenarios (list of int): List of number of scenarios
    env (object): Gurobi environment
    base_folder (str): Base folder to save the results
    Returns:
    None
    """
    if coeffs[0] is None:
        progress_bar = tqdm(total=len(alphas) * len(scenarios) * len(rds))
    else:
        progress_bar = tqdm(total=len(coeffs) * len(alphas) * len(scenarios) * len(rds))
    iteration = True

    for coeff in coeffs:
        for alpha in alphas:
            for i in range(len(rds)):
                rd = rds[i]
                rf = rfs[i]
                optimized_x, optimized_y, optimization_time = [], [], []
                for N_SCENARIO in scenarios:
                    if coeff is None:
                        progress_bar.set_description(f"Running simulation for "
                                                     f"alpha={alpha}, rd={rd}, rf={rf}, scenarios={N_SCENARIO}")
                    else:
                        progress_bar.set_description(f"Running simulation for "
                                                     f"coeff={coeff}, alpha={alpha}, rd={rd}, rf={rf}, scenarios={N_SCENARIO}")
                    # Define the probability of each scenario
                    pi = np.full(N_SCENARIO, 1 / N_SCENARIO)
                    # Check if the probabilities sum to 1
                    assert round(np.sum(pi), 5) == 1, "Probabilities must sum to 1"

                    # If model is DRO, set the fixed probabilities
                    if model == dro_mix_model:
                        # Define a discrete set of allowed probability values for DRO model
                        pi_dro = 1 / N_SCENARIO
                        fixed_probabilities = [pi_dro * i for i in model_parameters["allowed_probabilities"]]
                        # Add the fixed probabilities to the model parameters
                        model_parameters["fixed_probabilities"] = fixed_probabilities

                    # Define the trivial exchange cost for the company
                    # This can be computed in a smarter way if we consider that we can borrow money before usage!
                    trivial_exchange_cost = foreign_currency / s0
                    # Spot price time series
                    s = np.array([simulate_gbm(s0, rd, rf, sigma, t, n_steps) for _ in range(N_SCENARIO)])
                    # Last spot price of each scenario
                    s_t = s[:, -1]
                    # Forward price of the foreign currency
                    f = forward_price(s0, rd, rf, t)
                    # Call option price for each strike price
                    # Calculate the call option price for each strike price
                    c = [call_price_bsm(s0, ki, rd, rf, sigma, t) for ki in k]
                    # Compute the V@R
                    value_at_risk = compute_var(foreign_currency, s0, rd, rf, sigma, t, alpha=alpha)
                    # Plot the scenarios to verify V@R
                    # plot_scenarios(s, var=value_at_risk)

                    # Optimize model
                    x, y, losses, excess_losses, obj_val, times = model(
                        k, foreign_currency, pi, alpha, coeff, s_t, f, c,
                        value_at_risk, benchmark_cost, N_SCENARIO, env,
                        print_results=debug_model, model_parameters=model_parameters)

                    if save_loss_distributions:
                        try:
                            if coeff is None:
                                plot_loss_distribution(losses, value_at_risk, N_SCENARIO,
                                                       base_folder=base_folder, title=f"loss_alpha{alpha}_rd{rd}_rf{rf}")
                            else:
                                plot_loss_distribution(losses, value_at_risk, N_SCENARIO,
                                                       base_folder=base_folder, title=f"loss_coeff{coeff}_alpha{alpha}_rd{rd}_rf{rf}")
                        except Exception as e:
                            print(f"Error while saving loss distributions: {e}")

                    if single_run:
                        return x, y, losses, excess_losses, times

                    # Append results
                    optimized_x.append(x)
                    optimized_y.append(y)
                    optimization_time.append(times)

                    # Save results to file
                    iteration = save_to_file(
                        base_folder=base_folder,
                        alpha=alpha,
                        coeff=coeff,
                        rd=rd,
                        rf=rf,
                        n_scenarios=N_SCENARIO,
                        x=int(x),
                        y=[int(yi) for yi in y],
                        obj_val=int(obj_val),
                        losses=losses,
                        times=times,
                        first_iteration=iteration
                    )

                    # Update progress bar
                    progress_bar.update(1)

                # Plot the results
                # plot_results(alpha, rd, rf, k, optimized_x, optimized_y, optimization_time, base_folder, plot=False)