import os
import numpy as np
import matplotlib.pyplot as plt

def plot_scenarios(time_series, title="Generated Scenarios"):
    """
    Plot the time series data.

    Parameters:
    time_series (np.array): Each row is a time series, with the same length.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    for i, ts in enumerate(time_series):
        plt.plot(ts, label=f"Scenario {i+1}")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Spot Price")
    plt.show()

def plot_loss_distribution(losses, value_at_risk, n_scenarios):
    """
    Plot the distribution of losses and mark the Value at Risk (VaR).

    Parameters:
    losses (list of float): List of loss values.
    value_at_risk (float): Value at Risk (VaR) threshold.
    N_SCENARIOS (int): Number of scenarios.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(value_at_risk, color='red', linestyle='dashed', linewidth=2)
    plt.title(f"Loss Distribution (samples = {n_scenarios})")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend(["Value at Risk"])
    plt.show()

def plot_results(alpha, rd, rf, k, optimized_x, optimized_y, optimization_time, base_folder, plot=True):
    """
    Plot the results of the optimization and save the plots to files.

    Parameters:
    alpha (float): Confidence level.
    rd (float): Domestic risk-free rate.
    rf (float): Foreign risk-free rate.
    k (list of float): List of strike prices.
    optimized_x (list of float): List of optimized number of forward contracts.
    optimized_y (np.array): Array of optimized number of call options for each strike price.
    optimization_time (list of float): List of optimization times.
    base_folder (str): Base folder to save the results.
    plot (bool): Whether to display the plots.
    """
    # Check if the folder exists, if not create it
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    if not os.path.exists(f"{base_folder}/results"):
        os.makedirs(f"{base_folder}/results")
    if not os.path.exists(f"{base_folder}/time_measurements"):
        os.makedirs(f"{base_folder}/time_measurements")

    # Define the scenarios and convert the lists to numpy arrays
    scenarios = list(range(1, len(optimized_x) + 1))
    optimized_x = np.array(optimized_x)
    optimized_y = np.array(optimized_y)

    # Create the figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    # Define positions for alignment
    x_positions = np.arange(len(scenarios))  # X-axis positions for bar groups

    # Subplot 1: Line plot for forward contracts
    axes[0].plot(x_positions, optimized_x, marker='o', linestyle='-', linewidth=2, color='blue')
    axes[0].set_title(f"Number of Forward Contracts (alpha={alpha}, Rd={rd}, Rf={rf})", fontsize=14)
    axes[0].set_ylabel("Number of contracts", fontsize=12)
    axes[0].grid(alpha=0.5)

    # Subplot 2: Bar plot for call options
    width = 0.15  # Width of each bar
    for i, strike in enumerate(k):
        axes[1].bar(x_positions + i * width, optimized_y[:, i], width, label=f"Call Options (K={strike:.2f})")
    # Formatting the bar plot
    axes[1].set_title(f"Number of Call Options for Each Strike Price (alpha={alpha}, Rd={rd}, Rf={rf})", fontsize=14)
    axes[1].set_xlabel("Scenario", fontsize=12)
    axes[1].set_ylabel("Number of contracts", fontsize=12)
    axes[1].set_xticks(x_positions + (len(k) - 1) * width / 2)  # Center x-ticks
    axes[1].set_xticklabels(scenarios, fontsize=10)
    axes[1].legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(k), frameon=False)
    axes[1].grid(alpha=0.5, axis='y')
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig(f"{base_folder}/results/results_alpha{alpha}_rd{rd}_rf{rf}.png")
    plt.show() if plot else plt.close()

    # Plot the optimization time
    plt.figure(figsize=(10, 6))
    plt.plot(scenarios, optimization_time, marker='o', linestyle='-', linewidth=2, color='green')
    plt.title(f"Optimization Time for Different Number of Scenarios (alpha={alpha}, Rd={rd}, Rf={rf})", fontsize=14)
    plt.xlabel("Number of Scenarios", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    # Set log scale for y-axis
    plt.yscale('log')
    plt.grid(alpha=0.5)
    plt.savefig(f"{base_folder}/time_measurements/time_alpha{alpha}_rd{rd}_rf{rf}.png")
    plt.show() if plot else plt.close()
