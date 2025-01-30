import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_results(path):
    # Load the data from a file
    df = pd.read_csv(f"{path}/results.txt", delimiter=";")

    # Convert necessary columns to numeric
    df["n_scenarios"] = df["n_scenarios"].astype(int)
    df["OptimizedX"] = df["OptimizedX"].astype(int)

    # Convert OptimizedY from string representation of list to sum of values removing the brackets and commas
    df["OptimizedY"] = df["OptimizedY"].apply(lambda x: x[2:-1])
    df["OptimizedY"] = df["OptimizedY"].apply(lambda x: sum([int(i) for i in x.split(",")]))

    return df

def plot_call_forward(path, alpha_value):
    # Load the data
    df = load_results(path)

    # Filter for a specific Alpha value
    df_filtered = df[df["Alpha"] == alpha_value]

    # Get unique (Rd, Rf) combinations
    rd_rf_combinations = df_filtered[["Rd", "Rf"]].drop_duplicates()
    num_combinations = len(rd_rf_combinations)

    # Create subplots
    fig, axes = plt.subplots(num_combinations, 1, figsize=(10, 5 * num_combinations), sharex=True)

    # Ensure axes is iterable if only one subplot
    if num_combinations == 1:
        axes = [axes]

    # Iterate over each (Rd, Rf) combination and plot
    for ax, (_, row) in zip(axes, rd_rf_combinations.iterrows()):
        rd, rf = row["Rd"], row["Rf"]
        df_subset = df_filtered[(df_filtered["Rd"] == rd) & (df_filtered["Rf"] == rf)]

        # Get values for the stacked bar plot
        scenarios = df_subset["n_scenarios"].astype(str)  # Convert to string for x-axis labels
        forward_calls = df_subset["OptimizedX"].values
        function_calls = df_subset["OptimizedY"].values

        # Create stacked bar plot
        bottom = np.zeros(len(scenarios))
        bars1 = ax.bar(scenarios, forward_calls, label="Forward", color="blue", alpha=0.7, bottom=bottom)
        bottom += forward_calls
        bars2 = ax.bar(scenarios, function_calls, label="Calls", color="orange", alpha=0.7, bottom=bottom)

        # Add value labels on top of each bar
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=10,
                    color="black")

        for bar in bars2:
            height = bar.get_height() + bar.get_y()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom", fontsize=10,
                    color="black")

        # Formatting
        ax.set_title(f"Alpha={alpha_value}, Rd={rd}, Rf={rf}")
        ax.set_ylabel("Calls and Forwards")
        ax.legend()

    # Set common x-axis label
    axes[-1].set_xlabel("Number of Scenarios")

    # Adjust layout
    plt.tight_layout()
    # Add a title to the figure
    fig.suptitle(f"Optimization Results for Different (Rd, Rf) Combinations with Alpha={alpha_value}", fontsize=16)
    # Add more space between the subplots and the title
    plt.subplots_adjust(top=0.93)
    # Show plot
    plt.savefig(f"{path}/forward_call_alpha{alpha_value}.png")

# plot_call_forward(path="output/mix_model", alpha_value=0.95)

path1 = "basic_model"
path2 = "mix_model"

df1 = load_results(path1)
df2 = load_results(path2)

# Set the Alpha value to filter
alpha_value = 0.95  # Change this as needed
df1_filtered = df1[df1["Alpha"] == alpha_value]
df2_filtered = df2[df2["Alpha"] == alpha_value]

# Get unique (Rd, Rf) combinations from both datasets
rd_rf_combinations = pd.concat([df1_filtered[["Rd", "Rf"]], df2_filtered[["Rd", "Rf"]]]).drop_duplicates()
num_combinations = len(rd_rf_combinations)

# Define colors and line styles for different Coeff values
colors = ["blue", "red", "green", "purple", "orange", "brown"]
linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1))]

# Create subplots: 3 rows, 2 columns
fig, axes = plt.subplots(num_combinations, 2, figsize=(12, 4 * num_combinations), sharex=True)

# Ensure axes is iterable even if there's only one row
if num_combinations == 1:
    axes = [axes]

# Iterate over each (Rd, Rf) combination and plot
for i, (_, row) in enumerate(rd_rf_combinations.iterrows()):
    rd, rf = row["Rd"], row["Rf"]

    df1_subset = df1_filtered[(df1_filtered["Rd"] == rd) & (df1_filtered["Rf"] == rf)]
    df2_subset = df2_filtered[(df2_filtered["Rd"] == rd) & (df2_filtered["Rf"] == rf)]

    # Get unique Coeff values from each dataset independently
    coeff_values_1 = sorted(df1_subset["Coeff"].dropna().unique())
    coeff_values_2 = sorted(df2_subset["Coeff"].dropna().unique())

    # Convert scenarios to string for x-axis
    scenarios_1 = df1_subset["n_scenarios"].astype(str).unique()
    scenarios_2 = df2_subset["n_scenarios"].astype(str).unique()

    # Plot AvgLoss Comparison (Left Column)
    ax1 = axes[i, 0]
    if len(coeff_values_1) > 0:
        for j, coeff in enumerate(coeff_values_1):
            df_coeff = df1_subset[df1_subset["Coeff"] == coeff]

            avg_loss = df_coeff["AvgLoss"].values
            scenarios = df_coeff["n_scenarios"].astype(str)

            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            ax1.plot(scenarios, avg_loss, marker='o', label=f"{path1.split('/')[1]} Coeff={coeff}", linestyle=linestyle, color=color)
    else:
        ax1.plot(scenarios_1, df1_subset["AvgLoss"].values,
                 label=f"{path1.split('/')[1]}", linestyle='-', marker='o', color="black")

    if len(coeff_values_2) > 0:
        for j, coeff in enumerate(coeff_values_2):
            df_coeff = df2_subset[df2_subset["Coeff"] == coeff]

            avg_loss = df_coeff["AvgLoss"].values
            scenarios = df_coeff["n_scenarios"].astype(str)

            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            ax1.plot(scenarios_2, avg_loss, marker='s', label=f"{path2.split('/')[1]} Coeff={coeff}", linestyle=linestyle, color=color, alpha=0.7)
    else:
        ax1.plot(scenarios, df2_subset["AvgLoss"].values,
                 label=f"{path2.split('/')[1]}", linestyle='-', marker='s', color="black", alpha=0.7)

    ax1.set_title(f"AvgLoss Comparison (Rd={rd}, Rf={rf})")
    ax1.set_ylabel("AvgLoss")
    ax1.legend()

    # Plot StdDevLoss Comparison (Right Column)
    ax2 = axes[i, 1]
    if len(coeff_values_1) > 0:
        for j, coeff in enumerate(coeff_values_1):
            df_coeff = df1_subset[df1_subset["Coeff"] == coeff]

            stddev_loss = df_coeff["StdDevLoss"].values
            scenarios = df_coeff["n_scenarios"].astype(str)

            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            ax2.plot(scenarios, stddev_loss, marker='o', label=f"{path1.split('/')[1]} Coeff={coeff}", linestyle=linestyle, color=color)
    else:
        ax2.plot(scenarios_1, df1_subset["StdDevLoss"].values,
                 label=f"{path1.split('/')[1]}", linestyle='-', marker='o', color="black")

    if len(coeff_values_2) > 0:
        for j, coeff in enumerate(coeff_values_2):
            df_coeff = df2_subset[df2_subset["Coeff"] == coeff]

            stddev_loss = df_coeff["StdDevLoss"].values
            scenarios = df_coeff["n_scenarios"].astype(str)

            color = colors[j % len(colors)]
            linestyle = linestyles[j % len(linestyles)]

            ax2.plot(scenarios, stddev_loss, marker='s', label=f"{path2.split('/')[1]} Coeff={coeff}", linestyle=linestyle, color=color, alpha=0.7)
    else:
        ax2.plot(scenarios_2, df2_subset["StdDevLoss"].values,
                 label=f"{path2.split('/')[1]}", linestyle='-', marker='s', color="black", alpha=0.7)

    ax2.set_title(f"StdDevLoss Comparison (Rd={rd}, Rf={rf})")
    ax2.set_ylabel("StdDevLoss")
    ax2.legend()

# Set common x-axis label
for ax in axes[-1]:  # Last row
    ax.set_xlabel("Number of Scenarios")

# Adjust layout
plt.tight_layout()
fig.suptitle(f"Comparison of AvgLoss and StdDevLoss for Different (Rd, Rf) Combinations with Alpha={alpha_value}",
             fontsize=16)
plt.subplots_adjust(top=0.93)

# Show plot
plt.show()
