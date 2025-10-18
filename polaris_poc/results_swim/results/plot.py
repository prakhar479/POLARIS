def main():
    """Main function to load data, process, and plot."""
    try:
        df_sca = pd.read_csv(SCA_FILE)
        df_vec = pd.read_csv(VEC_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the correct directory.")
        return

    # --- Color Palette ---
    # Soft, aesthetic colors inspired by the stacked area chart
    color_cyan = "#9FDFFF"  # Soft cyan/light blue
    color_mint = "#7CEFB8"  # Mint/turquoise
    color_green = "#CFFFCA"  # Light green
    color_lavender = "#C8B5FF"  # Soft lavender/purple
    color_coral = "#FFB3BA"  # Soft coral/pink

    # --- 1. Process Data for Each Plot ---
    # The module and name strings have been updated based on your diagnostic output.

    # Plot 1: Total Requests/sec (from inter-arrival times)
    requests_raw = get_vector_data(df_vec, "SWIM.arrivalMonitor", "interArrival:vector")
    if requests_raw is not None and not requests_raw.empty:
        # Rate is 1 / time between arrivals. Avoid division by zero.
        requests_raw["value"] = 1 / requests_raw["value"].replace(0, np.nan)
    requests_avg = periodic_average(requests_raw, EVALUATION_PERIOD)

    # Plot 2: Active Servers (time-weighted average)
    servers_raw = get_vector_data(df_vec, "SWIM.monitor", "activeServers:vector")
    servers_avg = periodic_time_weighted_average(servers_raw, EVALUATION_PERIOD)

    # Plot 3: Brownout Factor / Dimmer (time-weighted average)
    dimmer_raw = get_vector_data(df_vec, "SWIM.monitor", "brownoutFactor:vector")
    dimmer_avg = periodic_time_weighted_average(dimmer_raw, EVALUATION_PERIOD)

    # Plot 4: Mean Response Time (simple periodic average of request lifetimes)
    response_raw = get_vector_data(df_vec, "SWIM.sink", "lifeTime:vector")
    response_avg = periodic_average(response_raw, EVALUATION_PERIOD)

    # Plot 5: Cumulative Utility
    utility_raw = get_vector_data(df_vec, "SWIM.monitor", "utilityPeriod:vector")
    utility_cumulative = pd.DataFrame()
    if utility_raw is not None and not utility_raw.empty:
        utility_cumulative = utility_raw.copy()
        utility_cumulative["value"] = utility_cumulative["value"].cumsum()

    # --- 2. Create the Plots ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.patch.set_facecolor("#FAFAFA")  # Light background for the figure
    fig.suptitle(
        "Web Infrastructure Simulation Analysis", fontsize=16, fontweight="600", color="#2C3E50"
    )

    # Plot 1: Requests/sec
    if not requests_avg.empty:
        axes[0].fill_between(
            requests_avg["time"], requests_avg["value"], alpha=0.6, color=color_cyan, linewidth=0
        )
        axes[0].plot(
            requests_avg["time"],
            requests_avg["value"],
            linestyle="-",
            linewidth=2,
            color="#4DB8E8",
            alpha=0.9,
        )
        axes[0].set_ylabel("Requests/sec", fontsize=11, fontweight="500", color="#34495E")
        axes[0].set_title(
            "Total Requests per Second", fontsize=12, fontweight="600", color="#2C3E50", pad=10
        )

    # Plot 2: Active Servers
    if not servers_avg.empty:
        axes[1].fill_between(
            servers_avg["time"], servers_avg["value"], alpha=0.6, color=color_mint, linewidth=0
        )
        axes[1].plot(
            servers_avg["time"],
            servers_avg["value"],
            linestyle="-",
            linewidth=2,
            color="#3ACC8F",
            alpha=0.9,
        )
        axes[1].set_ylabel("Servers", fontsize=11, fontweight="500", color="#34495E")
        axes[1].set_title(
            "Active Servers (Time-Weighted Avg)",
            fontsize=12,
            fontweight="600",
            color="#2C3E50",
            pad=10,
        )

    # Plot 3: Brownout Factor
    if not dimmer_avg.empty:
        axes[2].fill_between(
            dimmer_avg["time"], 1 - dimmer_avg["value"], alpha=0.6, color=color_green, linewidth=0
        )
        axes[2].plot(
            dimmer_avg["time"],
            1 - dimmer_avg["value"],
            linestyle="-",
            linewidth=2,
            color="#8FD97E",
            alpha=0.9,
        )
        axes[2].set_ylabel("Factor", fontsize=11, fontweight="500", color="#34495E")
        axes[2].set_title(
            "Brownout Factor (Time-Weighted Avg)",
            fontsize=12,
            fontweight="600",
            color="#2C3E50",
            pad=10,
        )
        axes[2].set_ylim(0, 1.05)  # Brownout factor is typically between 0 and 1

    # Plot 4: Response Time
    if not response_avg.empty:
        axes[3].fill_between(
            response_avg["time"], response_avg["value"], alpha=0.6, color=color_coral, linewidth=0
        )
        axes[3].plot(
            response_avg["time"],
            response_avg["value"],
            linestyle="-",
            linewidth=2,
            color="#FF7B82",
            alpha=0.9,
        )
        axes[3].set_ylabel("Time (s)", fontsize=11, fontweight="500", color="#34495E")
        axes[3].set_title(
            "Mean Response Time (Request Lifetime)",
            fontsize=12,
            fontweight="600",
            color="#2C3E50",
            pad=10,
        )

    # Plot 5: Cumulative Utility
    if not utility_cumulative.empty:
        axes[4].fill_between(
            utility_cumulative["time"],
            utility_cumulative["value"],
            alpha=0.6,
            color=color_lavender,
            linewidth=0,
        )
        axes[4].plot(
            utility_cumulative["time"],
            utility_cumulative["value"],
            linestyle="-",
            linewidth=2,
            color="#9B7FE8",
            alpha=0.9,
        )
        axes[4].set_ylabel("Total Utility", fontsize=11, fontweight="500", color="#34495E")
        axes[4].set_title(
            "Cumulative Utility", fontsize=12, fontweight="600", color="#2C3E50", pad=10
        )

    # --- 3. Common Formatting ---
    for ax in axes:  # Apply to all 5 plots now
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#E0E0E0", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(colors="#7F8C8D", which="both")

    plt.xlabel("Time (s)", fontsize=11, fontweight="500", color="#34495E")
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
