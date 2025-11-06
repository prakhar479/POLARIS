import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Define consistent color palette - matching the other plot.py file exactly
COLORS = {
    "cyan": "#00E5FF",  # Bright cyan/deep sky blue
    "teal": "#5BFCFF",  # Bright turquoise
    "periwinkle": "#6495ED",  # Bright cornflower blue
    "coral": "#FE9C8A",  # Bright tomato/coral
    "green": "#00E66F",
    "violet": "#9999FF",  # Bright medium purple
    "neutral": "#E0E0E0",  # Light gray for grid
    "threshold": "#95A5A6",  # Gray for threshold lines
}

# Set global font sizes to match annotation font size (14)
plt.rcParams.update(
    {
        "font.size": 13,  # Default font size
        "axes.labelsize": 13,  # Axis label font size
        "axes.titlesize": 13,  # Title font size
        "xtick.labelsize": 12,  # X-axis tick label font size
        "ytick.labelsize": 12,  # Y-axis tick label font size
        "legend.fontsize": 13,  # Legend font size
        "figure.titlesize": 13,  # Figure title font size
    }
)


def periodic_average(df, period):
    """Computes the average of an observed metric for each period"""
    start = np.floor(df["x"].min() / period) * period
    end = np.ceil(df["x"].max() / period) * period
    intervals = np.arange(start + period, end + period, period)

    # Find which interval each x value belongs to
    interval_indices = np.searchsorted(intervals, df["x"], side="right")

    # Group by interval and compute mean
    result = df.groupby(interval_indices)["y"].mean().reset_index(drop=True)

    # Create result dataframe
    avg_df = pd.DataFrame({"x": intervals[: len(result)], "y": result.values})

    return avg_df


def time_weighted_average(df, period):
    """Computes the time-weighted average of an observed metric for each period"""
    start = np.floor(df["x"].min() / period) * period
    end = np.ceil(df["x"].max() / period) * period

    if df["x"].max() == end:
        end = end + period

    intervals = np.arange(start, end + period, period)

    # Find missing intervals and add observations
    missing = intervals[~np.isin(intervals, df["x"].values)]

    if len(missing) > 0:
        missing_rows = []
        for t in missing:
            # Find last observation before time t
            mask = df["x"] < t
            if mask.any():
                last_y = df.loc[mask, "y"].iloc[-1]
                missing_rows.append({"x": t, "y": last_y})

        if missing_rows:
            df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

    # Sort by time
    df = df.sort_values("x").reset_index(drop=True)

    # Compute the end of the interval each observation belongs to
    interval_indices = np.searchsorted(intervals, df["x"].values, side="right") - 1
    end_of_interval = intervals[interval_indices] + period

    # Expand df with weight column
    if len(df) > 1:
        next_x = df["x"].shift(-1)
        weights = np.minimum(next_x[:-1], end_of_interval[:-1]) - df["x"][:-1]

        expdf = df[:-1].copy()
        expdf["weight"] = weights.values

        # Compute weighted mean for each interval
        expdf["interval"] = interval_indices[:-1]

        wm = (
            expdf.groupby("interval")
            .apply(lambda x: np.average(x["y"], weights=x["weight"]))
            .values
        )

        result = pd.DataFrame({"x": intervals[:-1][: len(wm)], "y": wm})

        return result

    return pd.DataFrame({"x": [], "y": []})


def period_utility_SEAMS2017A(
    max_servers,
    max_service_rate,
    arrival_rate_mean,
    dimmer,
    evaluation_period,
    RT_THRESHOLD,
    avg_response_time,
    avg_servers,
):
    """Utility function from SEAMS 2017 paper"""
    basic_revenue = 1
    opt_revenue = 1.5
    server_cost = 10
    precision = 1e-5

    max_throughput = max_servers * max_service_rate

    Ur = arrival_rate_mean * ((1 - dimmer) * basic_revenue + dimmer * opt_revenue)
    Uc = server_cost * (max_servers - avg_servers)
    UrOpt = arrival_rate_mean * opt_revenue

    if avg_response_time <= RT_THRESHOLD and Ur >= UrOpt - precision:
        utility = Ur + Uc
    elif avg_response_time <= RT_THRESHOLD:
        utility = Ur
    else:
        utility = min(0.0, arrival_rate_mean - max_throughput) * opt_revenue

    return utility


def period_utility_ICAC2016(
    max_servers,
    max_service_rate,
    arrival_rate_mean,
    dimmer,
    evaluation_period,
    RT_THRESHOLD,
    avg_response_time,
    avg_servers,
):
    """Utility function from ICAC 2016 paper"""
    basic_revenue = 1
    opt_revenue = 1.5
    max_throughput = max_servers * max_service_rate
    late_penalty = max_throughput * opt_revenue

    positive_utility = round(
        arrival_rate_mean * ((1 - dimmer) * basic_revenue + dimmer * opt_revenue)
    )

    if avg_response_time > RT_THRESHOLD or avg_response_time < 0:
        utility = min(0.0, arrival_rate_mean * opt_revenue - late_penalty)
    else:
        utility = positive_utility

    return utility


def read_vector(dbconn, vector_name, module_name=None):
    """Read vector data from SQLite database"""
    query = f"""SELECT simtimeRaw/1e12 as x, CAST(value as REAL) as y 
                FROM vector NATURAL JOIN vectorData
                WHERE vectorName='{vector_name}'"""

    if module_name is not None:
        query += f" AND moduleName='{module_name}'"

    df = pd.read_sql_query(query, dbconn)
    return df


def plot_results(
    config,
    folder="SWIM",
    run=0,
    save_as=None,
    instantaneous_utility=False,
    period_grid=False,
    utility_fc=period_utility_SEAMS2017A,
    brief=False,
):
    """Main plotting function that mirrors the R script logic"""
    USE_COMPUTED_UTILITY = True

    basedir = "./"

    # Read scalar database
    scalar_db_path = f"{basedir}{folder}/{config}-{run}.sca"
    sdb = sqlite3.connect(scalar_db_path)
    scalars = pd.read_sql_query("SELECT * FROM scalar", sdb)

    # Get network name
    network = scalars[scalars["scalarName"] == "maxServers"]["moduleName"].iloc[0]

    # Get scalar values
    boot_delay = scalars[scalars["scalarName"] == "bootDelay"]["scalarValue"].iloc[0]
    evaluation_period = scalars[scalars["scalarName"] == "evaluationPeriod"]["scalarValue"].iloc[0]
    RT_THRESHOLD_SEC = scalars[scalars["scalarName"] == "responseTimeThreshold"][
        "scalarValue"
    ].iloc[0]
    max_servers = scalars[scalars["scalarName"] == "maxServers"]["scalarValue"].iloc[0]
    max_service_rate = scalars[scalars["scalarName"] == "maxServiceRate"]["scalarValue"].iloc[0]

    # Read vector database
    vector_db_path = f"{basedir}{folder}/{config}-{run}.vec"
    vdb = sqlite3.connect(vector_db_path)

    servers = read_vector(vdb, "serverCost:vector")
    active_servers = read_vector(vdb, "activeServers:vector")
    dimmer_raw = read_vector(vdb, "brownoutFactor:vector")
    dimmer = dimmer_raw.copy()
    dimmer["y"] = 1 - dimmer["y"]

    responses = read_vector(vdb, "lifeTime:vector")
    low_responses = read_vector(vdb, "lifeTime:vector", module_name=f"{network}.sinkLow")

    pct_low = 100 * len(low_responses) / len(responses)
    pct_late = 100 * (responses["y"] > RT_THRESHOLD_SEC).sum() / len(responses)

    avg_interarrival = periodic_average(read_vector(vdb, "interArrival:vector"), evaluation_period)
    avg_arrival_rate = avg_interarrival.copy()
    avg_arrival_rate["y"] = 1 / avg_arrival_rate["y"]

    start = np.floor(servers["x"].min() / evaluation_period) * evaluation_period
    end = np.ceil(servers["x"].max() / evaluation_period) * evaluation_period

    # Find mean response time for each interval
    avg_response = periodic_average(responses, evaluation_period)

    if USE_COMPUTED_UTILITY:
        print(f"Computing utility with {utility_fc.__name__}")

        # Compute weighted mean for brownout factor
        dimmer_mean = time_weighted_average(dimmer, evaluation_period)
        dimmer_mean["x"] = dimmer_mean["x"] + evaluation_period

        # Compute weighted mean for servers
        servers_mean = time_weighted_average(servers, evaluation_period)
        servers_mean["x"] = servers_mean["x"] + evaluation_period

        # Trim all datasets
        avg_arrival_rate = avg_arrival_rate[avg_arrival_rate["x"] <= end]
        dimmer_mean = dimmer_mean[dimmer_mean["x"] <= end]
        servers_mean = servers_mean[servers_mean["x"] <= end]
        avg_response = avg_response[avg_response["x"] <= end]

        # Compute utility for each period
        utility_values = []
        for i in range(len(avg_response)):
            util = utility_fc(
                max_servers,
                max_service_rate,
                avg_arrival_rate["y"].iloc[i],
                dimmer_mean["y"].iloc[i],
                evaluation_period,
                RT_THRESHOLD_SEC,
                avg_response["y"].iloc[i],
                servers_mean["y"].iloc[i],
            )
            utility_values.append(util)

        utility = pd.DataFrame({"x": avg_response["x"].values, "y": utility_values})
        total_utility = utility["y"].sum()
    else:
        utility = read_vector(vdb, "utilityPeriod:vector")
        total_utility = scalars[scalars["scalarName"] == "utility:last"]["scalarValue"].iloc[0]

    # Create plots - only response time and utility
    num_plots = 2
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(num_plots, 1, figure=fig, hspace=0.2)

    plot_idx = 0

    # Plot 1: Response Time
    ax1 = fig.add_subplot(gs[plot_idx, 0])
    ax1.plot(avg_response["x"], avg_response["y"], color=COLORS["cyan"], alpha=0.85)
    ax1.axhline(
        y=RT_THRESHOLD_SEC, linestyle="--", color=COLORS["threshold"], linewidth=1.5, alpha=0.7
    )
    ax1.set_ylabel("resp. time (s)")
    ax1.set_ylim(top=6)
    ax1.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--", linewidth=1.5)
    ax1.set_xlim(start, end)
    plot_idx += 1

    # Plot 2: Cumulative Utility
    ax2 = fig.add_subplot(gs[plot_idx, 0], sharex=ax1)
    ax2.plot(utility["x"], np.cumsum(utility["y"]), color=COLORS["green"], alpha=0.85)
    ax2.set_ylabel("cum. utility")
    ax2.set_xlabel("time (s)")
    ax2.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--", linewidth=1.5)

    # Remove x-axis labels for all but bottom plot
    ax1.set_xlabel("")
    ax1.tick_params(labelbottom=False)

    # Calculate statistics
    avg_error = np.maximum(avg_response["y"] - RT_THRESHOLD_SEC, 0).sum() / (
        len(avg_response) * evaluation_period
    )
    overshoot = np.maximum(avg_response["y"] - RT_THRESHOLD_SEC, 0).max()

    # Prepare return values
    results = {
        "config": config,
        "total_utility": round(total_utility),
        "pct_optional": round(100 - pct_low, 1),
        "pct_late": round(pct_late, 1),
        "avg_servers": round(servers["y"].mean(), 1),
        "avg_error": avg_error,
        "overshoot": overshoot,
    }

    # Print table
    table_header = (
        "Config & Utility & % optional & % late & avg. servers & avg. error & overshoot\\\\\n"
    )
    table_row = f"{config} & {results['total_utility']} & {results['pct_optional']} & {results['pct_late']} & {results['avg_servers']} & {results['avg_error']} & {results['overshoot']} \\\\\n"
    print(table_header + table_row)

    plt.tight_layout()

    if save_as is not None:
        plt.savefig(save_as, dpi=2200, bbox_inches="tight")

    sdb.close()
    vdb.close()

    return fig, results


# def plot_overlapping_results(
#     folder1,
#     folder2,
#     folder3,
#     folder4,
#     config="sim",
#     run=1,
#     save_as=None,
#     utility_fc=period_utility_SEAMS2017A,
# ):
#     """
#     Creates overlapping plots from three different folders with different colors.
#     Includes vertical dotted reference lines at specified times.

#     Parameters:
#     folder1, folder2, folder3: Names of the three folders to compare
#     config: Configuration name (default: "sim")
#     run: Run number (default: 0)
#     save_as: Path to save the figure (optional)
#     utility_fc: Utility function to use (default: period_utility_SEAMS2017A)
#     """
#     # Define colors and labels for each folder
#     folder_colors = {folder1: COLORS["cyan"], folder2: COLORS["violet"], folder3: COLORS["green"]}
#     folder_labels = {
#         folder1: "Gemini Flash, Full Configuration",
#         folder2: "Gemini Pro, Full Configuration",
#         folder3: "GPT OSS, Full Configuration",
#         folder4: "GPT 5, Full Configuration",
#     }

#     # Store data from all folders
#     all_data = {}

#     # Read data from each folder
#     for folder in [folder1, folder2, folder3, folder4]:
#         basedir = "./"

#         # Read scalar database
#         scalar_db_path = f"{basedir}{folder}/{config}-{run}.sca"
#         sdb = sqlite3.connect(scalar_db_path)
#         scalars = pd.read_sql_query("SELECT * FROM scalar", sdb)

#         # Get parameters
#         network = scalars[scalars["scalarName"] == "maxServers"]["moduleName"].iloc[0]
#         evaluation_period = scalars[scalars["scalarName"] == "evaluationPeriod"][
#             "scalarValue"
#         ].iloc[0]
#         RT_THRESHOLD_SEC = scalars[scalars["scalarName"] == "responseTimeThreshold"][
#             "scalarValue"
#         ].iloc[0]
#         max_servers = scalars[scalars["scalarName"] == "maxServers"]["scalarValue"].iloc[0]
#         max_service_rate = scalars[scalars["scalarName"] == "maxServiceRate"]["scalarValue"].iloc[0]

#         # Read vector database
#         vector_db_path = f"{basedir}{folder}/{config}-{run}.vec"
#         vdb = sqlite3.connect(vector_db_path)

#         # Read all necessary vectors
#         servers = read_vector(vdb, "serverCost:vector")
#         active_servers = read_vector(vdb, "activeServers:vector")
#         dimmer_raw = read_vector(vdb, "brownoutFactor:vector")
#         dimmer = dimmer_raw.copy()
#         dimmer["y"] = 1 - dimmer["y"]

#         responses = read_vector(vdb, "lifeTime:vector")
#         avg_interarrival = periodic_average(
#             read_vector(vdb, "interArrival:vector"), evaluation_period
#         )
#         avg_arrival_rate = avg_interarrival.copy()
#         avg_arrival_rate["y"] = 1 / avg_arrival_rate["y"]

#         start = np.floor(servers["x"].min() / evaluation_period) * evaluation_period
#         end = np.ceil(servers["x"].max() / evaluation_period) * evaluation_period

#         avg_response = periodic_average(responses, evaluation_period)

#         # Compute utility (same logic as plot_results)
#         dimmer_mean = time_weighted_average(dimmer, evaluation_period)
#         dimmer_mean["x"] = dimmer_mean["x"] + evaluation_period

#         servers_mean = time_weighted_average(servers, evaluation_period)
#         servers_mean["x"] = servers_mean["x"] + evaluation_period

#         # Trim all datasets
#         avg_arrival_rate = avg_arrival_rate[avg_arrival_rate["x"] <= end]
#         dimmer_mean = dimmer_mean[dimmer_mean["x"] <= end]
#         servers_mean = servers_mean[servers_mean["x"] <= end]
#         avg_response = avg_response[avg_response["x"] <= end]

#         # Compute utility for each period
#         utility_values = []
#         for i in range(len(avg_response)):
#             util = utility_fc(
#                 max_servers,
#                 max_service_rate,
#                 avg_arrival_rate["y"].iloc[i],
#                 dimmer_mean["y"].iloc[i],
#                 evaluation_period,
#                 RT_THRESHOLD_SEC,
#                 avg_response["y"].iloc[i],
#                 servers_mean["y"].iloc[i],
#             )
#             utility_values.append(util)

#         utility = pd.DataFrame({"x": avg_response["x"].values, "y": utility_values})

#         # Store all data
#         all_data[folder] = {
#             "avg_arrival_rate": avg_arrival_rate,
#             "servers": servers,
#             "active_servers": active_servers,
#             "dimmer": dimmer,
#             "avg_response": avg_response,
#             "utility": utility,
#             "RT_THRESHOLD_SEC": RT_THRESHOLD_SEC,
#             "max_servers": max_servers,
#             "evaluation_period": evaluation_period,
#             "start": start,
#             "end": end,
#         }

#         sdb.close()
#         vdb.close()

#     # Find common time range
#     start = max([all_data[f]["start"] for f in all_data])
#     end = min([all_data[f]["end"] for f in all_data])

#     # Create plots
#     fig = plt.figure(figsize=(10, 12))
#     gs = GridSpec(4, 1, figure=fig, hspace=0.2)

#     # Plot 1: Servers
#     ax1 = fig.add_subplot(gs[0, 0])
#     for folder, color in folder_colors.items():
#         servers_data = all_data[folder]["servers"]
#         ax1.step(
#             servers_data["x"],
#             servers_data["y"],
#             where="post",
#             linestyle="--",
#             color=color,
#             linewidth=1.5,
#             alpha=0.4,
#         )
#     for folder, color in folder_colors.items():
#         active_servers_data = all_data[folder]["active_servers"]
#         ax1.step(
#             active_servers_data["x"],
#             active_servers_data["y"],
#             where="post",
#             linestyle="-",
#             color=color,
#             alpha=0.95,
#             linewidth=1.5,
#             label=folder_labels[folder],
#         )
#     ax1.set_ylabel("servers")
#     ax1.set_ylim(bottom=0)
#     ax1.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")
#     ax1.set_xlim(start, end)
#     ax1.legend(loc="lower right")

#     # Plot 2: Dimmer
#     ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
#     for folder, color in folder_colors.items():
#         data = all_data[folder]["dimmer"]
#         ax2.step(data["x"], data["y"], where="post", color=color, alpha=0.95, linewidth=1.5)
#     ax2.set_ylabel("dimmer")
#     ax2.set_ylim(0, 1)
#     ax2.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

#     # Plot 3: Response Time
#     ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
#     for folder, color in folder_colors.items():
#         data = all_data[folder]["avg_response"]
#         ax3.plot(data["x"], data["y"], color=color, alpha=0.95, linewidth=1.5)
#     ax3.axhline(
#         y=all_data[folder1]["RT_THRESHOLD_SEC"],
#         linestyle="--",
#         color=COLORS["threshold"],
#         linewidth=1.5,
#         alpha=0.7,
#     )
#     ax3.set_ylabel("resp. time (s)")
#     ax3.set_ylim(top=15)
#     ax3.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

#     # Plot 4: Cumulative Utility
#     ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
#     for folder, color in folder_colors.items():
#         data = all_data[folder]["utility"]
#         ax4.plot(data["x"], np.cumsum(data["y"]), color=color, alpha=0.95, linewidth=1.5)
#     ax4.set_ylabel("cum. utility")
#     ax4.set_xlabel("time (s)")
#     ax4.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

#     # Add vertical dotted lines
#     dotted_line_color = "#000000"  # Black color for dotted lines
#     dotted_line_style = ":"
#     dotted_line_width = 1.0

#     # Line 1: at t=2000s from servers=1 to bottom of response time
#     t = 2000
#     servers_y = ax1.transData.transform((0, 1))[1]  # y=1 in servers plot
#     response_bottom_y = ax3.transData.transform((0, ax3.get_ylim()[0]))[
#         1
#     ]  # bottom of response time

#     servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
#     response_y_fig = fig.transFigure.inverted().transform((0, response_bottom_y))[1]

#     x_fig = fig.transFigure.inverted().transform(ax1.transData.transform((t, 0)))[0]

#     fig.add_artist(
#         plt.Line2D(
#             [x_fig, x_fig],
#             [response_y_fig, servers_y_fig],
#             transform=fig.transFigure,
#             color=dotted_line_color,
#             linestyle=dotted_line_style,
#             linewidth=dotted_line_width,
#             alpha=0.6,
#         )
#     )

#     # Line 2: First time after 2500s where folder2's servers=2 and active_servers=1
#     if folder2 in all_data:
#         servers_data = all_data[folder2]["servers"]
#         active_servers_data = all_data[folder2]["active_servers"]

#         # Find times after 2500s where servers=2
#         after_2500_servers = servers_data[servers_data["x"] > 2500]
#         servers_eq_2 = after_2500_servers[after_2500_servers["y"] == 2]

#         if len(servers_eq_2) > 0:
#             # Check active_servers at the first such time
#             for idx in servers_eq_2.index:
#                 time_point = servers_data.loc[idx, "x"]

#                 # Find active_servers value at this time
#                 active_idx = np.searchsorted(active_servers_data["x"], time_point, side="right") - 1
#                 if active_idx >= 0 and active_idx < len(active_servers_data):
#                     active_value = active_servers_data["y"].iloc[active_idx]

#                     if active_value == 1:
#                         # Found the point! Draw the line down to utility plot
#                         servers_y = ax1.transData.transform((0, 1))[1]
#                         utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

#                         servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
#                         utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[
#                             1
#                         ]

#                         x_fig = fig.transFigure.inverted().transform(
#                             ax1.transData.transform((time_point, 0))
#                         )[0]

#                         fig.add_artist(
#                             plt.Line2D(
#                                 [x_fig, x_fig],
#                                 [utility_y_fig, servers_y_fig],
#                                 transform=fig.transFigure,
#                                 color=dotted_line_color,
#                                 linestyle=dotted_line_style,
#                                 linewidth=dotted_line_width,
#                                 alpha=0.6,
#                             )
#                         )
#                         ax1.annotate(
#                             "Scale up command\nwith latency=60s",
#                             xy=(time_point, 1),
#                             xytext=(time_point + 400, 1.5),
#                             fontsize=8,
#                             ha="left",
#                             va="center",
#                             style="italic",
#                             bbox=dict(
#                                 boxstyle="round,pad=0.4",
#                                 facecolor="#f8f9fa",
#                                 alpha=0.9,
#                                 edgecolor="#dee2e6",
#                                 linewidth=0.8,
#                             ),
#                             arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
#                         )
#                         break

#     # Line 3: First time after 4000s where folder3's dimmer < 0.9
#     if folder3 in all_data:
#         dimmer_data = all_data[folder3]["dimmer"]

#         # Find first time after 4000s where dimmer < 0.9
#         after_4000 = dimmer_data[dimmer_data["x"] > 4000]
#         dimmer_below = after_4000[after_4000["y"] < 0.9]

#         if len(dimmer_below) > 0:
#             time_point = dimmer_below["x"].iloc[0]
#             dimmer_value = dimmer_below["y"].iloc[0]

#             # Draw line from dimmer plot down to utility plot
#             dimmer_y = ax2.transData.transform((0, dimmer_value))[1]
#             utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

#             dimmer_y_fig = fig.transFigure.inverted().transform((0, dimmer_y))[1]
#             utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[1]

#             x_fig = fig.transFigure.inverted().transform(ax2.transData.transform((time_point, 0)))[
#                 0
#             ]

#             fig.add_artist(
#                 plt.Line2D(
#                     [x_fig, x_fig],
#                     [utility_y_fig, dimmer_y_fig],
#                     transform=fig.transFigure,
#                     color=dotted_line_color,
#                     linestyle=dotted_line_style,
#                     linewidth=dotted_line_width,
#                     alpha=0.6,
#                 )
#             )

#             # Add annotation for "Reduce dimmer since servers are at max"
#             ax2.annotate(
#                 "Reduce dimmer\n since servers are at max",
#                 xy=(time_point, dimmer_value),
#                 xytext=(time_point - 600, dimmer_value - 0.15),
#                 fontsize=8,
#                 ha="center",
#                 va="center",
#                 style="italic",
#                 bbox=dict(
#                     boxstyle="round,pad=0.4",
#                     facecolor="#f8f9fa",
#                     alpha=0.9,
#                     edgecolor="#dee2e6",
#                     linewidth=0.8,
#                 ),
#                 arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
#             )

#             # Get utility values at this time point for folder3 and folder1
#             folder3_utility_data = all_data[folder3]["utility"]
#             folder1_utility_data = all_data[folder1]["utility"]

#             # Find utility values closest to the time point
#             folder3_idx = np.argmin(np.abs(folder3_utility_data["x"] - time_point))
#             folder1_idx = np.argmin(np.abs(folder1_utility_data["x"] - time_point))

#             folder3_cumutil = np.cumsum(folder3_utility_data["y"])[folder3_idx]
#             folder1_cumutil = np.cumsum(folder1_utility_data["y"])[
#                 folder1_idx
#             ]  # Add annotation for folder3 (learning to avoid SLA violations)
#             # Add annotation for folder3 (learning to avoid SLA violations)
#             # Add annotation for folder3 (learning to avoid SLA violations) - point to where line meets folder3's utility
#             ax4.annotate(
#                 "Learns to avoid and reduce\nSLA violations over time",
#                 xy=(time_point, folder3_cumutil),
#                 xytext=(time_point - 800, folder3_cumutil + 500),
#                 fontsize=8,
#                 ha="center",
#                 va="center",
#                 style="italic",
#                 bbox=dict(
#                     boxstyle="round,pad=0.4",
#                     facecolor="#f8f9fa",
#                     alpha=0.9,
#                     edgecolor="#dee2e6",
#                     linewidth=0.8,
#                 ),
#                 arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
#             )

#             # Add annotation for folder1 (utility suffered) - point to where line meets folder1's utility
#             ax4.annotate(
#                 "Utility suffered due to\nfrequent and long SLA violations",
#                 xy=(time_point, folder1_cumutil),
#                 xytext=(time_point + 800, folder1_cumutil - 1500),
#                 fontsize=8,
#                 ha="center",
#                 va="center",
#                 style="italic",
#                 bbox=dict(
#                     boxstyle="round,pad=0.4",
#                     facecolor="#f8f9fa",
#                     alpha=0.9,
#                     edgecolor="#dee2e6",
#                     linewidth=0.8,
#                 ),
#                 arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
#             )
#             x_req = time_point + 800
#             y_req = folder1_cumutil - 1500

#     # Line 4: First time after 2500s where folder3's servers=3 and active_servers=2
#     if folder2 in all_data:
#         servers_data = all_data[folder2]["servers"]
#         active_servers_data = all_data[folder2]["active_servers"]

#         # Find times after 2500s where servers=3
#         after_2500_servers = servers_data[servers_data["x"] > 2500]
#         servers_eq_3 = after_2500_servers[after_2500_servers["y"] == 3]

#         if len(servers_eq_3) > 0:
#             # Check active_servers at the first such time
#             found = False
#             for i in range(len(servers_eq_3)):
#                 time_point = servers_eq_3["x"].iloc[i]

#                 # Find active_servers value at this time
#                 active_idx = (
#                     np.searchsorted(active_servers_data["x"].values, time_point, side="right") - 1
#                 )
#                 if active_idx >= 0 and active_idx < len(active_servers_data):
#                     active_value = active_servers_data["y"].iloc[active_idx]

#                     if active_value == 2:
#                         # Found the point! Draw the line down to utility plot
#                         print(f"Line 4: Drawing at time {time_point}, servers=3, active=2")
#                         servers_y = ax1.transData.transform((0, 2))[1]  # y=2 in servers plot
#                         utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

#                         servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
#                         utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[
#                             1
#                         ]

#                         x_fig = fig.transFigure.inverted().transform(
#                             ax1.transData.transform((time_point, 0))
#                         )[0]

#                         fig.add_artist(
#                             plt.Line2D(
#                                 [x_fig, x_fig],
#                                 [utility_y_fig, servers_y_fig],
#                                 transform=fig.transFigure,
#                                 color=dotted_line_color,
#                                 linestyle=dotted_line_style,
#                                 linewidth=dotted_line_width,
#                                 alpha=0.6,
#                             )
#                         )

#                         # Add annotation for "Add a server/scale up command with latency=60"

#                         found = True
#                         break

#             if not found:
#                 print(f"Line 4: No point found where servers=3 and active=2 after 2500s")
#                 print(f"Servers=3 times: {servers_eq_3['x'].values[:5]}")
#                 if len(servers_eq_3) > 0:
#                     for i in range(min(5, len(servers_eq_3))):
#                         t = servers_eq_3["x"].iloc[i]
#                         active_idx = (
#                             np.searchsorted(active_servers_data["x"].values, t, side="right") - 1
#                         )
#                         if active_idx >= 0:
#                             print(
#                                 f"  At time {t}: active_servers = {active_servers_data['y'].iloc[active_idx]}"
#                             )

#     # Remove x-axis labels for all but bottom plot
#     for ax in [ax1, ax2, ax3]:
#         ax.set_xlabel("")
#         ax.tick_params(labelbottom=False)

#     plt.tight_layout()

#     if save_as is not None:
#         plt.savefig(save_as, dpi=2200, bbox_inches="tight")

#     return fig


def plot_comparison_charts(folder1, folder2, folder3, config="sim", run=1, save_as_prefix=None):
    """
    Creates bar chart and spider chart comparing metrics from three folders.
    Uses subplots for bar chart to handle different scales.
    Parameters:
    folder1, folder2, folder3: Names of the three folders to compare
    config: Configuration name (default: "sim")
    run: Run number (default: 0)
    save_as_prefix: Prefix for saving figures (will save as prefix_bar.png and prefix_spider.png)
    """
    # Added imports to make the function self-contained
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.patches as mpatches
    from math import pi

    # Get results from each folder
    results = {}
    for folder in [folder1, folder2, folder3]:
        # Close any existing plots to avoid interference
        plt.close("all")
        # Get results without creating plots
        # Assuming plot_results and COLORS are defined elsewhere in your code
        _, result = plot_results(config, folder=folder, run=run, brief=True)
        results[folder] = result
        # Close the plots created by plot_results
        plt.close("all")

    # Define colors and labels for each folder
    folder_colors = {folder1: COLORS["cyan"], folder2: COLORS["violet"], folder3: COLORS["green"]}
    folder_labels = {
        folder1: "Gemini Flash, Full Configuration, Run 1",
        folder2: "Gemini Flash, Full Configuration, Run 2",
        folder3: "Gemini Flash, Full Configuration, Run 3",
    }
    # folder_labels = {
    #     folder1: "Gemini Flash, –ML, –Tools, –FC",
    #     folder2: "Gemini Flash, –Tools, –ML",
    #     folder3: "Gemini Flash, –Tools",
    # }

    # Prepare data for plotting
    metrics = ["total_utility", "pct_optional", "pct_late", "avg_servers"]
    metric_labels = ["Utility", "% Optional", "% Late", "Avg Servers"]

    # Create bar chart with subplots for different scales
    fig_bar, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Get values for this metric
        folders = list(folder_colors.keys())
        values = [results[folder][metric] for folder in folders]
        colors = [folder_colors[folder] for folder in folders]

        # --- MODIFICATION START ---
        # Define bar width and gap
        bar_width = 0.15  # Thin bars
        gap = 0.05  # <--- FIX 2: Add a slight space

        # Calculate x positions so bars have a slight gap
        # (distance between centers == bar_width + gap)
        x_positions = np.arange(len(folders)) * (bar_width + gap)

        # Center the whole group of bars horizontally
        x_centered = x_positions - np.mean(x_positions)

        # Create bars
        bars = ax.bar(
            x_centered,
            values,
            color=colors,
            alpha=0.4,
            width=bar_width,
            edgecolor=colors,
            linewidth=1.5,
        )

        # Customize each subplot
        ax.set_title(label, fontsize=12)

        # --- FIX 1: Tighter X-axis limits ---
        # Dynamically set X-axis limits to be just wider than the bar group
        min_x_edge = x_centered[0] - bar_width / 2
        max_x_edge = x_centered[-1] + bar_width / 2
        padding = 0.2  # Add a fixed padding to the edges (adjust as needed)
        ax.set_xlim(min_x_edge - padding, max_x_edge + padding)
        # --- END OF FIX ---

        ax.set_xticks(x_centered)  # Set ticks at the new bar centers
        # Remove x-axis labels
        ax.set_xticklabels([])
        ax.grid(True, alpha=0.3, axis="y")

        # Set specific y-axis limits for certain metrics
        if metric == "avg_servers":
            ax.set_ylim(0, 3)
        elif metric == "pct_late":
            ax.set_ylim(0, 20)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metric in ["pct_optional", "pct_late"]:
                label_text = f"{value:.1f}%"
            else:
                label_text = f"{value:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                label_text,
                ha="center",
                va="bottom",
                fontsize=14,
            )

    # Add a legend for folders
    legend_elements = [
        mpatches.Patch(
            facecolor=color, alpha=0.4, edgecolor=color, linewidth=1.5, label=folder_labels[folder]
        )
        for folder, color in folder_colors.items()
    ]

    fig_bar.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),  # Placed legend at 95% height
        ncol=3,
        frameon=True,
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # Subplots only use bottom 90%

    if save_as_prefix:
        plt.savefig(f"{save_as_prefix}_bar.png", dpi=2200, bbox_inches="tight")

    return fig_bar


# def plot_overlapping_results(
#     folder1,
#     folder2,
#     folder3,
#     folder4,
#     config="sim",
#     run=1,
#     save_as=None,
#     utility_fc=period_utility_SEAMS2017A,
# ):
#     """
#     Creates overlapping plots from three different folders with different colors.

#     Parameters:
#     folder1, folder2, folder3: Names of the three folders to compare
#     config: Configuration name (default: "sim")
#     run: Run number (default: 0)
#     save_as: Path to save the figure (optional)
#     utility_fc: Utility function to use (default: period_utility_SEAMS2017A)
#     """
#     # Define colors and labels for each folder
#     folder_colors = {
#         folder1: COLORS["cyan"],
#         folder2: COLORS["violet"],
#         folder3: COLORS["green"],
#         folder4: COLORS["coral"],
#     }
#     folder_labels = {
#         folder1: "Gemini Flash, Full Configuration",
#         folder2: "Gemini Pro, Full Configuration",
#         folder3: "GPT OSS, Full Configuration",
#         folder4: "GPT 5, Full Configuration",
#     }

#     # Store data from all folders
#     all_data = {}

#     # Read data from each folder
#     for folder in [folder1, folder2, folder3, folder4]:
#         basedir = "./"

#         # Read scalar database
#         scalar_db_path = f"{basedir}{folder}/{config}-{run}.sca"
#         sdb = sqlite3.connect(scalar_db_path)
#         scalars = pd.read_sql_query("SELECT * FROM scalar", sdb)

#         # Get parameters
#         network = scalars[scalars["scalarName"] == "maxServers"]["moduleName"].iloc[0]
#         evaluation_period = scalars[scalars["scalarName"] == "evaluationPeriod"][
#             "scalarValue"
#         ].iloc[0]
#         RT_THRESHOLD_SEC = scalars[scalars["scalarName"] == "responseTimeThreshold"][
#             "scalarValue"
#         ].iloc[0]
#         max_servers = scalars[scalars["scalarName"] == "maxServers"]["scalarValue"].iloc[0]
#         max_service_rate = scalars[scalars["scalarName"] == "maxServiceRate"]["scalarValue"].iloc[0]

#         # Read vector database
#         vector_db_path = f"{basedir}{folder}/{config}-{run}.vec"
#         vdb = sqlite3.connect(vector_db_path)

#         # Read all necessary vectors
#         servers = read_vector(vdb, "serverCost:vector")
#         active_servers = read_vector(vdb, "activeServers:vector")
#         dimmer_raw = read_vector(vdb, "brownoutFactor:vector")
#         dimmer = dimmer_raw.copy()
#         dimmer["y"] = 1 - dimmer["y"]

#         responses = read_vector(vdb, "lifeTime:vector")
#         avg_interarrival = periodic_average(
#             read_vector(vdb, "interArrival:vector"), evaluation_period
#         )
#         avg_arrival_rate = avg_interarrival.copy()
#         avg_arrival_rate["y"] = 1 / avg_arrival_rate["y"]

#         start = np.floor(servers["x"].min() / evaluation_period) * evaluation_period
#         end = np.ceil(servers["x"].max() / evaluation_period) * evaluation_period

#         avg_response = periodic_average(responses, evaluation_period)

#         # Compute utility (same logic as plot_results)
#         dimmer_mean = time_weighted_average(dimmer, evaluation_period)
#         dimmer_mean["x"] = dimmer_mean["x"] + evaluation_period

#         servers_mean = time_weighted_average(servers, evaluation_period)
#         servers_mean["x"] = servers_mean["x"] + evaluation_period

#         # Trim all datasets
#         avg_arrival_rate = avg_arrival_rate[avg_arrival_rate["x"] <= end]
#         dimmer_mean = dimmer_mean[dimmer_mean["x"] <= end]
#         servers_mean = servers_mean[servers_mean["x"] <= end]
#         avg_response = avg_response[avg_response["x"] <= end]

#         # Compute utility for each period
#         utility_values = []
#         for i in range(len(avg_response)):
#             util = utility_fc(
#                 max_servers,
#                 max_service_rate,
#                 avg_arrival_rate["y"].iloc[i],
#                 dimmer_mean["y"].iloc[i],
#                 evaluation_period,
#                 RT_THRESHOLD_SEC,
#                 avg_response["y"].iloc[i],
#                 servers_mean["y"].iloc[i],
#             )
#             utility_values.append(util)

#         utility = pd.DataFrame({"x": avg_response["x"].values, "y": utility_values})

#         # Store all data
#         all_data[folder] = {
#             "avg_arrival_rate": avg_arrival_rate,
#             "servers": servers,
#             "active_servers": active_servers,
#             "dimmer": dimmer,
#             "avg_response": avg_response,
#             "utility": utility,
#             "RT_THRESHOLD_SEC": RT_THRESHOLD_SEC,
#             "max_servers": max_servers,
#             "evaluation_period": evaluation_period,
#             "start": start,
#             "end": end,
#         }

#         sdb.close()
#         vdb.close()

#     # Find common time range
#     start = max([all_data[f]["start"] for f in all_data])
#     end = min([all_data[f]["end"] for f in all_data])

#     # Create plots - only utility plot
#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(1, 1, 1)

#     # Plot Cumulative Utility
#     for folder, color in folder_colors.items():
#         data = all_data[folder]["utility"]
#         ax.plot(
#             data["x"],
#             np.cumsum(data["y"]),
#             color=color,
#             alpha=0.95,
#             linewidth=1.5,
#             label=folder_labels[folder],
#         )

#     ax.set_ylabel("cum. utility")
#     ax.set_xlabel("time (s)")
#     ax.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")
#     ax.set_xlim(start, end)
#     ax.legend(loc="lower right")

#     plt.tight_layout()

#     if save_as is not None:
#         plt.savefig(save_as, dpi=2000, bbox_inches="tight")

#     return fig


def plot_overlapping_results(
    folder1,
    folder2,
    folder3,
    config="sim",
    run=1,
    save_as=None,
    utility_fc=period_utility_SEAMS2017A,
):
    """
    Creates overlapping plots from three different folders with different colors.
    Includes vertical dotted reference lines at specified times.

    Parameters:
    folder1, folder2, folder3: Names of the three folders to compare
    config: Configuration name (default: "sim")
    run: Run number (default: 0)
    save_as: Path to save the figure (optional)
    utility_fc: Utility function to use (default: period_utility_SEAMS2017A)
    """
    # Define colors and labels for each folder
    folder_colors = {folder1: COLORS["cyan"], folder2: COLORS["violet"], folder3: COLORS["green"]}
    folder_labels = {
        folder1: "Gemini Flash, –ML, –Tools, –FC",
        folder2: "Gemini Flash, –Tools, –ML",
        folder3: "Gemini Flash, –Tools",
    }

    # Store data from all folders
    all_data = {}

    # Read data from each folder
    for folder in [folder1, folder2, folder3]:
        basedir = "./"

        # Read scalar database
        scalar_db_path = f"{basedir}{folder}/{config}-{run}.sca"
        sdb = sqlite3.connect(scalar_db_path)
        scalars = pd.read_sql_query("SELECT * FROM scalar", sdb)

        # Get parameters
        network = scalars[scalars["scalarName"] == "maxServers"]["moduleName"].iloc[0]
        evaluation_period = scalars[scalars["scalarName"] == "evaluationPeriod"][
            "scalarValue"
        ].iloc[0]
        RT_THRESHOLD_SEC = scalars[scalars["scalarName"] == "responseTimeThreshold"][
            "scalarValue"
        ].iloc[0]
        max_servers = scalars[scalars["scalarName"] == "maxServers"]["scalarValue"].iloc[0]
        max_service_rate = scalars[scalars["scalarName"] == "maxServiceRate"]["scalarValue"].iloc[0]

        # Read vector database
        vector_db_path = f"{basedir}{folder}/{config}-{run}.vec"
        vdb = sqlite3.connect(vector_db_path)

        # Read all necessary vectors
        servers = read_vector(vdb, "serverCost:vector")
        active_servers = read_vector(vdb, "activeServers:vector")
        dimmer_raw = read_vector(vdb, "brownoutFactor:vector")
        dimmer = dimmer_raw.copy()
        dimmer["y"] = 1 - dimmer["y"]

        responses = read_vector(vdb, "lifeTime:vector")
        avg_interarrival = periodic_average(
            read_vector(vdb, "interArrival:vector"), evaluation_period
        )
        avg_arrival_rate = avg_interarrival.copy()
        avg_arrival_rate["y"] = 1 / avg_arrival_rate["y"]

        start = np.floor(servers["x"].min() / evaluation_period) * evaluation_period
        end = np.ceil(servers["x"].max() / evaluation_period) * evaluation_period

        avg_response = periodic_average(responses, evaluation_period)

        # Compute utility (same logic as plot_results)
        dimmer_mean = time_weighted_average(dimmer, evaluation_period)
        dimmer_mean["x"] = dimmer_mean["x"] + evaluation_period

        servers_mean = time_weighted_average(servers, evaluation_period)
        servers_mean["x"] = servers_mean["x"] + evaluation_period

        # Trim all datasets
        avg_arrival_rate = avg_arrival_rate[avg_arrival_rate["x"] <= end]
        dimmer_mean = dimmer_mean[dimmer_mean["x"] <= end]
        servers_mean = servers_mean[servers_mean["x"] <= end]
        avg_response = avg_response[avg_response["x"] <= end]

        # Compute utility for each period
        utility_values = []
        for i in range(len(avg_response)):
            util = utility_fc(
                max_servers,
                max_service_rate,
                avg_arrival_rate["y"].iloc[i],
                dimmer_mean["y"].iloc[i],
                evaluation_period,
                RT_THRESHOLD_SEC,
                avg_response["y"].iloc[i],
                servers_mean["y"].iloc[i],
            )
            utility_values.append(util)

        utility = pd.DataFrame({"x": avg_response["x"].values, "y": utility_values})

        # Store all data
        all_data[folder] = {
            "avg_arrival_rate": avg_arrival_rate,
            "servers": servers,
            "active_servers": active_servers,
            "dimmer": dimmer,
            "avg_response": avg_response,
            "utility": utility,
            "RT_THRESHOLD_SEC": RT_THRESHOLD_SEC,
            "max_servers": max_servers,
            "evaluation_period": evaluation_period,
            "start": start,
            "end": end,
        }

        sdb.close()
        vdb.close()

    # Find common time range
    start = max([all_data[f]["start"] for f in all_data])
    end = min([all_data[f]["end"] for f in all_data])

    # Create plots
    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(4, 1, figure=fig, hspace=0.2)

    # Plot 1: Servers
    ax1 = fig.add_subplot(gs[0, 0])
    for folder, color in folder_colors.items():
        servers_data = all_data[folder]["servers"]
        ax1.step(
            servers_data["x"],
            servers_data["y"],
            where="post",
            linestyle="--",
            color=color,
            linewidth=1.5,
            alpha=0.4,
        )
    for folder, color in folder_colors.items():
        active_servers_data = all_data[folder]["active_servers"]
        ax1.step(
            active_servers_data["x"],
            active_servers_data["y"],
            where="post",
            linestyle="-",
            color=color,
            alpha=0.95,
            linewidth=1.5,
            label=folder_labels[folder],
        )
    ax1.set_ylabel("servers")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")
    ax1.set_xlim(start, end)

    # Plot 2: Dimmer
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    for folder, color in folder_colors.items():
        data = all_data[folder]["dimmer"]
        ax2.step(data["x"], data["y"], where="post", color=color, alpha=0.95, linewidth=1.5)
    ax2.set_ylabel("dimmer")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

    # Plot 3: Response Time
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    for folder, color in folder_colors.items():
        data = all_data[folder]["avg_response"]
        ax3.plot(data["x"], data["y"], color=color, alpha=0.95, linewidth=1.5)
    ax3.axhline(
        y=all_data[folder1]["RT_THRESHOLD_SEC"],
        linestyle="--",
        color=COLORS["threshold"],
        linewidth=1.5,
        alpha=0.7,
    )
    ax3.set_ylabel("resp. time (s)")
    ax3.set_ylim(top=15)
    ax3.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

    # Plot 4: Cumulative Utility
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    for folder, color in folder_colors.items():
        data = all_data[folder]["utility"]
        ax4.plot(data["x"], np.cumsum(data["y"]), color=color, alpha=0.95, linewidth=1.5)
    ax4.set_ylabel("cum. utility")
    ax4.set_xlabel("time (s)")
    ax4.grid(True, alpha=0.7, color=COLORS["neutral"], linestyle="--")

    # Add vertical dotted lines
    dotted_line_color = "#000000"  # Black color for dotted lines
    dotted_line_style = ":"
    dotted_line_width = 1.0

    # Line 1: at t=2000s from servers=1 to bottom of response time
    t = 2000
    servers_y = ax1.transData.transform((0, 1))[1]  # y=1 in servers plot
    response_bottom_y = ax3.transData.transform((0, ax3.get_ylim()[0]))[
        1
    ]  # bottom of response time

    servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
    response_y_fig = fig.transFigure.inverted().transform((0, response_bottom_y))[1]

    x_fig = fig.transFigure.inverted().transform(ax1.transData.transform((t, 0)))[0]

    fig.add_artist(
        plt.Line2D(
            [x_fig, x_fig],
            [response_y_fig, servers_y_fig],
            transform=fig.transFigure,
            color=dotted_line_color,
            linestyle=dotted_line_style,
            linewidth=dotted_line_width,
            alpha=0.6,
        )
    )

    # Line 2: First time after 2500s where folder2's servers=2 and active_servers=1
    if folder2 in all_data:
        servers_data = all_data[folder2]["servers"]
        active_servers_data = all_data[folder2]["active_servers"]

        # Find times after 2500s where servers=2
        after_2500_servers = servers_data[servers_data["x"] > 2500]
        servers_eq_2 = after_2500_servers[after_2500_servers["y"] == 2]

        if len(servers_eq_2) > 0:
            # Check active_servers at the first such time
            for idx in servers_eq_2.index:
                time_point = servers_data.loc[idx, "x"]

                # Find active_servers value at this time
                active_idx = np.searchsorted(active_servers_data["x"], time_point, side="right") - 1
                if active_idx >= 0 and active_idx < len(active_servers_data):
                    active_value = active_servers_data["y"].iloc[active_idx]

                    if active_value == 1:
                        # Found the point! Draw the line down to utility plot
                        servers_y = ax1.transData.transform((0, 1))[1]
                        utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

                        servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
                        utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[
                            1
                        ]

                        x_fig = fig.transFigure.inverted().transform(
                            ax1.transData.transform((time_point, 0))
                        )[0]

                        fig.add_artist(
                            plt.Line2D(
                                [x_fig, x_fig],
                                [utility_y_fig, servers_y_fig],
                                transform=fig.transFigure,
                                color=dotted_line_color,
                                linestyle=dotted_line_style,
                                linewidth=dotted_line_width,
                                alpha=0.6,
                            )
                        )
                        ax1.annotate(
                            "Scale up command\nwith latency=60s",
                            xy=(time_point, 1),
                            xytext=(time_point + 400, 1.5),
                            ha="left",
                            va="center",
                            style="italic",
                            bbox=dict(
                                boxstyle="round,pad=0.4",
                                facecolor="#f8f9fa",
                                alpha=0.9,
                                edgecolor="#dee2e6",
                                linewidth=0.8,
                            ),
                            arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
                        )
                        break

    # Line 3: First time after 4000s where folder3's dimmer < 0.9
    if folder3 in all_data:
        dimmer_data = all_data[folder3]["dimmer"]

        # Find first time after 4000s where dimmer < 0.9
        after_4000 = dimmer_data[dimmer_data["x"] > 4000]
        dimmer_below = after_4000[after_4000["y"] < 0.9]

        if len(dimmer_below) > 0:
            time_point = dimmer_below["x"].iloc[0]
            dimmer_value = dimmer_below["y"].iloc[0]

            # Draw line from dimmer plot down to utility plot
            dimmer_y = ax2.transData.transform((0, dimmer_value))[1]
            utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

            dimmer_y_fig = fig.transFigure.inverted().transform((0, dimmer_y))[1]
            utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[1]

            x_fig = fig.transFigure.inverted().transform(ax2.transData.transform((time_point, 0)))[
                0
            ]

            fig.add_artist(
                plt.Line2D(
                    [x_fig, x_fig],
                    [utility_y_fig, dimmer_y_fig],
                    transform=fig.transFigure,
                    color=dotted_line_color,
                    linestyle=dotted_line_style,
                    linewidth=dotted_line_width,
                    alpha=0.6,
                )
            )

            # Add annotation for "Reduce dimmer since servers are at max"
            ax2.annotate(
                "Reduce dimmer\n since servers are at max",
                xy=(time_point, dimmer_value),
                xytext=(time_point - 600, dimmer_value - 0.25),
                ha="center",
                va="center",
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#f8f9fa",
                    alpha=0.9,
                    edgecolor="#dee2e6",
                    linewidth=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
            )

            # Get utility values at this time point for folder3 and folder1
            folder3_utility_data = all_data[folder3]["utility"]
            folder1_utility_data = all_data[folder1]["utility"]

            # Find utility values closest to the time point
            folder3_idx = np.argmin(np.abs(folder3_utility_data["x"] - time_point))
            folder1_idx = np.argmin(np.abs(folder1_utility_data["x"] - time_point))

            folder3_cumutil = np.cumsum(folder3_utility_data["y"])[folder3_idx]
            folder1_cumutil = np.cumsum(folder1_utility_data["y"])[
                folder1_idx
            ]  # Add annotation for folder3 (learning to avoid SLA violations)
            # Add annotation for folder3 (learning to avoid SLA violations)
            # Add annotation for folder3 (learning to avoid SLA violations) - point to where line meets folder3's utility
            ax4.annotate(
                "Learns to avoid and reduce\nSLA violations over time",
                xy=(time_point, folder3_cumutil),
                xytext=(time_point - 700, folder3_cumutil + 1400),
                ha="center",
                va="center",
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#f8f9fa",
                    alpha=0.9,
                    edgecolor="#dee2e6",
                    linewidth=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
            )

            # Add annotation for folder1 (utility suffered) - point to where line meets folder1's utility
            ax4.annotate(
                "Utility suffered due to\nfrequent and long SLA violations",
                xy=(time_point, folder1_cumutil),
                xytext=(time_point + 750, folder1_cumutil - 1500),
                ha="center",
                va="center",
                style="italic",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="#f8f9fa",
                    alpha=0.9,
                    edgecolor="#dee2e6",
                    linewidth=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="#6c757d", lw=1.0, alpha=0.8),
            )
            x_req = time_point + 800
            y_req = folder1_cumutil - 1500

    # Line 4: First time after 2500s where folder3's servers=3 and active_servers=2
    if folder2 in all_data:
        servers_data = all_data[folder2]["servers"]
        active_servers_data = all_data[folder2]["active_servers"]

        # Find times after 2500s where servers=3
        after_2500_servers = servers_data[servers_data["x"] > 2500]
        servers_eq_3 = after_2500_servers[after_2500_servers["y"] == 3]

        if len(servers_eq_3) > 0:
            # Check active_servers at the first such time
            found = False
            for i in range(len(servers_eq_3)):
                time_point = servers_eq_3["x"].iloc[i]

                # Find active_servers value at this time
                active_idx = (
                    np.searchsorted(active_servers_data["x"].values, time_point, side="right") - 1
                )
                if active_idx >= 0 and active_idx < len(active_servers_data):
                    active_value = active_servers_data["y"].iloc[active_idx]

                    if active_value == 2:
                        # Found the point! Draw the line down to utility plot
                        print(f"Line 4: Drawing at time {time_point}, servers=3, active=2")
                        servers_y = ax1.transData.transform((0, 2))[1]  # y=2 in servers plot
                        utility_bottom_y = ax4.transData.transform((0, ax4.get_ylim()[0]))[1]

                        servers_y_fig = fig.transFigure.inverted().transform((0, servers_y))[1]
                        utility_y_fig = fig.transFigure.inverted().transform((0, utility_bottom_y))[
                            1
                        ]

                        x_fig = fig.transFigure.inverted().transform(
                            ax1.transData.transform((time_point, 0))
                        )[0]

                        fig.add_artist(
                            plt.Line2D(
                                [x_fig, x_fig],
                                [utility_y_fig, servers_y_fig],
                                transform=fig.transFigure,
                                color=dotted_line_color,
                                linestyle=dotted_line_style,
                                linewidth=dotted_line_width,
                                alpha=0.6,
                            )
                        )

                        # Add annotation for "Add a server/scale up command with latency=60"

                        found = True
                        break

            if not found:
                print(f"Line 4: No point found where servers=3 and active=2 after 2500s")
                print(f"Servers=3 times: {servers_eq_3['x'].values[:5]}")
                if len(servers_eq_3) > 0:
                    for i in range(min(5, len(servers_eq_3))):
                        t = servers_eq_3["x"].iloc[i]
                        active_idx = (
                            np.searchsorted(active_servers_data["x"].values, t, side="right") - 1
                        )
                        if active_idx >= 0:
                            print(
                                f"  At time {t}: active_servers = {active_servers_data['y'].iloc[active_idx]}"
                            )

    # Remove x-axis labels for all but bottom plot
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    # Add legend at the bottom of the entire figure
    handles = []
    labels = []
    for folder, color in folder_colors.items():
        handles.append(
            plt.Line2D([0], [0], color=color, linewidth=1.5, label=folder_labels[folder])
        )
        labels.append(folder_labels[folder])

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_as is not None:
        plt.savefig(save_as, dpi=700, bbox_inches="tight")

    return fig


# Example usage:
if __name__ == "__main__":
    # Example with three folders
    # folder1 = "SWIM_clarknet_agentic_flash"
    # folder2 = "SWIM_clarknet_agentic_pro3"
    # folder3 = "SWIM_clarknet_agentic_gptoss2"
    # folder4 = "SWIM_clarknet_agentic_gpt5_2"
    folder1 = "SWIM_gpt_clarknet_ntools_nmeta_nfast"
    folder2 = "fastslow2"
    folder3 = "SWIM_clarknet_no_tools_flash_2"
    # Create overlapping plots
    fig_overlap = plot_overlapping_results(folder1, folder2, folder3, save_as="mf.png")
    # plt.show()

    # Create comparison charts
    # fig_bar = plot_comparison_charts(folder1, folder2, folder3, save_as_prefix="comparison")
    # plt.show()


# Example usage:
# fig = plot_results("myconfig", folder="SWIM", run=0, save_as="output.png")
# plt.show()
# def main():
#     # path is each folder of the form SWIM_* in the current directory for plot generation
#     import os
#     import glob

#     swim_dirs = glob.glob("SWIM_clarknet_agentic_gpt5_2")
#     for swim_dir in swim_dirs:
#         print(f"Processing directory: {swim_dir}")
#         # Call the plot_results function for each directory
#         fig = plot_results("sim", folder=swim_dir, run=1, save_as=f"{swim_dir}.png")


# if __name__ == "__main__":
#     main()
