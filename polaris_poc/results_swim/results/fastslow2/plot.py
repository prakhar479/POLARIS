import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
SCA_FILE = "sim-1s.csv"  # Scalar data (like sim-1s.sca)
VEC_FILE = "sim-1.csv"   # Vector data (like sim-1.vec)
EVALUATION_PERIOD = 60   # Time period in seconds for aggregation (from R script)

def get_vector_data(df, module_name, data_name):
    """
    Finds and extracts a specific vector data series from the dataframe.
    The vectime and vecvalue columns are parsed from strings into number arrays.
    """
    # Filter to find the specific row containing our vector data
    filtered_df = df[(df['module'] == module_name) & (df['name'] == data_name)]
    
    if filtered_df.empty:
        print(f"Warning: Vector data for module='{module_name}', name='{data_name}' not found.")
        return None

    # Get the first row (there should only be one)
    data_row = filtered_df.iloc[0]
    
    # Extract the space-separated string values
    time_str = data_row.get('vectime', '')
    value_str = data_row.get('vecvalue', '')

    # Check for missing data
    if pd.isna(time_str) or pd.isna(value_str) or not time_str.strip():
        print(f"Warning: Vector data for '{data_name}' is empty.")
        return None
        
    # Modern way to convert string of numbers to a NumPy array
    time_list = time_str.strip().split()
    value_list = value_str.strip().split()

    if not time_list or not value_list:
        print(f"Warning: Vector data for '{data_name}' is empty after splitting.")
        return None

    times = np.array(time_list, dtype=float)
    values = np.array(value_list, dtype=float)
    
    # Return as a clean pandas DataFrame
    return pd.DataFrame({'time': times, 'value': values})

def periodic_average(df, period):
    """
    Computes the simple average of a metric for each period.
    Equivalent to the R script's `periodicAverage` function.
    """
    if df is None or df.empty:
        return pd.DataFrame({'time': [], 'value': []})
        
    start_time = np.floor(df['time'].min() / period) * period
    end_time = np.ceil(df['time'].max() / period) * period
    
    # Create time bins for the given period
    bins = np.arange(start_time, end_time + period, period)
    labels = bins[:-1]
    
    # Assign each row to a time bin, ensuring the 'time_bin' column is categorical
    df['time_bin'] = pd.cut(df['time'], bins=bins, right=False, labels=labels)
    
    # Group by the bin and calculate the mean. Explicitly handle observed=True for categoricals.
    avg_df = df.groupby('time_bin', observed=True)['value'].mean()
    
    # Reindex to include all time bins, filling empty ones with 0
    avg_df = avg_df.reindex(labels, fill_value=0).reset_index()
    
    avg_df.rename(columns={'time_bin': 'time'}, inplace=True)
    
    # Adjust time to represent the end of the interval
    avg_df['time'] += period
    return avg_df

def periodic_time_weighted_average(df, period):
    """
    Computes the time-weighted average of a metric for each period.
    This is for step-like data where a value is held over time.
    Equivalent to the R script's `periodicTimeWeightedAverage` function.
    """
    if df is None or df.empty:
        return pd.DataFrame({'time': [], 'value': []})

    start_time = 0 # Assume simulation starts at or after 0
    end_time = np.ceil(df['time'].max() / period) * period
    intervals = np.arange(start_time, end_time + period, period)
    
    weighted_averages = []
    
    # Ensure the dataframe is sorted by time, which is crucial for this logic.
    df = df.sort_values('time').reset_index(drop=True)

    # Iterate through each evaluation period interval
    for i in range(len(intervals) - 1):
        interval_start, interval_end = intervals[i], intervals[i+1]
        
        # Find the last data point recorded *before* this interval began
        last_point_before = df[df['time'] < interval_start].tail(1)
        
        # Find all data points recorded *within* this interval
        points_in_interval = df[(df['time'] >= interval_start) & (df['time'] < interval_end)]
        
        # Combine them to get all data relevant to this interval's calculation
        interval_df = pd.concat([last_point_before, points_in_interval]).reset_index(drop=True)

        if interval_df.empty:
            weighted_averages.append({'time': interval_end, 'value': 0})
            continue

        event_times = interval_df['time'].to_numpy()
        event_values = interval_df['value'].to_numpy()
        
        effective_times = np.maximum(event_times, interval_start)
        time_points = np.append(effective_times, interval_end)
        durations = np.diff(time_points)
        
        total_duration = durations.sum()
        if total_duration > 0:
            weighted_avg = np.sum(event_values * durations) / total_duration
            weighted_averages.append({'time': interval_end, 'value': weighted_avg})
        else:
            # Fallback to the last known value if duration is zero
            last_value = event_values[-1] if len(event_values) > 0 else 0
            weighted_averages.append({'time': interval_end, 'value': last_value})

    return pd.DataFrame(weighted_averages)

def main():
    """Main function to load data, process, and plot."""
    try:
        df_sca = pd.read_csv(SCA_FILE)
        df_vec = pd.read_csv(VEC_FILE)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the correct directory.")
        return

    # --- 1. Process Data for Each Plot ---
    # The module and name strings have been updated based on your diagnostic output.

    # Plot 1: Total Requests/sec (from inter-arrival times)
    requests_raw = get_vector_data(df_vec, "SWIM.arrivalMonitor", "interArrival:vector")
    if requests_raw is not None and not requests_raw.empty:
        # Rate is 1 / time between arrivals. Avoid division by zero.
        requests_raw['value'] = 1 / requests_raw['value'].replace(0, np.nan)
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
        utility_cumulative['value'] = utility_cumulative['value'].cumsum()


    # --- 2. Create the Plots ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.suptitle('Web Infrastructure Simulation Analysis', fontsize=16)

    # Plot 1: Requests/sec
    if not requests_avg.empty:
        axes[0].plot(requests_avg['time'], requests_avg['value'], marker='o', linestyle='-', markersize=4)
    axes[0].set_ylabel("Requests/sec")
    axes[0].set_title("Total Requests per Second")

    # Plot 2: Active Servers
    if not servers_avg.empty:
      axes[1].plot(servers_avg['time'], servers_avg['value'], marker='o', linestyle='-', color='tab:orange', markersize=4)
    axes[1].set_ylabel("Servers")
    axes[1].set_title("Active Servers (Time-Weighted Avg)")

    # Plot 3: Brownout Factor
    if not dimmer_avg.empty:
      axes[2].plot(dimmer_avg['time'], 1-dimmer_avg['value'], marker='o', linestyle='-', color='tab:green', markersize=4)
    axes[2].set_ylabel("Factor")
    axes[2].set_title("Brownout Factor (Time-Weighted Avg)")
    axes[2].set_ylim(0, 1.05) # Brownout factor is typically between 0 and 1

    # Plot 4: Response Time
    if not response_avg.empty:
        axes[3].plot(response_avg['time'], response_avg['value'], marker='o', linestyle='-', color='tab:red', markersize=4)
    axes[3].set_ylabel("Time (s)")
    axes[3].set_title("Mean Response Time (Request Lifetime)")

    # Plot 5: Cumulative Utility
    if not utility_cumulative.empty:
        axes[4].plot(utility_cumulative['time'], utility_cumulative['value'], marker='o', linestyle='-', color='tab:purple', markersize=4)
    axes[4].set_ylabel("Total Utility")
    axes[4].set_title("Cumulative Utility")


    # --- 3. Common Formatting ---
    for ax in axes: # Apply to all 5 plots now
        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    
    plt.xlabel("Time (s)")
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()

