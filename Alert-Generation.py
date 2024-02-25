import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import datetime

# Database Connection Configuration
db_config = {
    'username': 'user_email@example.com',
    'password': 'your_password', 
    'account_name': 'your_account',
    'database_name': 'your_database',
    'schema_name': 'your_schema',
    'authenticator_url': 'https://your_authenticator_url',
    'role_name': 'your_role',
    'warehouse_name': 'your_warehouse',
}

# Creating engine and establishing connection
# engine = create_engine(f"snowflake://{db_config['username']}:{db_config['password']}@{db_config['account_name']}/{db_config['database_name']}/{db_config['schema_name']}?authenticator={db_config['authenticator_url']}&role={db_config['role_name']}&warehouse={db_config['warehouse_name']}")
# connection = engine.connect()

data_query_platform_1 = '''SELECT DATE, METRIC, PLATFORM, SUM(VALUE) AS TOTAL_VALUE
FROM DATA_TABLE_1
WHERE DATE > '2015-01-01' AND TO_DATE(DATE) < CURRENT_DATE() AND CONDITION = 'condition_1'
GROUP BY DATE, METRIC, PLATFORM;'''

data_query_platform_2 = '''SELECT DATE, METRIC, PLATFORM, SUM(VALUE) AS TOTAL_VALUE
FROM DATA_TABLE_2
WHERE DATE > '2015-01-01' AND TO_DATE(DATE) < CURRENT_DATE() AND CONDITION = 'condition_2'
GROUP BY DATE, METRIC, PLATFORM;'''

'''
raw_data_Platform1 = pd.read_sql_query(data_query_platform_1,conn)
raw_data_Platform2 = pd.read_sql_query(data_query_platform_2,conn)
'''

# Configuration
analysis_config = {
    'is_test_run': False,
    'confidence_interval': 0.1,
    'reduced_training_window': 0.2,  # Set to None to disable
}

def determine_time_granularity():
    current_day = datetime.datetime.now().weekday()
    return ['daily', 'weekly'] if current_day == 0 else ['daily']

def load_and_preprocess_data(file_paths, replacements):
    combined_data = pd.DataFrame()
    for path, replace_dict in file_paths.items():
        data = pd.read_csv(path)
        data['DATE'] = pd.to_datetime(data['DATE'])
        for original, new in replace_dict.items():
            data['METRIC'] = data['METRIC'].replace(original, new)
        combined_data = pd.concat([combined_data, data])
    return combined_data

# Define file paths and metric replacements in a dictionary for clarity
file_paths = {
    "C:/path/to/Platform1_data.csv": {'MetricA': 'MetricB', 'MetricC': 'MetricD', 'MetricE': 'MetricF'},
    "C:/path/to/Platform2_data.csv": {'MetricG': 'MetricH'},
}

def filter_data_by_platform(data, platforms):
    return data[data['PLATFORM'].isin(platforms)]

def send_slack_alert(message, channel="#alerts_channel", token="YOUR_SLACK_TOKEN"):
    client = WebClient(token=token)
    try:
        response = client.chat_postMessage(channel=channel, text=message)
        print("Alert sent successfully.")
    except SlackApiError as e:
        print(f"Error posting message to Slack: {e.response['error']}")

time_granularity = determine_time_granularity()
processed_data = filter_data_by_platform(raw_data, ['Platform1', 'Platform2', 'Platform3'])

# Convert the 'date' column to datetime
raw_data['DATE'] = pd.to_datetime(raw_data['DATE'])

def transform_data_to_array(data, metric):
    metric_data = data[data['METRIC'] == metric]
    values_array = metric_data[['VALUE']].to_numpy().T
    return values_array

def fill_missing_values(data, metrics, platforms, mark_temp_rows=False):
    """
    Fills missing values in the dataset for each metric and platform combination.
    
    Parameters:
    - data: DataFrame containing the processed data.
    - metrics: List of unique metrics to be considered.
    - platforms: List of unique platforms to be considered.
    - mark_temp_rows: Boolean indicating whether to mark newly added rows.
    
    Returns:
    - DataFrame with missing values filled.
    """
    # Initialize an empty DataFrame for storing missing rows
    missing_rows = pd.DataFrame(columns=['DATE', 'PLATFORM', 'METRIC', 'TOTAL_VALUE'])
    if mark_temp_rows:
        missing_rows['temp_added'] = False 

    for platform in platforms:
        for metric in metrics:
            mask = (data['PLATFORM'] == platform) & (data['METRIC'] == metric)
            existing_data = data[mask]
            last_observed_value = {}
            weeks_to_look_back = 4
            existing_data_lookup = existing_data.set_index('DATE')['TOTAL_VALUE']

            for date, row in existing_data.iterrows():
                for week in range(1, weeks_to_look_back + 1):
                    prev_week_date = row['DATE'] - pd.DateOffset(weeks=week)
                    if prev_week_date in existing_data_lookup:
                        last_observed_value[row['DATE']] = existing_data_lookup[prev_week_date]
                        break

            all_dates = pd.date_range(start=data['DATE'].min(), end=data['DATE'].max(), freq='D')
            missing_dates = all_dates.difference(existing_data['DATE'])

            for missing_date in missing_dates:
                prev_date = missing_date - pd.DateOffset(weeks=1)
                value = last_observed_value.get(prev_date, 0)
                missing_row = {'DATE': missing_date, 'PLATFORM': platform, 'METRIC': metric, 'TOTAL_VALUE': value}
                if mark_temp_rows:
                    missing_row['temp_added'] = True
                missing_rows = missing_rows.append(missing_row, ignore_index=True)

    # Combine the original data with the missing rows
    combined_data = pd.concat([data, missing_rows], ignore_index=True).sort_values(by='DATE')
    if mark_temp_rows:
        # Ensure existing data rows are not marked as temporary
        combined_data['temp_added'] = combined_data['temp_added'].fillna(False)

    return combined_data

def delay_adjustment(metric, delay_data, data_to_adjust):
    """
    Adjusts data based on delay information for a specific metric.
    
    Parameters:
    - metric: The specific metric to be adjusted.
    - delay_data: DataFrame containing delay information.
    - data_to_adjust: DataFrame containing the data to be adjusted.
    
    Returns:
    - DataFrame with adjusted values.
    """
    # Mapping metric to its corresponding cumulative column in delay data
    cumulative_column_map = {
        'MetricJ': 'reverse_cumulative_MetricJ_proportion',
        'MetricI': 'reverse_cumulative_MetricI_proportion'
    }
    cumulative_column = cumulative_column_map.get(metric, None)
    
    if cumulative_column is None:
        return data_to_adjust  # Return unmodified if metric not recognized

    most_recent_date = data_to_adjust['DATE'].max()
    
    for idx, row in data_to_adjust.iterrows():
        day_of_month = row['DATE'].day
        days_since_most_recent = (row['DATE'] - most_recent_date).days
        
        if days_since_most_recent > 100:
            continue  # Skip adjustment if the data point is too far in the past
        
        # Filter delay data for the current day of month
        candidate_rows = delay_data[delay_data['day_of_month'] == day_of_month]
        
        # Find the closest matching delay not exceeding the days since the most recent date
        closest_delay = candidate_rows.loc[candidate_rows['MetricJ_delay'] <= days_since_most_recent, 'reversal_delay'].max()
        
        if pd.notna(closest_delay):
            # Apply adjustment using the corresponding 'C' value
            C_value = candidate_rows.loc[candidate_rows['MetricJ_delay'] == closest_delay, cumulative_column].iloc[0]
            data_to_adjust.at[idx, 'TOTAL_VALUE'] = row['TOTAL_VALUE'] / C_value

    return data_to_adjust

def fill_missing_values_seasonality(data, metrics, platforms, mark_temp_rows=False):
    """
    Fills missing daily values based on the average of the same weekday in the past four weeks.
    
    Parameters:
    - data: DataFrame containing the processed data.
    - metrics: List of metrics to fill missing values for.
    - platforms: List of platforms to consider for filling missing values.
    - mark_temp_rows: Boolean to mark rows added to fill missing values.
    
    Returns:
    - DataFrame with missing values filled based on past weekday averages.
    """
    original_indexes = data.index
    
    filled_data = data.copy()
    all_dates = pd.date_range(start=filled_data['DATE'].min(), end=filled_data['DATE'].max(), freq='D')
    filled_data.set_index('DATE', inplace=True)
    
    for platform in platforms:
        for metric in metrics:
            for date in all_dates:
                if date not in filled_data.index or filled_data.loc[date, 'METRIC'] != metric or filled_data.loc[date, 'PLATFORM'] != platform:
                    # Calculate the average for the same weekday in the past four weeks
                    past_dates = [date - pd.DateOffset(weeks=w) for w in range(1, 5)]
                    past_values = filled_data.loc[(filled_data.index.isin(past_dates)) & (filled_data['METRIC'] == metric) & (filled_data['PLATFORM'] == platform), 'TOTAL_VALUE']
                    avg_value = past_values.mean() if not past_values.empty else 0
                    
                    filled_row = {'DATE': date, 'PLATFORM': platform, 'METRIC': metric, 'TOTAL_VALUE': avg_value}
                    if mark_temp_rows:
                        filled_row['temp_added'] = True
                    
                    filled_data = filled_data.append(filled_row, ignore_index=True)
    
    filled_data.sort_index(inplace=True)
    
    # Restore the original index order, if necessary
    filled_data = filled_data.loc[original_indexes]
    
    return filled_data


def systematically_add_missing_metric_rows(data, metrics, platforms, value_column='TOTAL_VALUE'):
    """
    Adds rows with a specified value (default 0) for systematically missing metrics across all platforms.

    Parameters:
    - data: DataFrame to which missing metric rows will be added.
    - metrics: List of metrics to handle for systematic absence.
    - platforms: List of platforms to consider for adding missing rows.
    - value_column: Name of the column where the value is stored.

    Returns:
    - DataFrame with missing metric rows added.
    """
    # Ensure data is sorted by date to correctly compute all_dates range
    data.sort_values('DATE', inplace=True)
    
    # Determine the complete date range within the dataset
    all_dates = pd.date_range(start=data['DATE'].min(), end=data['DATE'].max(), freq='D')
    missing_rows_list = []

    for platform in platforms:
        for metric in metrics:
            existing_dates = data.loc[(data['PLATFORM'] == platform) & (data['METRIC'] == metric), 'DATE'].unique()
            
            # Identify missing dates for the current metric and platform
            missing_dates = [date for date in all_dates if date not in existing_dates]
            
            # Prepare missing rows
            for missing_date in missing_dates:
                missing_row = {'DATE': missing_date, 'PLATFORM': platform, 'METRIC': metric, value_column: 0}
                missing_rows_list.append(missing_row)

    # Create a DataFrame for the missing rows and concatenate with the original data
    missing_rows_df = pd.DataFrame(missing_rows_list, columns=['DATE', 'PLATFORM', 'METRIC', value_column])
    updated_data = pd.concat([data, missing_rows_df], ignore_index=True).sort_values(by='DATE')
    
    return updated_data

# Apply the function to systematically add missing metric rows
unique_platforms = ['Platform1', 'Platform2', 'Platform3'] 
metrics_to_handle = ['MetricA', 'MetricB', 'MetricC', 'MetricD', 'MetricE']

processed_data = systematically_add_missing_metric_rows(processed_data, metrics_to_handle, unique_platforms, 'TOTAL_VALUE')

# After systematic addition, clean up any temporary rows and columns if needed
processed_data.drop(columns=['temp_added'], inplace=True, errors='ignore')  # Use errors='ignore' to avoid error if column doesn't exist

# Sort the DataFrame by 'DATE'
processed_data.sort_values(by='DATE', inplace=True)

# Initialize an empty DataFrame for calculated metrics
calculated_metrics = pd.DataFrame()

# Iterate over each unique platform and metric to perform calculations
for platform in processed_data['PLATFORM'].unique():
    for metric in processed_data['METRIC'].unique():
        platform_metric_data = processed_data[(processed_data['PLATFORM'] == platform) & (processed_data['METRIC'] == metric)]

        # Example calculation: MetricB percentage of MetricA
        if metric == "MetricB":
            result = platform_metric_data['TOTAL_VALUE'] / (processed_data[processed_data['METRIC'] == 'MetricA']['TOTAL_VALUE'] + platform_metric_data['TOTAL_VALUE'])
            result_df = result.reset_index(name='VALUE')
            result_df['METRIC'] = 'MetricB_Percentage_of_MetricA'
            result_df['PLATFORM'] = platform
            calculated_metrics = pd.concat([calculated_metrics, result_df], ignore_index=True)

# Note: Implement similar calculations for other metrics as required.

# Concatenate the original DataFrame with the new calculated metrics DataFrame
processed_data = pd.concat([processed_data, calculated_metrics], ignore_index=True).sort_values(by='DATE')

# Ensure all necessary metrics are present and add missing ones with 0 values as needed
# This step is especially important for newly calculated metrics
metrics_to_ensure = ['MetricB_Percentage_of_MetricA', 'MetricA', 'MetricB', 'MetricC', 'MetricD', 'MetricE', 'MetricI', 'MetricJ']
for metric in metrics_to_ensure:
    if metric not in processed_data['METRIC'].unique():
        missing_metric_df = pd.DataFrame({
            'DATE': processed_data['DATE'].unique(),
            'PLATFORM': platform,
            'METRIC': metric,
            'TOTAL_VALUE': 0
        })
        processed_data = pd.concat([processed_data, missing_metric_df], ignore_index=True)

# Sorting and cleanup
processed_data.sort_values(by=['DATE', 'PLATFORM', 'METRIC'], inplace=True

def calculate_new_metrics(data):
    """
    Parameters:
    - data: DataFrame containing the processed and initial calculated data.
    
    Returns:
    - DataFrame with new calculated metrics added.
    """
    # Ensure 'TOTAL_VALUE' is numeric for calculations
    data['TOTAL_VALUE'] = pd.to_numeric(data['TOTAL_VALUE'], errors='coerce')

    # Example calculation for MetricR for all data
    gross_margin_df = pd.DataFrame()
    for platform in data['PLATFORM'].unique():
        platform_data = data[data['PLATFORM'] == platform]
        net_revenue_data = platform_data[platform_data['METRIC'] == 'MetricC']
        total_cost_data = platform_data[platform_data['METRIC'] == 'MetricG']
        
        # Ensure matching index for proper subtraction
        total_cost_data = total_cost_data.set_index(['DATE', 'PLATFORM'])
        net_revenue_data = net_revenue_data.set_index(['DATE', 'PLATFORM'])
        
        # Calculation
        gross_margin = (net_revenue_data['TOTAL_VALUE'] - total_cost_data['TOTAL_VALUE']) / net_revenue_data['TOTAL_VALUE']
        gross_margin_df = pd.concat([gross_margin_df, gross_margin.reset_index()], ignore_index=True)
    
    gross_margin_df['METRIC'] = 'MetricR'  
    
    # Adding calculated gross margin data to the dataset
    data = pd.concat([data, gross_margin_df], ignore_index=True)

    # Drop any rows that resulted in NaN values during calculation
    data.dropna(subset=['TOTAL_VALUE'], inplace=True)
  
    
    return data

processed_data = calculate_new_metrics(processed_data)

# Ensure the dataset is sorted after adding new metrics
processed_data.sort_values(by=['DATE', 'PLATFORM', 'METRIC'], inplace=True)

# Define the list of alert metrics with completely generic names
alert_metrics = [
    'MetricO', 'MetricP', 'MetricA', 'MetricH', 'MetricC', 'MetricB', 
    'MetricS', 'MetricT', 'MetricR', 'MetricU', 'MetricV', 'MetricW', 
    'MetricX', 'MetricY'
]

def adjust_data_for_time_granularity(data, time_granularity_list, unique_platforms, reduced_training_window=False):
    adjusted_data = pd.DataFrame()

    for time_granularity in time_granularity_list:
        if time_granularity == 'weekly':
            metrics_to_adjust = ['MetricH', 'MetricC', 'MetricR', 'MetricY', 'MetricT', 'MetricS']
            adj_data = data[data['METRIC'].isin(metrics_to_adjust)]
            adj_data = fill_missing_values_seasonality(adj_data, metrics_to_adjust, unique_platforms, mark_temp_rows=True)
            
            # Convert data to weekly frequency
            adj_data['DATE'] = pd.to_datetime(adj_data['DATE'])
            adj_data.set_index('DATE', inplace=True)
            weekly_data = adj_data.groupby(['METRIC', 'PLATFORM']).resample('W').sum().reset_index()
            weekly_data.sort_values(by=['PLATFORM', 'METRIC', 'DATE'], inplace=True)

            # Adjust the TOTAL_VALUE for weekly aggregation
            weekly_data['TOTAL_VALUE'] /= 7

            adjusted_data = pd.concat([adjusted_data, weekly_data], ignore_index=True)
            
        elif time_granularity == 'daily':
            metrics_to_adjust_daily = ['MetricA', 'MetricO', 'MetricP', 'MetricV', 'MetricU', 'MetricW', 'MetricX', 'MetricY']
            adj_data_daily = data[data['METRIC'].isin(metrics_to_adjust_daily)]
            adj_data_daily = fill_missing_values_seasonality(adj_data_daily, metrics_to_adjust_daily, unique_platforms, mark_temp_rows=True)

            adjusted_data = pd.concat([adjusted_data, adj_data_daily], ignore_index=True)

    # Ensure all data is included, either adjusted or original
    remaining_data = data[~data['METRIC'].isin(metrics_to_adjust + metrics_to_adjust_daily)]
    adjusted_data = pd.concat([adjusted_data, remaining_data], ignore_index=True)

    return adjusted_data

# Applying the adjustment
time_granularity_list = determine_time_granularity() 
adjusted_processed_data = adjust_data_for_time_granularity(processed_data, time_granularity_list, unique_platforms, reduced_training_window=True)

# Cleanup: Removing temporary rows and sorting
adjusted_processed_data.drop(columns=['temp_added'], inplace=True, errors='ignore')  # Removing temporary marker if exists
adjusted_processed_data.sort_values(by=['DATE', 'PLATFORM', 'METRIC'], inplace=True)
def perform_seasonal_adjustment(data, adj_period):
    """
    Performs seasonal decomposition on data and adjusts values based on trend and seasonal components.

    Parameters:
    - data: DataFrame containing the adjusted data for analysis.
    - adj_period: The period for seasonal decomposition.

    Returns:
    - DataFrame with seasonally adjusted values.
    """
    adjusted_data = pd.DataFrame()
    
    for platform in data['PLATFORM'].unique():
        for metric in data['METRIC'].unique():
            metric_data = data.loc[(data['PLATFORM'] == platform) & (data['METRIC'] == metric)]
            
            # Ensure data is sorted by DATE for decomposition
            metric_data.sort_values('DATE', inplace=True)
            metric_data.set_index('DATE', inplace=True)
            
            if not metric_data.empty:
                decomposition = seasonal_decompose(metric_data['TOTAL_VALUE'], model='additive', period=adj_period, extrapolate_trend='freq')

                # Adjust data by removing the trend and seasonal components
                metric_data['TOTAL_VALUE_ADJ'] = metric_data['TOTAL_VALUE'] - (decomposition.trend + decomposition.seasonal)
                
                # Prepare adjusted data for merging
                adj_metric_data = metric_data.reset_index()
                adj_metric_data['METRIC'] = metric  # Ensure metric column is included
                adj_metric_data['PLATFORM'] = platform  # Ensure platform column is included
                
                adjusted_data = pd.concat([adjusted_data, adj_metric_data], ignore_index=True)

    # Cleanup
    adjusted_data.dropna(subset=['TOTAL_VALUE_ADJ'], inplace=True)
    adjusted_data.reset_index(drop=True, inplace=True)

    return adjusted_data

# Applying seasonal adjustment
# Assuming 'adj_period' has been defined based on the data's granularity (e.g., 7 for daily, 52 for weekly)
adjusted_processed_data = perform_seasonal_adjustment(adjusted_processed_data, adj_period=7) 

# Forecasting preparation
forecasting_span = range(1, 8)  # Example range for forecasting
retro_predictions = [7]  # Adjust for number of retroactive prediction as desired
    
def prepare_for_forecasting(adj_data, unique_platforms, unique_metrics, time_granularity, reduced_training_window_option):
    """
    Prepares data for forecasting by handling retro predictions and adjusting the dataset based on time granularity.

    Parameters:
    - adj_data: DataFrame containing the seasonally adjusted data.
    - unique_platforms: List of unique platforms in the data.
    - unique_metrics: List of unique metrics to forecast.
    - time_granularity: The granularity of the time series data ('daily' or 'weekly').
    - reduced_training_window_option: Option to use a reduced training window for forecasting.
    """
    forecasting_span = range(1, 8) if time_granularity == 'daily' else range(1, 5)
    retro_predictions = [7]  

    for retro_prediction in retro_predictions:
        for platform in unique_platforms:
            platform_data = adj_data.loc[adj_data['PLATFORM'] == platform]

            for metric in unique_metrics:
                metric_data = platform_data[platform_data['METRIC'] == metric]
                metric_data.sort_values('DATE', inplace=True)

                # Adjust data based on retro prediction logic
                if retro_prediction != 1:
                    # Cutting off the most recent dates for retro prediction
                    metric_data = metric_data[:-retro_prediction]

                # Prepare data for forecasting
                if time_granularity == 'daily' and reduced_training_window_option:
                    # Example adjustment for daily granularity with a reduced training window
                    metric_data = adjust_for_reduced_training_window(metric_data)

                # apply_forecasting_model(metric_data, forecasting_span)

def adjust_for_reduced_training_window(data):
    """
    Adjusts the dataset to use a reduced training window for forecasting.

    Parameters:
    - data: DataFrame of the metric data sorted by date.

    Returns:
    - Adjusted DataFrame based on the reduced training window.
    """
    # Use the last reduced_training_window% of data for the reduced training window
    cutoff_index = int(len(data) * reduced_training_window)
    return data.iloc[cutoff_index:]

unique_metrics_renamed = ['MetricS', 'MetricT', 'MetricR', 'MetricY', 'MetricV', 'MetricW', 'MetricX', 'MetricU']

prepare_for_forecasting(adjusted_processed_data, unique_platforms, unique_metrics_renamed, 'daily', True)
                
def perform_forecasting_and_generate_alerts(data, unique_platforms, unique_metrics, time_granularity, forecasting_span, alpha):
    """
    Performs forecasting using the autoARIMA model and generates alerts based on forecasted intervals.

    Parameters:
    - data: DataFrame containing the seasonally adjusted data ready for forecasting.
    - unique_platforms: List of unique platforms in the data.
    - unique_metrics: List of unique metrics to forecast.
    - time_granularity: The granularity of the data ('daily' or 'weekly').
    - forecasting_span: Range object indicating how many periods ahead to forecast.
    - alpha: Confidence interval for the forecast prediction.
    """
    alert_results = pd.DataFrame()

    for platform in unique_platforms:
        platform_data = data.loc[data['PLATFORM'] == platform]

        for metric in unique_metrics:
            metric_data = platform_data[platform_data['METRIC'] == metric]
            metric_data.sort_values('DATE', inplace=True)

            # Initialize for forecasting
            autoarima_predictions = []
            autoarima_pred_interval = np.empty((0, 2))
            threshold_counter = [0, 0]

            for forecasting_period in forecasting_span:
                train_data = metric_data[:-forecasting_period]
                test_data = metric_data[-forecasting_period:]

                # Fit the autoARIMA model
                autoarima_model = auto_arima(train_data['TOTAL_VALUE_ADJ'], start_p=1, start_q=1, max_p=2, max_q=2, m=7, seasonal=True,
                                             d=1, D=1, trace=False, error_action='ignore', suppress_warnings=True)
                autoarima_pred, autoarima_conf_int = autoarima_model.predict(n_periods=forecasting_period, return_conf_int=True, alpha=alpha)

                # Evaluate forecast against actual data
                actual_value = test_data['TOTAL_VALUE_ADJ'].iloc[-1]
                lower_bound = autoarima_conf_int[-1, 0]
                upper_bound = autoarima_conf_int[-1, 1]

                if actual_value > upper_bound:
                    threshold_counter[1] += 1  # Upper threshold breach
                elif actual_value < lower_bound:
                    threshold_counter[0] += 1  # Lower threshold breach

                # Generate alert if threshold breach exceeds tolerance
                if threshold_counter[1] > 1 or threshold_counter[0] > 1:
                    alert_message = f"{platform} {time_granularity} {'upper' if threshold_counter[1] > 1 else 'lower'} threshold breach for {metric} on {test_data.index[-1]}"
                    send_slack_alert(alert_message) 
                    
                # Store forecast results for analysis
                forecast_result = {
                    'DATE': test_data.index[-1],
                    'PLATFORM': platform,
                    'METRIC': metric,
                    'FORECAST_VALUE': autoarima_pred[-1],
                    'LOWER_BOUND': lower_bound,
                    'UPPER_BOUND': upper_bound,
                    'ACTUAL_VALUE': actual_value
                }
                alert_results = alert_results.append(forecast_result, ignore_index=True)

    return alert_results

# Apply the forecasting and generate alerts
unique_metrics_renamed = ['MetricS', 'MetricT', 'MetricR', 'MetricY', 'MetricV', 'MetricW', 'MetricX', 'MetricU']
forecasting_span = range(1, 8) 
alpha = 0.05  # Confidence interval for forecast prediction

alert_results = perform_forecasting_and_generate_alerts(adjusted_processed_data, unique_platforms, unique_metrics_renamed, 'daily', forecasting_span, alpha)

def finalize_and_save_alert_results(alert_results, test_run):
    """
    Finalizes the alert results by adding threshold order, pivoting the DataFrame, and saving the results.

    Parameters:
    - alert_results: DataFrame containing the forecasting and alert results.
    - test_run: Boolean indicating whether this is a test run or not.
    """
    # Add a column to indicate the order of threshold breaches
    alert_results['THRESHOLD_ORDER'] = alert_results.groupby(['DATE', 'RETRO_N_PREDICTION', 'METRIC', 'PLATFORM', 'TIME_GRANULARITY', 'VALUE']).cumcount() + 1

    # Pivot the DataFrame for a structured view on threshold breaches
    alert_results_pivot = alert_results.pivot_table(index=['DATE', 'RETRO_N_PREDICTION', 'METRIC', 'PLATFORM', 'TIME_GRANULARITY', 'VALUE'],
                                                    columns='THRESHOLD_ORDER',
                                                    values=['LOWER_THRESHOLD', 'UPPER_THRESHOLD'],
                                                    aggfunc='first').reset_index()

    # Dynamically generate column names based on unique threshold breach
    column_mapping = {1: 'FIRST', 2: 'SECOND', 3: 'THIRD', 4: 'FOURTH', 5: 'FIFTH', 6: 'SIXTH', 7: 'SEVENTH'}
    alert_results_pivot.columns = [f'{col[0]}_{column_mapping.get(col[1], col[1])}' if col[1] else col[0] for col in alert_results_pivot.columns]

    # Calculate conditions for lower and upper alert triggers
    alert_results_pivot['Lower Alert Triggered'] = alert_results_pivot.filter(like='LOWER_THRESHOLD').apply(lambda row: 1 if row.dropna().count() >= 3 else 0, axis=1)
    alert_results_pivot['Upper Alert Triggered'] = alert_results_pivot.filter(like='UPPER_THRESHOLD').apply(lambda row: 1 if row.dropna().count() >= 3 else 0, axis=1)

    # Save the results
    file_path = 'alert_results.xlsx' if not test_run else 'alert_results_test.xlsx'
    alert_results_pivot.to_excel(file_path, index=False)

    print(f"Alert results saved to {file_path}")

# Example call to the function
finalize_and_save_alert_results(alert_results, test_run=True)
