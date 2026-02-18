import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data and Filter for Next UK (Website Conversions) ---
file_data = "Multicat _ L180D spend_conversions + campaign objective w advertiser id filter-2025-11-25 14-04-50.csv"
df = pd.read_csv(file_data)

# Filter for the relevant advertiser and objective
advertiser_name = 'Next UK'
df_filtered = df[(df['Advertiser Name'] == advertiser_name) & (df['Objective Type'] == 'Website Conversions')].copy()

# Prepare for time series analysis (ds, y format needed for Prophet/time-series models)
df_filtered['ds'] = pd.to_datetime(df_filtered['p_date'])
df_filtered = df_filtered.sort_values('ds').set_index('ds')

# Calculate CPA for the model target variable (y)
df_filtered['CPA_Raw'] = df_filtered['Cost (USD)'] / df_filtered['Conversions']

# --- 2. Remove Outliers (IQR Method) ---
# Goal: Stop the model from overfitting on extreme, short-term efficiency spikes (anomalies).
Q1 = df_filtered['CPA_Raw'].quantile(0.25)
Q3 = df_filtered['CPA_Raw'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap/Floor outliers (a replacement method that preserves the overall data size)
df_filtered['CPA_Cleaned'] = df_filtered['CPA_Raw'].clip(lower=lower_bound, upper=upper_bound)
print(f"CPA Outliers Capped. Lower Bound: ${lower_bound:.2f}, Upper Bound: ${upper_bound:.2f}")

# --- 3. Dampen Data Anomaly (7-Day Rolling Mean) ---
# Goal: Smooth out daily noise and isolate the true underlying weekly trend for the time-series model.
df_filtered['CPA_Smoothed_7Day'] = df_filtered['CPA_Cleaned'].rolling(window=7, center=True).mean()

# The 'CPA_Smoothed_7Day' column is the new, cleaned time series data ('y')
# to be fed into your time series model (e.g., Prophet, ARIMA).

# --- 4. Prepare Final Data for New Model ---
# Rename the cleaned column to 'y' and reset the index for Prophet compatibility
df_model_input = df_filtered.reset_index()[['ds', 'CPA_Smoothed_7Day']].rename(columns={'CPA_Smoothed_7Day': 'y'})

# Drop the NaN values created by the rolling window calculation
df_model_input = df_model_input.dropna()

# Output a sample of the cleaned data
print("\nSample of Cleaned Data Ready for New Time-Series Model (Prophet/ARIMA):")
print(df_model_input.head())

# --- 5. Visualization of Cleaning Process ---
# (This code is for confirmation of cleaning effect, not part of the model)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_filtered.index, df_filtered['CPA_Raw'], label='1. CPA (Raw Anomalies)', color='red', alpha=0.5, linestyle='--')
plt.plot(df_filtered.index, df_filtered['CPA_Smoothed_7Day'], label='2. CPA (7-Day Rolling Average - Dampened)', color='green', linewidth=2)
plt.title('CPA Data Cleaning and Smoothing (Dampening Anomalies)')
plt.xlabel('Date (6-Month Lookback)')
plt.ylabel('CPA (USD)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cpa_data_cleaning_and_dampening.png')
