import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import os
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import norm
import math
from scipy.optimize import minimize

st.set_page_config(
    page_title="Real-Time Inventory Simulation Dashboard",
    page_icon="📦",
    layout="wide",
)

st.title("Real-Time Inventory Simulation Dashboard")

with st.container():
    st.write("This is inside the container")


# Function to perform double exponential smoothing
def double_exponential_smoothing(data):
    return ExponentialSmoothing(data, trend='add').fit(method='least_squares')

# Function to perform simple exponential smoothing
def simple_exponential_smoothing(data):
    return ExponentialSmoothing(data).fit(method='least_squares')

# List of file URLs on GitHub
file_paths = [
    "https://raw.githubusercontent.com/jessicakohkl/invsim/main/docs/grouped_inv_24m_R702010.csv",
    "https://raw.githubusercontent.com/jessicakohkl/invsim/main/docs/medications_dispensed_-_monthly_breakdown_2023_10_09_TEST.xlsx",
]
# Load data
data_dict = {}
for path in file_paths:
    # Use the base name (filename without extension) as the variable name
    variable_name = os.path.splitext(os.path.basename(path))[0]
    data_dict[variable_name] = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)

# Access individual DataFrames
medications_data = data_dict['medications_dispensed_-_monthly_breakdown_2023_10_09_TEST']
grouped_data = data_dict['grouped_inv_24m_R702010']

# Sidebar: Select medication from medications_data
makes = medications_data['medication_name'].drop_duplicates()
make_choice = st.sidebar.selectbox('Select medication:', makes)

# Check if 'class' column exists in grouped_data
if 'class' in grouped_data.columns:
    # Get the class of the selected medication
    class_value = grouped_data[grouped_data['medication_name'] == make_choice]['class'].iloc[0]
    st.sidebar.text(f"Class of selected medication: {class_value}")

    # Filter data based on selected medication
    selected_data = medications_data[medications_data['medication_name'] == make_choice]

    # Convert date column to datetime format
    selected_data.loc[:, 'year_and_month'] = pd.to_datetime(selected_data['year_and_month'], format='%Y-%m')

    # Check if selected_data is not empty
    if not selected_data.empty:
        # Sort the data by the datetime index
        selected_data = selected_data.sort_values('year_and_month')

        # Predictions for the next 3 months
        forecast_months = 3

        # Check if the required column exists
        if 'quantity_dispensed' in selected_data.columns:
            # Check if there's enough data to make a forecast
            if len(selected_data) >= forecast_months:
                # Generate forecast index using the last date in the data plus 1 month
                forecast_start_date = selected_data['year_and_month'].max() + pd.DateOffset(months=1)
                forecast_index = pd.date_range(forecast_start_date, periods=forecast_months, freq='MS')

                # Choose the appropriate model based on the class
                if class_value == 'A':
                    model_result = double_exponential_smoothing(selected_data['quantity_dispensed'])
                elif class_value == 'B':
                    model_result = simple_exponential_smoothing(selected_data['quantity_dispensed'])
                elif class_value == 'C':
                    model_result = double_exponential_smoothing(selected_data['quantity_dispensed'])
                else:
                    st.write("Unsupported class type.")

                # Make predictions with confidence intervals
                forecast_results = model_result.forecast(steps=forecast_months)

                # Extract point estimates and limits
                residuals = model_result.resid
                std_errors = np.std(residuals)
                lower_limits = forecast_results - 1.96 * std_errors
                upper_limits = forecast_results + 1.96 * std_errors

                # Create a DataFrame for confidence intervals
                confidence_intervals_df = pd.DataFrame({
                    'Lower Confidence Limit': lower_limits,
                    'Point Estimate': forecast_results,
                    'Upper Confidence Limit': upper_limits
                })

                # Display the confidence intervals table
                st.write("Confidence Intervals Table:")
                confidence_intervals_df.reset_index(inplace=True)
                confidence_intervals_df['index'] = forecast_index.strftime('%Y-%m')  # Use forecast_index for the correct date range
                confidence_intervals_df = confidence_intervals_df.rename(columns={'index': 'Time'})

                # Display the confidence intervals table
                st.write(confidence_intervals_df)

                # Plot the forecast with confidence intervals
                plt.figure(figsize=(12, 6))

                # Plot historical data
                plt.plot(selected_data['year_and_month'], selected_data['quantity_dispensed'], label='Historical Data', marker='o')

                # Plot forecast data with confidence intervals
                plt.plot(forecast_index, forecast_results, label='Forecast', linestyle='dashed', marker='o', color='orange')
                plt.fill_between(forecast_index, lower_limits, upper_limits, color='orange', alpha=0.2, label='Confidence Interval')

                # Format the x-axis to show dates properly
                plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
                plt.gca().xaxis.set_major_locator(matplotlib.dates.MonthLocator())

                plt.title(f'Forecast for {make_choice} with Confidence Intervals')
                plt.xlabel('Year and Month')
                plt.ylabel('Quantity Dispensed')
                plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
                plt.legend(loc='upper left')
                st.pyplot(plt)

            else:
                st.write("Not enough data to make a forecast.")
        else:
            st.write("Column 'quantity_dispensed' not found in selected data.")
    else:
        st.write("No data found for the selected medication.")
else:
    st.write("Class information not available for the selected medication.")


with st.container():
    st.write("This is inside another the container")


# Define parameters for each medicine (customize as per your data)
medicine_data = [
    {
        'medicine_name': 'Serratiopeptidase 5mg tab (Danzen)',
        'initial_inventory': 150000,
        'forecasted_demand': 164882.517995,         # Average monthly demand
        'upper_confidence_limit': 193030.597836,  # Upper confidence limit for demand
        'lower_confidence_limit': 136734.438154,  # Lower confidence limit for demand
        'lead_time_mean': 14,      # Average lead time in days
        'lead_time_std_dev': 2,   # Standard deviation of lead time
        'confidence_level': 0.95,
        'order_cost': 0.1,     # Cost per order
        'holding_cost_per_unit': 0.05,  # Holding cost per unit
    },
    {
        'medicine_name': '* Mometasone 0.1% cream (15g) (Elomet)',
        'initial_inventory': 100,
        'forecasted_demand': 150.442797,         # Average monthly demand
        'upper_confidence_limit': 186.492605,  # Upper confidence limit for demand
        'lower_confidence_limit': 114.39299,  # Lower confidence limit for demand
        'lead_time_mean': 14,      # Average lead time in days
        'lead_time_std_dev': 2,   # Standard deviation of lead time
        'confidence_level': 0.95,
        'order_cost': 15.8,     # Cost per order
        'holding_cost_per_unit': 0.05,  # Holding cost per unit
    },
    {
        'medicine_name': 'Azithromycin 250mg tab (Imexa)',
        'initial_inventory': 500,
        'forecasted_demand': 606.565514,         # Average monthly demand
        'upper_confidence_limit': 776.957756,  # Upper confidence limit for demand
        'lower_confidence_limit': 436.173272,  # Lower confidence limit for demand
        'lead_time_mean': 14,      # Average lead time in days
        'lead_time_std_dev': 2,   # Standard deviation of lead time
        'confidence_level': 0.95,
        'order_cost': 1.2,     # Cost per order
        'holding_cost_per_unit': 0.05,  # Holding cost per unit
    }
]

def calculate_optimal_restocking(medicine):
    forecasted_demand = medicine['forecasted_demand']
    upper_confidence_limit = medicine['upper_confidence_limit']
    lower_confidence_limit = medicine['lower_confidence_limit']
    lead_time_mean = medicine['lead_time_mean']
    lead_time_std_dev = medicine['lead_time_std_dev']
    order_cost = medicine['order_cost']
    holding_cost_per_unit = medicine['holding_cost_per_unit']

    # Parameters for simulation
    simulations = 1000

    # Simulate demand using a normal distribution
    demand_simulations_months = np.random.normal(forecasted_demand, scale=(upper_confidence_limit - lower_confidence_limit) / 6, size=simulations)
    demand_simulations_days = demand_simulations_months/30 

    # Calculate the restock point and restock quantity
    z = norm.ppf(medicine['confidence_level'])  # Z-score for the desired confidence level
    average_demand = np.mean(demand_simulations_days)
    # average_demand
    std_dev_demand = np.std(demand_simulations_months)
    # std_dev_demand
    restock_point = z * std_dev_demand * np.sqrt(lead_time_mean) + average_demand * lead_time_mean
    # restock_point
    safety_stock = z * std_dev_demand * np.sqrt(lead_time_mean)
    restock_quantity = max(0, restock_point - medicine['initial_inventory']/30)
    # restock_quantity

    # Calculate the total cost of the restocking policy
    total_cost = (order_cost * forecasted_demand / restock_quantity) + (holding_cost_per_unit * restock_quantity / 2)

    return restock_point, restock_quantity, total_cost,average_demand, std_dev_demand, safety_stock

lead_time = int(np.random.normal(medicine_data[0]['lead_time_mean'],medicine_data[0]['lead_time_std_dev']))

def simulate_inventory(medicine):
    initial_inventory = medicine['initial_inventory']
    restock_point, restock_quantity, _, _, _ ,safety_stock= calculate_optimal_restocking(medicine)
    
    # Lists to track inventory levels and costs
    inventory_levels = [initial_inventory]
    holding_costs = [0]  # Initialize holding cost
    lead_time = 0
    
    for day in range(1, 91):  # 90-day simulation
        # Simulate daily demand using a normal distribution
        daily_demand = np.random.normal(medicine['forecasted_demand'] / 30, 
                                        scale=(medicine['upper_confidence_limit'] - medicine['lower_confidence_limit']) / 6 / 30)
        
        inventory_level = inventory_levels[-1] - daily_demand
        
        if lead_time > 0:
            lead_time -= 1
            if lead_time == 0:
                inventory_level += restock_quantity
                holding_costs.append(holding_costs[-1] + inventory_levels[-1] * medicine['holding_cost_per_unit'])
        elif inventory_level <= restock_point:
            # If inventory reaches or falls below the reorder point, place an order
            lead_time = int(np.random.normal(medicine['lead_time_mean'],medicine['lead_time_std_dev']))
        
        inventory_levels.append(inventory_level)
    
    return inventory_levels, holding_costs, restock_point,safety_stock


inventory_data = []
holding_costs_data = []

# Sidebar options
# make_choice = st.sidebar.selectbox('Select medication:', makes)

selected_medicine = make_choice
# selected_medicine = st.sidebar.selectbox('Select Medicine', [medicine['medicine_name'] for medicine in medicine_data])

if selected_medicine:
    # Find the selected medicine
    selected_medicine_data = next((m for m in medicine_data if m['medicine_name'] == selected_medicine), None)

    if selected_medicine_data:
        # Calculate optimal restocking policy and simulate inventory movement
        optimal_restock_point, optimal_restock_quantity, total_cost, average_demand, std_dev_demand, safety_stock = calculate_optimal_restocking(selected_medicine_data)
        inventory_levels, holding_costs, restock_point, safety_stock = simulate_inventory(selected_medicine_data)

        # Display the simulation results in the sidebar
        st.sidebar.write('Simulation Results:')
        st.sidebar.write(f"Optimal Reorder Point: {optimal_restock_point:.2f}")
        st.sidebar.write(f"Optimal Reorder Quantity: {optimal_restock_quantity:.2f}")
        st.sidebar.write(f"Total Cost (including holding costs): {total_cost:.2f}")
        st.sidebar.write(f"Average Daily Demand: {average_demand:.2f}")
        st.sidebar.write(f"Standard Deviation of Daily Demand: {std_dev_demand:.2f}")
        st.sidebar.write(f"Safety Stock: {safety_stock:.2f}")

        # Plot the inventory movement for the selected medicine in the main content area
        st.write('Inventory Movement Plot:')
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, 92), inventory_levels, marker='o', linestyle='-')
        plt.axhline(y=restock_point, color='r', linestyle='--', label='Restock Point')
        plt.axhline(y=safety_stock, color='r', linestyle='--', label='Safety Stock')
        plt.xlabel('Day')
        plt.ylabel('Inventory Level')
        plt.title(f'Inventory Movement for {selected_medicine_data["medicine_name"]}')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.sidebar.write('Select a medicine from the sidebar to view its inventory simulation.')
