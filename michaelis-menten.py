import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy.special import expit
import io

# --- Main Analysis and Visualization Function (Cached for Performance) ---
# This function runs the core modeling and generates all outputs.
# Streamlit's cache decorator prevents re-running this heavy computation on every interaction.

@st.cache_data
def create_forecast_and_outputs(_df, spend_column, conversions_column, percentage_increase):
    """
    Runs the Bayesian Michaelis-Menten model, creates the plot, and generates the summary text.

    Args:
        _df (pd.DataFrame): The filtered and cleaned input DataFrame.
        spend_column (str): The name of the spend column.
        conversions_column (str): The name of the conversions column.
        percentage_increase (float): The budget increase percentage to forecast.

    Returns:
        tuple: A tuple containing the Matplotlib figure, the PNG image as bytes, 
               and a dictionary of summary results. Returns (None, None, {}) on error.
    """
    try:
        spend_obs = _df[spend_column].values
        conversions_obs = _df[conversions_column].values

        # --- Bayesian Michaelis-Menten Growth Model ---
        with pm.Model() as model:
            # Priors for the Michaelis-Menten function parameters: f(x) = (Vmax * x) / (Km + x)
            # Vmax (Maximum Conversions): The theoretical maximum number of conversions (asymptote).
            Vmax = pm.TruncatedNormal("Vmax", mu=conversions_obs.max() * 1.5, sigma=conversions_obs.max(), lower=conversions_obs.max())
            
            # Km (Half-saturation constant): The spend level at which half of Vmax is achieved.
            # Represents the point of diminishing returns.
            Km = pm.HalfNormal("Km", sigma=np.std(spend_obs) * 2)
            
            # Sigma: The noise or variability of the data around the curve.
            sigma = pm.HalfNormal("sigma", sigma=np.std(conversions_obs))

            # Expected conversions based on the Michaelis-Menten function
            mu = (Vmax * spend_obs) / (Km + spend_obs)

            # Likelihood of the observed data
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=conversions_obs)

            # Sample from the posterior distribution using a robust initialization method
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=1, init='advi+adapt_diag')

        # --- Generate Forecast and Analyze Results ---
        current_avg_spend = spend_obs.mean()
        potential_spend = current_avg_spend * (1 + percentage_increase / 100)
        
        budget_range = np.linspace(1, max(spend_obs.max(), potential_spend) * 1.2, 200)
        
        post = az.extract(trace, var_names=["Vmax", "Km", "sigma"])
        
        # Calculate the expected conversion value (mu) for each posterior sample across the budget range
        mu_curves = (post["Vmax"].values[:, np.newaxis] * budget_range) / (post["Km"].values[:, np.newaxis] + budget_range)
        mean_predictions = mu_curves.mean(axis=0)
        
        def predict_conversions(spend_value, trace):
            Vmax_samples = trace.posterior['Vmax'].values.flatten()
            Km_samples = trace.posterior['Km'].values.flatten()
            return ((Vmax_samples * spend_value) / (Km_samples + spend_value)).mean()

        current_avg_conv = predict_conversions(current_avg_spend, trace)
        potential_conv = predict_conversions(potential_spend, trace)
        
        current_cpa = current_avg_spend / current_avg_conv
        potential_cpa = potential_spend / potential_conv
        
        # Point of diminishing returns is represented by Km
        diminishing_returns_spend = trace.posterior['Km'].mean().item()

        # --- Create Graph ---
        plt.rcParams['font.family'] = 'Roboto'
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(budget_range, mean_predictions, c='royalblue', lw=3, label='Potential Conversion Growth Curve')
        ax.plot(current_avg_spend, current_avg_conv, 'o', color='darkorange', markersize=10, zorder=5, label='Current Average Position')
        ax.plot(potential_spend, potential_conv, '*', color='seagreen', markersize=15, zorder=5, label=f'Forecast at +{percentage_increase}% Spend')
        
        # Add dotted lines for the forecast point
        ax.vlines(x=potential_spend, ymin=0, ymax=potential_conv, color='seagreen', linestyle='--', alpha=0.7)
        ax.hlines(y=potential_conv, xmin=0, xmax=potential_spend, color='seagreen', linestyle='--', alpha=0.7)
        
        # Add a line for the point of diminishing returns (Km)
        ax.axvline(x=diminishing_returns_spend, color='crimson', linestyle='--', lw=2, label=f'Point of Diminishing Returns')

        ax.set_xlabel(f'Daily Spend ({spend_column})', fontsize=12)
        ax.set_ylabel('Predicted Daily Conversions', fontsize=12, color='royalblue')
        ax.tick_params(axis='y', labelcolor='royalblue')
        ax.set_ylim(0, max(mean_predictions.max(), potential_conv) * 1.1)
        ax.set_xlim(0, budget_range.max())

        ax.set_title(f'Spend vs. Conversions Forecast', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        ax.legend(loc='upper left')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_bytes = buf.getvalue()
        
        summary_results = {
            "current_spend": current_avg_spend,
            "potential_spend": potential_spend,
            "current_conv": current_avg_conv,
            "potential_conv": potential_conv,
            "current_cpa": current_cpa,
            "potential_cpa": potential_cpa,
            "percentage_increase": percentage_increase
        }
        
        return fig, image_bytes, summary_results

    except Exception as e:
        st.error(f"An error occurred during modeling: {e}")
        return None, None, {}

# --- Streamlit App UI ---

st.title('💸 Campaign Spend Forecaster 💸')

uploaded_file = st.file_uploader("Upload your campaign performance CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        # Define column names
        advertiser_id_column = 'Advertiser ID'
        account_name_column = 'Advertiser Name'
        objective_column = 'Objective Type'
        spend_column = 'Cost (USD)'
        conversions_column = 'Conversions'

        # Check if required columns exist
        required_cols = [advertiser_id_column, account_name_column, objective_column, spend_column, conversions_column]
        if not all(col in data.columns for col in required_cols):
            st.error(f"Error: CSV must contain all required columns: {required_cols}")
        else:
            # Get the single advertiser name from the file
            advertiser_name = data[account_name_column].iloc[0]

            # 3. Objective Selection from filtered data
            # Define objectives to exclude
            excluded_objectives = ['Reach', 'Video Views', 'Reach & Frequency', 'TopView', 'Top Feed', 'TikTok Pulse']
            
            # Filter for objectives with conversions > 0
            objective_conversions = data.groupby(objective_column)[conversions_column].sum()
            valid_objectives = objective_conversions[objective_conversions > 0].index.tolist()
            
            # Filter out the excluded objectives
            final_objectives = [obj for obj in valid_objectives if obj not in excluded_objectives]
            
            if not final_objectives:
                st.warning("No valid objectives with conversion data found for this Advertiser.")
            else:
                # Add "Select All" option
                objective_options = ["Select All"] + final_objectives
                selected_objective = st.selectbox("Choose an Objective to Analyze", options=objective_options)
                
                # 4. Percentage Increase Input
                percentage_increase = st.number_input("Enter Percentage Budget Increase (%)", min_value=0, max_value=200, value=20, step=5)

                # 5. Run Analysis Button
                if st.button("Generate Forecast"):
                    # Filter data for the final analysis based on objective selection
                    if selected_objective == "Select All":
                        analysis_df = data
                    else:
                        analysis_df = data[data[objective_column] == selected_objective]
                    
                    with st.spinner('Running forecasting model... This may take a few minutes.'):
                        fig, image_bytes, summary_res = create_forecast_and_outputs(
                            _df=analysis_df,
                            spend_column=spend_column,
                            conversions_column=conversions_column,
                            percentage_increase=percentage_increase
                        )

                    # 6. Display Outputs
                    if fig:
                        st.markdown("---")
                        st.markdown(f"## {advertiser_name} Campaign Analysis")
                        st.markdown(f"### Forecast for Objective: {selected_objective}")
                        st.markdown(f"## Expected Conversions: {summary_res['potential_conv']:.0f} (vs. {summary_res['current_conv']:.0f} currently)")
                        st.markdown(f"## Expected CPA: ${summary_res['potential_cpa']:.2f} (vs. ${summary_res['current_cpa']:.2f} currently)")
                        st.markdown("---")
                        
                        st.pyplot(fig)
                        
                        st.download_button(
                            label="Download Graph as PNG",
                            data=image_bytes,
                            file_name=f'forecast_{advertiser_name}_{selected_objective}.png',
                            mime='image/png'
                        )

                        # Analysis Bullet Points
                        st.markdown("---")
                        st.subheader("Analysis")
                        
                        # Use st.text to ensure consistent font and no markdown interpretation
                        summary_bullet_1 = (
                            f"• Current State: The campaigns are currently averaging ${summary_res['current_spend']:,.2f} in daily spend, "
                            f"resulting in approximately {summary_res['current_conv']:.0f} conversions at a CPA of ${summary_res['current_cpa']:.2f}."
                        )
                        st.text(summary_bullet_1)

                        summary_bullet_2 = (
                            f"• Growth Opportunity: Increasing the daily budget by {summary_res['percentage_increase']}% to ${summary_res['potential_spend']:,.2f} "
                            f"is forecasted to yield approximately {summary_res['potential_conv']:.0f} conversions. The model predicts this increase would be cost-effective, "
                            f"with an expected CPA of ${summary_res['potential_cpa']:.2f}."
                        )
                        st.text(summary_bullet_2)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
