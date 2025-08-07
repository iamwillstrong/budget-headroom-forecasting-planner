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
    Runs the Bayesian model, creates the plot, and generates the summary text.

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

        # Run Bayesian model
        with pm.Model() as model:
            L = pm.Normal("L", mu=conversions_obs.max() * 1.5, sigma=conversions_obs.max() * 0.5)
            k = pm.HalfNormal("k", sigma=1)
            x0 = pm.Normal("x0", mu=np.median(spend_obs), sigma=np.std(spend_obs) * 2)
            sigma = pm.HalfNormal("sigma", sigma=np.std(conversions_obs))
            mu = L * pm.math.sigmoid(k * (spend_obs - x0))
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=conversions_obs)
            trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, cores=1)

        # Process results
        current_avg_spend = spend_obs.mean()
        potential_spend = current_avg_spend * (1 + percentage_increase / 100)
        
        budget_range = np.linspace(1, max(spend_obs.max(), potential_spend) * 1.2, 200)
        post = az.extract(trace, var_names=["L", "k", "x0", "sigma"])
        mu_curves = post["L"].values[:, np.newaxis] * expit(post["k"].values[:, np.newaxis] * (budget_range - post["x0"].values[:, np.newaxis]))
        mean_predictions = mu_curves.mean(axis=0)
        modeled_cpa = budget_range / mean_predictions
        
        def predict_conversions(spend_value, trace):
            L_samples = trace.posterior['L'].values.flatten()
            k_samples = trace.posterior['k'].values.flatten()
            x0_samples = trace.posterior['x0'].values.flatten()
            return (L_samples * expit(k_samples * (spend_value - x0_samples))).mean()

        current_avg_conv = predict_conversions(current_avg_spend, trace)
        potential_conv = predict_conversions(potential_spend, trace)
        
        current_cpa = current_avg_spend / current_avg_conv
        potential_cpa = potential_spend / potential_conv

        # --- Create Graph ---
        plt.rcParams['font.family'] = 'Roboto'
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax1.plot(budget_range, mean_predictions, c='royalblue', lw=3, label='Potential Conversion Growth Curve')
        ax1.plot(current_avg_spend, current_avg_conv, 'o', color='darkorange', markersize=10, zorder=5, label='Current Average Position')
        ax1.plot(potential_spend, potential_conv, '*', color='seagreen', markersize=15, zorder=5, label=f'Forecast at +{percentage_increase}% Spend')
        
        # Add dotted lines for the forecast point
        ax1.vlines(x=potential_spend, ymin=0, ymax=potential_conv, color='seagreen', linestyle='--', alpha=0.7)
        ax1.hlines(y=potential_conv, xmin=0, xmax=potential_spend, color='seagreen', linestyle='--', alpha=0.7)

        ax1.set_xlabel(f'Daily Spend ({spend_column})', fontsize=12)
        ax1.set_ylabel('Predicted Daily Conversions', fontsize=12, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_ylim(0, max(mean_predictions.max(), potential_conv) * 1.1)
        ax1.set_xlim(0, budget_range.max())

        ax2 = ax1.twinx()
        ax2.plot(budget_range, modeled_cpa, c='gray', linestyle=':', lw=2, label='Modeled CPA')
        ax2.plot(current_avg_spend, current_cpa, 'o', color='darkorange', markersize=10, zorder=5)
        ax2.plot(potential_spend, potential_cpa, '*', color='seagreen', markersize=15, zorder=5)
        ax2.set_ylabel('Cost Per Acquisition (CPA)', fontsize=12, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, modeled_cpa[~np.isinf(modeled_cpa)].max() * 1.2)

        ax1.set_title(f'Spend vs. Conversions Forecast', fontsize=16, fontweight='bold')
        fig.tight_layout()
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
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

st.title('Campaign Spend vs. Conversion Forecaster')

uploaded_file = st.file_uploader("Upload your campaign performance CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        
        advertiser_id_column = 'Advertiser ID'
        objective_column = 'Objective Type'
        spend_column = 'Cost (USD)'
        conversions_column = 'Conversions'

        if advertiser_id_column not in data.columns or objective_column not in data.columns:
            st.error(f"Error: CSV must contain '{advertiser_id_column}' and '{objective_column}' columns.")
        else:
            advertiser_id_input = st.text_input("Enter Advertiser ID to analyze")

            if advertiser_id_input:
                data[advertiser_id_column] = data[advertiser_id_column].astype(str)
                advertiser_data = data[data[advertiser_id_column] == advertiser_id_input]

                if advertiser_data.empty:
                    st.warning("No data found for the specified Advertiser ID. Please check the ID and try again.")
                else:
                    objectives = advertiser_data[objective_column].dropna().unique().tolist()
                    if not objectives:
                        st.warning("No objectives found for this Advertiser ID.")
                    else:
                        selected_objective = st.selectbox("Choose an Objective to Analyze", options=objectives)
                        
                        # 4. Percentage Increase Input
                        percentage_increase = st.number_input("Enter Percentage Increase (%)", min_value=0, max_value=200, value=20, step=5)

                        # 5. Run Analysis Button
                        if st.button("Generate Forecast"):
                            analysis_df = advertiser_data[advertiser_data[objective_column] == selected_objective]
                            
                            with st.spinner('Running Bayesian model... This may take a few minutes.'):
                                fig, image_bytes, summary_res = create_forecast_and_outputs(
                                    _df=analysis_df,
                                    spend_column=spend_column,
                                    conversions_column=conversions_column,
                                    percentage_increase=percentage_increase
                                )

                            # 6. Display Outputs
                            if fig:
                                st.markdown("---")
                                st.markdown(f"## Expected Conversions: {summary_res['potential_conv']:.0f} (vs. {summary_res['current_conv']:.0f} currently)")
                                st.markdown(f"## Expected CPA: ${summary_res['potential_cpa']:.2f} (vs. ${summary_res['current_cpa']:.2f} currently)")
                                st.markdown("---")
                                
                                st.pyplot(fig)
                                
                                st.download_button(
                                    label="Download Graph as PNG",
                                    data=image_bytes,
                                    file_name=f'forecast_{advertiser_id_input}_{selected_objective}.png',
                                    mime='image/png'
                                )

                                # Analysis Bullet Points
                                st.markdown("---")
                                st.subheader("Analysis")
                                st.markdown(
                                    f" • **Current State**: The campaigns are currently averaging **${summary_res['current_spend']:,.2f}** in daily spend, "
                                    f"resulting in approximately **{summary_res['current_conv']:.0f}** conversions at a CPA of **${summary_res['current_cpa']:.2f}**."
                                )
                                st.markdown(
                                    f" • **Growth Opportunity**: Increasing the daily budget by **{summary_res['percentage_increase']}%** to **${summary_res['potential_spend']:,.2f}** "
                                    f"is forecasted to yield approximately **{summary_res['potential_conv']:.0f}** conversions. The model predicts this increase would be cost-effective, "
                                    f"with an expected CPA of **${summary_res['potential_cpa']:.2f}**."
                                )

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
