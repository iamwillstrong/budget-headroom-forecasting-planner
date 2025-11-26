import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# --- CONFIGURATION ---
st.set_page_config(page_title="TikTok Bidding Headroom Predictor", layout="wide")

# --- HELPER FUNCTIONS ---

def saturate_hill(spend, max_conversions, k):
    """
    Hill Function for Media Mix Modeling.
    """
    # FIX: Added 1e-10 (epsilon) to denominator to prevent division by zero 
    # without breaking array operations.
    return (max_conversions * spend) / (k + spend + 1e-10)

def format_currency(value):
    return f"${value:,.2f}"

# --- APP LAYOUT ---

st.title("📉 TikTok Campaign Headroom & Saturation Model")
st.markdown("""
**Goal:** Visualize the point of diminishing returns (Bidding Headroom).
**Instructions:** Upload your CSV with the standard schema (`p_date`, `Ad Group ID`, `Cost (USD)`, `Conversions`).
""")

# 1. SIDEBAR: DATA INPUT & CONTROLS
with st.sidebar:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload CSV Report", type=['csv'])
    
    st.markdown("---")
    st.header("2. Simulation Controls")
    
    # Slider: 0% to 300% 
    budget_increase_pct = st.slider(
        "Proposed Budget/Bid Increase (%)", 
        min_value=0, 
        max_value=300, 
        value=20, 
        step=5
    ) / 100.0

# 2. MAIN LOGIC
if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        
        # --- DATA MAPPING ---
        col_map = {
            'p_date': 'Date',
            'Cost (USD)': 'Spend', 
            'Conversions': 'Conversions', 
            'CPM': 'CPM',
            'Ad Group ID': 'Ad_Group_ID'
        }
        
        # Check if columns exist before renaming
        missing_cols = [key for key in col_map.keys() if key not in df.columns]
        if missing_cols:
            st.error(f"CSV is missing the following columns: {missing_cols}")
            st.stop()
            
        df = df.rename(columns=col_map)
        
        # Parse Dates
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        
        # --- FILTERING LOGIC ---
        
        # Get unique Ad Groups for dropdown, sorted
        unique_ad_groups = sorted(df['Ad_Group_ID'].unique().astype(str))
        options = ["All Data"] + unique_ad_groups
        
        # Dropdown Menu
        selected_options = st.multiselect(
            "Select Ad Groups to Analyze",
            options=options,
            default="All Data"
        )
        
        if not selected_options:
            st.warning("Please select at least one Ad Group or 'All Data'.")
            st.stop()

        # Filter Data
        if "All Data" in selected_options:
            filtered_df = df.copy()
            selection_label = "All Account Data"
        else:
            filtered_df = df[df['Ad_Group_ID'].astype(str).isin(selected_options)]
            selection_label = f"Selection ({len(selected_options)} Ad Groups)"

        # --- AGGREGATION ---
        daily_df = filtered_df.groupby('Date').agg({
            'Spend': 'sum',
            'Conversions': 'sum',
            'CPM': 'mean' 
        }).reset_index()

        # Remove zero spend days
        model_df = daily_df[daily_df['Spend'] > 0]
        
        if len(model_df) < 5:
            st.error("Not enough data points (days with spend) to generate a predictive curve. You need at least 5 days of data.")
            st.stop()

        # --- 3. THE MATH (CURVE FITTING) ---
        x_data = model_df['Spend']
        y_data = model_df['Conversions']

        # Initial parameter guesses
        p0 = [y_data.max() * 2, x_data.mean()]
        
        try:
            # Fit the Hill function
            popt, pcov = curve_fit(saturate_hill, x_data, y_data, p0=p0, maxfev=10000)
            max_conv_model, k_model = popt
        except Exception as e:
            st.warning("Could not fit a perfect curve (Linear data?). Showing best approximation.")
            max_conv_model = y_data.max() * 5
            k_model = x_data.max() * 5

        # --- 4. CALCULATE SCENARIOS ---
        
        # Current State
        current_avg_spend = x_data.mean()
        current_est_conv = saturate_hill(current_avg_spend, max_conv_model, k_model)
        # Avoid zero division in display math
        current_cpa = current_avg_spend / current_est_conv if current_est_conv > 1e-9 else 0
        
        # Future State
        new_spend = current_avg_spend * (1 + budget_increase_pct)
        new_est_conv = saturate_hill(new_spend, max_conv_model, k_model)
        new_cpa = new_spend / new_est_conv if new_est_conv > 1e-9 else 0
        
        # Marginal Metrics
        delta_spend = new_spend - current_avg_spend
        delta_conv = new_est_conv - current_est_conv
        marginal_cpa = delta_spend / delta_conv if delta_conv > 1e-9 else 0

        # --- 5. VISUALIZATION ---
        
        # Generate curve points
        x_range = np.linspace(0, x_data.max() * 2.5, 100) 
        y_range = saturate_hill(x_range, max_conv_model, k_model)

        fig = go.Figure()

        # Scatter: Actual Daily Data
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, 
            mode='markers', name='Daily Performance',
            marker=dict(color='gray', opacity=0.4, size=8)
        ))

        # Line: The Model Curve
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range, 
            mode='lines', name='Saturation Curve',
            line=dict(color='#ff0050', width=3)
        ))

        # Point: Current State
        fig.add_trace(go.Scatter(
            x=[current_avg_spend], y=[current_est_conv],
            mode='markers+text', name='Current Avg',
            text=['Current'], textposition="bottom right",
            marker=dict(color='blue', size=12, symbol='diamond')
        ))

        # Point: Future State
        fig.add_trace(go.Scatter(
            x=[new_spend], y=[new_est_conv],
            mode='markers+text', name='Projected',
            text=[f'+{int(budget_increase_pct*100)}% Spend'], textposition="top left",
            marker=dict(color='#00f2ea', size=14, symbol='star')
        ))

        fig.update_layout(
            title=f"Headroom Model: {selection_label}",
            xaxis_title="Daily Spend (USD)",
            yaxis_title="Daily Conversions",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- 6. REPORT ---
        
        # Color coding logic
        if marginal_cpa > (current_cpa * 1.5):
            status_color = "🔴" # Red
            status_msg = "High Saturation (Diminishing Returns)"
        elif marginal_cpa > (current_cpa * 1.15):
            status_color = "🟡" # Yellow
            status_msg = "Moderate Headroom"
        else:
            status_color = "🟢" # Green
            status_msg = "High Headroom (Scalable)"

        st.subheader("Predictive Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Avg CPA", format_currency(current_cpa))
        with col2:
            st.metric("Projected CPA", format_currency(new_cpa), delta=f"{((new_cpa-current_cpa)/(current_cpa+1e-9))*100:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Marginal CPA (Cost of New Conv)", format_currency(marginal_cpa))

        st.info(f"""
        **Status: {status_color} {status_msg}**
        
        Current average spend for this selection is **{format_currency(current_avg_spend)}**, yielding ~**{int(current_est_conv)} conversions**.
        
        By increasing budget by **{int(budget_increase_pct*100)}%** to **{format_currency(new_spend)}**:
        * You can expect to generate **{int(delta_conv)} additional conversions**.
        * These specific extra conversions will cost **{format_currency(marginal_cpa)}** each (Marginal CPA).
        """)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a CSV to begin.")
