import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from fpdf import FPDF
import tempfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart+ projected headroom", layout="wide")

# --- HELPER FUNCTIONS ---

def saturate_hill(spend, max_conversions, k):
    """
    Hill Function for Media Mix Modeling.
    """
    # Added epsilon to prevent division by zero
    return (max_conversions * spend) / (k + spend + 1e-10)

def format_currency(value):
    return f"${value:,.2f}"

def create_pdf_report(fig, report_data):
    """
    Generates a PDF report containing the chart and the analysis text.
    """
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Smart+ Headroom Report', 0, 1, 'C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    # 1. Save Plotly Figure as Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        try:
            # Explicit width/height to prevent cropping/scaling issues in PDF
            fig.write_image(tmpfile.name, scale=2, width=1200, height=600)
            # Margins
            pdf.image(tmpfile.name, x=10, y=30, w=190)
        except Exception as e:
            pdf.cell(0, 10, "Error rendering chart image. Ensure 'kaleido' is installed.", 0, 1)
        
        tmp_path = tmpfile.name

    # 2. Add Analysis Text
    pdf.set_y(130) 
    
    # Section: Current State
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Current State", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Selected Scope: {report_data['scope']}\n"
                             f"Current Daily Spend: {report_data['curr_spend']}\n"
                             f"Current Daily Conversions: {report_data['curr_conv']}\n"
                             f"Current CPA: {report_data['curr_cpa']}")
    pdf.ln(5)

    # Section: Projection
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Estimated Scenario", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Proposed Budget Increase: {report_data['pct_inc']}\n"
                             f"Est. New Daily Spend: {report_data['new_spend']}\n"
                             f"Est. Conversions: {report_data['new_conv']} (+{report_data['delta_conv']})\n"
                             f"Est. CPA: {report_data['new_cpa']}\n")
    pdf.ln(5)

    # Section: Marginal Efficiency
    pdf.set_font("Arial", 'B', 12)
    
    # Dynamic text color based on status
    if "High Saturation" in report_data['status']:
        pdf.set_text_color(255, 0, 0) # Red
    elif "Moderate" in report_data['status']:
        pdf.set_text_color(200, 150, 0) # Dark Gold
    else:
        pdf.set_text_color(0, 128, 0) # Green

    pdf.cell(0, 10, "Marginal Efficiency Status", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset color
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Est. Marginal CPA (Cost of extra results): {report_data['marg_cpa']}\n"
                             f"Status: {report_data['status']}")
    
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: All values are estimates based on historical data regression. Actual performance may vary due to auction dynamics and creative fatigue. These forecasts are not guaranteed.")

    try:
        os.remove(tmp_path)
    except:
        pass

    return pdf.output(dest='S').encode('latin-1')

# --- APP LAYOUT ---

st.title("Smart+ projected headroom")
st.markdown("""
**Instructions:** Upload your CSV with the required headers: 
`p_date`, `Ad Group Name`, `Objective Type`, `Is Catalog Ads`, `Cost (USD)`, `CPM`, `Conversions`
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
            'Ad Group Name': 'Ad_Group_Name',
            'Objective Type': 'Objective_Type',
            'Is Catalog Ads': 'Is_Catalog_Ads',
            'Cost (USD)': 'Spend', 
            'CPM': 'CPM',
            'Conversions': 'Conversions'
        }
        
        missing_cols = [key for key in col_map.keys() if key not in df.columns]
        if missing_cols:
            st.error(f"CSV is missing the following columns: {missing_cols}")
            st.stop()
            
        df = df.rename(columns=col_map)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        
        # --- FILTERING LOGIC ---
        
        # 1. AD GROUP SELECTION
        unique_ad_groups = sorted(df['Ad_Group_Name'].unique().astype(str))
        options = ["All Data"] + unique_ad_groups
        
        selected_ad_groups = st.multiselect(
            "Select Ad Groups to Analyze",
            options=options,
            default="All Data"
        )
        
        # 2. CATALOG TYPE SELECTION
        catalog_options = ["All", "S+ non catalog", "S+ catalog"]
        selected_catalog_type = st.selectbox(
            "Select Catalog Type",
            options=catalog_options
        )
        
        if not selected_ad_groups:
            st.warning("Please select at least one Ad Group.")
            st.stop()

        # Apply Ad Group Filter
        if "All Data" in selected_ad_groups:
            temp_df = df.copy()
            selection_label = "All Data"
        else:
            temp_df = df[df['Ad_Group_Name'].astype(str).isin(selected_ad_groups)]
            selection_label = f"Selection ({len(selected_ad_groups)} Ad Groups)"

        # Apply Catalog Filter
        if selected_catalog_type == "S+ catalog":
            filtered_df = temp_df[temp_df['Is_Catalog_Ads'].astype(str).str.strip().str.upper() == 'Y']
            selection_label += " | Catalog"
        elif selected_catalog_type == "S+ non catalog":
            filtered_df = temp_df[temp_df['Is_Catalog_Ads'].astype(str).str.strip().str.upper() == 'N']
            selection_label += " | Non-Catalog"
        else:
            filtered_df = temp_df

        # Check if data exists after filtering
        if filtered_df.empty:
            st.warning("No data matches the selected filters (Ad Group + Catalog Type).")
            st.stop()

        # --- AGGREGATION ---
        daily_df = filtered_df.groupby('Date').agg({
            'Spend': 'sum',
            'Conversions': 'sum',
            'CPM': 'mean' 
        }).reset_index()

        model_df = daily_df[daily_df['Spend'] > 0]
        
        if len(model_df) < 5:
            st.error("Not enough data points (days with spend) to generate a predictive curve.")
            st.stop()

        # --- 3. THE MATH ---
        x_data = model_df['Spend']
        y_data = model_df['Conversions']

        p0 = [y_data.max() * 2, x_data.mean()]
        
        try:
            popt, pcov = curve_fit(saturate_hill, x_data, y_data, p0=p0, maxfev=10000)
            max_conv_model, k_model = popt
        except:
            # Fallback (usually indicates linear data)
            max_conv_model = y_data.max() * 5
            k_model = x_data.max() * 5

        # --- SAFETY CHECK: IS THE DATA SUFFICIENT? ---
        # If the curve is effectively flat (linear), it means we haven't seen the saturation point.
        # We test this by checking the marginal CPA at a theoretical 300% spend increase.
        # If Marginal CPA at +300% is nearly identical to Current CPA, the model is guessing linear scale.
        
        test_spend_limit = x_data.mean() * 4.0 # 300% increase
        test_conv_limit = saturate_hill(test_spend_limit, max_conv_model, k_model)
        
        curr_test_spend = x_data.mean()
        curr_test_conv = saturate_hill(curr_test_spend, max_conv_model, k_model)
        
        # Calculate marginal CPA at the extreme limit
        test_marginal_cpa = (test_spend_limit - curr_test_spend) / (test_conv_limit - curr_test_conv)
        test_current_cpa = curr_test_spend / curr_test_conv if curr_test_conv > 0 else 0
        
        # If Marginal CPA at +300% spend is less than 1.1x current CPA, the curve is too flat.
        if test_marginal_cpa < (test_current_cpa * 1.1):
            st.error("⚠️ Insufficient Data for Rigorous Analysis")
            st.markdown("""
            **The provided data shows a linear relationship with no signs of diminishing returns.**
            
            This implies that the model cannot accurately predict a saturation point (headroom) because the ad groups have not yet been pushed hard enough to show efficiency drops. 
            
            **Action:** We cannot recommend a specific budget cap based on this file alone, as it would likely suggest unrealistic scaling (e.g., +300% with no CPA increase).
            """)
            
            # Show the raw data graph anyway so they can see the linearity
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers', name='Daily Performance'))
            fig.update_layout(title="Raw Data (Linear Trend Detected)", xaxis_title="Spend", yaxis_title="Conversions")
            st.plotly_chart(fig)
            st.stop()

        # --- 4. SCENARIOS ---
        current_avg_spend = x_data.mean()
        current_est_conv = saturate_hill(current_avg_spend, max_conv_model, k_model)
        current_cpa = current_avg_spend / current_est_conv if current_est_conv > 1e-9 else 0
        
        new_spend = current_avg_spend * (1 + budget_increase_pct)
        new_est_conv = saturate_hill(new_spend, max_conv_model, k_model)
        new_cpa = new_spend / new_est_conv if new_est_conv > 1e-9 else 0
        
        delta_spend = new_spend - current_avg_spend
        delta_conv = new_est_conv - current_est_conv
        marginal_cpa = delta_spend / delta_conv if delta_conv > 1e-9 else 0

        # --- 5. VISUALIZATION ---
        x_range = np.linspace(0, x_data.max() * 2.5, 100) 
        y_range = saturate_hill(x_range, max_conv_model, k_model)

        fig = go.Figure()

        # 1. Actual Data
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data, 
            mode='markers', name='Daily Performance',
            marker=dict(color='gray', opacity=0.3, size=7)
        ))

        # 2. The Curve (Legend Hidden)
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range, 
            mode='lines', name='Saturation Curve',
            showlegend=False,
            line=dict(color='#ff0050', width=3)
        ))

        # 3. Current State Point
        fig.add_trace(go.Scatter(
            x=[current_avg_spend], y=[current_est_conv],
            mode='markers+text', name='Current Avg',
            text=['Current'], textposition="top left",
            marker=dict(color='blue', size=12, symbol='diamond')
        ))

        # 4. Projected State Point
        fig.add_trace(go.Scatter(
            x=[new_spend], y=[new_est_conv],
            mode='markers+text', name='Projected',
            text=['Est. Future'], textposition="top left", # Changed text
            marker=dict(color='#00f2ea', size=14, symbol='star')
        ))

        # 5. DOTTED DROP LINES
        fig.add_shape(type="line",
            x0=current_avg_spend, y0=0, x1=current_avg_spend, y1=current_est_conv,
            line=dict(color="blue", width=1, dash="dot")
        )
        fig.add_shape(type="line",
            x0=0, y0=current_est_conv, x1=current_avg_spend, y1=current_est_conv,
            line=dict(color="blue", width=1, dash="dot")
        )
        fig.add_shape(type="line",
            x0=new_spend, y0=0, x1=new_spend, y1=new_est_conv,
            line=dict(color="#00f2ea", width=1, dash="dot")
        )
        fig.add_shape(type="line",
            x0=0, y0=new_est_conv, x1=new_spend, y1=new_est_conv,
            line=dict(color="#00f2ea", width=1, dash="dot")
        )

        fig.update_layout(
            title=f"Smart+ Headroom: {selection_label}",
            xaxis_title="Daily Spend (USD)",
            yaxis_title="Daily Conversions",
            template="plotly_white",
            height=500,
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- 6. REPORT & STATUS LOGIC ---
        
        # UPDATED STATUS LOGIC
        if marginal_cpa > (current_cpa * 2.0):
            status_color_icon = "🔴"
            status_msg = "Diminishing Returns (High Saturation)"
            delta_color_hex = "#ff2b2b" # Red
        elif marginal_cpa > (current_cpa * 1.5):
            status_color_icon = "🟡"
            status_msg = "Moderate Headroom"
            delta_color_hex = "#e6b800" # Dark Yellow
        else:
            status_color_icon = "🟢"
            status_msg = "Headroom (Scalable)"
            delta_color_hex = "#09ab3b" # Green

        st.subheader("Predictive Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # 1. Current CPA
        with col1:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Current Avg CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">{format_currency(current_cpa)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Projected CPA
        with col2:
            delta_val = ((new_cpa - current_cpa) / (current_cpa + 1e-9)) * 100
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: rgba(240, 242, 246, 0.5);">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Est. New CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">
                    {format_currency(new_cpa)}
                </p>
                <p style="font-size: 14px; margin: 0; color: {delta_color_hex}; font-weight: 600;">
                    ▲ {delta_val:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        # 3. Marginal CPA
        with col3:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Est. Marginal CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">{format_currency(marginal_cpa)}</p>
            </div>
            """, unsafe_allow_html=True)

        # UPDATED LANGUAGE & DISCLAIMER
        st.info(f"""
        **Status: {status_color_icon} {status_msg}**
        
        Current average spend for this selection is **{format_currency(current_avg_spend)}**, yielding ~**{int(current_est_conv)} conversions**.
        
        By increasing budget by **{int(budget_increase_pct*100)}%** to **{format_currency(new_spend)}**, the model estimates you **could** generate **{int(delta_conv)} additional conversions**.
        
        *Note: These specific extra conversions are estimated to cost **{format_currency(marginal_cpa)}** each (Marginal CPA).*
        
        ---
        *Disclaimer: All forecasts are estimates based on historical data regression. Actual performance may vary due to auction dynamics and creative fatigue. Results are not guaranteed.*
        """)

        # --- 7. PDF DOWNLOAD ---
        
        report_data = {
            "scope": selection_label,
            "curr_spend": format_currency(current_avg_spend),
            "curr_conv": str(int(current_est_conv)),
            "curr_cpa": format_currency(current_cpa),
            "pct_inc": f"{int(budget_increase_pct*100)}%",
            "new_spend": format_currency(new_spend),
            "new_conv": str(int(new_est_conv)),
            "delta_conv": str(int(delta_conv)),
            "new_cpa": format_currency(new_cpa),
            "marg_cpa": format_currency(marginal_cpa),
            "status": status_msg
        }

        st.write("---")
        st.write("### Export Report")
        
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf_bytes = create_pdf_report(fig, report_data)
                
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name="smart_plus_headroom_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a CSV to begin.")
