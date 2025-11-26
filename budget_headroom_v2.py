import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from fpdf import FPDF
import tempfile
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Ad Group Projected Headroom", layout="wide")

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
            self.cell(0, 10, 'Ad Group Headroom Report', 0, 1, 'C')
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    # 1. Save Plotly Figure as Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        try:
            fig.write_image(tmpfile.name, scale=2)
            pdf.image(tmpfile.name, x=10, y=30, w=190)
        except Exception as e:
            pdf.cell(0, 10, "Error rendering chart image. Ensure 'kaleido' is installed.", 0, 1)
        
        tmp_path = tmpfile.name

    # 2. Add Analysis Text
    pdf.set_y(140) 
    
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
    pdf.cell(0, 10, "Projected Scenario", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Proposed Budget Increase: {report_data['pct_inc']}\n"
                             f"New Daily Spend: {report_data['new_spend']}\n"
                             f"Projected Conversions: {report_data['new_conv']} (+{report_data['delta_conv']})\n"
                             f"Projected CPA: {report_data['new_cpa']}\n")
    pdf.ln(5)

    # Section: Marginal Efficiency
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 0, 0) # Red for emphasis
    pdf.cell(0, 10, "Marginal Efficiency (Diminishing Returns)", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset color
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=f"Marginal CPA (Cost of extra results): {report_data['marg_cpa']}\n"
                             f"Status: {report_data['status']}")

    try:
        os.remove(tmp_path)
    except:
        pass

    return pdf.output(dest='S').encode('latin-1')

# --- APP LAYOUT ---

st.title("Ad group projected headroom")
st.markdown("""
**Instructions:** Upload your CSV with the standard schema: `p_date`, `Ad Group Name`, `Cost (USD)`, `Conversions`.
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
            'Ad Group Name': 'Ad_Group_Name'
        }
        
        missing_cols = [key for key in col_map.keys() if key not in df.columns]
        if missing_cols:
            st.error(f"CSV is missing the following columns: {missing_cols}")
            st.stop()
            
        df = df.rename(columns=col_map)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
        
        # --- FILTERING LOGIC ---
        unique_ad_groups = sorted(df['Ad_Group_Name'].unique().astype(str))
        options = ["All Data"] + unique_ad_groups
        
        selected_options = st.multiselect(
            "Select Ad Groups to Analyze",
            options=options,
            default="All Data"
        )
        
        if not selected_options:
            st.warning("Please select at least one Ad Group.")
            st.stop()

        if "All Data" in selected_options:
            filtered_df = df.copy()
            selection_label = "All Account Data"
        else:
            filtered_df = df[df['Ad_Group_Name'].astype(str).isin(selected_options)]
            selection_label = f"Selection ({len(selected_options)} Ad Groups)"

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
            st.warning("Could not fit a perfect curve. Showing approximation.")
            max_conv_model = y_data.max() * 5
            k_model = x_data.max() * 5

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
            text=['Projected'], textposition="top left",
            marker=dict(color='#00f2ea', size=14, symbol='star')
        ))

        # 5. DOTTED DROP LINES
        # Line for Current Spend
        fig.add_shape(type="line",
            x0=current_avg_spend, y0=0, x1=current_avg_spend, y1=current_est_conv,
            line=dict(color="blue", width=1, dash="dot")
        )
        # Line for Current Conversions
        fig.add_shape(type="line",
            x0=0, y0=current_est_conv, x1=current_avg_spend, y1=current_est_conv,
            line=dict(color="blue", width=1, dash="dot")
        )
        # Line for Projected Spend
        fig.add_shape(type="line",
            x0=new_spend, y0=0, x1=new_spend, y1=new_est_conv,
            line=dict(color="#00f2ea", width=1, dash="dot")
        )
        # Line for Projected Conversions
        fig.add_shape(type="line",
            x0=0, y0=new_est_conv, x1=new_spend, y1=new_est_conv,
            line=dict(color="#00f2ea", width=1, dash="dot")
        )

        fig.update_layout(
            title=f"Ad Group Headroom: {selection_label}",
            xaxis_title="Daily Spend (USD)",
            yaxis_title="Daily Conversions",
            template="plotly_white",
            height=500,
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- 6. REPORT & STATUS LOGIC ---
        
        # Determine Color Codes
        # Using custom HEX to control the exact color of the 'chip' (delta)
        if marginal_cpa > (current_cpa * 2.0):
            status_color_icon = "🔴"
            status_msg = "Diminishing Returns (High Saturation)"
            delta_color_hex = "#ff2b2b" # Red
        elif marginal_cpa > (current_cpa * 1.5):
            status_color_icon = "🟡"
            status_msg = "Moderate Headroom"
            delta_color_hex = "#e6b800" # Dark Yellow/Gold (Readable on white)
        else:
            status_color_icon = "🟢"
            status_msg = "High Headroom (Scalable)"
            delta_color_hex = "#09ab3b" # Green

        st.subheader("Predictive Analysis")
        
        # Using HTML/CSS columns to enforce specific colors on the metrics
        col1, col2, col3 = st.columns(3)
        
        # 1. Current CPA (Standard HTML styling)
        with col1:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Current Avg CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">{format_currency(current_cpa)}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. Projected CPA (Custom HTML to match delta color to status)
        with col2:
            delta_val = ((new_cpa - current_cpa) / (current_cpa + 1e-9)) * 100
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: rgba(240, 242, 246, 0.5);">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Projected CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">
                    {format_currency(new_cpa)}
                </p>
                <p style="font-size: 14px; margin: 0; color: {delta_color_hex}; font-weight: 600;">
                    ▲ {delta_val:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

        # 3. Marginal CPA (Standard HTML styling)
        with col3:
            st.markdown(f"""
            <div style="padding: 10px;">
                <p style="font-size: 14px; color: #555; margin-bottom: 2px;">Marginal CPA</p>
                <p style="font-size: 26px; font-weight: bold; margin: 0;">{format_currency(marginal_cpa)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info(f"""
        **Status: {status_color_icon} {status_msg}**
        
        Current average spend for this selection is **{format_currency(current_avg_spend)}**, yielding ~**{int(current_est_conv)} conversions**.
        
        By increasing budget by **{int(budget_increase_pct*100)}%** to **{format_currency(new_spend)}**:
        * You can expect to generate **{int(delta_conv)} additional conversions**.
        * These specific extra conversions will cost **{format_currency(marginal_cpa)}** each (Marginal CPA).
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
                    file_name="tiktok_headroom_report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload a CSV to begin.")
