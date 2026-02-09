import streamlit as st
import pandas as pd
import sys
import os
import tempfile

# Add src directory to Python path to import arima_analyzer
# This assumes app.py is in arima_project/app/ and arima_analyzer.py is in arima_project/src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import arima_analyzer as arima

# --- Streamlit App Configuration ---
st.set_page_config(page_title="ARIMA Time Series Forecaster", layout="wide")

# --- Helper Functions (if any, or directly use arima_analyzer) ---

# --- Main Application ---
st.title("📈 ARIMA Time Series Forecaster")

st.markdown("""
This application allows you to upload your time series data (CSV),
perform ARIMA analysis, and visualize the forecasts.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    max_upload_mb = 10
    max_upload_bytes = max_upload_mb * 1024 * 1024
    if uploaded_file.size > max_upload_bytes:
        st.error(f"Uploaded file exceeds {max_upload_mb} MB limit. Please upload a smaller file.")
        st.stop()
    st.sidebar.subheader("CSV Column Names")
    # Infer columns for selection - this requires reading the CSV first
    try:
        # Read just the header or a few rows to get column names
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0) # Reset file pointer after reading

        available_columns = df_preview.columns.tolist()

        date_column = st.sidebar.selectbox("Select Date/Time Column", available_columns, index=0 if available_columns else -1)
        value_column = st.sidebar.selectbox("Select Value Column", available_columns, index=1 if len(available_columns) > 1 else -1)

        st.sidebar.subheader("ARIMA Parameters")
        # Basic model parameters - auto_arima will find optimal ones
        m_seasonality = st.sidebar.number_input("Seasonality Period (m)", min_value=1, value=12, help="e.g., 12 for monthly, 4 for quarterly, 1 for non-seasonal")
        forecast_steps = st.sidebar.number_input("Number of Steps to Forecast", min_value=1, value=12)

        run_analysis = st.sidebar.button("🚀 Run Analysis & Forecast")

    except Exception as e:
        st.error(f"Error reading CSV header: {e}")
        st.stop()

else:
    st.sidebar.info("Awaiting CSV file upload.")
    run_analysis = False # Disable button if no file

# --- Main Area for Results ---
if run_analysis and uploaded_file is not None:
    st.header("📊 Analysis Results")

    with st.spinner("Loading and processing data..."):
        # Load data using the arima_analyzer module
        # We need to save the uploaded file temporarily or pass its buffer
        # For simplicity, let's save it temporarily.
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name

            ts = arima.load_data(temp_file_path, date_column, value_column)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path) # Clean up temp file

    if ts is None:
        st.error("Failed to load or process time series data. Check column names and data format.")
        st.stop()

    st.subheader("Original Time Series Data")
    st.line_chart(ts, use_container_width=True)

    # Further steps (stationarity, model fitting, forecast) will be added here
    st.info("Further analysis steps (stationarity check, model fitting, forecasting) will be implemented here.")

    # Placeholder for plot
    # fig, ax = plt.subplots()
    # ax.plot(ts.index, ts.values)
    # st.pyplot(fig)

else:
    if not uploaded_file:
        st.info("Upload a CSV file and configure parameters in the sidebar to begin.")
    # If run_analysis is False but file is uploaded, it means button not clicked yet
    elif uploaded_file and not run_analysis:
        st.info("Click 'Run Analysis & Forecast' in the sidebar once ready.")

st.markdown("---")
st.markdown("Developed by an AI assistant.")
