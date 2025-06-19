import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from io import StringIO

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
    st.sidebar.subheader("CSV Column Names")
    # Infer columns for selection - this requires reading the CSV first
    try:
        # Read just the header or a few rows to get column names
        df_preview = pd.read_csv(uploaded_file, nrows=5)
        uploaded_file.seek(0) # Reset file pointer after reading

        available_columns = df_preview.columns.tolist()

        date_column = st.sidebar.selectbox("Select Date/Time Column", available_columns, index=0 if available_columns else -1, key="date_col")
        value_column = st.sidebar.selectbox("Select Value Column", available_columns, index=1 if len(available_columns) > 1 else -1, key="value_col")

        st.sidebar.subheader("ARIMA Parameters")
        # Basic model parameters - auto_arima will find optimal ones
        m_seasonality = st.sidebar.number_input("Seasonality Period (m)", min_value=1, value=12, help="e.g., 12 for monthly, 4 for quarterly, 1 for non-seasonal", key="m_seas")
        forecast_steps = st.sidebar.number_input("Number of Steps to Forecast", min_value=1, value=12, key="f_steps")

        run_analysis = st.sidebar.button("🚀 Run Analysis & Forecast", key="run_button")

    except Exception as e:
        st.error(f"Error reading CSV header: {e}")
        st.stop()

else:
    st.sidebar.info("Awaiting CSV file upload.")
    run_analysis = False # Disable button if no file

# --- Main Area for Results ---
if run_analysis and uploaded_file is not None:
    st.header("📊 Analysis Results")

    # Temporary file handling
    temp_file_path = os.path.join("temp_uploaded_data.csv")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ts = arima.load_data(temp_file_path, date_column, value_column)

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    if ts is None:
        st.error("Failed to load or process time series data. Check column names and data format.")
        st.stop()

    st.subheader("Original Time Series Data")
    st.line_chart(ts, use_container_width=True)

    with st.spinner("Performing time series analysis... This may take a moment."):
        # 1. Check Stationarity
        st.subheader("Stationarity Check (ADF Test)")
        with st.expander("View ADF Test Results for Original Series"):
            adf_output_buffer = StringIO()
            old_stdout_adf = sys.stdout
            sys.stdout = adf_output_buffer
            p_value = arima.check_stationarity(ts)
            sys.stdout = old_stdout_adf # Reset stdout
            st.text(adf_output_buffer.getvalue())
            if p_value is not None:
                if p_value <= 0.05:
                    st.success(f"Series is likely stationary (p-value: {p_value:.4f}).")
                else:
                    st.warning(f"Series is likely non-stationary (p-value: {p_value:.4f}). `auto_arima` will attempt to find 'd'.")
            else:
                st.error("Could not perform stationarity test.")

        # 2. Find Optimal Parameters
        st.subheader("Optimal ARIMA Parameters (via auto_arima)")
        optimal_order = None
        optimal_seasonal_order = None
        with st.expander("View auto_arima Process Details", expanded=False):
            auto_arima_output_buffer = StringIO()
            old_stdout_aa = sys.stdout
            sys.stdout = auto_arima_output_buffer
            try:
                optimal_order, optimal_seasonal_order = arima.find_optimal_parameters(
                    ts,
                    m=m_seasonality,
                    seasonal=True if m_seasonality > 1 else False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
            except Exception as e_aa:
                 st.error(f"Error during auto_arima execution: {e_aa}")
            finally:
                sys.stdout = old_stdout_aa # Reset stdout
            st.text(auto_arima_output_buffer.getvalue())

        if not optimal_order:
            st.error("Could not determine optimal ARIMA parameters. The process may have failed or data might be unsuitable.")
            st.stop()

        st.success(f"Optimal ARIMA order (p,d,q): {optimal_order}")
        if optimal_seasonal_order and m_seasonality > 1:
            st.success(f"Optimal Seasonal order (P,D,Q,m): {optimal_seasonal_order}")
        else:
            st.info("Non-seasonal model or m=1, so no seasonal order was determined or applied.")

        # 3. Split data - Using full series for training as per simplified app logic
        train_ts = ts

        # 4. Fit ARIMA Model
        st.subheader("ARIMA Model Fitting")
        fitted_model = None
        with st.spinner("Fitting ARIMA model..."):
            model_summary_buffer = StringIO()
            old_stdout_fit = sys.stdout
            sys.stdout = model_summary_buffer
            try:
                fitted_model = arima.fit_arima_model(
                    train_ts,
                    optimal_order,
                    seasonal_order=optimal_seasonal_order if m_seasonality > 1 else None
                )
            except Exception as e_fit:
                st.error(f"Error during model fitting: {e_fit}")
            finally:
                sys.stdout = old_stdout_fit # Reset stdout

        if fitted_model is None:
            st.error("Failed to fit the ARIMA model.")
            st.text(model_summary_buffer.getvalue())
            st.stop()

        st.success("ARIMA model fitted successfully!")
        with st.expander("View Model Summary"):
            st.text(model_summary_buffer.getvalue())

        # 5. Make Forecasts
        st.subheader(f"Forecast for Next {forecast_steps} Steps")
        forecast_df = None
        with st.spinner("Generating forecasts..."):
            try:
                forecast_df = arima.forecast(fitted_model, steps=forecast_steps)
            except Exception as e_fc:
                st.error(f"Error during forecasting: {e_fc}")

        if forecast_df is None:
            st.error("Failed to generate forecasts.")
            st.stop()

        st.success("Forecasts generated successfully!")
        st.dataframe(forecast_df)

        # 6. Plot results
        st.subheader("Forecast Visualization")
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot original data
            ax.plot(ts.index.to_timestamp() if isinstance(ts.index, pd.PeriodIndex) else ts.index, ts.values, label='Original Data')

            # Plot forecast
            ax.plot(forecast_df.index.to_timestamp() if isinstance(forecast_df.index, pd.PeriodIndex) else forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')

            if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
                ax.fill_between(forecast_df.index.to_timestamp() if isinstance(forecast_df.index, pd.PeriodIndex) else forecast_df.index,
                                forecast_df['lower_ci'], forecast_df['upper_ci'],
                                color='pink', alpha=0.3, label='Confidence Interval (95%)')

            ax.legend()
            ax.set_title('Time Series Forecast')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e_plot:
            st.error(f"Error plotting results: {e_plot}")

else:
    if not uploaded_file:
        st.info("Upload a CSV file and configure parameters in the sidebar to begin.")
    elif uploaded_file and not run_analysis:
        st.info("Click 'Run Analysis & Forecast' in the sidebar once ready.")

st.markdown("---")
st.markdown("Developed by an AI assistant.")
