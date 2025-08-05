import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from io import StringIO

# To use the arima_analyzer module from the src package,
# ensure the package is installed (e.g., `pip install -e .` from the `arima_project` root)
from src import arima_analyzer as arima

# --- Streamlit App Configuration ---
st.set_page_config(page_title="ARIMA Time Series Forecaster", layout="wide")

# --- Main Application ---
st.title("📈 ARIMA Time Series Forecaster")

st.markdown("""
This application allows you to upload your time series data (CSV), perform ARIMA analysis,
and visualize the forecasts. You can choose to train the model on your entire dataset
to predict future values, or hold back a portion of your data as a test set to
evaluate the model's forecasting accuracy.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    # Use a session state to avoid re-reading the file on every interaction
    if 'df_preview' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        try:
            st.session_state.df_preview = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

    df_preview = st.session_state.df_preview
    available_columns = df_preview.columns.tolist()

    st.sidebar.subheader("CSV Column Names")
    date_column = st.sidebar.selectbox("Select Date/Time Column", available_columns, index=0, key="date_col")
    value_column = st.sidebar.selectbox("Select Value Column", available_columns, index=1 if len(available_columns) > 1 else 0, key="value_col")

    st.sidebar.subheader("ARIMA Parameters")
    m_seasonality = st.sidebar.number_input("Seasonality Period (m)", min_value=1, value=12, help="e.g., 12 for monthly, 4 for quarterly, 1 for non-seasonal.", key="m_seas")

    st.sidebar.subheader("Train/Test Split")
    # Make sure test_size max value is safe
    max_test_size = len(df_preview) - 2 if len(df_preview) > 1 else 0
    test_size = st.sidebar.slider(
        "Test Set Size (Number of recent data points)",
        min_value=0,
        max_value=max_test_size,
        value=min(12, max_test_size),
        help="Set to 0 to train on all data and forecast future values."
    )

    st.sidebar.subheader("Forecasting")
    forecast_steps = st.sidebar.number_input(
        "Number of Steps to Forecast",
        min_value=1,
        value=12,
        key="f_steps",
        help="If using a test set, this is ignored and the forecast matches the test set size. Otherwise, this is the number of future steps to predict."
    )

    run_analysis = st.sidebar.button("🚀 Run Analysis & Forecast", key="run_button")

else:
    st.sidebar.info("Awaiting CSV file upload.")
    run_analysis = False

# --- Main Area for Results ---
if run_analysis and uploaded_file is not None:
    st.header("📊 Analysis Results")

    # Use a temporary file to pass to the analyzer module
    temp_file_path = "temp_uploaded_data.csv"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ts = arima.load_data(temp_file_path, date_column, value_column)
    os.remove(temp_file_path)

    if ts is None:
        st.error("Failed to load or process time series data. Check column names and data format.")
        st.stop()

    st.subheader("Original Time Series Data")
    st.line_chart(ts, use_container_width=True)

    with st.spinner("Performing time series analysis... This may take a moment."):
        # 1. Split data
        if test_size > 0 and len(ts) > test_size:
            train_ts = ts[:-test_size]
            test_ts = ts[-test_size:]
            st.subheader("Train/Test Data Split")
            st.write(f"Training data points: {len(train_ts)}")
            st.write(f"Test data points: {len(test_ts)}")
        else:
            train_ts = ts
            test_ts = None
            st.subheader("Training on Full Dataset")
            st.write("No test set selected. The model will be trained on the entire dataset.")

        # 2. Check Stationarity (on training data)
        st.subheader("Stationarity Check (ADF Test on Training Data)")
        with st.expander("View ADF Test Results"):
            adf_output_buffer = StringIO()
            sys.stdout = adf_output_buffer
            p_value = arima.check_stationarity(train_ts)
            sys.stdout = sys.__stdout__ # Reset stdout
            st.text(adf_output_buffer.getvalue())
            if p_value is not None:
                st.success(f"Series is likely {'stationary' if p_value <= 0.05 else 'non-stationary'} (p-value: {p_value:.4f}).")
            else:
                st.error("Could not perform stationarity test.")

        # 3. Find Optimal Parameters (on training data)
        st.subheader("Optimal ARIMA Parameters (via auto_arima)")
        with st.expander("View auto_arima Process Details"):
            auto_arima_output_buffer = StringIO()
            sys.stdout = auto_arima_output_buffer
            optimal_order, optimal_seasonal_order = arima.find_optimal_parameters(
                train_ts, m=m_seasonality, seasonal=(m_seasonality > 1), trace=True,
                error_action='ignore', suppress_warnings=True, stepwise=True
            )
            sys.stdout = sys.__stdout__
            st.text(auto_arima_output_buffer.getvalue())

        if not optimal_order:
            st.error("Could not determine optimal ARIMA parameters.")
            st.stop()

        st.success(f"Optimal ARIMA(p,d,q): {optimal_order}")
        if optimal_seasonal_order:
            st.success(f"Optimal Seasonal(P,D,Q,m): {optimal_seasonal_order}")

        # 4. Fit ARIMA Model
        st.subheader("ARIMA Model Fitting")
        with st.spinner("Fitting ARIMA model..."):
            model_summary_buffer = StringIO()
            sys.stdout = model_summary_buffer
            fitted_model = arima.fit_arima_model(train_ts, optimal_order, optimal_seasonal_order)
            sys.stdout = sys.__stdout__

        if fitted_model is None:
            st.error("Failed to fit the ARIMA model.")
            st.text(model_summary_buffer.getvalue())
            st.stop()

        st.success("ARIMA model fitted successfully!")
        with st.expander("View Model Summary"):
            st.text(model_summary_buffer.getvalue())

        # 5. Make Forecasts
        forecast_steps_actual = len(test_ts) if test_ts is not None else forecast_steps
        st.subheader(f"Forecasting {forecast_steps_actual} Steps Ahead")

        with st.spinner("Generating forecasts..."):
            forecast_df = arima.forecast(fitted_model, steps=forecast_steps_actual)

        if forecast_df is None:
            st.error("Failed to generate forecasts.")
            st.stop()

        st.dataframe(forecast_df)

        # 6. Evaluate Model on Test Set
        if test_ts is not None:
            st.subheader("Model Evaluation on Test Set")
            with st.spinner("Evaluating model..."):
                evaluation = arima.evaluate_model(test_ts, forecast_df['forecast'])
                if evaluation:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("RMSE", f"{evaluation['rmse']:.4f}")
                    col2.metric("MAE", f"{evaluation['mae']:.4f}")
                    col3.metric("MAPE", f"{evaluation['mape']:.2%}")
                else:
                    st.error("Could not evaluate the model.")

        # 7. Plot results
        st.subheader("Forecast Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(train_ts.index, train_ts.values, label='Training Data')
        if test_ts is not None:
            ax.plot(test_ts.index, test_ts.values, label='Test Data (Actual)', color='orange')

        ax.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red', linestyle='--')
        if 'lower_ci' in forecast_df.columns and 'upper_ci' in forecast_df.columns:
            ax.fill_between(forecast_df.index, forecast_df['lower_ci'], forecast_df['upper_ci'], color='pink', alpha=0.3, label='Confidence Interval (95%)')

        ax.legend()
        ax.set_title('Time Series Forecast')
        ax.set_xlabel('Time')
        ax.set_ylabel(value_column)
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    if not uploaded_file:
        st.info("Upload a CSV file and configure parameters in the sidebar to begin.")
    elif uploaded_file and not run_analysis:
        st.info("Click 'Run Analysis & Forecast' in the sidebar once ready.")

st.markdown("---")
st.markdown("Developed by an AI assistant, enhanced by Jules.")
