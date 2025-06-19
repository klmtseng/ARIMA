import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pmdarima as pm

def load_data(filepath, date_column, value_column):
    """
    Loads time series data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        date_column (str): Name of the column containing dates.
        value_column (str): Name of the column containing time series values.

    Returns:
        pd.Series: Time series data with DatetimeIndex.
                   Returns None if file not found or columns are not present.
    """
    try:
        data = pd.read_csv(filepath, parse_dates=[date_column], index_col=date_column)
        ts = data[value_column]
        ts = ts.asfreq(pd.infer_freq(ts.index)) # Infer frequency
        return ts
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except KeyError:
        print(f"Error: One or both columns ('{date_column}', '{value_column}') not found in the CSV.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def check_stationarity(timeseries):
    """
    Checks stationarity using the Augmented Dickey-Fuller test.

    Args:
        timeseries (pd.Series): The time series to check.

    Returns:
        float: The p-value from the ADF test.
    """
    if timeseries is None or timeseries.empty:
        print("Error: Timeseries is empty or None.")
        return None

    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries.dropna(), autolag='AIC') # dropna() to handle missing values if any
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

    p_value = dftest[1]
    if p_value <= 0.05:
        print("Conclusion: Series is stationary (p-value <= 0.05)")
    else:
        print("Conclusion: Series is non-stationary (p-value > 0.05)")
    return p_value

def make_stationary(timeseries, d=None):
    """
    Makes the time series stationary through differencing.

    Args:
        timeseries (pd.Series): The time series to make stationary.
        d (int, optional): The order of differencing. If None, differences once.

    Returns:
        tuple: (differenced_series (pd.Series), number_of_differences (int))
               Returns (None, 0) if timeseries is None or empty.
    """
    if timeseries is None or timeseries.empty:
        print("Error: Timeseries is empty or None for differencing.")
        return None, 0

    if d is None:
        # Automatic differencing until stationary (simplified: try differencing once)
        # A more robust approach would involve checking stationarity after each difference
        differenced_series = timeseries.diff().dropna()
        num_diffs = 1
        # Check stationarity again (optional, or could be done in a loop)
        # print("Checking stationarity after 1st differencing:")
        # check_stationarity(differenced_series)
        return differenced_series, num_diffs
    elif isinstance(d, int) and d >= 0:
        if d == 0:
            return timeseries, 0
        current_series = timeseries.copy()
        for _ in range(d):
            current_series = current_series.diff().dropna()
        return current_series, d
    else:
        print("Error: d must be a non-negative integer.")
        return timeseries, 0 # Return original if d is invalid

def find_optimal_parameters(timeseries, start_p=1, start_q=1, max_p=3, max_q=3, m=1,
                             start_P=0, seasonal=True, d=None, D=None, trace=True,
                             error_action='ignore', suppress_warnings=True, stepwise=True):
    """
    Finds optimal ARIMA (p,d,q)(P,D,Q,m) parameters using pmdarima.auto_arima.

    Args:
        timeseries (pd.Series): The time series data.
        start_p (int): Starting value of p.
        start_q (int): Starting value of q.
        max_p (int): Maximum value of p.
        max_q (int): Maximum value of q.
        m (int): The period for seasonal differencing.
        start_P (int): Starting value of P (seasonal).
        seasonal (bool): Whether to fit a seasonal ARIMA.
        d (int, optional): Order of first-differencing. If None, will be determined.
        D (int, optional): Order of seasonal differencing. If None, will be determined.
        trace (bool): Whether to print status messages.
        error_action (str): How to handle errors during model fitting.
        suppress_warnings (bool): Whether to suppress warnings.
        stepwise (bool): Whether to use the stepwise algorithm.

    Returns:
        tuple: Optimal (p, d, q) order and (P, D, Q, m) seasonal order if seasonal, else None.
               Returns None if timeseries is None or empty.
    """
    if timeseries is None or timeseries.empty:
        print("Error: Timeseries is empty or None for parameter search.")
        return None

    print(f"Finding optimal ARIMA parameters for series of length {len(timeseries)}...")

    try:
        auto_model = pm.auto_arima(timeseries.dropna(),
                                   start_p=start_p, start_q=start_q,
                                   max_p=max_p, max_q=max_q,
                                   m=m,
                                   start_P=start_P, seasonal=seasonal,
                                   d=d, D=D,
                                   trace=trace,
                                   error_action=error_action,
                                   suppress_warnings=suppress_warnings,
                                   stepwise=stepwise)

        print(f"Auto ARIMA summary:\n{auto_model.summary()}")
        print(f"Optimal order: {auto_model.order}")
        if seasonal:
            print(f"Optimal seasonal order: {auto_model.seasonal_order}")
            return auto_model.order, auto_model.seasonal_order
        else:
            return auto_model.order, None

    except Exception as e:
        print(f"An error occurred during auto_arima: {e}")
        # Fallback or simpler model suggestion if auto_arima fails
        # For simplicity, returning None. A more robust solution might try default orders.
        if d is None: # if d was not pre-determined, try to guess a simple one
            p_val = check_stationarity(timeseries)
            if p_val is not None and p_val > 0.05: # if non-stationary
                d_guessed = 1
                print(f"auto_arima failed. Based on ADF test, d={d_guessed} might be needed.")
            else:
                d_guessed = 0 # stationary or unable to test
        else: # d was provided
            d_guessed = d

        # Suggesting a very simple model as a fallback
        print("Falling back to a default order (1, d_guessed, 1) due to auto_arima error.")
        return (1, d_guessed, 1), None

def fit_arima_model(train_data, order, seasonal_order=None):
    """
    Fits an ARIMA model to the training data.

    Args:
        train_data (pd.Series): The training time series data.
        order (tuple): The (p, d, q) order for the ARIMA model.
        seasonal_order (tuple, optional): The (P, D, Q, m) seasonal order.
                                          Default is None for non-seasonal ARIMA.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: The fitted ARIMA model.
                                                        Returns None if fitting fails.
    """
    if train_data is None or train_data.empty:
        print("Error: Training data is empty or None.")
        return None
    if order is None:
        print("Error: ARIMA order cannot be None.")
        return None

    print(f"Fitting ARIMA model with order={order} and seasonal_order={seasonal_order}...")

    try:
        model = ARIMA(train_data.dropna(),
                      order=order,
                      seasonal_order=seasonal_order,
                      enforce_stationarity=False, # auto_arima might yield non-stationary params
                      enforce_invertibility=False) # auto_arima might yield non-invertible params

        fitted_model = model.fit()
        print("ARIMA model fitted successfully.")
        print(fitted_model.summary())
        return fitted_model
    except Exception as e:
        print(f"Error fitting ARIMA model: {e}")
        return None

def forecast(model, steps):
    """
    Makes forecasts for a given number of steps using the fitted model.

    Args:
        model (statsmodels.tsa.arima.model.ARIMAResultsWrapper): The fitted ARIMA model.
        steps (int): The number of steps to forecast ahead.

    Returns:
        pd.DataFrame: A DataFrame containing the forecast, standard error,
                      and confidence intervals. Returns None if forecasting fails.
    """
    if model is None:
        print("Error: Fitted model is None. Cannot make forecasts.")
        return None
    if not isinstance(steps, int) or steps <= 0:
        print("Error: Number of steps must be a positive integer.")
        return None

    print(f"Forecasting {steps} steps ahead...")

    try:
        forecast_results = model.get_forecast(steps=steps)

        forecast_values = forecast_results.predicted_mean
        conf_int = forecast_results.conf_int()

        # Create a DataFrame for easier handling
        forecast_df = pd.DataFrame({
            'forecast': forecast_values,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        })

        # If the original series had a DatetimeIndex, try to extend it for the forecast
        if isinstance(model.data.row_labels, pd.DatetimeIndex) and len(model.data.row_labels) > 0:
            last_date = model.data.row_labels[-1]
            freq = pd.infer_freq(model.data.row_labels)
            if freq:
                forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                forecast_df.index = forecast_index
            else: # If frequency cannot be inferred, use integer index
                print("Warning: Could not infer frequency from model's data index. Forecast index will be integer based.")

        print("Forecast generated successfully.")
        return forecast_df

    except Exception as e:
        print(f"Error during forecasting: {e}")
        return None

def evaluate_model(test_data, predictions):
    """
    Evaluates the model using RMSE and MAE.

    Args:
        test_data (pd.Series or np.array): The actual observed values.
        predictions (pd.Series or np.array): The predicted values from the model.

    Returns:
        dict: A dictionary containing 'rmse' and 'mae' if successful, else None.
              Returns None if inputs are invalid or lengths don't match.
    """
    if test_data is None or predictions is None:
        print("Error: Test data or predictions are None.")
        return None

    # Ensure inputs are numpy arrays for consistent processing
    if isinstance(test_data, pd.Series):
        test_data = test_data.values
    if isinstance(predictions, pd.Series):
        predictions = predictions.values

    if len(test_data) != len(predictions):
        print(f"Error: Length of test_data ({len(test_data)}) and predictions ({len(predictions)}) do not match.")
        # Try to align if predictions are longer (e.g. forecast included historical fit)
        if len(predictions) > len(test_data):
            print(f"Attempting to use the last {len(test_data)} predictions to match test_data length.")
            predictions = predictions[-len(test_data):]
            if len(test_data) != len(predictions): # Check again
                 print("Error: Still mismatched after attempting alignment.")
                 return None
        else: # test_data is longer or still mismatched
            return None

    print("Evaluating model performance...")
    try:
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)

        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        return {'rmse': rmse, 'mae': mae}
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None
