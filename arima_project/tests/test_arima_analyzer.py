import pytest
import pandas as pd
import numpy as np
from src import arima_analyzer as arima

# --- Fixtures ---
@pytest.fixture
def dummy_data_path(tmp_path):
    """Create a dummy CSV file for testing and return its path."""
    csv_content = """Date,Value
2023-01-01,100
2023-02-01,110
2023-03-01,120
2023-04-01,130
2023-05-01,140
"""
    p = tmp_path / "dummy_sales.csv"
    p.write_text(csv_content)
    return str(p)

@pytest.fixture
def non_existent_file_path():
    """Return a path to a file that does not exist."""
    return "non_existent_file.csv"

@pytest.fixture
def malformed_data_path(tmp_path):
    """Create a malformed CSV (wrong column names) and return its path."""
    csv_content = """Month,Sales
2023-01-01,100
"""
    p = tmp_path / "malformed_sales.csv"
    p.write_text(csv_content)
    return str(p)

@pytest.fixture
def sample_timeseries():
    """Create a sample time series for testing functions."""
    dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'])
    return pd.Series(data=[100, 110, 120, 130, 140], index=dates)

# --- Tests for load_data ---

def test_load_data_success(dummy_data_path):
    """Test that data is loaded successfully from a valid CSV."""
    ts = arima.load_data(dummy_data_path, date_column='Date', value_column='Value')
    assert ts is not None
    assert isinstance(ts, pd.Series)
    assert len(ts) == 5
    assert pd.api.types.is_datetime64_any_dtype(ts.index)
    assert ts.iloc[0] == 100

def test_load_data_file_not_found(non_existent_file_path):
    """Test that load_data returns None for a non-existent file."""
    ts = arima.load_data(non_existent_file_path, date_column='Date', value_column='Value')
    assert ts is None

def test_load_data_key_error(malformed_data_path):
    """Test that load_data returns None when column names are incorrect."""
    ts = arima.load_data(malformed_data_path, date_column='Date', value_column='Value')
    assert ts is None

# --- Tests for check_stationarity ---

def test_check_stationarity_stationary_series():
    """Test ADF check on a clearly stationary series."""
    stationary_series = pd.Series([1, 1.1, 1, 1.2, 1, 1.1, 1, 1.2])
    p_value = arima.check_stationarity(stationary_series)
    assert p_value is not None
    assert p_value <= 0.05

def test_check_stationarity_non_stationary_series(sample_timeseries):
    """Test ADF check on a non-stationary series (trend)."""
    p_value = arima.check_stationarity(sample_timeseries)
    assert p_value is not None
    assert p_value > 0.05

def test_check_stationarity_empty_series():
    """Test that the function handles empty series gracefully."""
    empty_series = pd.Series([], dtype=float)
    p_value = arima.check_stationarity(empty_series)
    assert p_value is None

# --- Tests for evaluate_model ---

def test_evaluate_model_basic(sample_timeseries):
    """Test basic evaluation metrics (RMSE, MAE, MAPE)."""
    # Using sample_timeseries as both actual and predicted, but with a slight shift
    test_data = sample_timeseries
    predictions = sample_timeseries + 2 # Simple constant error

    results = arima.evaluate_model(test_data, predictions)

    assert results is not None
    assert 'rmse' in results
    assert 'mae' in results
    assert 'mape' in results

    assert results['mae'] == pytest.approx(2.0)
    assert results['rmse'] == pytest.approx(2.0)

    # MAPE = mean(|(actual - predicted) / actual|)
    expected_mape = np.mean(np.abs((test_data - predictions) / test_data))
    assert results['mape'] == pytest.approx(expected_mape)

def test_evaluate_model_with_zeros():
    """Test that evaluation handles zeros in the actual data gracefully."""
    test_data = pd.Series([10, 0, 20, 5])
    predictions = pd.Series([12, 2, 18, 5])

    results = arima.evaluate_model(test_data, predictions)

    assert results is not None
    # MAPE should be calculated, potentially resulting in a large number or inf,
    # but the function should not crash. scikit-learn's implementation handles this.
    assert 'mape' in results
    assert np.isfinite(results['mape'])

def test_evaluate_model_mismatched_length():
    """Test that None is returned for mismatched input lengths."""
    test_data = pd.Series([1, 2, 3])
    predictions = pd.Series([1, 2])
    assert arima.evaluate_model(test_data, predictions) is None
