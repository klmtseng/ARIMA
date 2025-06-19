# ARIMA Time Series Forecasting Project

This project provides a Python-based solution for performing time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model. It includes a core analysis module, a sample dataset, a Jupyter Notebook example, and an interactive Streamlit web application for live demonstrations.

## Project Overview

The primary goal is to offer a reusable and understandable framework for ARIMA modeling. Key features include:
- Data loading and preprocessing.
- Stationarity testing (Augmented Dickey-Fuller test).
- Automatic determination of optimal ARIMA parameters (p, d, q) and seasonal parameters (P, D, Q, m) using `pmdarima.auto_arima`.
- Model fitting and summary.
- Forecasting with confidence intervals.
- Model evaluation (RMSE, MAE).
- A Jupyter Notebook to walk through an example analysis.
- A Streamlit web application to upload custom data and visualize forecasts interactively.

## Directory Structure

```
arima_project/
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   └── sample_monthly_sales.csv # Sample time series data
├── notebooks/
│   └── arima_example.ipynb # Jupyter Notebook example
├── src/
│   └── arima_analyzer.py   # Core ARIMA analysis module
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Setup Instructions

Follow these steps to set up and run the project locally:

1.  **Clone the repository (if applicable) or download the project files.**

2.  **Create a virtual environment (recommended):**
    Open your terminal or command prompt, navigate to the `arima_project` root directory, and run:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    With the virtual environment activated, install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Examples

### 1. Jupyter Notebook Example

The Jupyter Notebook provides a step-by-step guide through an ARIMA analysis using the sample data.

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  Start Jupyter Notebook server from the `arima_project` root directory:
    ```bash
    jupyter notebook
    ```
3.  In the Jupyter interface that opens in your browser, navigate to the `notebooks/` directory and open `arima_example.ipynb`.
4.  You can run the cells sequentially to see the analysis process.

### 2. Streamlit Live Demo Application

The Streamlit application allows you to upload your own CSV time series data and perform ARIMA forecasting interactively.

1.  Ensure your virtual environment is activated and dependencies are installed.
2.  Navigate to the `arima_project` root directory in your terminal.
3.  Run the Streamlit app using the following command:
    ```bash
    streamlit run app/app.py
    ```
4.  The application should open in your web browser. Use the sidebar to upload your CSV file, select the date and value columns, configure parameters, and click "Run Analysis & Forecast".

## Core Module: `src/arima_analyzer.py`

This module contains the core logic for the ARIMA analysis. Key functions include:

-   `load_data(filepath, date_column, value_column)`: Loads time series data from a CSV.
-   `check_stationarity(timeseries)`: Performs the ADF test.
-   `make_stationary(timeseries, d=None)`: Applies differencing (though `auto_arima` typically handles this).
-   `find_optimal_parameters(timeseries, ...)`: Uses `pmdarima.auto_arima` to find the best model order.
-   `fit_arima_model(train_data, order, seasonal_order=None)`: Fits the `statsmodels` ARIMA model.
-   `forecast(model, steps)`: Generates future predictions.
-   `evaluate_model(test_data, predictions)`: Calculates RMSE and MAE.

## Contributing

Feel free to fork this project, suggest improvements, or submit pull requests.

---
*This project was developed with assistance from an AI model.*
