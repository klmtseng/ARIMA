# ARIMA Time Series Analysis - Live Demo

This project is a fully interactive, front-end web application that simulates the process of ARIMA time series analysis. It allows users to select from classic time series datasets and instantly see the corresponding Python code, model summary, and a visualized forecast chart, providing a hands-on learning experience without needing a Python environment.

This entire demo is built with **HTML, CSS, and JavaScript** and runs completely in the browser.

## ✨ Features

* **Interactive Dataset Selection**: Choose from multiple well-known time series datasets like *International Airline Passengers* and *Monthly Sunspot Numbers*.
* **Instant Results Simulation**: Get immediate feedback upon clicking "Run Analysis," just like a real data science tool.
* **Code Display**: Shows the actual Python code (`pmdarima` library) required to perform the analysis on the selected dataset.
* **Model Summary**: Presents a detailed statistical summary of the fitted SARIMAX model, including coefficients, AIC, BIC, and other key metrics.
* **Interactive Forecast Chart**: Visualizes the results with a beautiful and responsive chart (using Chart.js) that displays:
    * Historical training data
    * Actual test data (for comparison)
    * ARIMA model forecast
    * 95% confidence interval

## 🚀 How to Use

This is a standalone web page. No installation or setup is required.

1.  **Download or Clone**: Get the `index.html` file from this repository.
    ```bash
    git clone [https://github.com/klmtseng/2D_Segment_Polyline.git](https://github.com/klmtseng/2D_Segment_Polyline.git)
    ```
2.  **Open in Browser**: Navigate to the project directory and open the `index.html` file in any modern web browser (like Chrome, Firefox, or Edge).
3.  **Interact**:
    * Use the dropdown menu to select a dataset.
    * Click the "Run Analysis" button to see the results.

That's it!

## 🛠️ Technologies Used

* **HTML5**: For the basic structure of the web page.
* **CSS3**: For styling and creating a clean, modern user interface.
* **JavaScript (ES6)**: To handle user interactions, manage data, and simulate the analysis process.
* **Chart.js**: For rendering the beautiful and interactive forecast charts.
* **date-fns Adapter for Chart.js**: To handle time-series axes correctly.

## 📊 Datasets Included

This demo uses pre-loaded data from several classic time series sources to simulate the analysis:

* **International Airline Passengers**: A famous dataset showing clear trend and multiplicative seasonality.
* **Monthly Sunspot Numbers**: A classic dataset exhibiting cyclical patterns.
* **US Monthly Retail Sales**: Features strong trend and seasonality with holiday peaks.
* **Melbourne Daily Minimum Temperatures**: A noisy dataset with annual seasonality.

---

> **Disclaimer**: This is a **simulation** designed for educational and demonstrative purposes. It does not perform real-time statistical calculations in the browser. The Python code, model summaries, and forecast data are pre-computed and stored within the JavaScript to create a seamless live demo experience.
