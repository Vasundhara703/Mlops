# ML Model Drift Monitoring Dashboard

**Student:** Vasundhara Mishra  
**Course:** BTech AIML | **Section:** C  
**Reg. No.:** 23FE10CAI00093

## Project overview
This project demonstrates a simple MLOps monitoring workflow. It compares a reference dataset with simulated incoming data, uses the two-sample Kolmogorov–Smirnov (KS) test to flag possible feature drift, displays model metrics, and allows the user to download a CSV report.

## Features
- Train a Random Forest classifier on scikit-learn's public Breast Cancer Wisconsin dataset.
- Simulate incoming data with adjustable shifts to selected features.
- Compare reference and incoming feature distributions.
- Show drift flags, KS statistics, p-values, and mean values.
- Display accuracy, precision, recall, and F1 score.
- Plot reference and incoming distributions.
- Export the drift table as CSV.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start the dashboard:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## How to demonstrate it
1. Set **Data drift strength** to `0.0`. Most features should not be shifted intentionally.
2. Increase it to `1.0` or `2.0`.
3. Keep the default shifted features or select others.
4. Observe the feature-level drift table, drift share, and distribution chart.
5. Download the CSV report.

## Method
For each numeric feature, the app compares the reference and incoming samples using the two-sample KS test. At a demo threshold of `p < 0.05`, the feature is flagged as having a statistically detectable distribution difference. The drift share is the number of flagged features divided by the total features monitored.

## Limitations
- The app simulates a production data shift; it is not connected to a live service.
- The demo uses a public teaching dataset and should not be used for medical decisions.
- Statistical drift does not automatically mean model performance has declined.
- The displayed performance metrics are illustrative: labels are reused from the held-out sample to keep the demo self-contained. A real deployment should evaluate predictions against verified labels when they become available.
- Repeated testing across many features can create false positives; production systems may need multiple-testing correction and agreed alert thresholds.

## Project structure
```text
ML_Model_Drift_Monitoring_Project/
├── app.py
├── monitoring.py
├── requirements.txt
├── README.md
├── report.md
└── tests/
    └── test_monitoring.py
```
