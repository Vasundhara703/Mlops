# Research Project Report: ML Model Drift Monitoring Dashboard

**Name:** Vasundhara Mishra  
**Course:** BTech AIML  
**Section:** C  
**Reg. No.:** 23FE10CAI00093

## 1. Abstract
Machine learning models can perform well during testing but behave differently when they receive new data. This project builds a small dashboard that demonstrates one part of MLOps: monitoring incoming data for possible drift. The dashboard compares a reference sample with simulated incoming data, applies the two-sample Kolmogorov–Smirnov (KS) test to numeric features, presents feature-level results, and displays basic model metrics. The project is intended for learning and demonstration, not for clinical or production use.

## 2. Problem Statement
After a model is deployed, the data it receives can change due to new user behaviour, different operating conditions, or changes in how data is collected. These changes are called data drift when the input distribution changes. If a team does not monitor incoming data, it may miss changes that could affect predictions. The project addresses this problem by providing a simple visual tool to compare reference data with incoming data and flag features that have changed statistically.

## 3. Objectives
- Understand the role of model monitoring in the MLOps lifecycle.
- Build a dashboard that compares reference and incoming data.
- Apply a statistical test to detect potential feature drift.
- Display results in a form that is easy to interpret.
- Provide a downloadable CSV report and a repeatable demonstration.

## 4. Methodology
The application uses the Breast Cancer Wisconsin dataset available through scikit-learn. A Random Forest classifier is trained on a training split. A held-out sample is used as reference data. To demonstrate drift without needing a live data feed, the user can select features and adjust the amount by which their values are shifted.

For each numeric feature, the application compares the reference and incoming samples using the two-sample KS test. The test compares their empirical distributions. If the p-value is below 0.05, the feature is flagged as showing a statistically detectable difference. The dashboard then calculates the percentage of monitored features that were flagged and plots the two distributions for a selected feature.

The dashboard also displays accuracy, precision, recall, and F1 score as an educational illustration. Because this project is a local simulation and does not receive fresh verified labels, these metrics should not be interpreted as a valid estimate of real production performance.

## 5. Implementation
The project is implemented in Python using Streamlit for the interactive interface, pandas and NumPy for data handling, scikit-learn for the dataset and classifier, SciPy for the KS test, and Matplotlib for visualisation.

The main interface contains:
1. Sidebar controls for drift strength, sample size, and selected features.
2. Summary metrics showing the number of monitored features, drifted features, and drift share.
3. A feature-level table with KS statistics, p-values, and means.
4. A model-metrics chart.
5. A histogram comparing reference and incoming data.
6. A button to download drift results as a CSV file.

## 6. Results and Expected Behaviour
When drift strength is set to zero, the incoming sample is not intentionally shifted. Results may still flag a feature by chance because statistical tests are not perfect. As the user increases the drift strength for selected features, their distributions move away from the reference distributions, and the KS test is more likely to flag them. The dashboard helps the user see which features changed and how their distributions differ.

No fixed accuracy or drift result is claimed because the result depends on the sample and user-selected controls. The application is designed to make the experiment repeatable and observable rather than to claim a benchmark improvement.

## 7. Real-World Relevance
The same general approach can be adapted to e-commerce recommendations, fraud detection, manufacturing sensors, and customer-support classification. For example, a shopping platform might compare recent customer or product features with the data used during model development. A drift alert would prompt the team to investigate the cause and evaluate whether model performance has changed.

In real applications, teams should combine drift checks with data-quality checks, performance monitoring when labels become available, privacy controls, and documented alert procedures. Drift alone does not prove that a model is inaccurate.

## 8. Challenges and Limitations
- The project uses simulated drift rather than a live production data stream.
- The KS test is designed for comparing distributions and does not directly measure prediction quality.
- Testing many features can increase the chance of false alarms.
- Model performance metrics require reliable ground-truth labels; the dashboard's metric display is only illustrative.
- The public dataset is for education and must not be used for real medical decisions.

## 9. Conclusion
This project demonstrates a practical MLOps idea: monitoring does not end when a model is deployed. Comparing incoming data with reference data can help identify changes early and guide further investigation. The dashboard provides a simple workflow for simulating drift, reviewing statistical results, and exporting a report. A production-ready extension could add a real data pipeline, alert notifications, historical monitoring, model version tracking, and a retraining process with validation and approval.

## 10. Tools and Technologies
- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- SciPy
- Matplotlib
- pytest

## 11. References
1. scikit-learn documentation, Breast Cancer Wisconsin dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
2. SciPy documentation, two-sample KS test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
3. Streamlit documentation: https://docs.streamlit.io/
