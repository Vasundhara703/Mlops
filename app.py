import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from monitoring import compare_datasets, summarize_drift

st.set_page_config(page_title="ML Model Drift Monitor", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y, list(X.columns)

@st.cache_resource
def train_model():
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test.reset_index(drop=True), y_test.reset_index(drop=True)

X, y, features = load_data()
model, baseline_X, baseline_y = train_model()

st.title("📊 ML Model Drift Monitoring Dashboard")
st.caption("A student project demonstrating how to detect changes in incoming data and check model quality.")

st.sidebar.header("Simulate production data")
drift_strength = st.sidebar.slider("Data drift strength", 0.0, 2.5, 0.8, 0.1)
sample_size = st.sidebar.slider("Number of incoming records", 50, 300, 150, 25)
selected_features = st.sidebar.multiselect(
    "Features to shift", features,
    default=["mean radius", "mean texture", "mean perimeter"]
)

rng = np.random.default_rng(7)
sampled_indices = baseline_X.sample(n=min(sample_size, len(baseline_X)), random_state=7).index
current_X = baseline_X.loc[sampled_indices].reset_index(drop=True).copy()
# Keep the corresponding labels for the same sampled rows in this demonstration.
current_y = baseline_y.loc[sampled_indices].reset_index(drop=True)
# Shift selected features by a fraction of their baseline standard deviation.
for feature in selected_features:
    std = float(baseline_X[feature].std())
    current_X[feature] = current_X[feature] + drift_strength * std

baseline_compare = baseline_X.sample(n=len(current_X), random_state=42).reset_index(drop=True)
drift_table = compare_datasets(baseline_compare, current_X, features)
drift_summary = summarize_drift(drift_table)

# Current sample labels are available here only to demonstrate metric monitoring.
predictions = model.predict(current_X)
metrics = {
    "Accuracy": accuracy_score(current_y, predictions),
    "Precision": precision_score(current_y, predictions, zero_division=0),
    "Recall": recall_score(current_y, predictions, zero_division=0),
    "F1 score": f1_score(current_y, predictions, zero_division=0),
}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Features monitored", len(features))
c2.metric("Features with drift", int(drift_table["Drift detected"].sum()))
c3.metric("Drift share", f"{100 * drift_summary['drift_share']:.1f}%")
c4.metric("Current sample accuracy", f"{metrics['Accuracy']:.3f}")

if drift_summary["drift_share"] >= 0.30:
    st.warning("High drift signal: investigate the changed features before relying on new predictions.")
elif drift_summary["drift_share"] > 0:
    st.info("Some drift was detected. Review the feature-level results.")
else:
    st.success("No drift detected by the selected statistical checks.")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Feature-level drift results")
    st.dataframe(
        drift_table.sort_values(["Drift detected", "KS p-value"], ascending=[False, True]),
        use_container_width=True,
        hide_index=True
    )
with right:
    st.subheader("Model quality metrics")
    metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Score": list(metrics.values())})
    st.bar_chart(metric_df.set_index("Metric"))
    st.caption("Metrics are illustrative because this local demo reuses held-out labels to keep the example self-contained.")

st.subheader("Compare feature distributions")
feature_to_plot = st.selectbox("Choose a feature", features, index=features.index("mean radius"))
fig, ax = plt.subplots()
ax.hist(baseline_compare[feature_to_plot], bins=20, alpha=0.65, label="Reference data")
ax.hist(current_X[feature_to_plot], bins=20, alpha=0.65, label="Incoming data")
ax.set_xlabel(feature_to_plot)
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Download monitoring report")
report_csv = drift_table.to_csv(index=False).encode("utf-8")
st.download_button("Download feature drift results (CSV)", report_csv, "drift_results.csv", "text/csv")

with st.expander("How to read this dashboard"):
    st.markdown("""
    - **KS p-value:** A small p-value (below 0.05 in this demo) suggests that the two samples may come from different distributions.
    - **Drift detected:** A flag based on the statistical threshold; it is a signal for investigation, not proof that the model is wrong.
    - **Drift share:** The fraction of monitored features flagged as changed.
    - **Important limitation:** The data is simulated from a public sample dataset. In a real system, labels may arrive later and production monitoring should protect private data.
    """)
