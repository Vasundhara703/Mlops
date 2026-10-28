import pandas as pd
from monitoring import compare_datasets, summarize_drift

def test_detects_shifted_feature():
    reference = pd.DataFrame({"a": list(range(100)), "b": list(range(100))})
    current = pd.DataFrame({"a": [x + 1000 for x in range(100)], "b": list(range(100))})
    result = compare_datasets(reference, current, ["a", "b"])
    assert bool(result.loc[result["Feature"] == "a", "Drift detected"].iloc[0]) is True

def test_summary_empty_table():
    empty = pd.DataFrame({"Drift detected": []})
    summary = summarize_drift(empty)
    assert summary["features_total"] == 0
    assert summary["drift_share"] == 0.0
