import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def compare_datasets(reference: pd.DataFrame, current: pd.DataFrame, features, alpha: float = 0.05) -> pd.DataFrame:
    """Compare numeric feature distributions with the two-sample Kolmogorov–Smirnov test."""
    rows = []
    for feature in features:
        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()
        if len(ref) < 2 or len(cur) < 2:
            statistic, p_value = np.nan, np.nan
            detected = False
        else:
            result = ks_2samp(ref, cur)
            statistic, p_value = float(result.statistic), float(result.pvalue)
            detected = bool(p_value < alpha)
        rows.append({
            "Feature": feature,
            "KS statistic": round(statistic, 4) if np.isfinite(statistic) else np.nan,
            "KS p-value": round(p_value, 6) if np.isfinite(p_value) else np.nan,
            "Drift detected": detected,
            "Reference mean": round(float(ref.mean()), 4) if len(ref) else np.nan,
            "Incoming mean": round(float(cur.mean()), 4) if len(cur) else np.nan,
        })
    return pd.DataFrame(rows)

def summarize_drift(drift_table: pd.DataFrame) -> dict:
    total = len(drift_table)
    flagged = int(drift_table["Drift detected"].sum()) if total else 0
    return {
        "features_total": total,
        "features_drifted": flagged,
        "drift_share": flagged / total if total else 0.0,
    }
