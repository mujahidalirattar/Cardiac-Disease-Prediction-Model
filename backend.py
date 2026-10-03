"""
backend.py : model loading, prediction, health indicators and retraining
for the Cardiac Health Risk dashboard. No web framework in here.

The saved model file is a dict:
    {"model": <classifier>, "scaler": <fitted scaler>, "feature_names": [...], "meta": {...}}
`meta` is optional (older files without it keep working).
"""

import os
import shutil
import datetime as dt
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

# ============================================================================= #
# PATHS AND CONSTANTS
# ============================================================================= #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "cardio_model.pkl"
TARGET_COL = "cardio"
DEFAULT_FEATURES = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                    "cholesterol", "gluc", "smoke", "alco", "active"]

# Plausible clinical ranges (same limits the input form enforces)
RANGES = {"age": (18, 100), "height": (100, 230), "weight": (30, 200),
          "ap_hi": (70, 240), "ap_lo": (40, 160)}


def resolve_model_path():
    """Look for the model next to this file, in a Model/ subfolder, then in the working directory."""
    for path in (os.path.join(BASE_DIR, MODEL_FILENAME),
                 os.path.join(BASE_DIR, "Model", MODEL_FILENAME),
                 os.path.join(os.getcwd(), MODEL_FILENAME)):
        if os.path.exists(path):
            return path
    return os.path.join(BASE_DIR, MODEL_FILENAME)


MODEL_PATH = resolve_model_path()


# ============================================================================= #
# MODEL LOADING (cached, so the file is read once and not on every click)
# ============================================================================= #

@lru_cache(maxsize=1)
def load_artifacts():
    """Loads the model, scaler and feature list from the .pkl file."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading model file: {e}")


def clear_cache():
    load_artifacts.cache_clear()


def model_status():
    """(ready, message). Never raises, so the UI can show a friendly status."""
    try:
        load_artifacts()
        return True, None
    except Exception as exc:
        return False, str(exc)


def feature_names():
    try:
        return list(load_artifacts()["feature_names"])
    except Exception:
        return list(DEFAULT_FEATURES)


def model_info():
    art = load_artifacts()
    meta = art.get("meta", {}) if isinstance(art, dict) else {}
    return {
        "algorithm": type(art["model"]).__name__,
        "scaler": type(art["scaler"]).__name__,
        "features": list(art["feature_names"]),
        "meta": meta,
        "file_mb": round(os.path.getsize(MODEL_PATH) / 1_048_576, 2) if os.path.exists(MODEL_PATH) else None,
    }


# ============================================================================= #
# PREDICTION
# ============================================================================= #

def predict_heart_disease(input_data: dict):
    """
    Accepts patient parameters dictionary, scales features, and returns prediction.
    """
    artifacts = load_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    names = artifacts["feature_names"]

    df_input = pd.DataFrame([input_data])[names]
    scaled_input = scaler.transform(df_input)

    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    risk_probability = float(probabilities[1] * 100)

    return {
        "has_disease": prediction == 1,
        "risk_percentage": round(risk_probability, 2),
        "confidence": round(float(np.max(probabilities)) * 100, 2),
    }


def predict_batch(frame: pd.DataFrame):
    """Risk percentage (0-100) for every row of a feature DataFrame."""
    art = load_artifacts()
    X = frame[art["feature_names"]]
    return art["model"].predict_proba(art["scaler"].transform(X))[:, 1] * 100.0


def feature_importance():
    """[(feature, value)] sorted ascending, or None if the model does not expose importances."""
    art = load_artifacts()
    model, names = art["model"], art["feature_names"]
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_))
    else:
        return None
    return sorted(zip(names, vals), key=lambda r: r[1])


# ============================================================================= #
# HEALTH INDICATORS (reference ranges, independent of the ML model)
# ============================================================================= #

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def bp_category(sys_bp, dia_bp):
    """Blood pressure stage following the AHA 2017 categories."""
    if sys_bp > 180 or dia_bp > 120:
        return "Hypertensive crisis"
    if sys_bp >= 140 or dia_bp >= 90:
        return "Hypertension stage 2"
    if sys_bp >= 130 or dia_bp >= 80:
        return "Hypertension stage 1"
    if sys_bp >= 120:
        return "Elevated"
    return "Normal"


def health_indicators(p: dict):
    """Derived numbers plus plain language observations for one patient dict."""
    bmi = p["weight"] / ((p["height"] / 100.0) ** 2)
    sys_bp, dia_bp = p["ap_hi"], p["ap_lo"]
    pulse_pressure = sys_bp - dia_bp
    mean_ap = dia_bp + pulse_pressure / 3.0
    bmi_cat, bp_cat = bmi_category(bmi), bp_category(sys_bp, dia_bp)

    obs = []  # (level, title, text) with level in good / watch / alert
    if bp_cat == "Normal":
        obs.append(("good", "Blood pressure", f"{sys_bp}/{dia_bp} mmHg is in the normal range."))
    elif bp_cat == "Elevated":
        obs.append(("watch", "Blood pressure", f"{sys_bp}/{dia_bp} mmHg is slightly elevated."))
    else:
        obs.append(("alert", "Blood pressure", f"{sys_bp}/{dia_bp} mmHg falls in: {bp_cat}."))
    if bmi_cat == "Normal":
        obs.append(("good", "Body mass index", f"BMI {bmi:.1f} is in the healthy range."))
    elif bmi_cat == "Overweight" or bmi_cat == "Underweight":
        obs.append(("watch", "Body mass index", f"BMI {bmi:.1f} is classed as {bmi_cat.lower()}."))
    else:
        obs.append(("alert", "Body mass index", f"BMI {bmi:.1f} is in the obese range."))
    for key, label in (("cholesterol", "Cholesterol"), ("gluc", "Glucose")):
        v = p[key]
        if v == 1:
            obs.append(("good", label, f"{label} level is normal."))
        elif v == 2:
            obs.append(("watch", label, f"{label} level is above normal."))
        else:
            obs.append(("alert", label, f"{label} level is well above normal."))
    if p["smoke"]:
        obs.append(("alert", "Smoking", "Smoking strongly raises cardiovascular risk."))
    if p["alco"]:
        obs.append(("watch", "Alcohol", "Regular alcohol intake can raise blood pressure."))
    if not p["active"]:
        obs.append(("watch", "Activity", "Low physical activity is a modifiable risk factor."))
    else:
        obs.append(("good", "Activity", "Physically active, which is protective."))
    if pulse_pressure > 60:
        obs.append(("watch", "Pulse pressure", f"{pulse_pressure} mmHg is wider than the usual 40-60 range."))

    return {"bmi": bmi, "bmi_cat": bmi_cat, "bp_cat": bp_cat,
            "pulse_pressure": pulse_pressure, "map": mean_ap, "observations": obs}


# ============================================================================= #
# TRAINING DATA PREPARATION
# ============================================================================= #

_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n"}


def read_table(file_bytes: bytes, filename: str):
    """Read an uploaded .csv (any separator, e.g. ; or ,) or .xlsx into a DataFrame."""
    import io
    name = filename.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")


def prepare_training_data(df_raw: pd.DataFrame, target: str = TARGET_COL, clean_outliers: bool = True):
    """Validate and clean an uploaded dataset. Returns (X, y, report). Raises ValueError if unusable."""
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    names = feature_names()
    report = {"rows_in": int(len(df)), "notes": []}

    missing = [c for c in names + [target] if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    df = df[names + [target]].copy()

    # Target to 0/1 (accepts 0/1, True/False, yes/no)
    t = df[target]
    if t.dtype == object or t.dtype == bool:
        t = t.astype(str).str.strip().str.lower().map({**{k: 1 for k in _TRUE}, **{k: 0 for k in _FALSE}})
    df[target] = pd.to_numeric(t, errors="coerce")

    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna()
    report["dropped_missing"] = int(before - len(df))

    # The public cardio dataset stores age in days; the dashboard works in years.
    if "age" in df.columns and len(df) and df["age"].median() > 150:
        df["age"] = df["age"] / 365.25
        report["notes"].append("Age looked like days, so it was converted to years.")

    if clean_outliers:
        before = len(df)
        mask = pd.Series(True, index=df.index)
        for col, (lo, hi) in RANGES.items():
            if col in df.columns:
                mask &= df[col].between(lo, hi)
        if "ap_hi" in df.columns and "ap_lo" in df.columns:
            mask &= df["ap_hi"] >= df["ap_lo"]
        df = df[mask]
        report["dropped_outliers"] = int(before - len(df))
    else:
        report["dropped_outliers"] = 0

    df = df[df[target].isin([0, 1])]
    report["rows_used"] = int(len(df))
    if len(df) < 200:
        raise ValueError(f"Only {len(df)} usable rows after cleaning. At least 200 are needed to train.")
    if df[target].nunique() < 2:
        raise ValueError("The target column has only one class. Both 0 and 1 are needed.")

    report["positive_rate"] = float(df[target].mean() * 100)
    return df[names].astype(float), df[target].astype(int), report


def dataset_template():
    """Tiny CSV text showing the expected columns."""
    cols = DEFAULT_FEATURES + [TARGET_COL]
    rows = [
        [50, 1, 165, 70, 120, 80, 1, 1, 0, 0, 1, 0],
        [61, 2, 172, 92, 160, 100, 3, 2, 1, 0, 0, 1],
    ]
    return ",".join(cols) + "\n" + "\n".join(",".join(str(v) for v in r) for r in rows) + "\n"


# ============================================================================= #
# TRAINING + DEPLOYMENT
# ============================================================================= #

ALGORITHMS = ["Random Forest", "Gradient Boosting (fast)", "Logistic Regression"]


def _build_model(algorithm, n_estimators, max_depth):
    if algorithm == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=n_estimators,
                                      max_depth=None if max_depth == 0 else max_depth,
                                      n_jobs=-1, random_state=42)
    if algorithm == "Gradient Boosting (fast)":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=n_estimators,
                                              max_depth=None if max_depth == 0 else max_depth,
                                              random_state=42)
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000)


def _metrics(model, Xs, y):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix, roc_curve)
    pred = model.predict(Xs)
    proba = model.predict_proba(Xs)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    step = max(1, len(fpr) // 300)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "cm": confusion_matrix(y, pred).tolist(),
        "fpr": fpr[::step].tolist(), "tpr": tpr[::step].tolist(),
    }


def train_model(X, y, algorithm="Random Forest", n_estimators=200, max_depth=0, test_size=0.2):
    """Train a new model and evaluate it on a held-out split. Nothing is saved yet."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = _build_model(algorithm, n_estimators, max_depth)
    model.fit(scaler.transform(X_tr), y_tr)

    Xs_te = scaler.transform(X_te)
    new_metrics = _metrics(model, Xs_te, y_te)

    # How does the model that is live right now do on this same test split?
    current_metrics = None
    try:
        art = load_artifacts()
        cur = art["model"]
        current_metrics = _metrics(cur, art["scaler"].transform(X_te[art["feature_names"]]), y_te)
    except Exception:
        pass

    # Feature importance
    names = list(X.columns)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    else:
        from sklearn.inspection import permutation_importance
        n = min(3000, len(X_te))
        r = permutation_importance(model, Xs_te[:n], y_te.iloc[:n], n_repeats=3, random_state=42, n_jobs=-1)
        imp = np.clip(r.importances_mean, 0, None)
    importances = sorted(zip(names, imp.tolist()), key=lambda r: r[1])

    meta = {"algorithm": algorithm, "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "rows": int(len(X)), "accuracy": new_metrics["accuracy"], "roc_auc": new_metrics["roc_auc"]}
    return {
        "artifacts": {"model": model, "scaler": scaler, "feature_names": names, "meta": meta},
        "metrics": new_metrics, "current_metrics": current_metrics, "importances": importances,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "algorithm": algorithm,
    }


def save_artifacts(artifacts: dict):
    """Make a trained model the live one. The previous file is kept as cardio_model.prev.pkl."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, MODEL_PATH.replace(".pkl", ".prev.pkl"))
    joblib.dump(artifacts, MODEL_PATH, compress=3)
    clear_cache()

# ============================================================================= #
# EVALUATION OF THE LIVE MODEL (added for the dashboard Insights page)
# ============================================================================= #

def evaluate_live_model(X, y):
    """
    Score the CURRENT live model on a labelled dataset.

    X : feature DataFrame (the Insights page builds it with prepare_training_data)
    y : 0/1 target series

    Returns the same metrics dict as training (accuracy, precision, recall, f1,
    roc_auc, confusion matrix, roc curve points) plus `probabilities`, the risk
    percentage for every row, so the UI can draw a distribution. The live model
    is only read here; nothing is retrained or saved.
    """
    art = load_artifacts()
    names = art["feature_names"]
    Xs = art["scaler"].transform(X[names])
    out = _metrics(art["model"], Xs, y)
    out["probabilities"] = (art["model"].predict_proba(Xs)[:, 1] * 100.0).tolist()
    out["n_rows"] = int(len(X))
    return out