import copy
import logging
import math
import os

import numpy as np
import pandas as pd

from scipy.stats import linregress

from mortality_prediction.dataloader import Gender
from mortality_prediction.dataloader import ParamType
from mortality_prediction.dataloader import Patient
from mortality_prediction.dataloader import ScalerParams
from mortality_prediction.dataloader import ScalerType
from mortality_prediction.utils import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Hour grid: 1, 2, ..., 48  (ceiling buckets; bucket N covers (N-1, N] hours)
HOURS = list(range(1, 49))

# Ordered time-series column names (one per ParamType, in enum definition order)
TS_COLS: list[str] = [p.value for p in ParamType]

STATIC_COLS: list[str] = ["age", "height_cm", "weight_kg", "gender"]

# Fixed column order for the output parquet
COLUMN_ORDER: list[str] = [
    "PatientID",
    "timestamp",
    *TS_COLS,
    *STATIC_COLS,
    "in_hospital_death",
]

_GENDER_FLOAT: dict[Gender, float] = {Gender.MALE: 1.0, Gender.FEMALE: 0.0}

# Ordered variable names for the triplet format (one-hot dimension = len(TRIPLET_VARS))
TRIPLET_VARS: list[str] = TS_COLS + STATIC_COLS


def collect_ts_values(patients: dict[str, Patient], param: ParamType) -> list[float]:
    """Collect all observed raw values for *param* across every patient's timeseries."""
    vals: list[float] = []
    for patient in patients.values():
        for tp in patient.timeseries:
            for m in tp.measurements:
                if m.param == param:
                    vals.append(
                        float(m.value.value if hasattr(m.value, "value") else m.value)
                    )
    return vals


def collect_static_values(patients: dict[str, Patient], field: str) -> list[float]:
    """Collect all non-null values of static field *field* across every patient."""
    vals: list[float] = []
    for patient in patients.values():
        v = getattr(patient.static, field)
        if v is not None:
            vals.append(float(v))
    return vals


def scale_value(param_name: str, value: float, params: ScalerParams) -> float:
    """Apply the fitted scaler for *param_name* to a single raw *value*."""
    if params.scaler_type == ScalerType.ROBUST:
        return (value - params.center) / params.scale
    else:  # LOG_STANDARD
        return (math.log1p(value) - params.mean) / params.std


def scale_patients(
    patients: dict[str, Patient],
    norm_params: dict[str, ScalerParams],
) -> dict[str, Patient]:
    """
    Return a deep copy of *patients* with every scalable measurement replaced
    by its scaled value.  Params absent from *norm_params* (e.g. GCS, MechVent,
    gender, icu_type) are left untouched.
    """
    scaled = copy.deepcopy(patients)
    for patient in scaled.values():
        for field in ("age", "height_cm", "weight_kg"):
            if field in norm_params:
                v = getattr(patient.static, field)
                if v is not None:
                    setattr(
                        patient.static,
                        field,
                        scale_value(field, float(v), norm_params[field]),
                    )
        for tp in patient.timeseries:
            for m in tp.measurements:
                if m.param == ParamType.GCS:
                    m.value = (15.0 - float(m.value)) / 12.0
                elif m.param.value in norm_params:
                    raw = float(m.value.value if hasattr(m.value, "value") else m.value)
                    m.value = scale_value(
                        m.param.value, raw, norm_params[m.param.value]
                    )
    return scaled


def _build_patient_df(patient_id: str, patient: Patient) -> pd.DataFrame:
    """
    Return a 48-row DataFrame (one row per hour 1-48) for a single patient.

    Timestamp binning: ceiling to the nearest hour so bucket N covers the
    half-open interval (N-1, N].  A measurement at exactly N hours falls
    into bucket N.  Measurements after hour 48 are dropped.  When multiple
    measurements fall into the same bucket the latest one wins.

    Missing observations are left as NaN / pd.NA.
    """
    hourly: dict[int, dict[str, float]] = {h: {} for h in HOURS}

    for tp in sorted(patient.timeseries, key=lambda x: x.time):
        hour = math.ceil(tp.time.total_seconds() / 3600)
        if hour < 1 or hour > 48:
            continue
        for m in tp.measurements:
            hourly[hour][m.param.value] = float(
                m.value.value if hasattr(m.value, "value") else m.value
            )

    rows = [{"PatientID": patient_id, "timestamp": h, **hourly[h]} for h in HOURS]
    df = pd.DataFrame(rows)

    # Static features — repeated at every row
    s = patient.static
    df["age"] = float(s.age) if s.age is not None else pd.NA
    df["height_cm"] = float(s.height_cm) if s.height_cm is not None else pd.NA
    df["weight_kg"] = float(s.weight_kg) if s.weight_kg is not None else pd.NA
    df["gender"] = _GENDER_FLOAT.get(s.gender, pd.NA) if s.gender is not None else pd.NA
    if s.in_hospital_death is None:
        raise ValueError(f"Patient {patient_id} has no in_hospital_death label.")
    df["in_hospital_death"] = float(s.in_hospital_death)

    return df


def build_dataset(patients: dict[str, Patient]) -> pd.DataFrame:
    """
    Combine all patients into a single DataFrame with the canonical column order.
    Columns not observed for any patient in this set are created as all-NA.
    """
    frames = [_build_patient_df(pid, p) for pid, p in patients.items()]
    df = pd.concat(frames, ignore_index=True)

    for col in COLUMN_ORDER:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[COLUMN_ORDER]

    df["timestamp"] = df["timestamp"].astype("int16")

    return df


def convert_to_table_format(
    set_name: str,
    patients: dict[str, Patient],
    fill_missing: bool = False,
) -> str:
    """
    Build the 48-row-per-patient hourly table and save it as a parquet file.

    If *fill_missing* is True, missing values are imputed in two passes:
      1. Forward-fill within each patient over the 48 hourly slots (carry the
         last known value forward in time).
      2. Fill any remaining NaN with 0.0 (covers slots that had no measurement
         at all before the first observation).

    Returns the path of the saved file.
    """
    logger.info(f"Building table format for {set_name} ({len(patients)} patients)...")
    df = build_dataset(patients)

    if fill_missing:
        fill_cols = [
            c
            for c in df.columns
            if c not in ("PatientID", "timestamp", "in_hospital_death")
        ]
        df[fill_cols] = df.groupby("PatientID", sort=False)[fill_cols].ffill()
        df[fill_cols] = df[fill_cols].fillna(0.0)

    suffix = "_table_imputed" if fill_missing else "_table_not_imputed"
    out_path = os.path.join(DATA_DIR, f"{set_name.replace('-', '_')}{suffix}.parquet")
    df.to_parquet(out_path, index=False)

    rows, cols = df.shape
    logger.info(f"Saved {rows} rows x {cols} columns -> {out_path}")
    return out_path


def _calc_slope(times: list[float], values: list[float]) -> float | None:
    """Robust linear trend: slope of OLS fit through all (time, value) pairs.
    Returns None when fewer than 2 points are available."""
    if len(values) < 2:
        return None
    slope, *_ = linregress(times, values)
    return float(slope)


def _calc_auc(times: list[float], values: list[float]) -> float | None:
    """Time-weighted area under the curve via the trapezoidal rule.
    Returns None when fewer than 2 points are available."""
    if len(values) < 2:
        return None
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    return float(np.trapezoid(v, t))


def _calc_mean_crossings(values: list[float]) -> int | None:
    """Number of times the signal crosses its own mean.
    Returns None when fewer than 2 points are available."""
    if len(values) < 2:
        return None
    v = np.asarray(values, dtype=float)
    mean_val = v.mean()
    crossings = int(((v[:-1] - mean_val) * (v[1:] - mean_val) < 0).sum())
    return crossings


def _build_patient_vector(patient_id: str, patient: Patient) -> dict:
    """
    Aggregate a single patient's timeseries into one flat feature vector.

    For each time-series param: min / max / last / mean / slope / auc / crossings.
    Static fields and label are appended at the end.
    Missing values (fewer than 2 observations) are left as pd.NA.
    """

    param_obs: dict[str, list[tuple[float, float]]] = {}
    for tp in sorted(patient.timeseries, key=lambda x: x.time):
        t_hours = tp.time.total_seconds() / 3600.0
        for m in tp.measurements:
            val = float(m.value.value if hasattr(m.value, "value") else m.value)
            param_obs.setdefault(m.param.value, []).append((t_hours, val))

    row: dict = {"PatientID": patient_id}
    for param in ParamType:
        obs = param_obs.get(param.value, [])
        if obs:
            times = [t for t, _ in obs]
            vals = [v for _, v in obs]
            row[f"{param.value}_min"] = min(vals)
            row[f"{param.value}_max"] = max(vals)
            row[f"{param.value}_last"] = vals[-1]
            row[f"{param.value}_mean"] = sum(vals) / len(vals)
        else:
            times, vals = [], []
            row[f"{param.value}_min"] = pd.NA
            row[f"{param.value}_max"] = pd.NA
            row[f"{param.value}_last"] = pd.NA
            row[f"{param.value}_mean"] = pd.NA

        slope = _calc_slope(times, vals)
        row[f"{param.value}_slope"] = slope if slope is not None else pd.NA

        auc_val = _calc_auc(times, vals)
        row[f"{param.value}_auc"] = auc_val if auc_val is not None else pd.NA

        crossings = _calc_mean_crossings(vals)
        row[f"{param.value}_crossings"] = crossings if crossings is not None else pd.NA

    s = patient.static
    row["age"] = float(s.age) if s.age is not None else pd.NA
    row["height_cm"] = float(s.height_cm) if s.height_cm is not None else pd.NA
    row["weight_kg"] = float(s.weight_kg) if s.weight_kg is not None else pd.NA
    row["gender"] = (
        _GENDER_FLOAT.get(s.gender, pd.NA) if s.gender is not None else pd.NA
    )
    if s.in_hospital_death is None:
        raise ValueError(f"Patient {patient_id} has no in_hospital_death label.")
    row["in_hospital_death"] = float(s.in_hospital_death)
    return row


def _patient_to_triplets(patient_id: str, patient: Patient) -> list[dict]:
    """
    Convert one patient into a list of (t, variable_idx, z_*, v) event dicts.

    - Static fields are emitted as t=0 events (None fields are skipped).
    - TS measurements use their actual recording time scaled to [0, 1].
    - The list is sorted by t.
    """
    _var_to_idx: dict[str, int] = {v: i for i, v in enumerate(TRIPLET_VARS)}
    n_vars = len(TRIPLET_VARS)

    events: list[tuple[float, str, float]] = []  # (t, var_name, value)

    s = patient.static
    for field in STATIC_COLS:
        raw = getattr(s, field)
        if raw is None:
            continue
        v = _GENDER_FLOAT[raw] if isinstance(raw, Gender) else float(raw)
        events.append((0.0, field, v))

    for tp in patient.timeseries:
        t = tp.time.total_seconds() / 3600.0 / 48.0
        for m in tp.measurements:
            v = float(m.value.value if hasattr(m.value, "value") else m.value)
            events.append((t, m.param.value, v))

    events.sort(key=lambda e: e[0])

    if s.in_hospital_death is None:
        raise ValueError(f"Patient {patient_id} has no in_hospital_death label.")
    label = float(s.in_hospital_death)

    rows = []
    for t, var_name, v in events:
        idx = _var_to_idx[var_name]
        one_hot = [0.0] * n_vars
        one_hot[idx] = 1.0
        row: dict = {"PatientID": patient_id, "t": t, "variable_idx": idx, "v": v}
        for i, col in enumerate(TRIPLET_VARS):
            row[f"z_{col}"] = one_hot[i]
        row["in_hospital_death"] = label
        rows.append(row)

    return rows


def convert_to_triplet_format(set_name: str, patients: dict[str, Patient]) -> str:
    """
    Build the triplet representation proposed by Horn et al. and save as parquet.

    Each row is one measurement event for one patient:

      PatientID      — patient identifier
      t              — time scaled to [0, 1]  (0 = admission, 1 = 48 h)
      variable_idx   — integer category index into TRIPLET_VARS (0 … N-1)
      v              — scaled measurement value
      z_{var_name}   — one-hot columns, one per entry in TRIPLET_VARS
      in_hospital_death — label (same value repeated for every row of a patient)

    Static fields are included as t=0 events.
    Rows are sorted by t within each patient.
    Values are expected to be already scaled before calling this function.

    Returns the path of the saved parquet file.
    """
    logger.info(f"Building triplet format for {set_name} ({len(patients)} patients)...")
    all_rows: list[dict] = []
    for pid, patient in patients.items():
        all_rows.extend(_patient_to_triplets(pid, patient))

    df = pd.DataFrame(all_rows)

    z_cols = [f"z_{v}" for v in TRIPLET_VARS]
    col_order = ["PatientID", "t", "variable_idx", "v", *z_cols, "in_hospital_death"]
    df = df[col_order]

    out_path = os.path.join(DATA_DIR, f"{set_name.replace('-', '_')}_triplet.parquet")
    df.to_parquet(out_path, index=False)

    n_rows, n_cols = df.shape
    n_patients = df["PatientID"].nunique()
    avg_events = n_rows / n_patients
    logger.info(
        f"Saved {n_rows} rows x {n_cols} columns -> {out_path}  "
        f"(avg {avg_events:.1f} events/patient)"
    )
    return out_path


def convert_to_vector_format(
    set_name: str,
    patients: dict[str, Patient],
    fill_missing: bool = False,
) -> str:
    """
    Build a one-row-per-patient vector table (min/max/last/mean per TS param
    plus statics and label) and save it as a parquet file.

    If *fill_missing* is True, all pd.NA values are replaced with 0.0.
    Returns the path of the saved file.
    """
    logger.info(f"Building vector format for {set_name} ({len(patients)} patients)...")
    df = pd.DataFrame([_build_patient_vector(pid, p) for pid, p in patients.items()])
    if fill_missing:
        df = df.fillna(0.0)

    suffix = "_vector_imputed" if fill_missing else "_vector_not_imputed"
    out_path = os.path.join(DATA_DIR, f"{set_name.replace('-', '_')}{suffix}.parquet")
    df.to_parquet(out_path, index=False)

    rows, cols = df.shape
    logger.info(f"Saved {rows} rows x {cols} columns -> {out_path}")
    return out_path